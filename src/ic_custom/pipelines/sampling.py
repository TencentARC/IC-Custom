import math
import sys
from typing import Callable, List, Optional, Union

import torch
from einops import rearrange, repeat
from torch import Tensor

from ..models.model import Flux
from ..modules.conditioner import HFEmbedder
from ..modules.image_embedders import ReduxImageEncoder

# -------------------------------------------------------------------------
# Progress bar
# -------------------------------------------------------------------------
import time

TGT_PREFIX = "[TARGET-SCENE]"


def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    """
    Simple progress bar for console output, with elapsed and estimated remaining time.

    Args:
        iteration: Current iteration (Int)
        total: Total iterations (Int)
        prefix: Prefix string (Str)
        suffix: Suffix string (Str)
        length: Bar length (Int)
        fill: Bar fill character (Str)
    """
    # Static variable to store start time
    if not hasattr(print_progress_bar, "_start_time") or iteration == 0:
        print_progress_bar._start_time = time.time()

    percent = f"{100 * (iteration / float(total)):.1f}%"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    elapsed = time.time() - print_progress_bar._start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    if iteration > 0:
        avg_time_per_iter = elapsed / iteration
        remaining = avg_time_per_iter * (total - iteration)
    else:
        remaining = 0
    remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining))

    time_info = f"Elapsed: {elapsed_str} | ETA: {remaining_str}"

    sys.stdout.write(f'\r{prefix} |{bar}| {percent} {suffix} {time_info}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

# -------------------------------------------------------------------------
# 1) sampling func
# -------------------------------------------------------------------------
def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
):
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):

    noise = torch.cat(
        [torch.randn(
        1,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed+i),
        ) 
        for i in range(num_samples)
    ],
    dim=0
    )
    return noise
    


# -------------------------------------------------------------------------
# prepare input func
# -------------------------------------------------------------------------
def _get_batch_size_and_prompt(prompt, img_shape):
    """
    Helper to determine batch size and prompt list.
    """
    bs, c, h, w = img_shape
    is_prompt_none = prompt is None
    return bs, prompt, is_prompt_none, h, w

def _make_img_ids(bs, h, w, device=None, dtype=None):
    """
    Helper to create image ids tensor.
    """
    img_ids = torch.zeros(h // 2, w // 2, 3, device=device, dtype=dtype)
    img_ids[..., 1] = torch.arange(h // 2)[:, None]
    img_ids[..., 2] = torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    return img_ids


def prepare(
    t5: HFEmbedder, 
    clip: HFEmbedder, 
    img: Tensor, 
    prompt: Union[str, List[str], None],
    num_images_per_prompt: int = 1,
):
    """
    Prepare the regular input for the Diffusion Transformer.
    """
    img_bs, prompt, is_prompt_none, h, w = _get_batch_size_and_prompt(prompt, img.shape)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    img_ids = _make_img_ids(img_bs, h, w, device=img.device, dtype=img.dtype)

    if isinstance(prompt, str):
        prompt = [prompt]

    txt_bs = len(prompt)
    if not is_prompt_none:
        prompt = [TGT_PREFIX + p for p in prompt]
        txt = t5(prompt)
        txt_ids = torch.zeros(txt_bs, txt.shape[1], 3, device=img.device, dtype=img.dtype)
        txt_vec = clip(prompt)
    else:
        
        
        txt = torch.zeros(txt_bs, 512, 4096, device=img.device, dtype=img.dtype)
        txt_ids = torch.zeros(txt_bs, 512, 3, device=img.device, dtype=img.dtype)
        txt_vec = torch.zeros(txt_bs, 768, device=img.device, dtype=img.dtype)


    if num_images_per_prompt > 1:
        txt = txt.repeat_interleave(num_images_per_prompt, dim=0)
        txt_ids = txt_ids.repeat_interleave(num_images_per_prompt, dim=0)
        txt_vec = txt_vec.repeat_interleave(num_images_per_prompt, dim=0)


    return {
        "img": img.to(img.device),
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "txt_vec": txt_vec.to(img.device),
    }


def prepare_with_redux(
    t5: HFEmbedder, 
    clip: HFEmbedder,
    image_encoder: ReduxImageEncoder,
    img: Tensor,
    img_ip: Tensor,
    prompt: Union[str, List[str], None],
    num_images_per_prompt: int = 1,
):
    img_bs, prompt, is_prompt_none, h, w = _get_batch_size_and_prompt(prompt, img.shape)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    img_ids = _make_img_ids(img_bs, h, w, device=img.device, dtype=img.dtype)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt_bs = len(prompt)

    if not is_prompt_none:
        prompt = [TGT_PREFIX + p for p in prompt]
        txt = torch.cat((t5(prompt), image_encoder(img_ip)), dim=1)
        txt_ids = torch.zeros(txt_bs, txt.shape[1], 3, device=img.device, dtype=img.dtype)
        txt_vec = clip(prompt)
    else:
        txt = torch.zeros(txt_bs, 512, 4096, device=img.device, dtype=img.dtype)
        txt_ids = torch.zeros(txt_bs, 512, 3, device=img.device, dtype=img.dtype)
        txt_vec = torch.zeros(txt_bs, 768, device=img.device, dtype=img.dtype)

    if num_images_per_prompt > 1:
        txt = txt.repeat_interleave(num_images_per_prompt, dim=0)
        txt_ids = txt_ids.repeat_interleave(num_images_per_prompt, dim=0)
        txt_vec = txt_vec.repeat_interleave(num_images_per_prompt, dim=0)

    return {
        "img": img.to(img.device),
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "txt_vec": txt_vec.to(img.device),
    }



def prepare_image_cond(
    ae,
    img_ref, 
    img_target,
    mask_target,
    dtype,
    device,
    num_images_per_prompt: int = 1,
):
    batch_size, _, _, _ = img_target.shape

    # Apply mask to target image
    mask_targeted_img = img_target * mask_target

    if mask_target.shape[1] == 3:
        mask_target = mask_target[:, 0 : 1, :, :]

    with torch.no_grad():
        autoencoder_dtype = next(ae.parameters()).dtype
        # Encode masked target image to latent space
        mask_targeted_latent = ae.encode(mask_targeted_img.to(autoencoder_dtype)).to(dtype)
        # Encode reference image to latent space
        reference_latent = ae.encode(img_ref.to(autoencoder_dtype)).to(dtype)

    # Repeat reference latent if batch size > 1
    if reference_latent.shape[0] == 1 and batch_size > 1:
        reference_latent = repeat(reference_latent, "1 ... -> bs ...", bs=batch_size)

    # Concatenate reference and target latents
    latent_concat = torch.cat((reference_latent, mask_targeted_latent), dim=-1)
    # Pack latents into 2x2 patches
    latent_packed = rearrange(latent_concat, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    # Create reference mask (all ones)
    reference_mask = torch.ones_like(img_ref)
    if reference_mask.shape[1] == 3:
        reference_mask = reference_mask[:, 0 : 1, :, :]

    # Concatenate reference and target masks
    mask_concat = torch.cat((reference_mask, mask_target), dim=-1)
    # Pack masks into 16x16 patches for image conditioning
    mask_16x16 = rearrange(mask_concat, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=16, pw=16)

    # Interpolate masks to latent space dimensions
    mask_latent = torch.nn.functional.interpolate(mask_concat, size=(latent_concat.shape[2] // 2, latent_concat.shape[3] // 2), mode='nearest')

    # Pack interpolated masks into 1x1 patches for mask conditioning
    mask_cond = rearrange(mask_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=1, pw=1)
    # Combine packed latents and masks for image conditioning
    img_cond = torch.cat((latent_packed, mask_16x16), dim=-1)

    if num_images_per_prompt > 1:
        img_cond = img_cond.repeat_interleave(num_images_per_prompt, dim=0)
        mask_cond = mask_cond.repeat_interleave(num_images_per_prompt, dim=0)
        latent_packed = latent_packed.repeat_interleave(num_images_per_prompt, dim=0)

    return {
        "img_cond": img_cond.to(device).to(dtype),
        "mask_cond": mask_cond.to(device).to(dtype),
        "img_latent": latent_packed.to(device).to(dtype),
    }


# -------------------------------------------------------------------------
# 2) denoise func
# -------------------------------------------------------------------------
def is_even_step(step: int) -> bool:
    """Check if the current step is odd."""
    return (step % 2 == 0)

def denoise(
    model, 
    img, 
    img_ids, 
    txt, 
    txt_ids, 
    txt_vec, 
    timesteps, 
    guidance: float = 4.0, 
    img_cond: Tensor = None,  
    mask_cond: Tensor = None, 
    img_latent: Tensor = None, 
    cond_w_regions: Optional[Union[List[int], int]] = None,
    mask_type_ids: Optional[Union[Tensor, int]] = None,
    height: int = 1024,
    width: int = 1024,
    use_background_preservation: bool = False,
    use_progressive_background_preservation: bool = True,
    background_blend_threshold: float = 0.8,
    true_gs: float = 1,
    timestep_to_start_cfg: int = 0,
    neg_txt: Tensor = None,
    neg_txt_ids: Tensor = None,
    neg_txt_vec: Tensor = None,
    show_progress: bool = False,
    use_flash_attention: bool = False,
    gradio_progress=None,
    ):
    do_true_cfg = true_gs > 1 and neg_txt is not None

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    v_gt = img - img_latent
    num_steps = len(timesteps[:-1])
    
    
    for step, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        if show_progress:
            print_progress_bar(step, num_steps, prefix='Denoising:', suffix=f'Step {step+1}/{num_steps}')
        
        # Update Gradio progress if available
        if gradio_progress is not None:
            # Map denoise progress to 0.2-0.8 range (since 0.0-0.2 is preprocessing, 0.8-1.0 is postprocessing)
            progress_value = (step / num_steps)
            gradio_progress(progress_value, desc=f"Denoising step {step+1}/{num_steps}")

        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        model_dtype = list(model.parameters())[0].dtype
        
        pred = model(
            img=torch.cat((img.to(model_dtype), img_cond.to(model_dtype)), dim=-1) if img_cond is not None else img.to(model_dtype),
            img_ids=img_ids.to(model_dtype),
            txt=txt.to(model_dtype),
            txt_ids=txt_ids.to(model_dtype),
            txt_vec=txt_vec.to(model_dtype),
            timesteps=t_vec.to(model_dtype),
            guidance=guidance_vec.to(model_dtype),
            cond_w_regions=cond_w_regions,
            mask_type_ids=mask_type_ids,
            height=height,
            width=width,
            use_flash_attention=use_flash_attention,
        )

        if do_true_cfg and step >= timestep_to_start_cfg:
            neg_perd = model(
                img=torch.cat((img.to(model_dtype), img_cond.to(model_dtype)), dim=-1) if img_cond is not None else img.to(model_dtype),
                img_ids=img_ids.to(model_dtype),
                txt=neg_txt.to(model_dtype),
                txt_ids=neg_txt_ids.to(model_dtype),
                txt_vec=neg_txt_vec.to(model_dtype),
                timesteps=t_vec.to(model_dtype),
                guidance=guidance_vec.to(model_dtype),
                cond_w_regions=cond_w_regions,
                mask_type_ids=mask_type_ids,
                height=height,
                width=width,
                use_flash_attention=use_flash_attention,
            )
            pred = neg_perd + true_gs * (pred - neg_perd)

        if use_background_preservation:
            is_early_phase = step <= num_steps * background_blend_threshold
            
            if is_early_phase:
                if use_progressive_background_preservation:
                    if is_even_step(step):
                        # Apply mask blending on odd steps in early phase
                        masked_latent = pred * (1 - mask_cond) + v_gt * mask_cond
                    else:
                        # Use prediction directly for even steps or late phase
                        masked_latent = pred
                else:
                    masked_latent = pred * (1 - mask_cond) + v_gt * mask_cond
            else:
                # Use prediction directly for even steps or late phase
                masked_latent = pred
                
            img = img + (t_prev - t_curr) * masked_latent
        else:
            img = img + (t_prev - t_curr) * pred

    if show_progress:
        print_progress_bar(num_steps, num_steps, prefix='Denoising:', suffix='Complete')

    return img