
import re
from typing import List, Optional, Union

import PIL
from PIL import Image
from einops import rearrange
from torch import Tensor

import numpy as np
import torch
from safetensors.torch import load_file as load_sft

from diffusers.image_processor import VaeImageProcessor

from ..modules.layers import (
    SingleStreamBlockLoraProcessor,
    DoubleStreamBlockLoraProcessor,
)
from ..pipelines.sampling import denoise, prepare_image_cond, get_noise, get_schedule, prepare, prepare_with_redux, unpack
from ..utils.model_utils import (
    load_ae,
    load_clip,
    load_ic_custom,
    load_t5,
    load_redux
)


PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]


class ICCustomPipeline:
    def __init__(
        self, 
        clip_path: str = "clip-vit-large-patch14", 
        t5_path: str = "t5-v1_1-xxl", 
        siglip_path: str = "siglip-so400m-patch14-384", 
        ae_path: str = "flux-fill-dev", 
        dit_path: str = "flux1-fill-dev", 
        redux_path: str = "flux1-redux-dev", 
        lora_path: str = "dit_lora_0x1561",
        img_txt_in_path: str = "dit_txt_img_in_0x1561",
        boundary_embeddings_path: str = "dit_boundary_embeddings_0x1561",
        task_register_embeddings_path: str = "dit_task_register_embeddings_0x1561",
        network_alpha: int = None,
        double_blocks_idx: str = None,
        single_blocks_idx: str = None,
        device: torch.device = torch.device("cuda"), 
        offload: bool = False,
        weight_dtype: torch.dtype = torch.bfloat16,
        show_progress: bool = False,
        use_flash_attention: bool = False,
):
        self.device = device
        self.offload = offload
        self.weight_dtype = weight_dtype
        
        self.clip = load_clip(clip_path, self.device if not offload else "cpu", dtype=self.weight_dtype)
        self.t5 = load_t5(t5_path, self.device if not offload else "cpu", max_length=512, dtype=self.weight_dtype)

        self.ae = load_ae(ae_path, device="cpu" if offload else self.device)
      
        self.model = load_ic_custom(dit_path, device="cpu" if offload else self.device, dtype=self.weight_dtype)

        self.image_encoder = load_redux(redux_path, siglip_path, device="cpu" if offload else self.device, dtype=self.weight_dtype)

        self.vae_scale_factor = 8

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.mask_processor = VaeImageProcessor(resample="nearest", do_normalize=False)

        self.set_lora(lora_path, network_alpha, double_blocks_idx, single_blocks_idx)
        self.set_img_txt_in(img_txt_in_path)
        self.set_boundary_embeddings(boundary_embeddings_path)
        self.set_task_register_embeddings(task_register_embeddings_path)

        self.model.to(self.device)

        self.show_progress = show_progress
        self.use_flash_attention = use_flash_attention

    def set_show_progress(self, show_progress: bool):
        self.show_progress = show_progress

    def set_use_flash_attention(self, use_flash_attention: bool):
        self.use_flash_attention = use_flash_attention

    def set_pipeline_offload(self, offload: bool):
        self.offload = offload

    def set_pipeline_gradient_checkpointing(self, enable: bool):
        def _recursive_set_gradient_checkpointing(module):
            self.model._set_gradient_checkpointing(module, enable)
            for child in module.children():
                _recursive_set_gradient_checkpointing(child)
    
        _recursive_set_gradient_checkpointing(self.model)

    def get_lora_rank(self, weights):
        for k in weights.keys():
            if k.endswith(".down.weight"):
                return weights[k].shape[0]

    def load_model_weights(self, weights: dict, strict: bool = False):
        model_state_dict = self.model.state_dict()
        update_dict = {k: v for k, v in weights.items() if k in model_state_dict}
        missing_keys = [k for k in weights if k not in model_state_dict]
        assert len(missing_keys) == 0, f"Some keys in the file are not found in the model: {missing_keys}"
        self.model.load_state_dict(update_dict, strict=strict)

    def set_lora(
        self, 
        lora_path: str = None, 
        network_alpha: int = None, 
        double_blocks_idx: str = None, 
        single_blocks_idx: str = None,
        ):
        assert lora_path is not None, "lora_path is required"
        weights = load_sft(lora_path)
        self.update_model_with_lora(weights, network_alpha, double_blocks_idx, single_blocks_idx)

    def update_model_with_lora(
        self, 
        weights, 
        network_alpha, 
        double_blocks_idx, 
        single_blocks_idx,
        ):
        rank = self.get_lora_rank(weights)
        network_alpha = network_alpha if network_alpha is not None else rank
        lora_attn_procs = {}

        if double_blocks_idx is None:
            double_blocks_idx = []
        else:
            double_blocks_idx = [int(idx) for idx in double_blocks_idx.split(",")]

        if single_blocks_idx is None:
            single_blocks_idx = list(range(38))
        else:
            single_blocks_idx = [int(idx) for idx in single_blocks_idx.split(",")]
            
        for name, attn_processor in self.model.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))

            if name.startswith("double_blocks") and layer_index in double_blocks_idx:
                # print("setting LoRA Processor for", name)
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
                    dim=3072, rank=rank, network_alpha=network_alpha
                )
            elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
                # print("setting LoRA Processor for", name)
                lora_attn_procs[name] = SingleStreamBlockLoraProcessor(
                    dim=3072, rank=rank, network_alpha=network_alpha
                )
            else:
                lora_attn_procs[name] = attn_processor

        self.model.set_attn_processor(lora_attn_procs)

        self.load_model_weights(weights, strict=False)

    def set_img_txt_in(self, img_txt_in_path: str):
        assert img_txt_in_path is not None, "img_txt_in_path is required"
        weights = load_sft(img_txt_in_path)
        self.load_model_weights(weights, strict=False)

    def set_boundary_embeddings(self, boundary_embeddings_path: str):
        assert boundary_embeddings_path is not None, "boundary_embeddings_path is required"
        weights = load_sft(boundary_embeddings_path)
        self.load_model_weights(weights, strict=False)

    def set_task_register_embeddings(self, task_register_embeddings_path: str):
        assert task_register_embeddings_path is not None, "task_register_embeddings_path is required"
        weights = load_sft(task_register_embeddings_path)
        self.load_model_weights(weights, strict=False)

    def offload_model_to_cpu(self, *models):
        for model in models:
            if model is not None:
                model.to("cpu")

    def prepare_image(
        self,
        image,
        device,
        dtype,
        width=None,
        height=None,
    ):
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image = image.to(device=device, dtype=dtype)
        return image

    def prepare_mask(
        self,
        mask,
        device,
        dtype,
        width: int = None,
        height: int = None,
    ):
        if isinstance(mask, torch.Tensor):
            pass
        else:
            mask = self.mask_processor.preprocess(mask, height=height, width=width)

        mask = mask.to(device=device, dtype=dtype)
        return mask

    def __call__(
        self,
        prompt: Union[str, List[str], None],
        width: int = 512,
        height: int = 512,
        guidance: float = 4,
        num_steps: int = 50,
        seed: int = 123456789,
        true_gs: float = 1,
        neg_prompt: Optional[Union[str, List[str], None]] = None,
        timestep_to_start_cfg: int = 0,
        img_ref: Optional[PipelineImageInput] = None,
        img_target: Optional[PipelineImageInput] = None,
        mask_target: Optional[PipelineImageInput] = None,
        img_ip: Optional[PipelineImageInput] = None,
        cond_w_regions: Optional[Union[List[int], int]] = None,
        mask_type_ids: Optional[Union[Tensor, int]] = None,
        use_background_preservation: bool = False,
        use_progressive_background_preservation: bool = True,
        background_blend_threshold: float = 0.8,
        num_images_per_prompt: int = 1,
        gradio_progress=None,
        ):

        width = 16 * (width // 16)
        height = 16 * (height // 16)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        img_ref = self.prepare_image(
            img_ref, 
            self.device, 
            self.weight_dtype,
        )
        img_target = self.prepare_image(
            img_target, 
            self.device, 
            self.weight_dtype,
        )
        mask_target = self.prepare_mask(
            mask_target, 
            self.device, 
            self.weight_dtype,
        )
        if num_images_per_prompt > 1:
            mask_type_ids = mask_type_ids.repeat_interleave(num_images_per_prompt, dim=0)

        return self.forward(
            batch_size,
            num_images_per_prompt,
            prompt,
            width,
            height,
            guidance,
            num_steps,
            seed,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            neg_prompt=neg_prompt,
            img_ref=img_ref,
            img_target=img_target,
            mask_target=mask_target,
            img_ip=img_ip,
            cond_w_regions=cond_w_regions,
            mask_type_ids=mask_type_ids,
            use_background_preservation=use_background_preservation,
            use_progressive_background_preservation=use_progressive_background_preservation,
            background_blend_threshold=background_blend_threshold,
            gradio_progress=gradio_progress,
        )

    def forward(
        self,
        batch_size,
        num_images_per_prompt,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        timestep_to_start_cfg,
        true_gs,
        neg_prompt,
        img_ref,
        img_target,
        mask_target,
        img_ip,
        cond_w_regions,
        mask_type_ids,
        use_background_preservation,
        use_progressive_background_preservation,
        background_blend_threshold,
        gradio_progress=None,
    ):
        has_neg_prompt = neg_prompt is not None 
        do_true_cfg = true_gs > 1 and has_neg_prompt

        x = get_noise(
            batch_size * num_images_per_prompt, height, width, device=self.device,
            dtype=self.weight_dtype, seed=seed
        )
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        timesteps = get_schedule(
            num_steps,
            image_seq_len,
            shift=True,
        )

        with torch.no_grad():
            if self.offload:
                self.t5, self.clip, self.image_encoder = self.t5.to(self.device), self.clip.to(self.device), self.image_encoder.to(self.device)

            if self.image_encoder is not None:
                inp_cond = prepare_with_redux(t5=self.t5, clip=self.clip, image_encoder=self.image_encoder, img=x, img_ip=img_ip, prompt=prompt, num_images_per_prompt=num_images_per_prompt)
            else:
                inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt, num_images_per_prompt=num_images_per_prompt)


            neg_inp_cond = None
            if do_true_cfg:
                if self.image_encoder is not None:
                    neg_inp_cond = prepare_with_redux(t5=self.t5, clip=self.clip, image_encoder=self.image_encoder, img=x, img_ip=img_ip, prompt=neg_prompt, num_images_per_prompt=num_images_per_prompt)
                else:
                    neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt, num_images_per_prompt=num_images_per_prompt)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip, self.image_encoder)
                self.model = self.model.to(self.device)
                self.ae.encoder = self.ae.encoder.to(self.device)

            inp_img_cond = prepare_image_cond(
                ae=self.ae,  
                img_ref=img_ref, 
                img_target=img_target,
                mask_target=mask_target,
                dtype=self.weight_dtype,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                )
          
            x = denoise(
                self.model,
                img=inp_cond['img'],
                img_ids=inp_cond['img_ids'],
                txt=inp_cond['txt'],
                txt_ids=inp_cond['txt_ids'],
                txt_vec=inp_cond['txt_vec'],
                timesteps=timesteps,
                guidance=guidance,
                img_cond=inp_img_cond['img_cond'],
                mask_cond=inp_img_cond['mask_cond'],
                img_latent=inp_img_cond['img_latent'],
                cond_w_regions=cond_w_regions,
                mask_type_ids=mask_type_ids,
                height=height,
                width=width,
                use_background_preservation=use_background_preservation,
                use_progressive_background_preservation=use_progressive_background_preservation,
                background_blend_threshold=background_blend_threshold,
                true_gs=true_gs,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'] if neg_inp_cond is not None else None,
                neg_txt_ids=neg_inp_cond['txt_ids'] if neg_inp_cond is not None else None,
                neg_txt_vec=neg_inp_cond['txt_vec'] if neg_inp_cond is not None else None,
                show_progress=self.show_progress,
                use_flash_attention=self.use_flash_attention,
                gradio_progress=gradio_progress,
            )
                

            if self.offload:
                self.offload_model_to_cpu(self.model, self.ae.encoder)
                self.ae.decoder = self.ae.decoder.to(x.device)

            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)

            if self.offload:
                self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1, "b c h w -> b h w c")

        output_imgs_target = []
        for i in range(x1.shape[0]):
            output_img = Image.fromarray((127.5 * (x1[i] + 1.0)).cpu().byte().numpy())
            img_target_height, img_target_width = img_target.shape[2], img_target.shape[3]
            output_img_target = output_img.crop((
                output_img.width - img_target_width,
                output_img.height - img_target_height,
                output_img.width,
                output_img.height
            ))
            output_imgs_target.append(output_img_target)

        return output_imgs_target

