import argparse
import os
from datetime import datetime
from typing import Any, Optional

from omegaconf import OmegaConf
from PIL import Image

import torch

from ic_custom.pipelines.ic_custom_pipeline import ICCustomPipeline


def ensure_divisible_by_value(image_pil: Image.Image, value: int = 8, interpolate: Image.Resampling = Image.Resampling.LANCZOS) -> Image.Image:
    """
    Ensure the image dimensions are divisible by value.

    Args:
        image_pil (Image.Image): The image to ensure divisible by value.
        value (int): The value to ensure divisible by.
    """

    w, h = image_pil.size
    w = (w // value) * value
    h = (h // value) * value
    image_pil = image_pil.resize((w, h), interpolate)
    return image_pil

def resize_paired_image(
    reference_image: Image.Image, 
    target_image: Image.Image, 
    mask_target: Image.Image, 
    ) -> tuple[Image.Image, Image.Image, Image.Image]:

    ref_w, ref_h = reference_image.size
    target_w, target_h = target_image.size

    # resize the ref image to the same height as the target image and ensure the ratio remains the same
    if ref_h != target_h:
        scale_ratio = target_h / ref_h
        reference_image = reference_image.resize((int(ref_w * scale_ratio), target_h), interpolate=Image.Resampling.LANCZOS)

    #  Ensure the image dimensions are divisible by 16.
    reference_image = ensure_divisible_by_value(reference_image, value=16, interpolate=Image.Resampling.LANCZOS)
    target_image = ensure_divisible_by_value(target_image, value=16, interpolate=Image.Resampling.LANCZOS)
    mask_target = ensure_divisible_by_value(mask_target, value=16, interpolate=Image.Resampling.NEAREST)

    return reference_image, target_image, mask_target


def prepare_input_images(
    img_ref_path: str,
    img_target_path: Optional[str] = None,
    mask_target_path: Optional[str] = None,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:

    img_ref = Image.open(img_ref_path)

    # Initialize img_target as a pure white image and mask_target as a pure black image, same size as reference
    img_target = Image.new("RGB", img_ref.size, (255, 255, 255))
    mask_target = Image.new("RGB", img_ref.size, (0, 0, 0))

    if img_target_path is not None:
        img_target = Image.open(img_target_path)
    if mask_target_path is not None:
        mask_target = Image.open(mask_target_path)

    img_ref, img_target, mask_target = resize_paired_image(img_ref, img_target, mask_target)

    return img_ref, img_target, mask_target


def get_mask_type_ids(mask_type: str) -> int:
    if mask_type.lower() == "pos-free":
        return 0
    elif mask_type.lower() == "pos-aware-precise":
        return 1
    elif mask_type.lower() == "pos-aware-drawn":
        return 2
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")


def concat_image(
    img_ref: Image.Image,
    img_target: Image.Image,
    output_img: Image.Image,
) -> Image.Image:
    concat_img = Image.new("RGB", (img_ref.width + img_target.width + output_img.width, output_img.height))
    concat_img.paste(img_ref, (0, 0))
    concat_img.paste(img_target, (img_ref.width, 0))
    concat_img.paste(output_img, (img_ref.width + img_target.width, 0))

    return concat_img


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="IC-Custom Inference.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=False,
        help="Hugging Face token",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        required=False,
        default=os.path.expanduser("~/.cache/huggingface/hub"),
        help="Cache directory to save the models, default is ~/.cache/huggingface/hub",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    if args.hf_token is not None:
        os.environ["HF_TOKEN"] = args.hf_token

    if args.hf_cache_dir is not None:
        os.environ["HF_HUB_CACHE"] = args.hf_cache_dir

    inference_config = cfg.inference_config
    model_config = cfg.model_config
    checkpoint_config = cfg.checkpoint_config

    os.makedirs(inference_config.output_dir, exist_ok=True)

    # import ipdb; ipdb.set_trace()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    pipeline = ICCustomPipeline(
        clip_path=checkpoint_config.clip_path,
        t5_path=checkpoint_config.t5_path,
        siglip_path=checkpoint_config.siglip_path,
        ae_path=checkpoint_config.ae_path,
        dit_path=checkpoint_config.dit_path,
        redux_path=checkpoint_config.redux_path,
        lora_path=checkpoint_config.lora_path,
        img_txt_in_path=checkpoint_config.img_txt_in_path,
        boundary_embeddings_path=checkpoint_config.boundary_embeddings_path,
        task_register_embeddings_path=checkpoint_config.task_register_embeddings_path,
        network_alpha=model_config.network_alpha,
        double_blocks_idx=model_config.double_blocks,
        single_blocks_idx=model_config.single_blocks,
        device=device,
        weight_dtype=weight_dtype,
    )
    pipeline.set_pipeline_offload(inference_config.offload)
    pipeline.set_show_progress(inference_config.show_progress)

    img_ref, img_target, mask_target = prepare_input_images(
        img_ref_path=inference_config.img_ref_path,
        img_target_path=inference_config.img_target_path,
        mask_target_path=inference_config.mask_target_path,
    )
    img_ip = img_ref.copy()

    width, height = img_target.size[0] + img_ref.size[0], img_target.size[1]

    cond_w_regions = [img_ref.size[0]]
    mask_type_ids = get_mask_type_ids(inference_config.custom_mask_type)


    with torch.no_grad():
        output_img: Image.Image = pipeline(
            prompt=inference_config.prompt,
            width=width,
            height=height,
            guidance=inference_config.guidance,
            num_steps=inference_config.num_steps,
            seed=inference_config.seed,
            img_ref=img_ref,
            img_target=img_target,
            mask_target=mask_target,
            img_ip=img_ip,
            cond_w_regions=cond_w_regions,
            mask_type_ids=mask_type_ids,
            use_background_preservation=inference_config.use_background_preservation,
            use_progressive_background_preservation=inference_config.use_progressive_background_preservation,
            background_blend_threshold=inference_config.background_blend_threshold,
            true_gs=inference_config.true_gs,
            neg_prompt=inference_config.neg_prompt,
        )[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_name = inference_config.img_ref_path.split("/")[-2]
        output_dir = os.path.join(inference_config.output_dir, save_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{timestamp}_tgt.png")
        output_img.save(output_path)

        concat_img = concat_image(img_ref, img_target, output_img)
        concat_img.save(os.path.join(output_dir, f"{timestamp}_concat.png"))

        print("The output image is saved to: ", f"{output_dir}/{timestamp}_tgt.png")


if __name__ == "__main__":
    main()


