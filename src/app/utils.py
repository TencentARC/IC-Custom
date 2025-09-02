import os
import sys
import base64
from io import BytesIO
from typing import Optional

from PIL import Image

import numpy as np
import torch

from segment_anything import SamPredictor, sam_model_registry
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from huggingface_hub import hf_hub_download

sys.path.append(os.getcwd())
import BEN2


## Ordinary function
def resize(image: Image.Image, 
            target_width: int, 
            target_height: int,
            interpolate: Image.Resampling = Image.Resampling.LANCZOS,
            return_type: str = "pil") -> Image.Image | np.ndarray:
    """
    Crops and resizes an image while preserving the aspect ratio.

    Args:
        image (Image.Image): Input PIL image to be cropped and resized.
        target_width (int): Target width of the output image.
        target_height (int): Target height of the output image.
        interpolate (Image.Resampling): The interpolation method.
        return_type (str): The type of the output image.

    Returns:
        Image.Image: Cropped and resized image.
    """
    # Original dimensions
    resized_image = image.resize((target_width, target_height), interpolate)
    if return_type == "pil":
        return resized_image
    elif return_type == "np":
        return np.array(resized_image)
    else:
        raise ValueError(f"Invalid return type: {return_type}")


def resize_long_edge(
    image: Image.Image, 
    long_edge_size: int, 
    interpolate: Image.Resampling = Image.Resampling.LANCZOS,
    return_type: str = "pil"
    ) -> np.ndarray | Image.Image:
    """
    Resize the long edge of the image to the long_edge_size.

    Args:
        image (Image.Image): The image to resize.
        long_edge_size (int): The size of the long edge.
        interpolate (Image.Resampling): The interpolation method.

    Returns:
        np.ndarray: The resized image.
    """
    w, h = image.size
    scale_ratio = long_edge_size / max(h, w)
    output_w = int(w * scale_ratio)
    output_h = int(h * scale_ratio)
    image = resize(image, target_width=int(output_w), target_height=int(output_h), interpolate=interpolate, return_type=return_type)
    return image


def ensure_divisible_by_value(
    image: Image.Image | np.ndarray, 
    value: int = 8, 
    interpolate: Image.Resampling = Image.Resampling.NEAREST,
    return_type: str = "np"
    ) -> np.ndarray | Image.Image:
    """
    Ensure the image dimensions are divisible by value.

    Args:
        image_pil (Image.Image): The image to ensure divisible by value.
        value (int): The value to ensure divisible by.
        interpolate (Image.Resampling): The interpolation method.
        return_type (str): The type of the output image.

    Returns:
        np.ndarray | Image.Image: The resized image.
    """

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    w, h = image.size

    w = (w // value) * value
    h = (h // value) * value
    image = resize(image, w, h, interpolate=interpolate, return_type=return_type)
    return image


def resize_paired_image(
    image_reference: np.ndarray, 
    image_target: np.ndarray, 
    mask_target: np.ndarray,
    force_resize_long_edge: bool = False,
    return_type: str = "np"
    ) -> tuple[np.ndarray | Image.Image, np.ndarray | Image.Image, np.ndarray | Image.Image]:

    if isinstance(image_reference, np.ndarray):
        image_reference = Image.fromarray(image_reference)
    if isinstance(image_target, np.ndarray):
        image_target = Image.fromarray(image_target)
    if isinstance(mask_target, np.ndarray):
        mask_target = Image.fromarray(mask_target)

    if force_resize_long_edge:
        image_reference = resize_long_edge(image_reference, 1024, interpolate=Image.Resampling.LANCZOS, return_type=return_type)
        image_target = resize_long_edge(image_target, 1024, interpolate=Image.Resampling.LANCZOS, return_type=return_type)
        mask_target = resize_long_edge(mask_target, 1024, interpolate=Image.Resampling.NEAREST, return_type=return_type)

    if isinstance(image_reference, Image.Image):
        ref_width, ref_height = image_reference.size
        target_width, target_height = image_target.size
    else:
        ref_height, ref_width = image_reference.shape[:2]
        target_width, target_height = image_target.shape[:2]

    # resize the ref image to the same height as the target image and ensure the ratio remains the same
    if ref_height != target_height:
        scale_ratio = target_height / ref_height
        image_reference = resize(image_reference, int(ref_width * scale_ratio), target_height, interpolate=Image.Resampling.LANCZOS, return_type=return_type)

    if return_type == "pil": 
        image_reference = Image.fromarray(image_reference) if isinstance(image_reference, np.ndarray) else image_reference
        image_target = Image.fromarray(image_target) if isinstance(image_target, np.ndarray) else image_target
        mask_target = Image.fromarray(mask_target) if isinstance(mask_target, np.ndarray) else mask_target
        return image_reference, image_target, mask_target
    else:
        image_reference = np.array(image_reference) if isinstance(image_reference, Image.Image) else image_reference
        image_target = np.array(image_target) if isinstance(image_target, Image.Image) else image_target
        mask_target = np.array(mask_target) if isinstance(mask_target, Image.Image) else mask_target
        return image_reference, image_target, mask_target


def prepare_input_images(
    img_ref: np.ndarray,
    custmization_mode: str,
    img_target: Optional[np.ndarray] = None,
    mask_target: Optional[np.ndarray] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    force_resize_long_edge: bool = False,
    return_type: str = "np"
    ) -> tuple[np.ndarray | Image.Image, np.ndarray | Image.Image, np.ndarray | Image.Image]:


    if custmization_mode.lower() == "position-free":
        img_target = np.ones_like(img_ref) * 255
        mask_target = np.zeros_like(img_ref)

    if isinstance(width, int) and isinstance(height, int):
        img_ref = resize(Image.fromarray(img_ref), width, height, interpolate=Image.Resampling.LANCZOS, return_type=return_type)
        img_target = resize(Image.fromarray(img_target), width, height, interpolate=Image.Resampling.LANCZOS, return_type=return_type)
        mask_target = resize(Image.fromarray(mask_target), width, height, interpolate=Image.Resampling.NEAREST, return_type=return_type)
    else:
        img_ref, img_target, mask_target = resize_paired_image(img_ref, img_target, mask_target, force_resize_long_edge, return_type=return_type)

    img_ref = ensure_divisible_by_value(img_ref, value=16, interpolate=Image.Resampling.LANCZOS, return_type=return_type)
    img_target = ensure_divisible_by_value(img_target, value=16, interpolate=Image.Resampling.LANCZOS, return_type=return_type)
    mask_target = ensure_divisible_by_value(mask_target, value=16, interpolate=Image.Resampling.NEAREST, return_type=return_type)

    return img_ref, img_target, mask_target


def get_mask_type_ids(custmization_mode: str, input_mask_mode: str) -> int:
    if custmization_mode.lower() == "position-free":
        return torch.tensor([0])
    elif custmization_mode.lower() == "position-aware":
        if "precise" in input_mask_mode.lower():
            return torch.tensor([1])
        else:
            return torch.tensor([2])
    else: 
        raise ValueError(f"Invalid custmization mode: {custmization_mode}")


def scale_image(image_np, is_mask: bool = False):
    """
    Scale the image to the range of [-1, 1] if not a mask, otherwise scale to [0, 1].

    Args:
        image_np (np.ndarray): Input image.
        is_mask (bool): Whether the image is a mask.
    Returns:
        np.ndarray: Scaled image.
    """
    if is_mask:
        image_np = image_np / 255.0
    else:
        image_np = image_np / 255.0
        image_np = image_np * 2 - 1
    return image_np


def get_sam_predictor(sam_ckpt_path, device):
    """
    Get the SAM predictor.
    Args:
        sam_ckpt_path (str): The path to the SAM checkpoint.
        device (str): The device to load the model on.
    Returns:
        SamPredictor: The SAM predictor.
    """
    if not os.path.exists(sam_ckpt_path):
        sam_ckpt_path = hf_hub_download(repo_id="HCMUE-Research/SAM-vit-h", filename="sam_vit_h_4b8939.pth")

    sam = sam_model_registry['vit_h'](checkpoint=sam_ckpt_path).to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    return predictor


def image_to_base64(img):
    """
    Convert an image to a base64 string.
    Args:
        img (PIL.Image.Image): The image to convert.
    Returns:
        str: The base64 string.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def get_vlm(vlm_ckpt_path, device, torch_dtype):
    """
    Get the VLM pipeline.
    Args:
        vlm_ckpt_path (str): The path to the VLM checkpoint.
        device (str): The device to load the model on.
        torch_dtype (torch.dtype): The data type of the model.
    Returns:
        tuple: The processor and model.
    """
    if not os.path.exists(vlm_ckpt_path):
        vlm_ckpt_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    vlm_ckpt_path, torch_dtype=torch_dtype).to(device)
    processor = AutoProcessor.from_pretrained(vlm_ckpt_path)


    return processor, model


def construct_vlm_gen_prompt(image_target, image_reference, target_mask, custmization_mode):
    """
    Construct the VLM generation prompt.
    Args:
        image_target (np.ndarray): The target image.
        image_reference (np.ndarray): The reference image.
        target_mask (np.ndarray): The target mask.
        custmization_mode (str): The customization mode.
    Returns:
        list: The messages.
    """
    if custmization_mode.lower() == "position-free":
        image_reference_pil = Image.fromarray(image_reference.astype(np.uint8))
        image_reference_base_64 = image_to_base64(image_reference_pil)
        messages = [
            {
                "role": "system", 
                "content": "I will input a reference image. Please identify the main subject/object in this image and generate a new description by placing this subject in a completely different scene or context. For example, if the reference image shows a rabbit sitting in a garden surrounded by green leaves and roses, you could generate a description like 'The rabbit is sitting on a rocky cliff overlooking a serene ocean, with the sun setting behind it, casting a warm glow over the scene'. Please directly output the new description without explaining your thought process. The description should not exceed 256 tokens."
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{image_reference_base_64}"
                    },
                ],
            }
        ]
        return messages
    else:
        image_reference_pil = Image.fromarray(image_reference.astype(np.uint8))
        image_reference_base_64 = image_to_base64(image_reference_pil)

        target_mask_binary = target_mask > 127.5
        masked_image_target = image_target * target_mask_binary
        masked_image_target_pil = Image.fromarray(masked_image_target.astype(np.uint8))
        masked_image_target_base_64 = image_to_base64(masked_image_target_pil)
        

        messages = [
            {
                "role": "system", 
                "content": "I will input a reference image and a target image with its main subject area masked (in black). Please directly describe the scene where the main subject/object from the reference image is placed into the masked area of the target image. Focus on describing the final combined scene, making sure to clearly describe both the object from the reference image and the background/environment from the target image. For example, if the reference shows a white cat with orange stripes on a beach and the target shows a masked area in a garden with blooming roses and tulips, directly describe 'A white cat with orange stripes sits elegantly among the vibrant red roses and yellow tulips in the lush garden, surrounded by green foliage.' The description should not exceed 256 tokens."
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{image_reference_base_64}"
                    },
                    {
                        "type": "image",
                        "image": f"data:image;base64,{masked_image_target_base_64}"
                    }
                ],
            }
        ]
        return messages


def construct_vlm_polish_prompt(prompt):
    """
    Construct the VLM polish prompt.
    Args:
        prompt (str): The prompt to polish.
    Returns:
        list: The messages.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can polish the text prompt to make it more specific, detailed, and complete. Please directly output the polished prompt without explaining your thought process. The prompt should not exceed 256 tokens."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    return messages


def run_vlm(vlm_processor, vlm_model, messages, device):
    """
    Run the VLM.
    Args:
        vlm_processor (torch.nn.Module): The VLM processor.
        vlm_model (torch.nn.Module): The VLM model.
        messages (list): The messages.
        device (str): The device to run the model on.
    Returns:
        str: The output text.
    """
    text = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    # Inference
    generated_ids = vlm_model.generate(**inputs, do_sample=True, num_beams=4, temperature=1.5, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text


def get_ben2_model(ben2_model_path, device):
    """
    Get the BEN2 model.
    Args:
        ben2_model_path (str): The path to the BEN2 model.
        device (str): The device to load the model on.
    Returns:
        BEN2: The BEN2 model.
    """
    if not os.path.exists(ben2_model_path):
        ben2_model_path = hf_hub_download(repo_id="PramaLLC/BEN2", filename="BEN2_Base.pth")

    ben2_model = BEN2.BEN_Base().to(device)
    ben2_model.loadcheckpoints(model_path=ben2_model_path)
    return ben2_model


def make_dict_img_mask(img_path, mask_path):
    """
    Make a dictionary of the image and mask for gr.ImageEditor.
    Keep interface, not used in the gradio app.
    Args:
        img_path (str): The path to the image.
        mask_path (str): The path to the mask.
    Returns:
        dict: The dictionary of the image and mask.
    """
    from PIL import ImageOps
    background = Image.open(img_path).convert("RGBA")
    layers = [
        Image.merge("RGBA", (
            Image.new("L", Image.open(mask_path).size, 255),  # R channel
            Image.new("L", Image.open(mask_path).size, 255),  # G channel
            Image.new("L", Image.open(mask_path).size, 255),  # B channel
            ImageOps.invert(Image.open(mask_path).convert("L"))  # Inverted alpha channel
        ))
    ]
    # Combine layers with background by replacing the alpha channel
    background = np.array(background.convert("RGB"))
    _, _, _, layer_alpha = layers[0].split()
    layer_alpha = np.array(layer_alpha)[:,:,np.newaxis]
    composite = background * (1 - (layer_alpha > 0)) + np.ones_like(background) * (layer_alpha > 0) * 255

    
    composite = Image.fromarray(composite.astype("uint8")).convert("RGBA")
    return {
        'background': background,
        'layers': layers,
        'composite': composite
    }