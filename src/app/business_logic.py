#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Business logic functions for IC-Custom application.
"""
import numpy as np
import torch
import cv2
import gradio as gr
from PIL import Image
from datetime import datetime
import json
import os
from scipy.ndimage import binary_dilation, binary_erosion

from constants import (
    DEFAULT_BACKGROUND_BLEND_THRESHOLD, DEFAULT_SEED, DEFAULT_NUM_IMAGES,
    DEFAULT_GUIDANCE, DEFAULT_TRUE_GS, DEFAULT_NUM_STEPS, DEFAULT_ASPECT_RATIO,
    DEFAULT_DILATION_KERNEL_SIZE, DEFAULT_MARKER_SIZE, DEFAULT_MARKER_THICKNESS,
    DEFAULT_MASK_ALPHA, DEFAULT_COLOR_ALPHA, TIMESTAMP_FORMAT, SEGMENTATION_COLORS, SEGMENTATION_MARKERS
)

# Global holder for SAM mobile predictor injected from the app layer
MOBILE_PREDICTOR = None
BEN2_MODEL = None   # ben2 model injected from the app layer    

def set_mobile_predictor(predictor):
    """Inject SAM mobile predictor into this module without changing function signatures."""
    global MOBILE_PREDICTOR
    MOBILE_PREDICTOR = predictor

def set_ben2_model(ben2_model):
    """Inject ben2 model into this module without changing function signatures."""
    global BEN2_MODEL
    BEN2_MODEL = ben2_model

    
def init_image_target_1(target_image):
    """Initialize UI state when a target image is uploaded."""

    # Handle both PIL Image (image_target_1) and ImageEditor dict (image_target_2)
    try:
        if isinstance(target_image, dict) and 'composite' in target_image:
            # ImageEditor format (user-drawn mask)
            image_target_state = np.array(target_image['composite'].convert("RGB"))
        else:
            # PIL Image format (precise mask)
            image_target_state = np.array(target_image.convert("RGB"))
    except Exception as e:
        # If there's an error processing the image, skip initialization
        print(f"Warning: Error processing target image in init_image_target: {e}")
        return (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            gr.skip(), gr.skip(), gr.update(value="-1")
        )
    
    selected_points = []
    mask_target_state = None
    prompt = None
    mask_gallery = []
    result_gallery = []
    use_background_preservation = False
    background_blend_threshold = DEFAULT_BACKGROUND_BLEND_THRESHOLD
    seed = DEFAULT_SEED
    num_images_per_prompt = DEFAULT_NUM_IMAGES
    guidance = DEFAULT_GUIDANCE
    true_gs = DEFAULT_TRUE_GS
    num_steps = DEFAULT_NUM_STEPS
    aspect_ratio_val = gr.update(value=DEFAULT_ASPECT_RATIO)

    return (image_target_state, selected_points, mask_target_state, prompt, 
            mask_gallery, result_gallery, use_background_preservation, 
            background_blend_threshold, seed, num_images_per_prompt, guidance, 
            true_gs, num_steps, aspect_ratio_val)


def init_image_target_2(target_image):
    """Initialize UI state when a target image is uploaded."""
    # Handle both PIL Image (image_target_1) and ImageEditor dict (image_target_2)
    try:
        if isinstance(target_image, dict) and 'composite' in target_image:
            # ImageEditor format (user-drawn mask)
            image_target_state = np.array(target_image['composite'].convert("RGB"))
        else:
            # PIL Image format (precise mask)
            image_target_state = np.array(target_image.convert("RGB"))
    except Exception as e:
        # If there's an error processing the image, skip initialization
        print(f"Warning: Error processing target image in init_image_target: {e}")
        return (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            gr.skip(), gr.skip(), gr.update(value="-1")
        )
    
    selected_points = gr.skip()
    mask_target_state = gr.skip()
    prompt = gr.skip()
    mask_gallery = gr.skip()
    result_gallery = gr.skip()
    use_background_preservation = gr.skip()
    background_blend_threshold = gr.skip()
    seed = gr.skip()
    num_images_per_prompt = gr.skip()
    guidance = gr.skip()
    true_gs = gr.skip()
    num_steps = gr.skip()
    aspect_ratio_val = gr.skip()

    return (image_target_state, selected_points, mask_target_state, prompt, 
            mask_gallery, result_gallery, use_background_preservation, 
            background_blend_threshold, seed, num_images_per_prompt, guidance, 
            true_gs, num_steps, aspect_ratio_val)


def init_image_reference(image_reference):
    """Initialize all UI states when a reference image is uploaded."""
    image_reference_state = np.array(image_reference.convert("RGB"))
    image_reference_ori_state = image_reference_state
    image_reference_rmbg_state = None
    image_target_state = None
    mask_target_state = None
    prompt = None
    mask_gallery = []
    result_gallery = []
    image_target_1_val = None
    image_target_2_val = None
    selected_points = []
    input_mask_mode_val = gr.update(value="Precise mask")
    seg_ref_mode_val = gr.update(value="Full Ref")
    move_to_center = False
    use_background_preservation = False
    background_blend_threshold = DEFAULT_BACKGROUND_BLEND_THRESHOLD
    seed = DEFAULT_SEED
    num_images_per_prompt = DEFAULT_NUM_IMAGES
    guidance = DEFAULT_GUIDANCE
    true_gs = DEFAULT_TRUE_GS
    num_steps = DEFAULT_NUM_STEPS
    aspect_ratio_val = gr.update(value=DEFAULT_ASPECT_RATIO)

    return (
        image_reference_ori_state, image_reference_rmbg_state, image_target_state,
        mask_target_state, prompt, mask_gallery, result_gallery, image_target_1_val,
        image_target_2_val, selected_points, input_mask_mode_val, seg_ref_mode_val,
        move_to_center, use_background_preservation, background_blend_threshold,
        seed, num_images_per_prompt, guidance, true_gs, num_steps, aspect_ratio_val,
    )


def undo_seg_points(orig_img, sel_pix):
    """Remove the latest segmentation point and recompute the preview mask."""
    if len(sel_pix) != 0:
        temp = orig_img.copy()
        sel_pix.pop()
        # Online show seg mask
        if len(sel_pix) != 0:
            temp, output_mask = segmentation(temp, sel_pix, MOBILE_PREDICTOR, SEGMENTATION_COLORS, SEGMENTATION_MARKERS)
            output_mask_pil = Image.fromarray(output_mask.astype("uint8"))
            masked_img_pil = Image.fromarray(np.where(output_mask > 0, orig_img, 0).astype("uint8"))
            mask_gallery = [masked_img_pil, output_mask_pil]
        else:
            output_mask = None
            mask_gallery = []
        return temp.astype(np.uint8), output_mask, mask_gallery
    else:
        gr.Warning("Nothing to Undo")
        return orig_img, None, []


def segmentation(img, sel_pix, mobile_predictor, colors, markers):
    """Run SAM-based segmentation given selected points and return previews."""
    points = []
    labels = []
    for p, l in sel_pix:
        points.append(p)
        labels.append(l)
    
    mobile_predictor.set_image(img if isinstance(img, np.ndarray) else np.array(img))
    with torch.no_grad():
        masks, _, _ = mobile_predictor.predict(
            point_coords=np.array(points), 
            point_labels=np.array(labels), 
            multimask_output=False
        )

    output_mask = np.ones((masks.shape[1], masks.shape[2], 3)) * 255
    for i in range(3):
        output_mask[masks[0] == True, i] = 0.0

    mask_all = np.ones((masks.shape[1], masks.shape[2], 3))
    color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
        mask_all[masks[0] == True, i] = color_mask[i]
    
    masked_img = img / 255 * DEFAULT_MASK_ALPHA + mask_all * DEFAULT_COLOR_ALPHA
    masked_img = masked_img * 255
    
    # Draw points
    for point, label in sel_pix:
        cv2.drawMarker(
            masked_img, point, colors[label], 
            markerType=markers[label], 
            markerSize=DEFAULT_MARKER_SIZE, 
            thickness=DEFAULT_MARKER_THICKNESS
        )
    
    return masked_img, output_mask


def get_point(img, sel_pix, evt: gr.SelectData):
    """Handle a user click on the target image to add a foreground point."""
    if evt is None or not hasattr(evt, 'index'):
        gr.Warning(f"Event object missing index attribute. Event type: {type(evt)}")
        return img, None, []
    
    sel_pix.append((evt.index, 1))  # append the foreground_point
    # Online show seg mask
    global MOBILE_PREDICTOR
    masked_img_seg, output_mask = segmentation(img, sel_pix, MOBILE_PREDICTOR, SEGMENTATION_COLORS, SEGMENTATION_MARKERS)

    # Apply dilation to output_mask
    output_mask = 1 - output_mask
    kernel = np.ones((DEFAULT_DILATION_KERNEL_SIZE, DEFAULT_DILATION_KERNEL_SIZE), np.uint8)
    output_mask = cv2.dilate(output_mask, kernel, iterations=1)
    output_mask = 1 - output_mask

    output_mask_binary = output_mask / 255

    masked_img_seg = masked_img_seg.astype("uint8")
    output_mask = output_mask.astype("uint8")

    masked_img = img * output_mask_binary
    masked_img_pil = Image.fromarray(masked_img.astype("uint8"))
    output_mask_pil = Image.fromarray(output_mask.astype("uint8"))
    outputs_gallery = [masked_img_pil, output_mask_pil]

    return masked_img_seg, output_mask, outputs_gallery


def get_brush(img):
    """Extract a mask from ImageEditor brush layers or composite/background diff."""
    if img is None or not isinstance(img, dict):
        return gr.skip(), gr.skip()
    
    layers = img.get("layers", [])
    background = img.get('background', None)
    composite = img.get('composite', None)

    output_mask = None
    if layers and layers[0] is not None and background is not None:
        output_mask = 255 - np.array(layers[0].convert("RGB")).astype(np.uint8)
    elif composite is not None and background is not None:
        comp_rgb = np.array(composite.convert("RGB")).astype(np.int16)
        bg_rgb = np.array(background.convert("RGB")).astype(np.int16)
        diff = np.abs(comp_rgb - bg_rgb)
        painted = (diff.sum(axis=2) > 0).astype(np.uint8)
        output_mask = (1 - painted) * 255
        output_mask = np.repeat(output_mask[:, :, None], 3, axis=2).astype(np.uint8)
    else:
        return gr.skip(), gr.skip()

    if len(np.unique(output_mask)) == 1:
        return gr.skip(), gr.skip()

    img = np.array(background.convert("RGB")).astype(np.uint8)

    output_mask_binary = output_mask / 255
    masked_img = img * output_mask_binary
    masked_img_pil = Image.fromarray(masked_img.astype("uint8"))
    output_mask_pil = Image.fromarray(output_mask.astype("uint8"))
    mask_gallery = [masked_img_pil, output_mask_pil]
    
    return output_mask, mask_gallery


def random_mask_func(mask, dilation_type='square', dilation_size=20):
    """Utility to dilate/erode/box/ellipse expand a binary mask."""
    binary_mask = mask[:,:,0] < 128

    if dilation_type == 'square_dilation':
        structure = np.ones((dilation_size, dilation_size), dtype=bool)
        dilated_mask = binary_dilation(binary_mask, structure=structure)
    elif dilation_type == 'square_erosion':
        structure = np.ones((dilation_size, dilation_size), dtype=bool)
        dilated_mask = binary_erosion(binary_mask, structure=structure)
    elif dilation_type == 'bounding_box':
        # Find the most left top and left bottom point
        rows, cols = np.where(binary_mask)
        if len(rows) == 0 or len(cols) == 0:
            return mask  # return original mask if no valid points

        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        # Create a bounding box
        dilated_mask = np.zeros_like(binary_mask, dtype=bool)
        dilated_mask[min_row:max_row + 1, min_col:max_col + 1] = True

    elif dilation_type == 'bounding_ellipse':
        # Find the most left top and left bottom point
        rows, cols = np.where(binary_mask)
        if len(rows) == 0 or len(cols) == 0:
            return mask  # return original mask if no valid points

        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        # Calculate the center and axis length of the ellipse
        center = ((min_col + max_col) // 2, (min_row + max_row) // 2)
        a = (max_col - min_col) // 2  # half long axis
        b = (max_row - min_row) // 2  # half short axis

        # Create a bounding ellipse
        y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
        ellipse_mask = ((x - center[0])**2 / a**2 + (y - center[1])**2 / b**2) <= 1
        dilated_mask = np.zeros_like(binary_mask, dtype=bool)
        dilated_mask[ellipse_mask] = True
    else:
        raise ValueError("dilation_type must be 'square', 'ellipse', 'bounding_box', or 'bounding_ellipse'")

    # Use binary dilation
    dilated_mask = 1 - dilated_mask
    dilated_mask = np.uint8(dilated_mask[:,:,np.newaxis]) * 255
    dilated_mask = np.concatenate([dilated_mask, dilated_mask, dilated_mask], axis=2)
    return dilated_mask


def dilate_mask(mask, image):
    """Dilate the target mask for robustness and preview the result."""
    if mask is None:
        gr.Warning("Please input the target mask first")
        return None, None
    
    mask = random_mask_func(mask, dilation_type='square_dilation', dilation_size=DEFAULT_DILATION_KERNEL_SIZE)
    masked_img = image * (mask > 0)
    return mask, [masked_img, mask]


def erode_mask(mask, image):
    """Erode the target mask and preview the result."""
    if mask is None:
        gr.Warning("Please input the target mask first")
        return None, None
    
    mask = random_mask_func(mask, dilation_type='square_erosion', dilation_size=DEFAULT_DILATION_KERNEL_SIZE)
    masked_img = image * (mask > 0)
    return mask, [masked_img, mask]


def bounding_box(mask, image):
    """Create bounding box mask and preview the result."""
    if mask is None:
        gr.Warning("Please input the target mask first")
        return None, None
    
    mask = random_mask_func(mask, dilation_type='bounding_box', dilation_size=DEFAULT_DILATION_KERNEL_SIZE)
    masked_img = image * (mask > 0)
    return mask, [masked_img, mask]


def change_input_mask_mode(input_mask_mode, custmization_mode):
    """Change visibility of input mask mode components."""

    if custmization_mode == "Position-free":
        return (
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(visible=False),
            )
    elif input_mask_mode.lower() == "precise mask":
        return (
            gr.update(visible=True), 
            gr.update(visible=False), 
            gr.update(visible=True), 
            )
    elif input_mask_mode.lower() == "user-drawn mask":
        return (
            gr.update(visible=False), 
            gr.update(visible=True), 
            gr.update(visible=False), 
            )
    else:
        gr.Warning("Invalid input mask mode")
        return (
            gr.skip(), gr.skip(), gr.skip()
            )

def change_custmization_mode(custmization_mode, input_mask_mode):
    """Change visibility and interactivity based on customization mode."""


    if custmization_mode.lower() == "position-free":
        return (gr.update(interactive=False, visible=False),
                gr.update(interactive=False, visible=False),
                gr.update(interactive=False, visible=False),
                gr.update(interactive=False, visible=False),
                gr.update(interactive=False, visible=False),
                gr.update(interactive=False, visible=False),
                gr.update(value="<s>Select a input mask mode</s>", visible=False),
                gr.update(value="<s>Input target image & mask (Iterate clicking or brushing until the target is covered)</s>", visible=False),
                gr.update(value="<s>View or modify the target mask</s>", visible=False),
                gr.update(value="3. Input text prompt (necessary)"),
                gr.update(value="4. Submit and view the output"),
                gr.update(visible=False),
                gr.update(visible=False),
                
                )
    else:
        if input_mask_mode.lower() == "precise mask":
            return (gr.update(interactive=True, visible=True),
                    gr.update(interactive=True, visible=False),
                    gr.update(interactive=True, visible=True),
                    gr.update(interactive=True, visible=True),
                    gr.update(interactive=True, visible=True),
                    gr.update(interactive=True, visible=True),
                    gr.update(value="3. Select a input mask mode", visible=True),
                    gr.update(value="4. Input target image & mask (Iterate clicking or brushing until the target is covered)", visible=True),
                    gr.update(value="6. View or modify the target mask", visible=True),
                    gr.update(value="5. Input text prompt (optional)", visible=True),
                    gr.update(value="7. Submit and view the output", visible=True),
                    gr.update(visible=True, value="Precise mask"),
                    gr.update(visible=True),
                    )
        elif input_mask_mode.lower() == "user-drawn mask":
            return (gr.update(interactive=True, visible=False),
                    gr.update(interactive=True, visible=True),
                    gr.update(interactive=False, visible=False),
                    gr.update(interactive=True, visible=True),
                    gr.update(interactive=True, visible=True),
                    gr.update(interactive=True, visible=True),
                    gr.update(value="3. Select a input mask mode", visible=True),
                    gr.update(value="4. Input target image & mask (Iterate clicking or brushing until the target is covered)", visible=True),
                    gr.update(value="6. View or modify the target mask", visible=True),
                    gr.update(value="5. Input text prompt (optional)", visible=True),
                    gr.update(value="7. Submit and view the output", visible=True),
                    gr.update(visible=True, value="User-drawn mask"),
                    gr.update(visible=True),
                    )



def change_seg_ref_mode(seg_ref_mode, image_reference_state, move_to_center):
    """Change segmentation reference mode and handle background removal."""
    if image_reference_state is None:
        gr.Warning("Please upload the reference image first")
        return None, None
    
    global BEN2_MODEL
    
    if seg_ref_mode == "Full Ref":
        return image_reference_state, None
    else:
        if BEN2_MODEL is None:
            gr.Warning("Please enable ben2 for mask reference first")
            return gr.skip(), gr.skip()

        image_reference_pil = Image.fromarray(image_reference_state)
        image_reference_pil_rmbg = BEN2_MODEL.inference(image_reference_pil, move_to_center=move_to_center)
        image_reference_rmbg = np.array(image_reference_pil_rmbg)
        return image_reference_rmbg, image_reference_rmbg


def vlm_auto_generate(image_target_state, image_reference_state, mask_target_state, 
                      custmization_mode, vlm_processor, vlm_model, device,
                      construct_vlm_gen_prompt, run_vlm):
    """Auto-generate prompt using VLM."""
    if custmization_mode == "Position-aware":
        if image_target_state is None or mask_target_state is None:
            gr.Warning("Please upload the target image and get mask first")
            return None
    
    if image_reference_state is None:
        gr.Warning("Please upload the reference image first")
        return None

    if vlm_processor is None:
        gr.Warning("Please enable vlm for prompt first")
        return None

    messages = construct_vlm_gen_prompt(image_target_state, image_reference_state, mask_target_state, custmization_mode)
    output_text = run_vlm(vlm_processor, vlm_model, messages, device)
    return output_text


def vlm_auto_polish(prompt, custmization_mode, vlm_processor, vlm_model, device,
                     construct_vlm_polish_prompt, run_vlm):
    """Auto-polish prompt using VLM."""
    if prompt is None:
        gr.Warning("Please input the text prompt first")
        return None

    if custmization_mode == "Position-aware":
        gr.Warning("Polishing only works in position-free mode")
        return prompt
    
    if vlm_processor is None:
        gr.Warning("Please enable vlm for prompt first")
        return None

    messages = construct_vlm_polish_prompt(prompt)
    output_text = run_vlm(vlm_processor, vlm_model, messages, device)
    return output_text


def save_results(output_img, image_reference, image_target, mask_target, prompt,
                custmization_mode, input_mask_mode, seg_ref_mode, seed, guidance,
                num_steps, num_images_per_prompt, use_background_preservation,
                background_blend_threshold, true_gs, assets_cache_dir):
    """Save generated results and metadata."""
    save_name = datetime.now().strftime(TIMESTAMP_FORMAT)
    results = []
    
    for i in range(num_images_per_prompt):
        save_dir = os.path.join(assets_cache_dir, save_name)
        os.makedirs(save_dir, exist_ok=True)

        output_img[i].save(os.path.join(save_dir, f"img_gen_{i}.png"))
        image_reference.save(os.path.join(save_dir, f"img_ref_{i}.png"))
        image_target.save(os.path.join(save_dir, f"img_target_{i}.png"))
        mask_target.save(os.path.join(save_dir, f"mask_target_{i}.png"))

        with open(os.path.join(save_dir, f"hyper_params_{i}.json"), "w") as f:
            json.dump({
                "prompt": prompt,
                "custmization_mode": custmization_mode,
                "input_mask_mode": input_mask_mode,
                "seg_ref_mode": seg_ref_mode,
                "seed": seed,
                "guidance": guidance,
                "num_steps": num_steps,
                "num_images_per_prompt": num_images_per_prompt,
                "use_background_preservation": use_background_preservation,
                "background_blend_threshold": background_blend_threshold,
                "true_gs": true_gs,
            }, f)

        results.append(output_img[i])

    return results
