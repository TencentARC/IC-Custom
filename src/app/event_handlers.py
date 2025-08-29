#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Event handlers for IC-Custom application.
"""
import gradio as gr


def setup_event_handlers(
    # UI components
    input_mask_mode, image_target_1, image_target_2, undo_target_seg_button,
    custmization_mode, dilate_button, erode_button, bounding_box_button,
    mask_gallery, md_input_mask_mode, md_target_image, md_mask_operation,
    md_prompt, md_submit, result_gallery, image_target_state, mask_target_state,
    seg_ref_mode, image_reference_ori_state, move_to_center,
    image_reference, image_reference_rmbg_state,
    # Functions
    change_input_mask_mode, change_custmization_mode, change_seg_ref_mode,
    init_image_target_1,  init_image_target_2, init_image_reference,
    get_point, undo_seg_points, get_brush,
    # VLM buttons (UI components)
    vlm_generate_btn, vlm_polish_btn,
    # VLM functions
    vlm_auto_generate, vlm_auto_polish,
    dilate_mask, erode_mask, bounding_box,
    run_model,
    # Other components
    selected_points, prompt,
    use_background_preservation, background_blend_threshold, seed,
    num_images_per_prompt, guidance, true_gs, num_steps, aspect_ratio,
    submit_button,
    # extra state
    eg_idx,
):
    """Setup all event handlers for the application."""
    
    # Change input mask mode: precise mask or user-drawn mask
    input_mask_mode.change(
        change_input_mask_mode,
        [input_mask_mode, custmization_mode],
        [image_target_1, image_target_2, undo_target_seg_button]
    )

    # Change customization mode: pos-aware or pos-free
    custmization_mode.change(
        change_custmization_mode,
        [custmization_mode, input_mask_mode],
        [image_target_1, image_target_2, undo_target_seg_button, dilate_button, 
         erode_button, bounding_box_button, md_input_mask_mode, 
         md_target_image, md_mask_operation, md_prompt, md_submit, input_mask_mode, mask_gallery]
    )

    # Remove background for reference image
    seg_ref_mode.change(
        change_seg_ref_mode,
        [seg_ref_mode, image_reference_ori_state, move_to_center],
        [image_reference, image_reference_rmbg_state]
    )

    # Initialize components only on user upload (not programmatic updates)
    image_target_1.upload(
        init_image_target_1,
        [image_target_1],
        [image_target_state, selected_points, prompt, mask_target_state, mask_gallery, 
         result_gallery, use_background_preservation, background_blend_threshold, seed, 
         num_images_per_prompt, guidance, true_gs, num_steps, aspect_ratio]
    )

    image_target_2.upload(
        init_image_target_2,
        [image_target_2],
        [image_target_state, selected_points, prompt, mask_target_state, mask_gallery, 
         result_gallery, use_background_preservation, background_blend_threshold, seed, 
         num_images_per_prompt, guidance, true_gs, num_steps, aspect_ratio]
    )

    image_reference.upload(
        init_image_reference,
        [image_reference],
        [image_reference_ori_state, image_reference_rmbg_state, image_target_state, 
         mask_target_state, prompt, mask_gallery, result_gallery, image_target_1, 
         image_target_2, selected_points, input_mask_mode, seg_ref_mode, move_to_center,
         use_background_preservation, background_blend_threshold, seed, 
         num_images_per_prompt, guidance, true_gs, num_steps, aspect_ratio]
    )

    # SAM for image_target_1
    image_target_1.select(
        get_point,
        [image_target_state, selected_points],
        [image_target_1, mask_target_state, mask_gallery],
    )

    undo_target_seg_button.click(
        undo_seg_points,
        [image_target_state, selected_points],
        [image_target_1, mask_target_state, mask_gallery]
    )

    # Brush for image_target_2
    image_target_2.change(
        get_brush,
        [image_target_2],
        [mask_target_state, mask_gallery],
    )

    # VLM auto generate
    vlm_generate_btn.click(
        vlm_auto_generate,
        [image_target_state, image_reference_ori_state, mask_target_state, custmization_mode],
        [prompt]
    )

    # VLM auto polish
    vlm_polish_btn.click(
        vlm_auto_polish,
        [prompt, custmization_mode],
        [prompt]
    )

    # Mask operations
    dilate_button.click(
        dilate_mask,
        [mask_target_state, image_target_state],
        [mask_target_state, mask_gallery]
    )

    erode_button.click(
        erode_mask,
        [mask_target_state, image_target_state],
        [mask_target_state, mask_gallery]
    )

    bounding_box_button.click(
        bounding_box,
        [mask_target_state, image_target_state],
        [mask_target_state, mask_gallery]
    )

    # Run function
    ips = [
        image_target_state, mask_target_state, image_reference_ori_state,
        image_reference_rmbg_state, prompt, seed, guidance, true_gs, num_steps,
        num_images_per_prompt, use_background_preservation, background_blend_threshold,
        aspect_ratio, custmization_mode, seg_ref_mode, input_mask_mode,
    ]

    submit_button.click(
        fn=run_model,
        inputs=ips,
        outputs=[result_gallery, seed, prompt],
        show_progress=True,
    )


