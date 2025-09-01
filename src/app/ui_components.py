#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
UI components construction for IC-Custom application.
"""
import gradio as gr
from constants import (
    ASPECT_RATIO_LABELS, 
    DEFAULT_ASPECT_RATIO,
    DEFAULT_BRUSH_SIZE
)


def create_theme():
    """Create and configure the Gradio theme."""
    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )
    return theme


def create_css():
    """Create custom CSS for the application."""
    from stylesheets import get_css
    return get_css()


def create_header_section():
    """Create the header section with title and description."""
    from metainfo import head, getting_started
    
    with gr.Row():
        gr.HTML(head)

    with gr.Accordion(label="🚀 Getting Started:", open=True, elem_id="accordion"):
        with gr.Row(equal_height=True):
            gr.HTML(getting_started)


def create_customization_section():
    """Create the customization mode selection section."""
    with gr.Row():
        # Add a note to remind users to click Clear before starting
        md_custmization_mode = gr.Markdown(
            "1. Select a Customization Mode\n\n*Tip: Please click the Clear button first to reset all states before starting a new task.*"
        )
    with gr.Row():
        custmization_mode = gr.Radio(
            ["Position-aware", "Position-free"],
            value="Position-aware",
            scale=1,
            elem_id="customization_mode_radio",
            show_label=False,
            label="Customization Mode",
        )
    return custmization_mode, md_custmization_mode


def create_image_input_section():
    """Create image input section optimized for left column layout."""
    # Reference image section
    md_image_reference = gr.Markdown("2. Input reference image")
    with gr.Group():
        image_reference = gr.Image(
            label="Reference Image", 
            type="pil", 
            interactive=True,
            height=320,
            container=True,
            elem_id="reference_image"
        )
    
    # Input mask mode selection
    md_input_mask_mode = gr.Markdown("3. Select input mask mode")
    with gr.Group():
        input_mask_mode = gr.Radio(
            ["Precise mask", "User-drawn mask"],
            value="Precise mask",
            elem_id="input_mask_mode_radio",
            show_label=False,
            label="Input Mask Mode",
        )
    
    # Target image section
    md_target_image = gr.Markdown("4. Input target image & mask (Iterate clicking or brushing until the target is covered)")
    
    # Precise mask mode
    with gr.Group():
        image_target_1 = gr.Image(
            type="pil", 
            label="Target Image (precise mask)", 
            interactive=True, 
            visible=True,
            height=500,
            container=True,
            elem_id="target_image_1"
        )
        with gr.Row():
            undo_target_seg_button = gr.Button(
                'Undo seg', 
                elem_id="undo_btnSEG", 
                visible=True,
                size="sm",
                scale=1
            )

    # User-drawn mask mode
    with gr.Group():
        image_target_2 = gr.ImageEditor( 
            label="Target Image (user-drawn mask)",
            type="pil",
            brush=gr.Brush(colors=["#FFFFFF"], default_size=DEFAULT_BRUSH_SIZE, color_mode="fixed"),
            layers=False,
            interactive=True,
            sources=["upload", "clipboard"],
            placeholder="Please click here or the icon to upload the image.",
            visible=False,
            height=500,
            container=True,
            elem_id="target_image_2",
            fixed_canvas=True,
        )
    
    return (image_reference, input_mask_mode, image_target_1, image_target_2, 
            undo_target_seg_button, md_image_reference, md_input_mask_mode, md_target_image)


def create_prompt_section():
    """Create the text prompt input section with improved layout."""
    md_prompt = gr.Markdown("5. Input text prompt (optional)")
    with gr.Group():
        prompt = gr.Textbox(
            placeholder="Please input the description for the target scene.", 
            value="", 
            lines=2, 
            show_label=False, 
            label="Text Prompt",
            container=True,
            elem_id="text_prompt"
        )

        with gr.Row():
            vlm_generate_btn = gr.Button(
                "🤖 VLM Auto-generate", 
                scale=1, 
                elem_id="vlm_generate_btn",
                variant="secondary"
            )
            vlm_polish_btn = gr.Button(
                "✨ VLM Auto-polish", 
                scale=1, 
                elem_id="vlm_polish_btn",
                variant="secondary"
            )
    
    return prompt, vlm_generate_btn, vlm_polish_btn, md_prompt


def create_advanced_options_section():
    """Create the advanced options section."""
    with gr.Accordion("Advanced Options", open=False, elem_id="accordion1"):
        with gr.Group():
            aspect_ratio = gr.Dropdown(
                label="Output aspect ratio", 
                choices=ASPECT_RATIO_LABELS, 
                value=DEFAULT_ASPECT_RATIO,
                interactive=True,
                allow_custom_value=False,
                filterable=False,
                elem_id="aspect_ratio_dropdown"
            )

        with gr.Group():
            seg_ref_mode = gr.Radio(
                label="Segmentation mode", 
                choices=["Full Ref", "Masked Ref"], 
                value="Full Ref", 
                elem_id="seg_ref_mode_radio"
            )
            move_to_center = gr.Checkbox(label="Move object to center", value=False, elem_id="move_to_center_checkbox")
        
        with gr.Group():
            with gr.Row():
                use_background_preservation = gr.Checkbox(label="Use background preservation", value=False, elem_id="use_bg_preservation_checkbox")
                background_blend_threshold = gr.Slider(
                    label="Background blend threshold", 
                    minimum=0, 
                    maximum=1, 
                    step=0.1, 
                    value=0.5
                )

        with gr.Group():
            with gr.Row():        
                seed = gr.Slider(
                    label="Seed (-1 for random): ", 
                    minimum=-1, 
                    maximum=2147483647, 
                    step=1, 
                    value=-1, 
                    scale=4
                )
                
                num_images_per_prompt = gr.Slider(
                    label="Num samples", 
                    minimum=1, 
                    maximum=4, 
                    step=1, 
                    value=1, 
                    scale=1
                )
            
        with gr.Group():
            with gr.Row():
                guidance = gr.Slider(
                    label="Guidance scale",
                    minimum=10,
                    maximum=65,
                    step=1,
                    value=40,
                )
                num_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=60,
                    step=1,
                    value=32,
                )
            with gr.Row():
                true_gs = gr.Slider(
                    label="True GS",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=3,
                )
    
    return (aspect_ratio, seg_ref_mode, move_to_center, use_background_preservation,
            background_blend_threshold, seed, num_images_per_prompt, guidance, num_steps, true_gs)


def create_mask_operation_section():
    """Create mask operation section optimized for right column (outputs)."""
    md_mask_operation = gr.Markdown("6. View or modify the target mask")

    with gr.Group():
        # Mask gallery with responsive layout
        mask_gallery = gr.Gallery(
            label='Mask Preview', 
            show_label=False, 
            interactive=False,
            columns=2,
            rows=1,
            height="auto",
            object_fit="contain",
            preview=True,
            allow_preview=True,
            selected_index=0,
            elem_id="mask_gallery",
            elem_classes=["custom-gallery", "responsive-gallery"],
            container=True,
            show_fullscreen_button=False
        )
        
        # Mask operation buttons - horizontal layout
        with gr.Row():
            dilate_button = gr.Button(
                '🔍 Dilate', 
                elem_id="dilate_btn",
                variant="secondary",
                size="sm",
                scale=1
            )
            erode_button = gr.Button(
                '🔽 Erode', 
                elem_id="erode_btn",
                variant="secondary", 
                size="sm",
                scale=1
            )
            bounding_box_button = gr.Button(
                '📦 Bounding box', 
                elem_id="bounding_box_btn",
                variant="secondary",
                size="sm",
                scale=1
            )
    
    return mask_gallery, dilate_button, erode_button, bounding_box_button, md_mask_operation


def create_output_section():
    """Create the output section optimized for right column."""
    md_submit = gr.Markdown("7. Submit and view the output")
    
    # Generation controls at top for better workflow
    with gr.Group():
        with gr.Row():
            submit_button = gr.Button(
                "💫 Generate", 
                variant="primary", 
                scale=3,
                size="lg"
            )
            clear_btn = gr.ClearButton(
                scale=1,
                variant="secondary",
                value="🗑️ Clear"
            )
    
    # Results gallery with responsive layout
    with gr.Group():
        result_gallery = gr.Gallery(
            label='Generated Results', 
            show_label=False, 
            interactive=False,
            columns=1,
            rows=1, 
            height="auto",
            object_fit="contain",
            preview=True,
            allow_preview=True,
            selected_index=0,
            elem_id="result_gallery",
            elem_classes=["custom-gallery", "responsive-gallery"],
            container=True,
            show_fullscreen_button=False
        )
    
    return result_gallery, submit_button, clear_btn, md_submit


def create_examples_section(examples_list, inputs, outputs, fn):
    """Create the examples section with required arguments."""
    examples = gr.Examples(
        examples=examples_list,
        inputs=inputs,
        outputs=outputs,
        fn=fn,
        cache_examples=False,
        examples_per_page=10,
        run_on_click=True,
    )
    return examples


def create_citation_section():
    """Create the citation section."""
    from metainfo import citation
    
    with gr.Row():
        gr.Markdown(citation)
