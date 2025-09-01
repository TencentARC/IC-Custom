#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
IC-Custom Gradio Application

This module defines the UI and glue logic to run the IC-Custom pipeline
via Gradio. The code aims to keep UI text user-friendly while keeping the
implementation readable and maintainable.
"""
import os
import sys
import numpy as np
import torch
import gradio as gr

from PIL import Image
import time

# Add current directory to path for imports
sys.path.append(os.getcwd() + '/src/app')

# Import modular components
from config import parse_args, load_config, setup_environment
from ui_components import (
    create_theme, create_css, create_header_section, create_customization_section,
    create_image_input_section, create_prompt_section, create_advanced_options_section,
    create_mask_operation_section, create_output_section, create_examples_section,
    create_citation_section
)
from event_handlers import setup_event_handlers
from business_logic import (
    init_image_target_1, init_image_target_2, init_image_reference,
    undo_seg_points, segmentation, get_point, get_brush,
    dilate_mask, erode_mask, bounding_box,
    change_input_mask_mode, change_custmization_mode, change_seg_ref_mode,
    vlm_auto_generate, vlm_auto_polish, save_results, set_mobile_predictor,
    set_ben2_model,
)

# Import other dependencies
from utils import (
    get_sam_predictor, get_vlm, construct_vlm_gen_prompt, 
    construct_vlm_polish_prompt, run_vlm, get_ben2_model, 
    prepare_input_images, get_mask_type_ids
)
from examples import GRADIO_EXAMPLES, MASK_TGT, IMG_GEN
from ic_custom.pipelines.ic_custom_pipeline import ICCustomPipeline


def initialize_models(args, cfg, device, weight_dtype):
    """Initialize all required models."""
    # Load IC-Custom pipeline
    pipeline = ICCustomPipeline(
        clip_path=cfg.checkpoint_config.clip_path if os.path.exists(cfg.checkpoint_config.clip_path) else "clip-vit-large-patch14",
        t5_path=cfg.checkpoint_config.t5_path if os.path.exists(cfg.checkpoint_config.t5_path) else "t5-v1_1-xxl",
        siglip_path=cfg.checkpoint_config.siglip_path if os.path.exists(cfg.checkpoint_config.siglip_path) else "siglip-so400m-patch14-384",
        ae_path=cfg.checkpoint_config.ae_path if os.path.exists(cfg.checkpoint_config.ae_path) else "flux-fill-dev-ae",
        dit_path=cfg.checkpoint_config.dit_path if os.path.exists(cfg.checkpoint_config.dit_path) else "flux-fill-dev-dit",
        redux_path=cfg.checkpoint_config.redux_path if os.path.exists(cfg.checkpoint_config.redux_path) else "flux1-redux-dev",
        lora_path=cfg.checkpoint_config.lora_path if os.path.exists(cfg.checkpoint_config.lora_path) else "dit_lora_0x1561",
        img_txt_in_path=cfg.checkpoint_config.img_txt_in_path if os.path.exists(cfg.checkpoint_config.img_txt_in_path) else "dit_txt_img_in_0x1561",
        boundary_embeddings_path=cfg.checkpoint_config.boundary_embeddings_path if os.path.exists(cfg.checkpoint_config.boundary_embeddings_path) else "dit_boundary_embeddings_0x1561",
        task_register_embeddings_path=cfg.checkpoint_config.task_register_embeddings_path if os.path.exists(cfg.checkpoint_config.task_register_embeddings_path) else "dit_task_register_embeddings_0x1561",
        network_alpha=cfg.model_config.network_alpha,
        double_blocks_idx=cfg.model_config.double_blocks,
        single_blocks_idx=cfg.model_config.single_blocks,
        device=device,
        weight_dtype=weight_dtype,
    )
    pipeline.set_pipeline_offload(True)
    pipeline.set_show_progress(True)

    # Load SAM predictor
    mobile_predictor = get_sam_predictor(cfg.checkpoint_config.sam_path, device)

    # Load VLM if enabled
    vlm_processor, vlm_model = None, None
    if args.enable_vlm_for_prompt:
        vlm_processor, vlm_model = get_vlm(
            cfg.checkpoint_config.vlm_path,
            device=device,
            torch_dtype=weight_dtype,
        )

    # Load BEN2 model if enabled
    ben2_model = None
    if args.enable_ben2_for_mask_ref:
        ben2_model = get_ben2_model(cfg.checkpoint_config.ben2_path, device)

    return pipeline, mobile_predictor, vlm_processor, vlm_model, ben2_model


@torch.no_grad()
def run_model(
    image_target_state, mask_target_state, image_reference_ori_state,
    image_reference_rmbg_state, prompt, seed, guidance, true_gs, num_steps,
    num_images_per_prompt, use_background_preservation, background_blend_threshold,
    aspect_ratio, custmization_mode, seg_ref_mode, input_mask_mode,
    pipeline, assets_cache_dir,
    progress=gr.Progress()
):
    """Run IC-Custom pipeline with current UI state and return images."""
    start_ts = time.time()
    progress(0, desc="Starting generation...")
    # Select reference image and check inputs
    if seg_ref_mode == "Masked Ref":
        image_reference_state = image_reference_rmbg_state
    else:
        image_reference_state = image_reference_ori_state

    if image_reference_state is None:
        gr.Warning('Please upload the reference image')
        return None, seed, gr.update(placeholder="Last Input: " + prompt, value="")

    if image_target_state is None and custmization_mode != "Position-free":
        gr.Warning('Please upload the target image and mask it')
        return None, seed, gr.update(placeholder="Last Input: " + prompt, value="")

    if custmization_mode == "Position-aware" and mask_target_state is None:
        gr.Warning('Please select/draw the target mask')
        return None, seed, gr.update(placeholder=prompt, value="")

   
    mask_type_ids = get_mask_type_ids(custmization_mode, input_mask_mode)
    
    from constants import ASPECT_RATIO_TEMPLATE
    output_w, output_h = ASPECT_RATIO_TEMPLATE[aspect_ratio]
    image_reference, image_target, mask_target = prepare_input_images(
        image_reference_state, custmization_mode, image_target_state, mask_target_state,
        width=output_w, height=output_h,
        force_resize_long_edge="long edge" in aspect_ratio,
        return_type="pil"
    )

    gr.Info(f"Output WH resolution: {image_target.size[0]}px x {image_target.size[1]}px")
    # Run the model
    if seed == -1:
        seed = torch.randint(0, 2147483647, (1,)).item()

    width, height = image_target.size[0] + image_reference.size[0], image_target.size[1]

    output_img = pipeline(
        prompt=prompt, width=width, height=height, guidance=guidance,
        num_steps=num_steps, seed=seed, img_ref=image_reference,
        img_target=image_target, mask_target=mask_target, img_ip=image_reference,
        cond_w_regions=[image_reference.size[0]], mask_type_ids=mask_type_ids,
        use_background_preservation=use_background_preservation,
        background_blend_threshold=background_blend_threshold, true_gs=true_gs,
        neg_prompt="worst quality, normal quality, low quality, low res, blurry,",
        num_images_per_prompt=num_images_per_prompt,
        gradio_progress=progress,
    )
    
    # Save results
    results = save_results(
        output_img, image_reference, image_target, mask_target, prompt,
        custmization_mode, input_mask_mode, seg_ref_mode, seed, guidance,
        num_steps, num_images_per_prompt, use_background_preservation,
        background_blend_threshold, true_gs, assets_cache_dir
    )
    elapsed = time.time() - start_ts
    progress(1.0, desc=f"Completed in {elapsed:.2f}s!")
    gr.Info(f"Finished in {elapsed:.2f}s")

    return results, -1, gr.update(placeholder=f"Last Input ({elapsed:.2f}s): " + prompt, value="")


def example_pipeline(
    image_reference, image_target_1, image_target_2, custmization_mode,
    input_mask_mode, seg_ref_mode, prompt, seed, true_gs, eg_idx, num_steps, guidance
):
    """Handle example loading in the UI."""

    if seg_ref_mode == "Full Ref":
        image_reference_ori_state = np.array(image_reference.convert("RGB"))
        image_reference_rmbg_state = None
        image_reference_state = image_reference_ori_state
    else:
        image_reference_rmbg_state = np.array(image_reference.convert("RGB"))
        image_reference_ori_state = None
        image_reference_state = image_reference_rmbg_state

    if custmization_mode == "Position-aware":
        if input_mask_mode == "Precise mask":
            image_target_state = np.array(image_target_1.convert("RGB"))
        else:
            image_target_state = np.array(image_target_2['composite'].convert("RGB"))
        mask_target_state = np.array(Image.open(MASK_TGT[int(eg_idx)]))
    else:  # Position-free mode
        # For Position-free, use the target image from IMG_TGT1 and corresponding mask
        image_target_state = np.array(image_target_1.convert("RGB"))
        mask_target_state = np.array(Image.open(MASK_TGT[int(eg_idx)]))

    mask_target_binary = mask_target_state / 255
    masked_img = image_target_state * mask_target_binary
    masked_img_pil = Image.fromarray(masked_img.astype("uint8"))
    output_mask_pil = Image.fromarray(mask_target_state.astype("uint8"))

    if custmization_mode == "Position-aware":
        mask_gallery = [masked_img_pil, output_mask_pil]
    else:
        mask_gallery = gr.skip()

    result_gallery = [Image.open(IMG_GEN[int(eg_idx)]).convert("RGB")]


    if custmization_mode == "Position-free":
        return (image_reference_ori_state, image_reference_rmbg_state, image_target_state,
                mask_target_state, mask_gallery, result_gallery, 
                gr.update(visible=False), gr.update(visible=False))

    if input_mask_mode == "Precise mask":
        return (image_reference_ori_state, image_reference_rmbg_state, image_target_state,
                mask_target_state, mask_gallery, result_gallery, 
                gr.update(visible=True), gr.update(visible=False))
    else:
        # Ensure ImageEditor has a proper background so brush + undo work
        try:
            bg_img = image_target_2.get('background') or image_target_2.get('composite')
        except Exception:
            bg_img = image_target_2

        return (
            image_reference_ori_state, image_reference_rmbg_state, image_target_state,
            mask_target_state, mask_gallery, result_gallery,
            gr.update(visible=False),
            gr.update(visible=True, value={"background": bg_img, "layers": [], "composite": bg_img}),
        )



def create_application(pipeline, vlm_processor, vlm_model, assets_cache_dir):
    """Create the main Gradio application."""
    # Create theme and CSS
    theme = create_theme()
    css = create_css()
    
    with gr.Blocks(theme=theme, css=css) as demo:

        with gr.Column(elem_id="global_glass_container"):
            
            # Create UI sections
            create_header_section()

            # Hidden components
            eg_idx = gr.Textbox(label="eg_idx", visible=False, value="-1") 

            # State variables
            image_target_state = gr.State(value=None)
            mask_target_state = gr.State(value=None)
            image_reference_ori_state = gr.State(value=None)
            image_reference_rmbg_state = gr.State(value=None)
            selected_points = gr.State(value=[])


            # Main UI content with optimized left-right layout
            with gr.Column(elem_id="glass_card"):
                # Top section - Mode selection (full width)
                custmization_mode, md_custmization_mode = create_customization_section()
                
                # Main layout: Left for inputs, Right for outputs
                with gr.Row(equal_height=False):
                    # LEFT COLUMN - ALL INPUTS
                    with gr.Column(scale=3, min_width=400):
                        # Image input section
                        (image_reference, input_mask_mode, image_target_1, image_target_2,
                            undo_target_seg_button, md_image_reference, md_input_mask_mode, 
                            md_target_image) = create_image_input_section()
                        
                        # Text prompt section
                        prompt, vlm_generate_btn, vlm_polish_btn, md_prompt = create_prompt_section()
                        
                        # Advanced options (collapsible)
                        (aspect_ratio, seg_ref_mode, move_to_center, use_background_preservation,
                            background_blend_threshold, seed, num_images_per_prompt, guidance,
                            num_steps, true_gs) = create_advanced_options_section()
                        
                    # RIGHT COLUMN - ALL OUTPUTS
                    with gr.Column(scale=2, min_width=350):
                        # Mask preview and operations
                        (mask_gallery, dilate_button, erode_button, bounding_box_button,
                            md_mask_operation) = create_mask_operation_section()
                        
                        # Generation controls and results
                        result_gallery, submit_button, clear_btn, md_submit = create_output_section()

                with gr.Row(elem_id="glass_card"):
                    # Examples section
                    examples = create_examples_section(
                        GRADIO_EXAMPLES,
                        inputs=[
                            image_reference,
                            image_target_1,
                            image_target_2,
                            custmization_mode,
                            input_mask_mode,
                            seg_ref_mode,
                            prompt,
                            seed,
                            true_gs,
                            eg_idx,
                            num_steps,
                            guidance,
                        ],
                        outputs=[
                            image_reference_ori_state, 
                            image_reference_rmbg_state, 
                            image_target_state, 
                            mask_target_state, 
                            mask_gallery, 
                            result_gallery, 
                            image_target_1, 
                            image_target_2,
                        ],
                        fn=example_pipeline,
                    )
            
            with gr.Row(elem_id="glass_card"):
                # Citation section
                create_citation_section()

        # Setup event handlers
        setup_event_handlers(
            ## UI components
            input_mask_mode, image_target_1, image_target_2, undo_target_seg_button,
            custmization_mode, dilate_button, erode_button, bounding_box_button,
            mask_gallery, md_input_mask_mode, md_target_image, md_mask_operation,
            md_prompt, md_submit, result_gallery, image_target_state, mask_target_state,
            seg_ref_mode, image_reference_ori_state, move_to_center,
            image_reference, image_reference_rmbg_state,
            ## Functions
            change_input_mask_mode, change_custmization_mode, 
            lambda seg_ref_mode, img_ref, move_to_center: change_seg_ref_mode(
                seg_ref_mode, img_ref, move_to_center
            ),
            init_image_target_1, init_image_target_2, init_image_reference,
            get_point, undo_seg_points,
            get_brush,
            # VLM buttons
            vlm_generate_btn, vlm_polish_btn,
            # VLM functions
            lambda img_target, img_ref, mask_target, cust_mode: vlm_auto_generate(
                img_target, img_ref, mask_target, cust_mode, vlm_processor, vlm_model,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                construct_vlm_gen_prompt, run_vlm
            ),
            lambda prompt, cust_mode: vlm_auto_polish(
                prompt, cust_mode, vlm_processor, vlm_model,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                construct_vlm_polish_prompt, run_vlm
            ),
            dilate_mask, erode_mask, bounding_box,
            lambda *args: run_model(*args, pipeline, assets_cache_dir),
            ## Other components
            selected_points, prompt,
            use_background_preservation, background_blend_threshold, seed,
            num_images_per_prompt, guidance, true_gs, num_steps, aspect_ratio,
            submit_button,
            eg_idx,
        )

        # Setup clear button
        clear_btn.add(
            [image_reference, image_target_1,image_target_2, mask_gallery, result_gallery,
            selected_points, image_target_state, mask_target_state, prompt,
            image_reference_ori_state, image_reference_rmbg_state]
        )

    return demo


def main():
    """Main entry point for the application."""
    # Parse arguments and load config
    args = parse_args()
    cfg = load_config(args.config)
    setup_environment(args)

    # Initialize device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    pipeline, mobile_predictor, vlm_processor, vlm_model, ben2_model = initialize_models(
        args, cfg, device, weight_dtype
    )
    # Inject mobile predictor into business logic module so get_point can access it without lambdas
    set_mobile_predictor(mobile_predictor)
    set_ben2_model(ben2_model)

    # Create and launch the application
    demo = create_application(
        pipeline, vlm_processor, vlm_model, args.assets_cache_dir
    )
    
    # Launch the demo
    demo.launch(server_name="0.0.0.0", server_port=12345)


if __name__ == "__main__":
    main()