#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Configuration management for IC-Custom application.
"""
import os
import argparse
from omegaconf import OmegaConf


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="IC-Custom App.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/app/app.yaml",
        required=False,
        help="path to config",
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
    parser.add_argument(
        "--assets_cache_dir",
        type=str,
        required=False,
        default="results/app",
        help="Cache directory to save the results, default is results/app",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save results",
    )
    parser.add_argument(
        "--enable_ben2_for_mask_ref",
        action="store_true",
        help="Enable ben2 for mask reference",
    )
    parser.add_argument(
        "--enable_vlm_for_prompt",
        action="store_true",
        help="Enable vlm for prompt",
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from file."""
    return OmegaConf.load(config_path)


def setup_environment(args):
    """Setup environment variables from command line arguments."""
    if args.hf_token is not None:
        os.environ["HF_TOKEN"] = args.hf_token

    if args.hf_cache_dir is not None:
        os.environ["HF_HUB_CACHE"] = args.hf_cache_dir
