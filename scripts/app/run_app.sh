#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


HF_TOKEN=$1
HF_CACHE_DIR=$2
ASSETS_CACHE_DIR=$3

if [ -z "$HF_TOKEN" ] && [ -z "$HF_CACHE_DIR" ] && [ -z "$ASSETS_CACHE_DIR" ]; then
    python src/app/app.py \
        --config configs/app/app.yaml \
        --save_results \
        --enable_ben2_for_mask_ref \
        # --enable_vlm_for_prompt \
else
    python src/app/app.py \
        --config configs/app/app.yaml \
        --hf_token $HF_TOKEN \
        --hf_cache_dir $HF_CACHE_DIR \
        --assets_cache_dir $ASSETS_CACHE_DIR \
        --save_results \
        --enable_ben2_for_mask_ref \
        # --enable_vlm_for_prompt \
fi

