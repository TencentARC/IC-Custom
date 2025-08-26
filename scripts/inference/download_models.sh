#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_TOKEN=$1
HF_CACHE_DIR=$2

if [ -z "$HF_TOKEN" ] && [ -z "$HF_CACHE_DIR" ]; then
    python src/inference/download_models.py \
        --local_dir models
else
    python src/inference/download_models.py \
        --hf_token $HF_TOKEN \
        --hf_cache_dir $HF_CACHE_DIR \
        --local_dir models
fi