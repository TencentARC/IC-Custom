#!/bin/bash


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_TOKEN=$1
HF_CACHE_DIR=$2

export CUDA_VISIBLE_DEVICES=0

config_file=configs/inference/inference.yaml

if [ -z "$HF_TOKEN" ] && [ -z "$HF_CACHE_DIR" ]; then
    python src/inference/inference.py --config $config_file
else
    python src/inference/inference.py --config $config_file --hf_token $HF_TOKEN --hf_cache_dir $HF_CACHE_DIR
fi