#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_TOKEN=$1
HF_CACHE_DIR=$2

export CUDA_VISIBLE_DEVICES=0

# Inference 01
config_file=configs/inference/inference_01.yaml

if [ -z "$HF_TOKEN" ] && [ -z "$HF_CACHE_DIR" ]; then
    python src/inference/inference.py --config $config_file
else
    python src/inference/inference.py --config $config_file --hf_token $HF_TOKEN --hf_cache_dir $HF_CACHE_DIR
fi

echo "Inference 01 completed"

# Inference 02
# config_file=configs/inference/inference_02.yaml

# if [ -z "$HF_TOKEN" ] && [ -z "$HF_CACHE_DIR" ]; then
#     python src/inference/inference.py --config $config_file
# else
#     python src/inference/inference.py --config $config_file --hf_token $HF_TOKEN --hf_cache_dir $HF_CACHE_DIR
# fi

# echo "Inference 02 completed"