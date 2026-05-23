#!/bin/bash
# Multi-GPU video captioning script — wraps caption_all_videos.py
#
# Usage: bash run_caption.sh [NUM_GPUS] --model MODEL [EXTRA_ARGS]
# Examples:
#   bash run_caption.sh 4 --model Qwen/Qwen3-VL-32B-Instruct --skip_existing
#   bash run_caption.sh 2 --model llava-hf/llava-v1.6-mistral-7b-hf --dataset EgoSchema
#   bash run_caption.sh 1 --model gpt-4o --num_threads 32 --skip_existing
#
# Note: --model is required by caption_all_videos.py. The script auto-routes
# OpenAI (gpt-*) to API backend and Qwen/LLaVA to local transformers backend.

NUM_GPUS=${1:-4}
shift  # remaining args are forwarded to caption_all_videos.py

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting video captioning with $NUM_GPUS GPUs"
echo "Extra arguments: $@"

torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    "$HERE/caption_all_videos.py" \
    --skip_existing \
    "$@"

echo "Done!"
