#!/bin/bash
# Stage 2 — EMC-guided VideoQA inference (paper Table 2).
#
# Runs new_video_qa_inference.py across every (model, dataset) pair, twice each:
#   - without EMC (original video + original question)
#   - with EMC    (compressed video + rewritten question, sourced from Stage 1 output)
#
# OpenAI API models are launched with plain python + multi-threading; local
# models are launched with torchrun for multi-GPU inference. Already-completed
# (model, dataset, mode) cells are auto-skipped via check_already_completed().
#
# Usage:
#   bash run_emc_guided_inference.sh [NUM_GPUS] [NUM_THREADS]
#
# Defaults: NUM_GPUS=1 (per local-model torchrun), NUM_THREADS=200 (per API call batch).
#
# Prerequisite: Stage 1 must have produced ReSimplifyIt_simple_*.json files in
# the repo dir (i.e. run_emc_process.sh has been executed).

NUM_GPUS=${1:-1}
NUM_THREADS=${2:-200}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Models tested in paper Table 2 (local VLMs + OpenAI API baselines)
MODELS=(
    "InternVL3.5-8B" "InternVL3.5-14B"
    "Qwen2.5-VL-3B"  "Qwen2.5-VL-7B"
    "Qwen3-VL-4B"    "Qwen3-VL-32B"
    "LLaVA-OneVision-4B" "LLaVA-OneVision-8B"
    "GPT-4o" "GPT-4.1-mini" "GPT-4-turbo"
)

DATASETS=(
    "EgoSchema" "LVBench" "MLVU" "Video-MME"
    "ActivityNet-QA" "NExT-QA" "NExT-OE" "EMCompress"
)

OPENAI_MODELS=("GPT-4o" "GPT-4.1-mini" "GPT-4-turbo")

is_openai_model() {
    local model=$1
    for api_model in "${OPENAI_MODELS[@]}"; do
        [ "$model" == "$api_model" ] && return 0
    done
    return 1
}

total_experiments=$((${#MODELS[@]} * ${#DATASETS[@]} * 2))
current=0

echo "========================================"
echo "Stage 2 — EMC-guided VideoQA inference"
echo "Local-model GPUs: $NUM_GPUS    API threads: $NUM_THREADS"
echo "Start: $(date)"
echo "========================================"

run_cell() {
    local model=$1 dataset=$2 emc_flag=$3 mode_label=$4
    current=$((current + 1))
    echo ""
    echo "[$current/$total_experiments] $model | $dataset | $mode_label   $(date +%H:%M:%S)"

    local emc_bool=False
    [ "$emc_flag" == "--emc" ] && emc_bool=True

    if python -c "from new_video_qa_inference import check_already_completed; \
                  completed, msg = check_already_completed('$model', '$dataset', $emc_bool, False); \
                  print(msg) if msg else None; \
                  exit(0 if completed else 1)" 2>/dev/null; then
        echo "  → skip (already completed)"
        return
    fi

    if is_openai_model "$model"; then
        python new_video_qa_inference.py --model "$model" --dataset "$dataset" --num_threads "$NUM_THREADS" $emc_flag
    else
        torchrun --nproc_per_node="$NUM_GPUS" new_video_qa_inference.py --model "$model" --dataset "$dataset" $emc_flag
    fi
}

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        run_cell "$model" "$dataset" "" "Original"
        run_cell "$model" "$dataset" "--emc" "EMC"
    done
done

echo ""
echo "========================================"
echo "All experiments done. End: $(date)"
echo "========================================"
