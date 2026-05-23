#!/usr/bin/env python3
"""
Quick test script to verify all baseline models can run on all datasets.
Only processes 1 sample per combination to check functionality.
Skips ActivityNet-QA and EMCompress.
"""

import sys
import json
import torch
from decord import VideoReader, cpu
import new_video_qa_inference as bvi

# Test each model type with different dataset types (skip ActivityNet-QA and EMCompress)
test_cases = [
    ("Qwen2.5-VL-3B", "EgoSchema"),      # qwen2_vl + MCQ
    ("Qwen2.5-VL-3B", "NExT-OE"),        # qwen2_vl + Open-ended
    ("Qwen3-VL-4B", "MLVU"),             # qwen3_vl + MCQ
    ("InternVL3.5-8B", "Video-MME"),     # internvl + MCQ
    ("LLaVA-OneVision-4B", "LVBench"),   # llava_onevision + MCQ
    ("LLaVA-OneVision-4B", "NExT-QA"),   # llava_onevision + MCQ
]

results = {}

for model_name, dataset_name in test_cases:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Testing: {model_name} on {dataset_name}")
    print(sep)

    try:
        # Load dataset
        from emc_utils.utils import load_test_split
        dataset_config = bvi.DATASET_CONFIGS[dataset_name]
        data = load_test_split(dataset_config["data_dir"])

        is_open_ended = bvi.is_open_ended_dataset(data)
        dtype = "Open-ended QA" if is_open_ended else "Multiple Choice QA"
        print(f"Dataset type: {dtype}")

        # Load model
        model_config = bvi.MODEL_CONFIGS[model_name]
        print(f"Loading {model_name}...")
        model_type = model_config["model_type"]
        model_dict = bvi.MODEL_LOADERS[model_type](model_config["path"])
        infer_func = bvi.MODEL_INFERENCERS[model_type]
        print("Model loaded.")

        # Test on 1 sample
        num_frames = model_config["num_frames"]
        key, item = list(data.items())[0]
        print(f"Processing {key}...")

        video_path = item["video_path"]
        question = item["question"]
        options = item.get("options", [])

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        frame_indices = bvi.sample_frame_indices_uniform(total_frames, num_frames)
        frames = bvi.load_video_frames(video_path, frame_indices)

        if is_open_ended:
            prompt = bvi.format_open_ended_prompt(question)
        else:
            prompt = bvi.format_mcq_prompt(question, options)

        response = infer_func(model_dict, frames, prompt)
        print(f"Response: {response[:150]}...")
        print("SUCCESS!")
        results[(model_name, dataset_name)] = "SUCCESS"

        # Clear GPU memory
        del model_dict
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        results[(model_name, dataset_name)] = f"FAILED: {e}"

# Print summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for (model, dataset), status in results.items():
    print(f"{model} + {dataset}: {status[:50]}")
