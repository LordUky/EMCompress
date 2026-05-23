"""
Baseline Video QA Inference Script for Multiple Video-LLM Models
Supports: InternVL3.5-8B/14B, Qwen2.5-VL-3B/7B, Qwen3-VL-4B/32B, LLaVA-OneVision-1.5-4B/8B
         GPT-4o, GPT-4.1-mini, GPT-4-turbo (OpenAI API baselines with 8 uniformly sampled frames)
Two experiment modes: Original video + question, EMC-processed video + question

For open-ended QA datasets (no "options" field), uses GPT-4o to evaluate and score responses.
For MCQ datasets, extracts answer letter and checks correctness.
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import torch
import torch.distributed as dist
from decord import VideoReader, cpu
from openai import OpenAI
from config import openai_api_key, DATASETS_DIR, MODELS_HF_DIR, MODELS_LOCAL_DIR, BASE64_CACHE_DIR
from emc_utils.utils import load_test_split
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import signal
from functools import wraps
import base64
from io import BytesIO
import threading
import time
import hashlib
import pickle

# ============ Timeout Utilities ============
class TimeoutError(Exception):
    pass

def timeout(seconds):
    """Decorator to add timeout to a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Inference timed out after {seconds} seconds")

            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator

# Global variable for device (set during distributed init)
DEVICE_MAP = "auto"

# InternVL constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_HERE = os.path.dirname(os.path.abspath(__file__))

# ============ Configuration ============
MODEL_CONFIGS = {
    "InternVL3.5-8B": {
        "path": os.path.join(MODELS_HF_DIR, "models--OpenGVLab--InternVL3_5-8B/snapshots/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79"),
        "num_frames": 16,
        "sampling": "uniform",
        "model_type": "internvl",
    },
    "InternVL3.5-14B": {
        "path": os.path.join(MODELS_HF_DIR, "models--OpenGVLab--InternVL3_5-14B/snapshots/a1e37197b393ce9eec9df700fef65c11f4a6ffbd"),
        "num_frames": 16,
        "sampling": "uniform",
        "model_type": "internvl",
    },
    "Qwen2.5-VL-3B": {
        "path": os.path.join(MODELS_HF_DIR, "models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"),
        "num_frames": 16,
        "sampling": "uniform",
        "model_type": "qwen2_vl",
    },
    "Qwen2.5-VL-7B": {
        "path": os.path.join(MODELS_HF_DIR, "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"),
        "num_frames": 16,
        "sampling": "uniform",
        "model_type": "qwen2_vl",
    },
    "Qwen3-VL-4B": {
        "path": os.path.join(MODELS_LOCAL_DIR, "Qwen3-VL-4B-Instruct"),
        "num_frames": 16,
        "sampling": "uniform",
        "model_type": "qwen3_vl",
    },
    "Qwen3-VL-32B": {
        "path": os.path.join(MODELS_HF_DIR, "models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb"),
        "num_frames": 16,
        "sampling": "uniform",
        "model_type": "qwen3_vl",
    },
    "LLaVA-OneVision-4B": {
        "path": os.path.join(MODELS_LOCAL_DIR, "LLaVA-OneVision-1.5-4B-Instruct"),
        "num_frames": 16,
        "sampling": "uniform",
        "model_type": "llava_onevision",
    },
    "LLaVA-OneVision-8B": {
        "path": os.path.join(MODELS_LOCAL_DIR, "LLaVA-OneVision-1.5-8B-Instruct"),
        "num_frames": 16,
        "sampling": "uniform",
        "model_type": "llava_onevision",
    },
    # OpenAI API baselines (8 uniformly sampled frames)
    "GPT-4o": {
        "path": None,  # API model, no local path
        "num_frames": 8,
        "sampling": "uniform",
        "model_type": "openai_api",
        "api_model_name": "gpt-4o",
    },
    "GPT-4.1-mini": {
        "path": None,  # API model, no local path
        "num_frames": 8,
        "sampling": "uniform",
        "model_type": "openai_api",
        "api_model_name": "gpt-4.1-mini",
    },
    "GPT-4-turbo": {
        "path": None,  # API model, no local path
        "num_frames": 8,
        "sampling": "uniform",
        "model_type": "openai_api",
        "api_model_name": "gpt-4-turbo",
    },
}

DATASET_CONFIGS = {
    "EgoSchema": {
        "data_dir": os.path.join(DATASETS_DIR, "EgoSchema"),
        "emc_result": os.path.join(_HERE, "ReSimplifyIt_simple_EgoSchema.json"),
    },
    "LVBench": {
        "data_dir": os.path.join(DATASETS_DIR, "LVBench"),
        "emc_result": os.path.join(_HERE, "ReSimplifyIt_simple_LVBench.json"),
    },
    "MLVU": {
        "data_dir": os.path.join(DATASETS_DIR, "MLVU"),
        "emc_result": os.path.join(_HERE, "ReSimplifyIt_simple_MLVU.json"),
    },
    "Video-MME": {
        "data_dir": os.path.join(DATASETS_DIR, "Video-MME"),
        "emc_result": os.path.join(_HERE, "ReSimplifyIt_simple_Video-MME.json"),
    },
    "ActivityNet-QA": {
        "data_dir": os.path.join(DATASETS_DIR, "ActivityNet-QA"),
        "emc_result": os.path.join(_HERE, "ReSimplifyIt_simple_ActivityNet-QA.json"),
    },
    "NExT-QA": {
        "data_dir": os.path.join(DATASETS_DIR, "NExT-QA"),
        "emc_result": os.path.join(_HERE, "ReSimplifyIt_simple_NExT-QA.json"),
    },
    "NExT-OE": {
        "data_dir": os.path.join(DATASETS_DIR, "NExT-OE"),
        "emc_result": os.path.join(_HERE, "ReSimplifyIt_simple_NExT-OE.json"),
    },
    "EMCompress": {
        "data_dir": os.path.join(DATASETS_DIR, "EMCompress"),
        "emc_result": os.path.join(_HERE, "ReSimplifyIt_simple_EMCompress.json"),
    },
}

OUTPUT_DIR = os.path.join(_HERE, "baseline_inference_results")

# GPT-4o evaluation prompt for open-ended QA
EVAL_PROMPT_TEMPLATE = '''
You are a helpful assistant to rate answer for a Video QA question.

You will be given the ground truth answer and the candidate answer, and the original input question. Please evaluate the output based on the following three criterias:

    1. Relevance: to minimize unwanted information
    - in this criteria, a candidate output gets full mark if it doesn't contain any information (phrases, concepts, etc.) out of the scope of the original input question.
    - the more information it contains that is not mentioned in the original input question, the more marks are deducted.

    2. Simplicity: to minimize tangential information
    - in this criteria, a candidate output gets full mark if doesn't contain any information (phrases, concepts, etc.) that is included in the original input question but not included in the ground truth compressed question. In other words, whether the question sentence is fully compressed.
    - the more information it contains that is included in the original input question but not included in the ground truth question, the more marks will be deducted.

    3. Completeness: to minimize over-compression
    - in this criteria, a candidate output gets full mark if it contains all information included in the ground truth compressed question.
    - the more information contained in the ground truth output question is found missing in the output compressed question, the more marks will be deducted.

    Here is the original input question: {question}, and
    here is the ground truth answer: {ground_truth}, and
    here is the candidate answer: {candidate}.

    Please rate the candidate answer on a scale from 0 to 100, with 0 being the worst and 100 being the best (full mark).
    Now, rate the quality of the candidate answer based on all the information above.
    Return your answer in this json format: {{"score": [your score, from 0 to 100]}}.
'''


def get_video_duration(video_path):
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return None


def sample_frame_indices_uniform(total_frames, num_frames):
    """Uniform sampling of frame indices."""
    if total_frames <= num_frames:
        return list(range(total_frames))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return indices.tolist()


def sample_frame_indices_from_segments(segments, total_frames, fps, num_frames, sampling_mode="uniform"):
    """
    Sample frame indices from EMC segments.
    segments: [[start1, end1], [start2, end2], ...]  in seconds
    Returns frame indices that fall within the segments.
    """
    if not segments or segments == [[]]:
        return sample_frame_indices_uniform(total_frames, num_frames)

    valid_frames = []
    for seg in segments:
        if len(seg) != 2:
            continue
        start_sec, end_sec = seg
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))
        valid_frames.extend(range(start_frame, end_frame + 1))

    if not valid_frames:
        return sample_frame_indices_uniform(total_frames, num_frames)

    valid_frames = sorted(set(valid_frames))

    if len(valid_frames) <= num_frames:
        return valid_frames
    indices = np.linspace(0, len(valid_frames) - 1, num_frames, dtype=int)
    return [valid_frames[i] for i in indices]


def load_video_frames(video_path, frame_indices):
    """Load specific frames from video using decord."""
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(frame_indices)
    if hasattr(frames, 'asnumpy'):
        frames = frames.asnumpy()
    elif hasattr(frames, 'numpy'):
        frames = frames.numpy()
    else:
        frames = np.array(frames)
    return [Image.fromarray(f) for f in frames]


def format_mcq_prompt(question, options):
    """Format multiple choice question prompt."""
    prompt = f"{question}\n\n"
    for opt in options:
        prompt += f"{opt}\n"
    prompt += "\nYour answer should be a single letter, indicating the correct choice."
    return prompt


def format_open_ended_prompt(question):
    """Format open-ended question prompt."""
    return f"{question}\n\nPlease provide a concise answer."


def is_open_ended_dataset(data):
    """Check if dataset is open-ended (no options field or empty options)."""
    for key, item in data.items():
        options = item.get("options", [])
        if options and len(options) > 0:
            return False
        return True
    return True


# ============ Model Loading Functions ============

def get_device_map():
    """Get the appropriate device map based on distributed setting."""
    global DEVICE_MAP
    return DEVICE_MAP


def load_internvl(model_path):
    """Load InternVL3.5 model."""
    from transformers import AutoModel, AutoTokenizer

    device_map = get_device_map()
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    return {"model": model, "tokenizer": tokenizer}


def load_qwen2_vl(model_path):
    """Load Qwen2.5-VL model."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    device_map = get_device_map()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path)

    return {"model": model, "processor": processor}


def load_qwen3_vl(model_path):
    """Load Qwen3-VL model."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device_map = get_device_map()
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path)

    return {"model": model, "processor": processor}


def load_llava_onevision(model_path):
    """Load LLaVA-OneVision 1.5 model."""
    from transformers import AutoProcessor, AutoModelForCausalLM

    device_map = get_device_map()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",  # Disable flash attention for compatibility
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    return {"model": model, "processor": processor}


# ============ Model Inference Functions ============

def build_internvl_transform(input_size):
    """Build transform for InternVL."""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find closest aspect ratio for InternVL dynamic preprocess."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess_internvl(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamic preprocess for InternVL."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def infer_internvl(model_dict, frames, prompt):
    """Run inference with InternVL3.5."""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    input_size = 448
    max_num = 1  # For video, use 1 tile per frame to save memory
    transform = build_internvl_transform(input_size)

    # Process frames
    pixel_values_list = []
    num_patches_list = []
    for frame in frames:
        frame = frame.convert('RGB') if frame.mode != 'RGB' else frame
        tiles = dynamic_preprocess_internvl(frame, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list, dim=0).to(model.device, dtype=model.dtype)

    # Build video prefix with frame labels
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(frames))])
    question = video_prefix + prompt

    generation_config = dict(max_new_tokens=256, do_sample=False)

    response = model.chat(
        tokenizer, pixel_values, question, generation_config,
        num_patches_list=num_patches_list, history=None, return_history=False
    )

    return response


def infer_qwen2_vl(model_dict, frames, prompt):
    """Run inference with Qwen2.5-VL."""
    model = model_dict["model"]
    processor = model_dict["processor"]

    # Build message with video frames
    content = []
    for frame in frames:
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def infer_qwen3_vl(model_dict, frames, prompt):
    """Run inference with Qwen3-VL."""
    # Same as Qwen2.5-VL
    return infer_qwen2_vl(model_dict, frames, prompt)


def infer_llava_onevision(model_dict, frames, prompt):
    """Run inference with LLaVA-OneVision 1.5."""
    model = model_dict["model"]
    processor = model_dict["processor"]

    # LLaVA-OneVision 1.5 uses same format as Qwen2.5-VL with video input
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Remove any extra kwargs that the model doesn't accept
    generate_kwargs = {k: v for k, v in inputs.items()
                       if k in ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']}

    with torch.inference_mode():
        output_ids = model.generate(
            **generate_kwargs,
            do_sample=False,
            max_new_tokens=256,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# ============ OpenAI API Model Functions ============

def load_openai_api(model_path, api_model_name):
    """Load OpenAI API client (no local model loading needed)."""
    client = OpenAI(api_key=openai_api_key)
    return {"client": client, "api_model_name": api_model_name}


def get_cache_key(video_path, frame_indices):
    """
    Generate a unique cache key based on video path and frame indices.
    The key is a hash of the video path and sorted frame indices.
    """
    # Use video path and frame indices to create a unique identifier
    key_str = f"{video_path}:{sorted(frame_indices)}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cache_path(cache_key):
    """Get the file path for a cache entry."""
    # Use subdirectories based on first 2 characters of hash to avoid too many files in one directory
    subdir = cache_key[:2]
    cache_dir = os.path.join(BASE64_CACHE_DIR, subdir)
    return os.path.join(cache_dir, f"{cache_key}.pkl")


def get_video_metadata_fast(video_path):
    """
    Get video metadata (total_frames, fps) using cv2.
    This is ~10x faster than decord (0.008s vs 0.07s).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps


def frame_indices_match(cached_indices, requested_indices, fps, tolerance_sec=1.0):
    """
    Check if cached frame indices match requested indices within a tolerance window.
    This handles potential precision differences in frame sampling across runs.

    Args:
        cached_indices: Frame indices from cache
        requested_indices: Frame indices we want
        fps: Video FPS (to convert tolerance from seconds to frames)
        tolerance_sec: Tolerance window in seconds (default 1.0s)

    Returns:
        True if indices match within tolerance
    """
    if len(cached_indices) != len(requested_indices):
        return False

    tolerance_frames = int(fps * tolerance_sec)

    for cached, requested in zip(cached_indices, requested_indices):
        if abs(cached - requested) > tolerance_frames:
            return False

    return True


# ============ Video-to-Cache Index for Fast Lookup ============
VIDEO_CACHE_INDEX = {}  # video_path -> list of (cache_key, frame_indices)
VIDEO_CACHE_INDEX_LOCK = threading.Lock()
VIDEO_CACHE_INDEX_FILE = os.path.join(BASE64_CACHE_DIR, "_video_index.pkl")
VIDEO_CACHE_INDEX_LOADED = False


def _load_video_cache_index():
    """Load video cache index from disk (called once)."""
    global VIDEO_CACHE_INDEX, VIDEO_CACHE_INDEX_LOADED
    if VIDEO_CACHE_INDEX_LOADED:
        return
    with VIDEO_CACHE_INDEX_LOCK:
        if VIDEO_CACHE_INDEX_LOADED:
            return
        if os.path.exists(VIDEO_CACHE_INDEX_FILE):
            try:
                with open(VIDEO_CACHE_INDEX_FILE, 'rb') as f:
                    VIDEO_CACHE_INDEX = pickle.load(f)
            except Exception:
                VIDEO_CACHE_INDEX = {}
        VIDEO_CACHE_INDEX_LOADED = True


def _save_video_cache_index():
    """Save video cache index to disk."""
    try:
        os.makedirs(BASE64_CACHE_DIR, exist_ok=True)
        with open(VIDEO_CACHE_INDEX_FILE, 'wb') as f:
            pickle.dump(VIDEO_CACHE_INDEX, f)
    except Exception:
        pass


def _update_video_cache_index(video_path, cache_key, frame_indices):
    """Add entry to the video cache index."""
    global VIDEO_CACHE_INDEX
    _load_video_cache_index()
    with VIDEO_CACHE_INDEX_LOCK:
        if video_path not in VIDEO_CACHE_INDEX:
            VIDEO_CACHE_INDEX[video_path] = []
        # Check if already exists
        for ck, fi in VIDEO_CACHE_INDEX[video_path]:
            if ck == cache_key:
                return
        VIDEO_CACHE_INDEX[video_path].append((cache_key, list(frame_indices)))
        # Save periodically
        if sum(len(v) for v in VIDEO_CACHE_INDEX.values()) % 50 == 0:
            _save_video_cache_index()


def load_base64_cache(video_path, frame_indices, fps=30.0):
    """
    Load cached base64 encoded frames if available.
    Returns a list of base64 strings or None if cache miss.

    Uses tolerance-based matching to handle precision differences.
    Uses index for fast lookup instead of directory scanning.
    """
    cache_key = get_cache_key(video_path, frame_indices)
    cache_path = get_cache_path(cache_key)

    # First try exact match
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            if (cached_data.get('video_path') == video_path and
                cached_data.get('frame_indices') == list(frame_indices)):
                return cached_data.get('base64_frames')
        except Exception:
            pass

    # Try fuzzy match using index (much faster than directory scanning)
    _load_video_cache_index()

    with VIDEO_CACHE_INDEX_LOCK:
        cached_entries = VIDEO_CACHE_INDEX.get(video_path, [])

    for ck, cached_indices in cached_entries:
        if frame_indices_match(cached_indices, frame_indices, fps):
            # Found a match in index, load the cache file
            cp = get_cache_path(ck)
            if os.path.exists(cp):
                try:
                    with open(cp, 'rb') as f:
                        cached_data = pickle.load(f)
                    return cached_data.get('base64_frames')
                except Exception:
                    continue

    return None


def save_base64_cache(video_path, frame_indices, base64_frames):
    """
    Save base64 encoded frames to cache and update index.
    """
    cache_key = get_cache_key(video_path, frame_indices)
    cache_path = get_cache_path(cache_key)

    # Create directory if needed
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)

    try:
        cached_data = {
            'video_path': video_path,
            'frame_indices': list(frame_indices),
            'base64_frames': base64_frames
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        # Update index for fast lookup
        _update_video_cache_index(video_path, cache_key, frame_indices)
    except Exception as e:
        # Silently fail cache save - not critical
        print(f"Warning: Failed to save base64 cache: {e}")


def encode_image_to_base64(pil_image):
    """Encode a PIL image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_frames_to_base64_cached(video_path, frame_indices, frames, fps=30.0):
    """
    Encode frames to base64 with caching support.
    First checks cache, if miss, encodes and saves to cache.

    Args:
        video_path: Path to the video file
        frame_indices: List of frame indices that were sampled
        frames: List of PIL images
        fps: Video FPS for tolerance-based cache matching

    Returns:
        List of base64 encoded strings
    """
    # Try to load from cache (with tolerance matching)
    cached = load_base64_cache(video_path, frame_indices, fps)
    if cached is not None:
        return cached

    # Cache miss - encode frames
    base64_frames = [encode_image_to_base64(frame) for frame in frames]

    # Save to cache for future use
    save_base64_cache(video_path, frame_indices, base64_frames)

    return base64_frames


def infer_openai_api(model_dict, frames, prompt, video_path=None, frame_indices=None):
    """
    Run inference with OpenAI API models (GPT-4o, GPT-4.1-mini, GPT-4-turbo).

    Args:
        model_dict: Dictionary with 'client' and 'api_model_name'
        frames: List of PIL images
        prompt: The prompt to send
        video_path: Optional - video path for caching
        frame_indices: Optional - frame indices for caching

    If video_path and frame_indices are provided, base64 encoding will be cached.
    """
    client = model_dict["client"]
    api_model_name = model_dict["api_model_name"]

    # Build the system prompt
    system_prompt = (
        "You are a Video Question Answering assistant. "
        "You will be shown 8 images that are uniformly sampled frames from a video. "
        "Please analyze these frames carefully to understand the video content and answer the question."
    )

    # Build the user message content with images
    content = []

    # Add frame description
    content.append({
        "type": "text",
        "text": "Here are 8 uniformly sampled frames from a video:"
    })

    # Get base64 encoded frames (with caching if video_path and frame_indices provided)
    if video_path is not None and frame_indices is not None:
        base64_frames = encode_frames_to_base64_cached(video_path, frame_indices, frames)
    else:
        base64_frames = [encode_image_to_base64(frame) for frame in frames]

    # Add each frame as base64 encoded image
    for base64_image in base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "auto"
            }
        })

    # Add the question/prompt
    content.append({
        "type": "text",
        "text": f"\n{prompt}"
    })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]

    # Call the API
    response = client.chat.completions.create(
        model=api_model_name,
        messages=messages,
        temperature=0,
        max_tokens=256
    )

    return response.choices[0].message.content


# Model dispatcher
MODEL_LOADERS = {
    "internvl": load_internvl,
    "qwen2_vl": load_qwen2_vl,
    "qwen3_vl": load_qwen3_vl,
    "llava_onevision": load_llava_onevision,
    "openai_api": load_openai_api,
}

MODEL_INFERENCERS = {
    "internvl": infer_internvl,
    "qwen2_vl": infer_qwen2_vl,
    "qwen3_vl": infer_qwen3_vl,
    "llava_onevision": infer_llava_onevision,
    "openai_api": infer_openai_api,
}


def extract_answer_letter(response):
    """Extract single letter answer from model response."""
    response = response.strip().upper()
    for letter in ['A', 'B', 'C', 'D', 'E', 'F']:
        if letter in response:
            return letter
    return response[0] if response else ""


def extract_mcq_answer_with_gpt(question, options, model_response):
    """Use GPT-3.5-turbo to extract the answer letter from model response."""
    try:
        client = OpenAI(api_key=openai_api_key)

        options_text = "\n".join(options)
        prompt = f"""The following is a multiple choice question:

{question}

Options:
{options_text}

A video language model was asked this question and gave the following response:
"{model_response}"

Based on the model's response, which option (A, B, C, D, E, or F) did it choose?
Output ONLY the single letter of the correct option. Do not include any explanation or punctuation."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract answer letters from text responses to multiple choice questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5
        )

        answer = response.choices[0].message.content.strip().upper()
        for letter in ['A', 'B', 'C', 'D', 'E', 'F']:
            if letter in answer:
                return letter
        return answer[0] if answer else ""
    except Exception as e:
        print(f"GPT extraction error: {e}")
        return extract_answer_letter(model_response)


def evaluate_open_ended_with_gpt4o(question, ground_truth, candidate):
    """Use GPT-4o to evaluate open-ended QA response and return a score."""
    try:
        client = OpenAI(api_key=openai_api_key)

        eval_prompt = EVAL_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            candidate=candidate
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0,
            max_tokens=100
        )

        result_text = response.choices[0].message.content.strip()

        # Parse the JSON response
        import re
        json_match = re.search(r'\{.*?"score".*?\}', result_text, re.DOTALL)
        if json_match:
            result_json = json.loads(json_match.group())
            return result_json.get("score", 0)

        # Fallback: try to extract number
        num_match = re.search(r'\d+', result_text)
        if num_match:
            return int(num_match.group())

        return 0
    except Exception as e:
        print(f"GPT-4o evaluation error: {e}")
        return 0


def run_inference(model_name, dataset_name, use_emc=False, force_redo=False):
    """Run inference for a model on a dataset."""
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    if rank == 0:
        print(f"World Size: {world_size}")

    # Load dataset
    dataset_config = DATASET_CONFIGS[dataset_name]
    data = load_test_split(dataset_config["data_dir"])

    # Check if open-ended
    is_open_ended = is_open_ended_dataset(data)
    if rank == 0:
        print(f"Dataset type: {'Open-ended QA' if is_open_ended else 'Multiple Choice QA'}")

    # Prepare output paths
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix = "_emc" if use_emc else "_original"
    output_path = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset_name}{suffix}.json")
    rank_output_path = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset_name}{suffix}_rank{rank}.json")

    # Load EMC results if needed
    emc_data = None
    if use_emc:
        emc_path = dataset_config["emc_result"]
        if not os.path.exists(emc_path):
            raise FileNotFoundError(f"EMC result not found at {emc_path}")
        with open(emc_path) as f:
            raw_emc = json.load(f)
        if dataset_name not in raw_emc:
            raise ValueError(f"No EMC data for {dataset_name} in {emc_path}")
        emc_data = raw_emc[dataset_name]

    # Load model (only if there's work to do)
    model_config = MODEL_CONFIGS[model_name]
    model_type = model_config["model_type"]

    if model_type == "openai_api":
        # OpenAI API models don't need local loading
        if rank == 0:
            print(f"Initializing {model_name} (OpenAI API)...")
        api_model_name = model_config["api_model_name"]
        model_dict = MODEL_LOADERS[model_type](None, api_model_name)
    else:
        if rank == 0:
            print(f"Loading {model_name} from {model_config['path']}...")
        model_dict = MODEL_LOADERS[model_type](model_config["path"])

    infer_func = MODEL_INFERENCERS[model_type]

    if rank == 0:
        print(f"Model loaded.")

    # Load existing results for incremental processing
    results = {}
    if not force_redo and os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        if rank == 0:
            print(f"Loaded {len(results)} existing results from {output_path}")

    num_frames = model_config["num_frames"]
    sampling_mode = model_config["sampling"]

    # Shard data by rank
    data_items = list(data.items())
    items_per_rank = len(data_items) // world_size
    start_idx = rank * items_per_rank
    end_idx = start_idx + items_per_rank if rank < world_size - 1 else len(data_items)
    my_data = dict(data_items[start_idx:end_idx])

    if rank == 0:
        print(f"Total items: {len(data_items)}, Rank {rank} processing: {len(my_data)} items")

    correct = 0
    total = 0
    total_score = 0

    for key, item in tqdm(my_data.items(), desc=f"Rank {rank}: {model_name} on {dataset_name}", disable=rank!=0):
        if not force_redo and key in results:
            if is_open_ended:
                total_score += results[key].get("score", 0)
            else:
                if results[key].get("correct"):
                    correct += 1
            total += 1
            continue

        try:
            video_path = item["video_path"]
            question = item["question"]
            answer = item["answer"]
            options = item.get("options", [])

            # Get video info
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            fps = vr.get_avg_fps()

            # Determine frame indices
            if use_emc and emc_data:
                if key not in emc_data:
                    raise KeyError(f"Key {key} not found in EMC data")
                emc_item = emc_data[key]
                if emc_item.get("status") != "succeeded":
                    raise ValueError(f"EMC status for {key} is not 'succeeded': {emc_item.get('status')}")

                segments = emc_item.get("screened_timestamps", [[]])
                question = emc_item.get("screened_question", question)
                frame_indices = sample_frame_indices_from_segments(
                    segments, total_frames, fps, num_frames, sampling_mode
                )
            else:
                frame_indices = sample_frame_indices_uniform(total_frames, num_frames)

            # Load frames
            frames = load_video_frames(video_path, frame_indices)

            # Format prompt based on dataset type
            if is_open_ended:
                prompt = format_open_ended_prompt(question)
            else:
                prompt = format_mcq_prompt(question, options)

            # Run inference (with timeout for EMCompress)
            print(f"\n>>> Processing key: {key} | video: {os.path.basename(video_path)}")
            if dataset_name == "EMCompress":
                # Apply 60-second timeout for EMCompress
                @timeout(60)
                def infer_with_timeout():
                    return infer_func(model_dict, frames, prompt)
                response = infer_with_timeout()
            else:
                response = infer_func(model_dict, frames, prompt)
            print(f"Response: {response[:200]}...")

            if is_open_ended:
                # Evaluate with GPT-4o
                score = evaluate_open_ended_with_gpt4o(question, answer, response)
                total_score += score
                total += 1

                results[key] = {
                    "video_path": video_path,
                    "question": question,
                    "answer": answer,
                    "prediction": response,
                    "score": score,
                    "num_frames_used": len(frame_indices),
                }
            else:
                # Extract answer letter
                cleaned = ''.join(c for c in response.strip().upper() if c.isalpha())
                if len(cleaned) == 1 and cleaned in 'ABCDEF':
                    pred_letter = cleaned
                else:
                    pred_letter = extract_mcq_answer_with_gpt(question, options, response)

                gt_letter = answer.strip().upper()[0] if answer.strip() else ""
                is_correct = pred_letter == gt_letter
                if is_correct:
                    correct += 1
                total += 1

                results[key] = {
                    "video_path": video_path,
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "gt_letter": gt_letter,
                    "prediction": pred_letter,
                    "raw_response": response,
                    "correct": is_correct,
                    "num_frames_used": len(frame_indices),
                }

            # Save rank results
            with open(rank_output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            if rank == 0:
                print(f"Error processing {key}: {e}")
            import traceback
            traceback.print_exc()
            results[key] = {"error": str(e)}
            with open(rank_output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            continue

    # Save rank results
    with open(rank_output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if is_open_ended:
        avg_score = total_score / total if total > 0 else 0
        if rank == 0:
            print(f"\nRank {rank} Results: avg_score={avg_score:.2f} over {total} samples")
    else:
        accuracy = correct / total if total > 0 else 0
        if rank == 0:
            print(f"\nRank {rank} Results: {correct}/{total} = {accuracy:.4f}")

    if rank == 0:
        print(f"Saved to {rank_output_path}")

    # Synchronize all ranks
    if is_distributed:
        dist.barrier()

    # Merge results on rank 0
    if rank == 0:
        print("\nMerging results from all ranks...")
        all_results = {}
        for r in range(world_size):
            rank_file = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset_name}{suffix}_rank{r}.json")
            if os.path.exists(rank_file):
                with open(rank_file) as f:
                    all_results.update(json.load(f))

        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Delete rank files
        for r in range(world_size):
            rank_file = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset_name}{suffix}_rank{r}.json")
            if os.path.exists(rank_file):
                os.remove(rank_file)
                print(f"Deleted temporary file: {rank_file}")

        # Calculate final metrics
        if is_open_ended:
            scores = [v.get("score", 0) for v in all_results.values() if "score" in v]
            final_avg_score = sum(scores) / len(scores) if scores else 0
            print(f"\nFinal Results: avg_score={final_avg_score:.2f} over {len(scores)} samples")
        else:
            total_correct = sum(1 for v in all_results.values() if v.get("correct"))
            total_count = len([v for v in all_results.values() if "correct" in v])
            final_accuracy = total_correct / total_count if total_count > 0 else 0
            print(f"\nFinal Results: {total_correct}/{total_count} = {final_accuracy:.4f}")

        print(f"Saved to {output_path}")

        return final_avg_score if is_open_ended else final_accuracy

    return avg_score if is_open_ended else accuracy


def run_inference_openai_multithread(model_name, dataset_name, use_emc=False, force_redo=False, num_threads=200):
    """
    Run inference for OpenAI API models using multi-threading.
    This is optimized for API-based models that don't require GPU.
    """
    print(f"\n{'='*60}")
    print(f"Running OpenAI API inference with {num_threads} threads")
    print(f"Model: {model_name}, Dataset: {dataset_name}, EMC: {use_emc}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset_config = DATASET_CONFIGS[dataset_name]
    data = load_test_split(dataset_config["data_dir"])

    # Check if open-ended
    is_open_ended = is_open_ended_dataset(data)
    print(f"Dataset type: {'Open-ended QA' if is_open_ended else 'Multiple Choice QA'}")

    # Prepare output paths
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix = "_emc" if use_emc else "_original"
    output_path = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset_name}{suffix}.json")

    # Load EMC results if needed
    emc_data = None
    if use_emc:
        emc_path = dataset_config["emc_result"]
        if not os.path.exists(emc_path):
            raise FileNotFoundError(f"EMC result not found at {emc_path}")
        with open(emc_path) as f:
            raw_emc = json.load(f)
        if dataset_name not in raw_emc:
            raise ValueError(f"No EMC data for {dataset_name} in {emc_path}")
        emc_data = raw_emc[dataset_name]

    # Load model config
    model_config = MODEL_CONFIGS[model_name]
    api_model_name = model_config["api_model_name"]
    num_frames = model_config["num_frames"]
    print(f"Initializing {model_name} (OpenAI API: {api_model_name})...")

    # Load existing results for incremental processing
    results = {}
    if not force_redo and os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {output_path}")

    # Determine keys to process
    if force_redo:
        all_keys = list(data.keys())
    else:
        all_keys = [key for key in data.keys() if key not in results]

    print(f"Total samples: {len(data)}, Samples to process: {len(all_keys)}")

    if len(all_keys) == 0:
        print("All samples already processed. Skipping.")
        return

    # Thread-safe data structures
    results_lock = threading.Lock()
    keys_lock = threading.Lock()
    pbar = tqdm(total=len(all_keys), desc=f"{model_name} on {dataset_name}")

    def thread_worker():
        """Worker function for each thread."""
        # Each thread creates its own OpenAI client
        client = OpenAI(api_key=openai_api_key)

        while True:
            # Get next key to process
            with keys_lock:
                if len(all_keys) == 0:
                    return
                key = all_keys.pop(0)

            retry_count = 3
            while retry_count > 0:
                try:
                    item = data[key]
                    video_path = item["video_path"]
                    question = item["question"]
                    answer = item["answer"]
                    options = item.get("options", [])

                    # Get video metadata using fast cv2 method (0.008s vs 0.07s with decord)
                    total_frames, fps = get_video_metadata_fast(video_path)

                    # Determine frame indices
                    if use_emc and emc_data:
                        if key not in emc_data:
                            raise KeyError(f"Key {key} not found in EMC data")
                        emc_item = emc_data[key]
                        if emc_item.get("status") != "succeeded":
                            raise ValueError(f"EMC status for {key} is not 'succeeded': {emc_item.get('status')}")

                        segments = emc_item.get("screened_timestamps", [[]])
                        question = emc_item.get("screened_question", question)
                        frame_indices = sample_frame_indices_from_segments(
                            segments, total_frames, fps, num_frames, "uniform"
                        )
                    else:
                        frame_indices = sample_frame_indices_uniform(total_frames, num_frames)

                    # Try to load from cache first (with tolerance matching)
                    cached_base64 = load_base64_cache(video_path, frame_indices, fps)

                    if cached_base64 is not None:
                        # Cache hit - skip video frame loading entirely
                        base64_frames = cached_base64
                        frames = None  # Not needed
                    else:
                        # Cache miss - load frames from video
                        frames = load_video_frames(video_path, frame_indices)

                    # Format prompt based on dataset type
                    if is_open_ended:
                        prompt = format_open_ended_prompt(question)
                    else:
                        prompt = format_mcq_prompt(question, options)

                    # Build the system prompt
                    system_prompt = (
                        "You are a Video Question Answering assistant. "
                        "You will be shown 8 images that are uniformly sampled frames from a video. "
                        "Please analyze these frames carefully to understand the video content and answer the question."
                    )

                    # Build the user message content with images
                    content = []
                    content.append({
                        "type": "text",
                        "text": "Here are 8 uniformly sampled frames from a video:"
                    })

                    # Get base64 encoded frames (use cached if available, otherwise encode and cache)
                    if cached_base64 is None:
                        # Cache miss - encode frames and save to cache
                        base64_frames = encode_frames_to_base64_cached(video_path, frame_indices, frames, fps)
                    # else: base64_frames already set from cache hit above

                    # Add each frame as base64 encoded image
                    for base64_image in base64_frames:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "auto"
                            }
                        })

                    # Add the question/prompt
                    content.append({
                        "type": "text",
                        "text": f"\n{prompt}"
                    })

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ]

                    # Call the API
                    response_obj = client.chat.completions.create(
                        model=api_model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=256
                    )
                    response = response_obj.choices[0].message.content

                    # Process response
                    if is_open_ended:
                        # Evaluate with GPT-4o
                        score = evaluate_open_ended_with_gpt4o(question, answer, response)

                        result = {
                            "video_path": video_path,
                            "question": question,
                            "answer": answer,
                            "prediction": response,
                            "score": score,
                            "num_frames_used": len(frame_indices),
                        }
                    else:
                        # Extract answer letter
                        cleaned = ''.join(c for c in response.strip().upper() if c.isalpha())
                        if len(cleaned) == 1 and cleaned in 'ABCDEF':
                            pred_letter = cleaned
                        else:
                            pred_letter = extract_mcq_answer_with_gpt(question, options, response)

                        gt_letter = answer.strip().upper()[0] if answer.strip() else ""
                        is_correct = pred_letter == gt_letter

                        result = {
                            "video_path": video_path,
                            "question": question,
                            "options": options,
                            "answer": answer,
                            "gt_letter": gt_letter,
                            "prediction": pred_letter,
                            "raw_response": response,
                            "correct": is_correct,
                            "num_frames_used": len(frame_indices),
                        }

                    # Save result thread-safely
                    with results_lock:
                        results[key] = result
                        # Save periodically (every result for safety)
                        with open(output_path, 'w') as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)

                    break  # Success, exit retry loop

                except Exception as e:
                    retry_count -= 1
                    if retry_count <= 0:
                        # Record error
                        with results_lock:
                            results[key] = {"error": str(e)}
                            with open(output_path, 'w') as f:
                                json.dump(results, f, indent=2, ensure_ascii=False)
                        print(f"\nError processing {key}: {e}")
                    else:
                        time.sleep(1)  # Brief pause before retry
                        continue

            pbar.update(1)

    # Start threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=thread_worker)
        threads.append(t)

    for t in threads:
        t.start()
        time.sleep(0.05)  # Small delay to avoid API rate limit issues at start

    # Wait for all threads to complete
    for t in threads:
        t.join()

    pbar.close()

    # Calculate final metrics
    if is_open_ended:
        scores = [v.get("score", 0) for v in results.values() if "score" in v]
        final_avg_score = sum(scores) / len(scores) if scores else 0
        print(f"\nFinal Results: avg_score={final_avg_score:.2f} over {len(scores)} samples")
    else:
        total_correct = sum(1 for v in results.values() if v.get("correct"))
        total_count = len([v for v in results.values() if "correct" in v])
        final_accuracy = total_correct / total_count if total_count > 0 else 0
        print(f"\nFinal Results: {total_correct}/{total_count} = {final_accuracy:.4f}")

    print(f"Saved to {output_path}")


def check_already_completed(model_name, dataset_name, use_emc, force_redo):
    """
    Check if all samples are already processed before initializing distributed.
    Returns (is_completed, message) tuple.
    """
    if force_redo:
        return False, None

    # Load dataset to get all keys
    dataset_config = DATASET_CONFIGS[dataset_name]
    data = load_test_split(dataset_config["data_dir"])

    # Check output file
    suffix = "_emc" if use_emc else "_original"
    output_path = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset_name}{suffix}.json")

    if not os.path.exists(output_path):
        return False, None

    with open(output_path) as f:
        existing_results = json.load(f)

    # Check if all keys are processed
    missing_keys = [key for key in data.keys() if key not in existing_results]

    if len(missing_keys) == 0:
        # All done, compute metrics for display
        is_open_ended = is_open_ended_dataset(data)
        if is_open_ended:
            scores = [v.get("score", 0) for v in existing_results.values() if "score" in v]
            avg_score = sum(scores) / len(scores) if scores else 0
            msg = f"All {len(data)} samples already processed. Skipping.\n" \
                  f"Existing results: avg_score={avg_score:.2f} over {len(scores)} samples"
        else:
            total_correct = sum(1 for v in existing_results.values() if v.get("correct"))
            total_count = len([v for v in existing_results.values() if "correct" in v])
            accuracy = total_correct / total_count if total_count > 0 else 0
            msg = f"All {len(data)} samples already processed. Skipping.\n" \
                  f"Existing results: {total_correct}/{total_count} = {accuracy:.4f}"
        return True, msg

    return False, f"Found {len(existing_results)} existing results, {len(missing_keys)} samples remaining."


def main():
    parser = argparse.ArgumentParser(description="Baseline Video QA Inference")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to evaluate on")
    parser.add_argument("--emc", action="store_true",
                        help="Use EMC-processed results")
    parser.add_argument("--force_redo", action="store_true",
                        help="Force redo all inference, ignore existing results")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    parser.add_argument("--num_threads", type=int, default=200,
                        help="Number of threads for OpenAI API models (default: 200)")
    args = parser.parse_args()

    # Check if this is an OpenAI API model
    model_config = MODEL_CONFIGS[args.model]
    is_openai_api = model_config["model_type"] == "openai_api"

    if is_openai_api:
        # Use multi-threaded inference for OpenAI API models
        print(f"\n{'='*60}")
        print(f"Model: {args.model}, Dataset: {args.dataset}, EMC: {args.emc}, Force Redo: {args.force_redo}")
        print(f"Using multi-threaded inference with {args.num_threads} threads")
        print(f"{'='*60}")

        run_inference_openai_multithread(
            args.model, args.dataset, args.emc, args.force_redo, args.num_threads
        )
    else:
        # Use original logic for local models
        # Early exit check BEFORE distributed initialization (only on rank 0 or single process)
        # For torchrun, only rank 0 should print, but all ranks should exit
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Model: {args.model}, Dataset: {args.dataset}, EMC: {args.emc}, Force Redo: {args.force_redo}")
            print(f"{'='*60}")

        is_completed, msg = check_already_completed(args.model, args.dataset, args.emc, args.force_redo)
        if is_completed:
            if rank == 0:
                print(msg)
            return  # Exit without initializing distributed

        if msg and rank == 0:
            print(msg)

        # Initialize distributed if using torchrun
        global DEVICE_MAP
        if args.local_rank != -1 or 'RANK' in os.environ:
            if 'RANK' in os.environ:
                rank = int(os.environ['RANK'])
                local_rank = int(os.environ['LOCAL_RANK'])
                world_size = int(os.environ['WORLD_SIZE'])
            else:
                rank = args.local_rank
                local_rank = args.local_rank
                world_size = torch.cuda.device_count()

            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')

            # Set device map to only use the current GPU for this rank
            DEVICE_MAP = {"": f"cuda:{local_rank}"}

            if rank == 0:
                print(f"Initialized distributed training: world_size={world_size}, rank={rank}")
                print(f"Each rank uses its own GPU (device_map per rank)")

        run_inference(args.model, args.dataset, args.emc, args.force_redo)

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
