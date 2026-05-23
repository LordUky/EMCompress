"""
EMC Simple Baseline (ReSimplifyIt_simple)
- Single Agent (GPT-5.2), single round
- Full trajectory logging: every LLM call, every caption request, timestamps, etc.
- Multi-threaded for parallel API calls
- Uses Qwen-VL-Plus via DashScope for image captioning
- Uses GPT-4o for planning

Usage:
    python run_emc_simple_baseline.py --dataset ActivityNet-QA --num_threads 50
    python run_emc_simple_baseline.py --dataset all --num_threads 50
"""

import cv2
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import time
import openai
from openai import OpenAI
import math
import json
import base64
import threading
import argparse
import copy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Suppress OpenCV/FFmpeg warnings and reduce read attempts to avoid long hangs
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "512"

VIDEO_IO_TIMEOUT = 15  # seconds max for any single video I/O operation

from config import openai_api_key, DATASETS_DIR

_HERE = os.path.dirname(os.path.abspath(__file__))

# ============ Parse Arguments ============
parser = argparse.ArgumentParser(description='Run EMC Simple Baseline on datasets')
parser.add_argument('--dataset', type=str, required=True,
                    help='Dataset name or "all" for all 7 datasets')
parser.add_argument('--force_redo', action='store_true',
                    help='Force redo all samples even if already processed')
parser.add_argument('--num_threads', type=int, default=50,
                    help='Number of threads for parallel processing')
args = parser.parse_args()

ALL_DATASETS = [
    "ActivityNet-QA", "EMCompress", "EgoSchema",
    "LVBench", "MLVU", "Video-MME", "NExT-OE"
]

datasets_to_run = ALL_DATASETS if args.dataset == "all" else [args.dataset]

# ============ Configuration ============
CAPTION_PROMPT = "Describe this image in detail. Include subtitle if present."

CORE_MODEL = "gpt-4o"

# Initialize clients
client = OpenAI(api_key=openai_api_key)

# Base paths
DATA_BASE = DATASETS_DIR
OUTPUT_BASE = os.path.join(_HERE, "results_simple_baseline")
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Thread locks
cache_lock = threading.Lock()
output_lock = threading.Lock()

import tempfile

def _atomic_json_write(filepath, data):
    """Write JSON atomically via temp file + rename to prevent corruption on kill."""
    dirn = os.path.dirname(filepath) or '.'
    fd, tmp = tempfile.mkstemp(dir=dirn, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, filepath)
    except:
        try:
            os.unlink(tmp)
        except:
            pass
        raise


# ============ Video / Frame Utilities ============

def _get_video_info_inner(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        duration = frame_count / fps
        return duration, fps
    return None, None


def get_video_info(video_path):
    """Get video info with timeout protection."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_get_video_info_inner, video_path)
        try:
            return future.result(timeout=VIDEO_IO_TIMEOUT)
        except (FuturesTimeoutError, Exception):
            return None, None


def _extract_frame_inner(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    if not ret:
        ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None


def extract_frame_at_timestamp(video_path, timestamp):
    """Extract frame with timeout protection."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_extract_frame_inner, video_path, timestamp)
        try:
            return future.result(timeout=VIDEO_IO_TIMEOUT)
        except (FuturesTimeoutError, Exception):
            return None
    return None


def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============ Caption Utilities ============

def get_caption_from_vlm(image):
    """Call Qwen-VL-Plus API to get image caption."""
    base64_image = image_to_base64(image)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": CAPTION_PROMPT}
            ]
        }],
        max_tokens=500,
        temperature=0.2
    )
    return response.choices[0].message.content


def load_caption_cache(caption_cache_dir, video_id):
    """Load caption cache for a video."""
    cache_path = os.path.join(caption_cache_dir, f"{video_id}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_caption_cache(caption_cache_dir, video_id, cache_dict):
    """Save caption cache for a video (thread-safe)."""
    cache_path = os.path.join(caption_cache_dir, f"{video_id}.json")
    with cache_lock:
        existing = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except:
                pass
        existing.update(cache_dict)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)


def get_caption_with_cache_and_log(video_path, video_id, timestamp, vid_duration, caption_cache_dir):
    """
    Get caption for a frame with caching. Returns (caption, actual_timestamp, was_cached).
    """
    timestamp = round(timestamp)
    max_ts = max(0, int(vid_duration) - 1)
    timestamp = max(0, min(timestamp, max_ts))
    ts_key = str(timestamp)

    cache = load_caption_cache(caption_cache_dir, video_id)
    if ts_key in cache:
        return cache[ts_key], timestamp, True  # cached

    frame = extract_frame_at_timestamp(video_path, timestamp)
    if frame is None:
        caption = "[Failed to extract frame]"
    else:
        caption = get_caption_from_vlm(frame)

    save_caption_cache(caption_cache_dir, video_id, {ts_key: caption})
    return caption, timestamp, False  # not cached


def get_initial_captions_for_video(video_path, video_id, vid_duration, caption_cache_dir, num_samples=20):
    """Get initial captions for evenly sampled frames. Returns (captions_dict, log_entries)."""
    max_ts = max(0, int(vid_duration) - 1)
    if max_ts == 0:
        timestamps = [0]
    else:
        step = max(1, max_ts // num_samples)
        timestamps = list(range(0, max_ts + 1, step))[:num_samples]

    captions = {}
    log_entries = []
    for ts in timestamps:
        cap, actual_ts, was_cached = get_caption_with_cache_and_log(
            video_path, video_id, ts, vid_duration, caption_cache_dir
        )
        captions[str(actual_ts)] = cap
        log_entries.append({
            "requested_timestamp": ts,
            "actual_timestamp": actual_ts,
            "caption": cap,
            "was_cached": was_cached
        })

    return captions, log_entries


# ============ LLM Utilities ============

def call_llm(conversation, model=CORE_MODEL):
    """Call gpt-4o. Returns (response_text, usage_dict)."""
    completion = client.chat.completions.create(
        model=model,
        messages=conversation,
        max_tokens=1024
    )
    response_text = completion.choices[0].message.content
    usage = {
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "total_tokens": completion.usage.total_tokens,
    }
    return response_text, usage


def pickout_and_jsonize(s):
    """Extract JSON from 【...】 delimiters."""
    temp = s.split('【')[-1]
    temp = temp.split('】')[0]
    return json.loads(temp)


# ============ Prompt Template ============

INITIAL_PROMPT_TEMPLATE = '''
You are a assistant for the video question answering process, in which a candidate is presented with a video and a question for them to answer.\
Your objective is to help the candidate so that he will be able to give the answer with watching the shortest posible sub-clip(s) of the video. \
Your task is to cut the video to acquire this sub-clip(s) and also to modify the question, so that the candidate directly answering your modified question with presented only this sub-clip(s) of the video would be equivalent to answering the original question with presented the original whole uncut video. \
For example, when answering the question "what did the person do after putting down the dog", you will need to provide the sub-clip starting from approximately the last frame of the person put down the dog till the end, and the corresponding revised question would be "what did the person do in this video". \
In this way, if the the man in the video went preparing food after putting down the dog, then the answer "he went prepare food" would be simultaneously correct for both the original and revised questions. \
You will need to cut the video in the form of providing me the timestamps, which is a list of [start, end] unit clips in the unit of second. \
A tool (python function) will be helping you to get the frame caption of at a certain timestamp (in the unit of second). Whenever you need to call this tool, send a message in this json format: 【{"decision": "tool", "parameter": [timestamp you need]}】. For example, if you want the caption at 3.5s of the video, return 【{"decision": "tool", "parameter": 3.5}】. The corresponding caption will be returned to you. \
Whenever you think you are confident enough to provide the timestamp, return 【{"decision": "end", "timestamps": [your result timestamps], "revised_question": [your revised question]}】. For example, if you decide to preserve the sub-clip from 1.4s to 4.8s with the revised prompt being "what did the person do in this video", then return 【{"decision": "end", "timestamps": [[1.4, 4.8]], "revised_question": "what did the person do in this video"}】.

Before we formally begin, here is a set of original captions with their timestamps provided for you to have an overall rough understanding of the video: [initial_captions_flag].
Also, the frame rate of this video is [frame_rate_flag] frames per second, and the total duration is [duration_flag] seconds.

[sys_usr_split]Now let's begin! and the original question is "[original_question_flag]".
'''


# ============ Main Processing ============

def process_dataset(dataset_name):
    """Process a single dataset with the simple baseline pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")

    dataset_dir = f"{DATA_BASE}/{dataset_name}"
    caption_cache_dir = f"{dataset_dir}/captions_1fps"
    os.makedirs(caption_cache_dir, exist_ok=True)

    output_path = os.path.join(OUTPUT_BASE, f"simple_{dataset_name}.json")
    log_path = os.path.join(OUTPUT_BASE, f"simple_{dataset_name}_log.json")

    from emc_utils.utils import load_test_split
    try:
        input_data = load_test_split(dataset_dir)
    except FileNotFoundError as e:
        print(f"  Dataset files not found in {dataset_dir}: {e}, skipping.")
        return

    # Load or initialize outputs and logs
    if os.path.exists(output_path) and not args.force_redo:
        with open(output_path, 'r') as f:
            outputs = json.load(f)
    else:
        outputs = {}

    if os.path.exists(log_path) and not args.force_redo:
        with open(log_path, 'r') as f:
            all_logs = json.load(f)
    else:
        all_logs = {}

    # Determine which keys to process
    if args.force_redo:
        all_keys = list(input_data.keys())
    else:
        all_keys = []
        for key in input_data.keys():
            if key not in outputs or outputs[key].get('status') != 'succeeded' or key not in all_logs:
                all_keys.append(key)

    print(f"  Total samples: {len(input_data)}, to process: {len(all_keys)}")
    if len(all_keys) == 0:
        print(f"  Nothing to process for {dataset_name}.")
        return

    pbar = tqdm(total=len(all_keys), desc=dataset_name)

    def thread_run():
        while True:
            try:
                key = all_keys.pop(0)
            except IndexError:
                return

            sample_log = {
                "sample_id": key,
                "dataset": dataset_name,
                "start_time": datetime.now().isoformat(),
                "model": CORE_MODEL,
                "caption_model": "gpt-4o",
                "initial_captions": [],
                "conversation_turns": [],
                "tool_calls": [],
                "llm_calls": [],
                "result": None,
                "error": None,
            }

            fault_count = 5
            final_timestamps = None
            revised_question = None
            status = None

            while True:
                try:
                    item = input_data[key]
                    video_path = item['video_path']
                    question = item['question']
                    video_id = os.path.splitext(os.path.basename(video_path))[0]

                    sample_log["video_path"] = video_path
                    sample_log["original_question"] = question
                    sample_log["video_id"] = video_id

                    # Get video info
                    vid_duration, frame_rate = get_video_info(video_path)
                    if vid_duration is None:
                        raise Exception(f"Cannot open video: {video_path}")
                    sample_log["vid_duration"] = vid_duration
                    sample_log["frame_rate"] = frame_rate

                    # Get initial captions
                    initial_captions, init_cap_logs = get_initial_captions_for_video(
                        video_path, video_id, vid_duration, caption_cache_dir, num_samples=20
                    )
                    sample_log["initial_captions"] = init_cap_logs

                    # Build prompt
                    initial_prompt = INITIAL_PROMPT_TEMPLATE.replace(
                        "[initial_captions_flag]", str(initial_captions)
                    ).replace(
                        "[frame_rate_flag]", str(round(frame_rate, 2))
                    ).replace(
                        "[duration_flag]", str(int(vid_duration))
                    ).replace(
                        "[original_question_flag]", question
                    )

                    sys_prompt, usr_prompt = initial_prompt.split('[sys_usr_split]')

                    conversation = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt}
                    ]
                    sample_log["conversation_turns"].append({"role": "system", "content": sys_prompt})
                    sample_log["conversation_turns"].append({"role": "user", "content": usr_prompt})

                    turn_idx = 0
                    while True:
                        if len(conversation) > 100:
                            final_timestamps = [[0, vid_duration]]
                            revised_question = question
                            status = 'max_turns_exceeded'
                            break

                        err_count = 0
                        while True:
                            llm_call_log = {
                                "turn": turn_idx,
                                "call_time": datetime.now().isoformat(),
                                "model": CORE_MODEL,
                            }
                            response_text, usage = call_llm(conversation)
                            llm_call_log["response"] = response_text
                            llm_call_log["usage"] = usage
                            sample_log["llm_calls"].append(llm_call_log)

                            yes = True
                            try:
                                j = pickout_and_jsonize(response_text)
                                yes = yes and 'decision' in j
                                if j['decision'] == 'tool':
                                    yes = yes and isinstance(j["parameter"], (int, float)) and j["parameter"] >= 0
                                elif j['decision'] == 'end':
                                    yes = yes and "timestamps" in j and "revised_question" in j
                                    if j["timestamps"] and isinstance(j["timestamps"], list):
                                        for ele in j["timestamps"]:
                                            if isinstance(ele, list) and len(ele) >= 2:
                                                yes = yes and ele[0] >= 0 and ele[1] >= 0
                            except Exception as e:
                                err_count += 1
                                yes = False
                                if err_count >= 10:
                                    j = {'decision': 'end', 'timestamps': [[0, vid_duration]], 'revised_question': question}
                                    break
                            if yes:
                                conversation.append({"role": "assistant", "content": response_text})
                                sample_log["conversation_turns"].append({"role": "assistant", "content": response_text})
                                break

                        if j['decision'] == 'end':
                            final_timestamps = j["timestamps"]
                            revised_question = j['revised_question']

                            # Validate timestamps
                            if final_timestamps and isinstance(final_timestamps, list):
                                validated = []
                                for ts_pair in final_timestamps:
                                    if isinstance(ts_pair, list) and len(ts_pair) >= 2:
                                        start = max(0, min(ts_pair[0], vid_duration))
                                        end = max(0, min(ts_pair[1], vid_duration))
                                        if start <= end:
                                            validated.append([start, end])
                                final_timestamps = validated if validated else [[0, vid_duration]]
                            else:
                                final_timestamps = [[0, vid_duration]]

                            status = 'succeeded'
                            break
                        else:
                            # Tool call: get caption
                            request_ts = j["parameter"]
                            tool_call_log = {
                                "turn": turn_idx,
                                "requested_timestamp": request_ts,
                                "call_time": datetime.now().isoformat(),
                            }

                            cap, actual_ts, was_cached = get_caption_with_cache_and_log(
                                video_path, video_id, request_ts, vid_duration, caption_cache_dir
                            )
                            tool_call_log["actual_timestamp"] = actual_ts
                            tool_call_log["caption"] = cap
                            tool_call_log["was_cached"] = was_cached
                            sample_log["tool_calls"].append(tool_call_log)

                            if round(request_ts) != actual_ts:
                                cap = f"[Note: timestamp {request_ts} was adjusted to {actual_ts}] {cap}"

                            conversation.append({"role": "user", "content": cap})
                            sample_log["conversation_turns"].append({"role": "user", "content": cap})

                        turn_idx += 1

                    if final_timestamps in ([[]], [], None) or not isinstance(final_timestamps, list):
                        raise Exception('final_timestamps is empty or not a list')

                    break  # success, exit fault retry loop

                except Exception as e:
                    fault_count -= 1
                    sample_log["error"] = str(e)
                    if fault_count <= 0:
                        final_timestamps = [[0, vid_duration if vid_duration else 180]]
                        revised_question = question if question else ""
                        status = 'failed'
                        break
                    else:
                        time.sleep(1)
                        continue

            sample_log["end_time"] = datetime.now().isoformat()
            sample_log["result"] = {
                "screened_timestamps": final_timestamps,
                "screened_question": revised_question,
                "status": status,
            }

            # Save results and logs (thread-safe)
            with output_lock:
                outputs[key] = {
                    "screened_timestamps": final_timestamps,
                    "screened_question": revised_question,
                    "status": status,
                }
                all_logs[key] = sample_log
                try:
                    _atomic_json_write(output_path, outputs)
                    _atomic_json_write(log_path, all_logs)
                except:
                    pass

            pbar.update(1)

    # Launch threads
    num_threads = args.num_threads
    if num_threads > 1:
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=thread_run)
            threads.append(t)
            t.start()
            time.sleep(0.1)
        for t in threads:
            t.join()
    else:
        thread_run()

    pbar.close()
    print(f"  Done! Results: {output_path}")
    print(f"  Logs:    {log_path}")


# ============ Main Entry ============

if __name__ == "__main__":
    print(f"EMC Simple Baseline")
    print(f"Model: {CORE_MODEL}")
    print(f"Caption model: gpt-4o")
    print(f"Datasets: {datasets_to_run}")
    print(f"Threads: {args.num_threads}")
    print()

    for ds in datasets_to_run:
        process_dataset(ds)

    print("\nAll datasets processed!")
