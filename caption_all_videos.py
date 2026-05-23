"""
Video Frame Caption Script — multi-backend (OpenAI API or HF local VLM).

Sampling strategy:
- Videos < 5 min: 1 frame per second
- Videos 5 min ~ 1 hour: 1 frame per 2 seconds
- Videos > 1 hour: 1 frame per 5 seconds

Output format:
- One JSON file per video: {video_id}.json
- JSON content: {"0": "caption for frame at 0s", "1": ..., ...}

Supported backends (auto-detected by --model):
- OpenAI API:       gpt-4o, gpt-4, gpt-5.2, gpt-4.1-mini, gpt-4-turbo, ... (anything starting with gpt-/o)
- Qwen-VL family:   Qwen/Qwen2.5-VL-{3B,7B,...}-Instruct, Qwen/Qwen3-VL-{4B,32B,...}-Instruct
- LLaVA-1.5 family: llava-hf/llava-1.5-{7,13}b-hf
- LLaVA-NeXT/1.6:   llava-hf/llava-v1.6-mistral-7b-hf, llava-hf/llava-v1.6-vicuna-13b-hf

Launching:
- API backend (single process, multi-thread):
    python caption_all_videos.py --model gpt-4o --num_threads 32
- Local VLM (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 caption_all_videos.py --model Qwen/Qwen3-VL-32B-Instruct
"""

import os
import json
import argparse
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DATASETS_DIR, openai_api_key
from emc_utils.utils import load_test_split


# ============ Configuration ============

DATASETS = {
    name: {
        "data_dir": os.path.join(DATASETS_DIR, name),
        "caption_output_dir": os.path.join(DATASETS_DIR, name, "captions_1fps"),
    }
    for name in ["EgoSchema", "LVBench", "MLVU", "Video-MME"]
}

CAPTION_PROMPT = "Describe this image in detail. Focus on the main objects, actions, and scene."


# ============ Frame extraction ============

def get_sample_interval(duration_seconds):
    """1 fps if <5min, 0.5 fps if <1h, else 0.2 fps."""
    if duration_seconds < 300:
        return 1
    elif duration_seconds < 3600:
        return 2
    else:
        return 5


def extract_frames_at_timestamps(video_path, timestamps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = {}
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ret, frame = cap.read()
        if ret:
            frames[ts] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps if fps > 0 else None


def get_timestamps_for_video(video_path):
    duration = get_video_duration(video_path)
    if duration is None:
        return [], None
    interval = get_sample_interval(duration)
    return list(range(0, int(duration), interval)), duration


# ============ Captioner backends ============

class _Captioner:
    """Common interface: .caption(PIL.Image) -> str."""
    def caption(self, image):
        raise NotImplementedError


class OpenAICaptioner(_Captioner):
    """OpenAI vision API. Works for gpt-4o, gpt-4-turbo, gpt-5.2, gpt-4.1-mini, etc."""
    def __init__(self, model_name):
        from openai import OpenAI
        import base64, io
        self._b64 = base64
        self._io = io
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model_name
        # gpt-5* / o1 / o3 reasoning models use max_completion_tokens; older gpt-4* use max_tokens
        self._uses_max_completion = model_name.startswith("gpt-5") or model_name.startswith("o")

    def _to_b64(self, image):
        buf = self._io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return self._b64.b64encode(buf.getvalue()).decode("utf-8")

    def caption(self, image):
        b64 = self._to_b64(image)
        kwargs = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }],
        }
        if self._uses_max_completion:
            kwargs["max_completion_tokens"] = 256
        else:
            kwargs["max_tokens"] = 256
            kwargs["temperature"] = 0.2
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content


class QwenVLCaptioner(_Captioner):
    """Qwen2.5-VL / Qwen3-VL family."""
    def __init__(self, model_path, device):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        self._process_vision_info = process_vision_info
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def caption(self, image):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


class LlavaCaptioner(_Captioner):
    """LLaVA-1.5 (llava-hf/llava-1.5-{7,13}b-hf)."""
    def __init__(self, model_path, device):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def caption(self, image):
        conv = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}],
        }]
        prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.bfloat16)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.processor.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


class LlavaNextCaptioner(_Captioner):
    """LLaVA-NeXT / 1.6 (llava-hf/llava-v1.6-{mistral-7b,vicuna-13b}-hf)."""
    def __init__(self, model_path, device):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        self.device = device
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device,
        )
        self.processor = LlavaNextProcessor.from_pretrained(model_path)

    def caption(self, image):
        conv = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}],
        }]
        prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.bfloat16)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.processor.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def make_captioner(model, device):
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return OpenAICaptioner(model), False  # is_local=False
    low = model.lower()
    if "qwen" in low:
        return QwenVLCaptioner(model, device), True
    if any(k in low for k in ("llava-v1.6", "llava-1.6", "llava-next")):
        return LlavaNextCaptioner(model, device), True
    if "llava" in low:
        return LlavaCaptioner(model, device), True
    raise ValueError(
        f"Unsupported model: {model}. Supported families: gpt-* (OpenAI API), "
        f"Qwen-VL, LLaVA-1.5, LLaVA-NeXT."
    )


# ============ Per-video pipeline ============

def process_single_video(video_path, output_path, captioner, num_threads=1):
    timestamps, duration = get_timestamps_for_video(video_path)
    if not timestamps:
        print(f"  Warning: could not get timestamps for {video_path}")
        return False
    print(f"  Duration: {duration:.1f}s, Interval: {get_sample_interval(duration)}s, Frames: {len(timestamps)}")

    try:
        frames = extract_frames_at_timestamps(video_path, timestamps)
    except Exception as e:
        print(f"  Error extracting frames: {e}")
        return False

    captions = {}

    def _do(ts):
        try:
            return ts, captioner.caption(frames[ts])
        except Exception as e:
            return ts, f"[caption_error: {e}]"

    work = [ts for ts in timestamps if ts in frames]
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            for fut in tqdm(as_completed([ex.submit(_do, ts) for ts in work]),
                            total=len(work), desc="  Captioning", leave=False):
                ts, cap = fut.result()
                captions[str(ts)] = cap
    else:
        for ts in tqdm(work, desc="  Captioning", leave=False):
            _, cap = _do(ts)
            captions[str(ts)] = cap

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    return True


def collect_all_videos():
    all_videos, seen = [], set()
    for dataset_name, config in DATASETS.items():
        data_dir = config["data_dir"]
        caption_output_dir = config["caption_output_dir"]
        try:
            data = load_test_split(data_dir)
        except FileNotFoundError as e:
            print(f"Warning: dataset files not found in {data_dir} ({e}), skipping {dataset_name}")
            continue
        for key, item in data.items():
            vp = item.get("video_path", "")
            if vp and vp not in seen:
                seen.add(vp)
                vid_id = os.path.splitext(os.path.basename(vp))[0]
                all_videos.append((vp, os.path.join(caption_output_dir, f"{vid_id}.json"), dataset_name))
    return all_videos


def main():
    parser = argparse.ArgumentParser(description="Caption video frames with a configurable VLM backend.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model id. OpenAI: gpt-4o/gpt-5.2/.... HF: Qwen/Qwen3-VL-32B-Instruct, "
                             "llava-hf/llava-1.5-7b-hf, llava-hf/llava-v1.6-mistral-7b-hf, ...")
    parser.add_argument("--dataset", type=str, default=None, help="Process only this dataset (e.g. EgoSchema)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip videos with existing caption JSON")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Parallel threads per process (only meaningful for OpenAI API backend)")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))

    print(f"Process {local_rank}/{world_size} — model={args.model}")

    captioner, is_local = make_captioner(args.model, device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    all_videos = collect_all_videos()
    if args.dataset:
        all_videos = [(v, o, d) for v, o, d in all_videos if d == args.dataset]
    if args.skip_existing:
        all_videos = [(v, o, d) for v, o, d in all_videos if not os.path.exists(o)]
    print(f"Total videos to process: {len(all_videos)}")

    videos_for_this_rank = all_videos[local_rank::world_size]
    print(f"Rank {local_rank} will process {len(videos_for_this_rank)} videos")

    success, failed = 0, 0
    for i, (vp, op, ds) in enumerate(videos_for_this_rank):
        print(f"\n[{i+1}/{len(videos_for_this_rank)}] {os.path.basename(vp)} ({ds})")
        if not os.path.exists(vp):
            print(f"  Video not found: {vp}")
            failed += 1
            continue
        try:
            if process_single_video(vp, op, captioner, num_threads=args.num_threads):
                success += 1
                print(f"  Saved to: {op}")
            else:
                failed += 1
        except Exception as e:
            print(f"  Error: {e}")
            failed += 1

    print(f"\n{'='*50}\nRank {local_rank} done. Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
