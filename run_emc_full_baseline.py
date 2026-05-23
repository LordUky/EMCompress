"""
EMC Full Baseline (ReSimplifyIt)
- Multi-Agent (Launcher → Validator → Viewer), multi-round iterative pipeline
- Full trajectory logging: every LLM call per agent, every caption, timestamps, etc.
- Multi-threaded for parallel API calls
- Uses Qwen-VL-Plus via DashScope for image captioning (on-the-fly, with caching)
- Uses GPT-4o for all agents

Usage:
    python run_emc_full_baseline.py --dataset ActivityNet-QA --num_threads 30
    python run_emc_full_baseline.py --dataset all --num_threads 30
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
from math import floor
import json
import base64
import re
import threading
import argparse
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
parser = argparse.ArgumentParser(description='Run EMC Full Baseline on datasets')
parser.add_argument('--dataset', type=str, required=True,
                    help='Dataset name or "all" for all 7 datasets')
parser.add_argument('--force_redo', action='store_true',
                    help='Force redo all samples even if already processed')
parser.add_argument('--num_threads', type=int, default=30,
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
TOOL_MODEL = "gpt-4o"  # same model for all agents

client = OpenAI(api_key=openai_api_key)

DATA_BASE = DATASETS_DIR
OUTPUT_BASE = os.path.join(_HERE, "results_full_baseline")
os.makedirs(OUTPUT_BASE, exist_ok=True)

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


# ============ Video / Frame / Caption Utilities ============

def _get_video_info_inner(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps, fps
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
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None


def extract_frame_at_timestamp(video_path, timestamp):
    """Extract frame with timeout protection."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_extract_frame_inner, video_path, timestamp)
        try:
            return future.result(timeout=VIDEO_IO_TIMEOUT)
        except (FuturesTimeoutError, Exception):
            return None


def image_to_base64(image):
    import io
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_caption_from_vlm(image):
    b64 = image_to_base64(image)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": CAPTION_PROMPT}
        ]}],
        max_tokens=500, temperature=0.2
    )
    return resp.choices[0].message.content


def load_caption_cache(caption_cache_dir, video_id):
    path = os.path.join(caption_cache_dir, f"{video_id}.json")
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_caption_cache(caption_cache_dir, video_id, updates):
    path = os.path.join(caption_cache_dir, f"{video_id}.json")
    with cache_lock:
        existing = {}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except:
                pass
        existing.update(updates)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)


def get_caption_at_physical_ts(video_path, video_id, physical_ts, vid_duration, caption_cache_dir):
    """Get caption at a physical timestamp. Returns (caption, was_cached)."""
    ts_key = str(round(max(0, min(physical_ts, vid_duration - 1))))
    cache = load_caption_cache(caption_cache_dir, video_id)
    if ts_key in cache:
        return cache[ts_key], True
    frame = extract_frame_at_timestamp(video_path, float(ts_key))
    if frame is None:
        caption = "[Failed to extract frame]"
    else:
        caption = get_caption_from_vlm(frame)
    save_caption_cache(caption_cache_dir, video_id, {ts_key: caption})
    return caption, False


# ============ LLM Utilities ============

def call_llm(conversation, model=CORE_MODEL, log_list=None):
    """Call LLM. Returns response text. Appends to log_list if provided."""
    completion = client.chat.completions.create(
        model=model,
        messages=conversation,
        max_tokens=1024
    )
    resp = completion.choices[0].message.content
    usage = {
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "total_tokens": completion.usage.total_tokens,
    }
    if log_list is not None:
        log_list.append({
            "time": datetime.now().isoformat(),
            "model": model,
            "response": resp,
            "usage": usage,
        })
    return resp


def call_llm_and_modify_conv(message, conversation, model=CORE_MODEL, log_list=None):
    """Append user message, call LLM, append assistant response. Returns response text."""
    conversation.append({"role": "user", "content": message})
    resp = call_llm(conversation, model=model, log_list=log_list)
    conversation.append({"role": "assistant", "content": resp})
    return resp


def extract_dict_from_response(response):
    pattern = re.compile(r'\{(.*?)\}', re.DOTALL)
    match = pattern.search(response)
    content = match.group(1)
    return json.loads('{' + content + '}')


# ============ Prompt Templates (from original ReSimplifyIt.py) ============

PROMPT_LAUNCHER = """
Given a video question answering case, i.e. a video, a question, and an answer, sometimes it is possible to cut the video by only keeping a sub-clip or several sub-clips (concatenate if so) to be the video output and simultaneously modify the text question to be the paired text output while keeping the answer consistent and unchanged, so that the answer is still completely compatible to the modified question and the obtained video clip.
We define this action as "screening". By doing screening, the video becomes shorter, and therefore introduces lower cost to the downstream video question answering model. When modifying the question, note that the downstream video question answering model will only be seeing the sub-clip and think that the shown sub-clip is the whole video, so make sure the modified question is perfectly paired and aligned with and adapted to the modification.
For example, provided the original video and its paired question "what happened from 00:30 to 00:45 in this video", we can achieve screening by: keep the clip from 00:30 to 00:45, and simultaneously change the quesion to "what happened in this video", so that answering the (video, question) pair before or after the transformation should be equivalent, thereby persering answer consistency;
For another example, provided the original video and its paired question "what is shown after [event A]", we can achieve screening by: keep the clip after [event A] ended, and simultaneously change the question to "what is shown in this video" to make the answer compatible;
For one more example, provided the original video and its paired question "What is shown in the video apart from [event A] and [event B]", we can achieve screening by: abort the clip(s) showing [event A] and [event B] and keep the remaining clip(s), and simultaneously change the question to "what is shown in this video" to make the answer compatible.

Sometimes, such screening can be repeatedly applied sequentially by several rounds, where the input in each round is the output video clip and the output modified question of its preceeding round. As mentioned, the answer consistency should be always ensured throughout the whole process, and is always completely compatible to the resulting video clip and modified question of every round. If succeeded, each round would make the video shorter and making the situation closer to the optimal case.
For example, provided the original video and its paired question "what event is the first shot after 01:54 showing in this video", we might break the screening process in two rounds:
In the first round, we keep the video clip from 01:54 to the end, and simultaneously change the question to "what event is the first shot of this video showing";
In the second round, where the input (video, question) pair is the output of the first round, we keep the clip from the start to the ending of the first show, and simultaneously change the question to "what event is this video showing".

Your duty is to help complete the screening. Due to some limit, you are only provided with the text question but not the video input.
Your task is to initiate an immediate plan for the screening operation, including a natrual language instruction telling which video clip(s) to keep (similar to examples above) and the paired modified text question. If you think the screening may compose more than one round, you only need to perform one round next.
As you cannot see the actual video, your plan might fail, if the downstream video editting agent taking and executing your instruction on video clipping finds it infeasible. Therefore, your plan is refered to as a 'trial'.
In your current situation, some rounds or trials of the screening process might have already been conducted, and privided as following information:
1, the 'failure history': it is about the history of failed previous trials of the screening of your current round. Here is the failure history for you (an empty indicates that you are making the first trial of this round): [failure_history_flag];
2, the 'success history': it is about the history of the one success trial of all previous round. Here is the failure history for you (an empty indicates that you are at the first round): [success_history_flag].

Now, here is the original question of this round for video question answering: "[quesion_flag]"
Again, please tell your plan which describes what part of the video should be kept. Also, give the modified question under the assumption that the video is processed smoothly according to your plan.

If you feel that there is no room to make such screening (e.g. when the question is being general like "what is this video about") so you feel that you shouldn't make any plan, you should decide to terminate the process.


Hints:
1. Before processing, remember to take a look in the 'failure history' and 'success history' information;
2. If you find that the success history contains a case whose modified_question and the description both significantly overlap with the ones you are about to make, then you should avoid making the same plan again. In this case, you should switch to a clear reasonable sensible alternative plan, or make a decision to terminate the modification process if you can't confidently find one;
3. If you find that the failure history contains a case whose modified_question and the description both significantly overlap with the ones you are about to make, or that any of the "reason" in the failure history records is going to make your attempt fail, then you should avoid making the same plan again. In this case, you should switch to a clear reasonable sensible alternative plan, or make a decision to terminate the modification process if you can't confidently find one.

Return your plan in this json format (keep in mind here, that your response should be in json format):
{"decision": [your decision, either "process" or "terminate"], "modified_question": [the modified question, or "N/A" if your previous decision is "terminate"], "description": [Description of what part of the video should be kept as wanted sub-clip. The description will be passed to downstream processer to validate. Return "N/A" if your decision is "terminate".]}
"""

PROMPT_VALIDATOR = """
You will be given a natrual language instruction telling you to trim a video, which instruction itself might be infeasible. The reason it might be infeasible is because the agent who gave the instruction had no access to the actual video content, so it might be infeasible if take the actual video contenet into consideration.

Therefore, I need you to be a helpful assistant to confirm if the trimming plan is feasible.
Specifically, your job is to act as a validator to validate whether it is feasible (whether the video content really supports the plan).
If it is feasible, you will need to implement the plan and return the resulting sub-clip of the plan in the form of a two-layer list. For example, [[1, 5]] means the resulting sub-clip is from 1s to 5s of original video, and [[1, 5], [11, 15]] means the resulting sub-clip is the concatenation of the two sub-clips (from 1s to 5s, and from 11s to 15s, respectively.)
If it is not feasible, you will need to tell the reason.

Here are some examples and explanations for infeasible trimming instructions:

[start of examples]

[example instruction 1]: "keep the video clip from 16s to 39s and discard everything else."
[possible reason for infeasibility]: total video length is 30s, so 39s exceeds upper limit.

[example instruction 2]: "keep the video clip after the boy left the room."
[possible reason for infeasibility]: the content "the boy left the room" doesn't exist in the video.

[example instruction 3]: "keep the video clip before the door opens."
[possible reason for infeasibility]: the content "the door opens" is already at the start of the video.

[end of examples]

You can invoke a viewer multiple times to acquire the video content (partial content each time), which viewer is a downstream module prepared to assist you.

Note that the viewer is capable to deal with two type of questions:

1. snippet rough scanning, e.g. "what is the video about from xxx second to xxx second"?
2. localizing, e.g. "which segment contains xxx (event/object)?"

Therefore, if [your decision] is "view", it is strongly advised that [your message] follows one of the above two example templates.

Here is the trimming instruction you need to validate: [plan_flag].
The video length is [video_length_flag] seconds, and its frame rate is [frame_rate_flag] fps.

Now, in each following turn of this conversation, you need to give your response in this json format: {"decision": [your decision], "message": [your message]}. This works as follows:
[your decision]: either "succeeded", "failed", or "view". "succeeded" means that the plan is successfully implemented, "failed" means that it is not, and by "view" you invoke the viewer to provide partial video content for you. If you choose "view", the viewer will take your message and return as you requested, and the conversation will continue. If you choose "succeeded" or "failed", it will be your final decision and the conversation will end.
[your message]: if you choose "succeeded" as your decision, then it should be the two-layer list as mentioned before, as the video edit result of the plan. If you choose "failed", this should be a brief reason on the failure (e.g. requested timestamp exceeds video length, video doesn't have the object/event needed, etc.). If you choose "view", this should be the question to ask the viewer about the video content.

Hint: it is not always necessary to invoke the viewer. For example, if the instruction is "keep the video clip from 10s to 20s", you can directly return "succeeded" with [[10, 20]] as the message, if the video length is not shorter than 20s.

For your ease of decision, here are some initial frame captions and their timestamps for you (in the form of key value pairs, where the value is the frame caption and the key is its corresponding timestamp, in the unit of second): [initial_captions_flag].
[sys_usr_split]

Now let's start!
"""

PROMPT_VIEWER = """
You are a helpful and smart assistant that can respond to an upstream request a video by invoking tools. The length of this video is [video_length_flag].

Here is the content of the request: [validator_request_flag].

Here are the tools you can access (you might access them multiple times if you want):

1. scan(start, end): Return the overall caption of the video snippet (clip) between start and end timestamp, which are the parameters with the unit of second.
2. localize(query): Return the video location, in the form of a timestamp range given by the start and end timestamp, which contains the visual content of the query (which query might be an object, event, etc.)
3. get_image_cap(timestamp): parameter "timestamp" is an integer in the unit of second. Return the caption of the video frame at the given timestamp (regard the frame as an image).

Now, in each turn of the following conversation, your response should be in the following json format: {"decision": [your decision], "message": [your message]}. This works as follows:
[your decision]: either "tool" (if you wish to call tool in this round) or "respond" (if you feel that you are able to respond to the upstream module's request by comprehending your current knowledge acquired about the video.).
[your message]: if your decision is "tool", then this should only contain tool calling following given format, e.g. 'get_image_cap(10)', 'scan(21, 35)', 'localize("kicking the ball")'. If your decision is "respond", then this should be your response to the upstream request.

[sys_usr_split]

Now let's start!
"""

PROMPT_GET_CAP_SNIPPET = """
You are a helpful assistant that can infer the content of a video snippet by looking at language descriptions of some of its frames. in the following content, the "key" is the timestamp in the unit of second, and "value" is the image caption of the frame at the corresponding second.

[scatter_captions_flag]
[sys_usr_split]
Now infer the content of the video snippet and give me a coherent caption of it. Please directly return the snippet caption, containing as much information as possible.
"""


# ============ Core Pipeline Classes (with logging) ============

class FullBaselineEMC:
    """Top-level controller for a single data sample's EMC processing."""

    def __init__(self, video_path, video_id, question, vid_duration, frame_rate, caption_cache_dir):
        self.video_path = video_path
        self.video_id = video_id
        self.current_question = question
        self.original_question = question
        self.vid_physical_duration = vid_duration
        self.frame_rate = frame_rate
        self.caption_cache_dir = caption_cache_dir

        self.current_cut_segment = [[0, vid_duration]]
        self.current_videolength = vid_duration

        self.failure_history = []
        self.success_history = []

        self.terminate = False
        self.launch_num_threshold = 2  # max rounds

        # Logging
        self.log = {
            "rounds": [],
            "caption_calls": [],  # all caption API calls
        }

    def set_current_segment_and_update_length(self, segment):
        self.current_cut_segment = segment
        self.current_videolength = sum([e[1] - e[0] for e in segment])

    def mapping_v2p_singletimestamp(self, virtual_timestamp):
        accumulated = 0
        for start, end in self.current_cut_segment:
            seg_dur = end - start
            if accumulated <= virtual_timestamp < accumulated + seg_dur:
                return start + (virtual_timestamp - accumulated)
            accumulated += seg_dur
        return max(0, self.current_cut_segment[-1][1] - 1) if self.current_cut_segment else 0

    def mapping_p2v_singletimestamp(self, physical_timestamp):
        accumulated = 0
        for seg in self.current_cut_segment:
            if physical_timestamp > seg[1] + 0.5:
                accumulated += seg[1] - seg[0]
            else:
                return accumulated + physical_timestamp - seg[0]
        return accumulated

    def mapping_v2p_timestampssegments(self, virtual_timestamps):
        original_intervals = []
        for v_start, v_end in virtual_timestamps:
            accumulated = 0
            for start, end in self.current_cut_segment:
                seg_dur = end - start
                if accumulated < v_end and accumulated + seg_dur > v_start:
                    seg_start = max(v_start - accumulated, 0) + start
                    seg_end = min(v_end - accumulated, seg_dur) + start
                    original_intervals.append([seg_start, seg_end])
                accumulated += seg_dur
        return original_intervals

    def get_caption_at_physical(self, physical_ts):
        """Get caption at a physical timestamp, with logging."""
        cap, was_cached = get_caption_at_physical_ts(
            self.video_path, self.video_id, physical_ts,
            self.vid_physical_duration, self.caption_cache_dir
        )
        self.log["caption_calls"].append({
            "physical_timestamp": round(physical_ts, 2),
            "caption": cap,
            "was_cached": was_cached,
            "time": datetime.now().isoformat(),
        })
        return cap

    def get_initial_captions_uniform(self, num=10):
        """Get uniform captions for current virtual video. Returns dict {v_ts_str: caption}."""
        v_dur = self.current_videolength
        if v_dur <= 0:
            return {}
        step = max(1, floor(v_dur) // num)
        v_timestamps = list(range(0, floor(v_dur), step))[:num]
        # shift to center of each bin
        half_step = step // 2
        v_timestamps = [min(t + half_step, floor(v_dur) - 1) for t in v_timestamps]

        captions = {}
        for vt in v_timestamps:
            pt = self.mapping_v2p_singletimestamp(vt)
            cap = self.get_caption_at_physical(pt)
            captions[str(vt)] = cap
        return captions

    def run(self):
        round_idx = 0
        while not self.terminate and round_idx < 5:  # safety max 5 rounds
            round_log = {
                "round_index": round_idx,
                "input_question": self.current_question,
                "input_segment": list(self.current_cut_segment),
                "input_videolength": self.current_videolength,
                "trials": [],
                "outcome": None,
            }

            trial_max = 2
            trial_count = 0

            while trial_count <= trial_max:
                trial_log = {
                    "trial_index": trial_count,
                    "launcher": None,
                    "validator": None,
                    "viewer_sessions": [],
                    "outcome": None,
                }

                # === LAUNCHER ===
                launcher_log = {"llm_calls": [], "decision": None, "modified_question": None, "plan": None}
                ready_prompt = PROMPT_LAUNCHER.replace(
                    "[quesion_flag]", self.current_question
                ).replace(
                    "[failure_history_flag]", json.dumps(self.failure_history)
                ).replace(
                    "[success_history_flag]", json.dumps(self.success_history)
                )

                conv = [
                    {"role": "system", "content": "You are a smart and helpful assistant"},
                    {"role": "user", "content": ready_prompt}
                ]

                resp = call_llm(conv, log_list=launcher_log["llm_calls"])
                try:
                    parsed = extract_dict_from_response(resp)
                    launcher_decision = parsed["decision"]
                    launcher_modified_q = parsed.get("modified_question", "N/A")
                    launcher_plan = parsed.get("description", "N/A")
                except Exception as e:
                    launcher_decision = "terminate"
                    launcher_modified_q = "N/A"
                    launcher_plan = f"Parse error: {e}"

                launcher_log["decision"] = launcher_decision
                launcher_log["modified_question"] = launcher_modified_q
                launcher_log["plan"] = launcher_plan
                trial_log["launcher"] = launcher_log

                if launcher_decision == "terminate":
                    trial_log["outcome"] = "launcher_terminated"
                    round_log["trials"].append(trial_log)
                    self.terminate = True
                    break

                # === VALIDATOR ===
                validator_log = {"llm_calls": [], "initial_captions": {}, "turns": [], "decision": None, "message": None}

                init_caps = self.get_initial_captions_uniform(num=10)
                validator_log["initial_captions"] = init_caps

                ready_val = PROMPT_VALIDATOR.replace(
                    "[plan_flag]", launcher_plan
                ).replace(
                    "[video_length_flag]", str(floor(self.current_videolength))
                ).replace(
                    "[frame_rate_flag]", str(floor(self.frame_rate))
                ).replace(
                    "[initial_captions_flag]", str(init_caps)
                )
                val_sys, val_usr = ready_val.split("[sys_usr_split]")

                val_conv = [{"role": "system", "content": val_sys}]
                newest_val_msg = val_usr

                val_turn = 0
                while val_turn < 20:  # safety limit
                    resp = call_llm_and_modify_conv(newest_val_msg, val_conv, log_list=validator_log["llm_calls"])
                    try:
                        parsed = extract_dict_from_response(resp)
                        val_decision = parsed["decision"]
                        val_message = parsed["message"]
                    except Exception as e:
                        val_decision = "failed"
                        val_message = f"Parse error: {e}"

                    validator_log["turns"].append({
                        "turn": val_turn,
                        "decision": val_decision,
                        "message": str(val_message)[:2000],
                    })

                    if val_decision in ("succeeded", "failed"):
                        break
                    elif val_decision == "view":
                        # === VIEWER ===
                        viewer_session_log = {"request": val_message, "llm_calls": [], "tool_calls": [], "response": None}

                        ready_view = PROMPT_VIEWER.replace(
                            "[validator_request_flag]", str(val_message)
                        ).replace(
                            "[video_length_flag]", str(self.current_videolength)
                        )
                        view_sys, view_usr = ready_view.split("[sys_usr_split]")

                        view_conv = [{"role": "system", "content": view_sys}]
                        newest_view_msg = view_usr

                        view_turn = 0
                        while view_turn < 15:
                            resp = call_llm_and_modify_conv(newest_view_msg, view_conv, log_list=viewer_session_log["llm_calls"])
                            try:
                                parsed_v = extract_dict_from_response(resp)
                                v_decision = parsed_v["decision"]
                                v_message = parsed_v["message"]
                            except:
                                v_decision = "respond"
                                v_message = "Unable to process request."

                            if v_decision == "respond":
                                viewer_session_log["response"] = str(v_message)
                                break
                            elif v_decision == "tool":
                                tool_output = self._execute_viewer_tool(str(v_message), viewer_session_log["tool_calls"])
                                newest_view_msg = tool_output
                            else:
                                viewer_session_log["response"] = "Unknown viewer decision."
                                break
                            view_turn += 1

                        if viewer_session_log["response"] is None:
                            viewer_session_log["response"] = "Viewer max turns exceeded."

                        trial_log["viewer_sessions"].append(viewer_session_log)
                        newest_val_msg = viewer_session_log["response"]
                    else:
                        val_decision = "failed"
                        val_message = "Unknown validator decision"
                        break
                    val_turn += 1

                validator_log["decision"] = val_decision
                validator_log["message"] = str(val_message)[:5000]
                trial_log["validator"] = validator_log

                # Process validator result
                if val_decision == "failed":
                    lesson = {
                        "failed_modified_question": launcher_modified_q,
                        "corresponding_failed_plan": launcher_plan,
                        "reason": str(val_message)[:1000]
                    }
                    self.failure_history.append(lesson)
                    trial_log["outcome"] = "validator_failed"
                    trial_count += 1
                elif val_decision == "succeeded":
                    try:
                        if isinstance(val_message, str):
                            val_message = json.loads(val_message)
                        physical_segment = self.mapping_v2p_timestampssegments(list(val_message))
                        if not physical_segment:
                            raise ValueError("Empty physical segment")
                        self.set_current_segment_and_update_length(physical_segment)
                        self.current_question = launcher_modified_q
                        self.success_history.append({
                            "succeeded_modified_question": launcher_modified_q,
                            "corresponding_succeeded_plan": launcher_plan
                        })
                        self.failure_history = []
                        trial_log["outcome"] = "succeeded"
                    except Exception as e:
                        trial_log["outcome"] = f"segment_parse_error: {e}"
                        trial_count += 1
                else:
                    trial_log["outcome"] = "unknown"
                    trial_count += 1

                round_log["trials"].append(trial_log)

                if trial_log["outcome"] == "succeeded":
                    break

            # If all trials exhausted
            if trial_count > trial_max:
                self.terminate = True
                round_log["outcome"] = "all_trials_exhausted"
            elif self.terminate:
                round_log["outcome"] = "terminated"
            else:
                round_log["outcome"] = "round_succeeded"

            self.log["rounds"].append(round_log)
            round_idx += 1

    def _execute_viewer_tool(self, tool_call_str, tool_log_list):
        """Execute a viewer tool call string like get_image_cap(10) or scan(5, 15)."""
        log_entry = {
            "tool_call": tool_call_str,
            "time": datetime.now().isoformat(),
            "result": None,
        }

        try:
            tool_call_str = tool_call_str.strip()

            if tool_call_str.startswith("get_image_cap"):
                # Parse: get_image_cap(10)
                ts = float(re.search(r'get_image_cap\((\d+\.?\d*)\)', tool_call_str).group(1))
                physical_ts = self.mapping_v2p_singletimestamp(ts)
                if physical_ts > self.vid_physical_duration + 2:
                    result = "Error. This timestamp exceeds the video length."
                else:
                    result = self.get_caption_at_physical(physical_ts)
                log_entry["parsed"] = {"function": "get_image_cap", "virtual_ts": ts, "physical_ts": round(physical_ts, 2)}

            elif tool_call_str.startswith("scan"):
                # Parse: scan(5, 15)
                match = re.search(r'scan\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)', tool_call_str)
                v_start, v_end = float(match.group(1)), float(match.group(2))
                p_start = self.mapping_v2p_singletimestamp(v_start)
                p_end = self.mapping_v2p_singletimestamp(v_end)
                if p_start > self.vid_physical_duration + 2:
                    result = "Error. The starting timestamp exceeds the video length."
                else:
                    # Get scattered captions and synthesize
                    skip = max(1, floor((v_end - v_start) / 4))
                    caps = ""
                    for vt in range(int(v_start), int(v_end + 1), skip):
                        pt = self.mapping_v2p_singletimestamp(vt)
                        cap = self.get_caption_at_physical(pt)
                        if "Error" in cap:
                            break
                        caps += f"{vt}: {cap}\n"

                    # Synthesize snippet caption using LLM
                    synth_prompt = PROMPT_GET_CAP_SNIPPET.replace("[scatter_captions_flag]", caps)
                    synth_sys, synth_usr = synth_prompt.split("[sys_usr_split]")
                    synth_conv = [
                        {"role": "system", "content": synth_sys},
                        {"role": "user", "content": synth_usr}
                    ]
                    result = call_llm(synth_conv, model=TOOL_MODEL)
                log_entry["parsed"] = {"function": "scan", "v_start": v_start, "v_end": v_end}

            elif tool_call_str.startswith("localize"):
                # Parse: localize("kicking the ball")
                match = re.search(r'localize\(["\'](.+?)["\']\)', tool_call_str)
                query = match.group(1)
                log_entry["parsed"] = {"function": "localize", "query": query}

                # 3-stage LLM localization (faithful to original ReSimplifyIt.py)
                # No GPU/clustering available, use except-branch: uniform sampling for initial captions
                v_dur = self.current_videolength
                k = 10  # init_n_clusters
                center_v_times = list(range(floor(v_dur)))[::max(1, round(floor(v_dur) / k))]
                center_p_times = [self.mapping_v2p_singletimestamp(vt) for vt in center_v_times]
                initial_caps_dict = {}
                for i, pt in enumerate(center_p_times):
                    vt = center_v_times[i] if i < len(center_v_times) else self.mapping_p2v_singletimestamp(pt)
                    cap = self.get_caption_at_physical(pt)
                    initial_caps_dict[f'frame caption at {vt}s'] = cap
                str_initial_captions = str(initial_caps_dict)

                # === Stage 1: Propose 5 candidate timestamps ===
                propose_prompt = f"""You are a smart assistant to find some timestamps of a video that corresponds to a natrual language query as accurate and informative as possible. Here, the query is: "{query}".

In fact, you will need to propose five of these timestamps.

A tool (python function) will be helping you to get the frame caption of at a certain timestamp (in the unit of second). Whenever you need to call this tool, send a message in this json format: {{"decision": "tool", "parameter": [timestamp you need, as an integer]}}. For example, if you need the caption at 3s of the video, return {{"decision": "tool", "parameter": 3}}, then it will be returned to you.
Whenever you think you are confident enough to provide the timestamps, return {{"decision": "end", "timestamps": [your result five timestamps]}}. For example, if you decide that timestamp 5s, 28s, 97s, 112s, and 343s are the most possible timestamps to contain the content of the query, then return {{"decision": "end", "timestamps": [5, 28, 97, 112, 343]}}.

Before we formally begin, here is a set of original captions with their timestamps provided for you to have an overall rough understanding of the video: {str_initial_captions}. (so you don't have to call tool to get captions of these timestamps, as they are already provided here.)
Also, the frame rate of this video is {round(self.frame_rate, 2)} frames per second, and the total duration is {floor(v_dur)}.

Note that the frame captions are captions of the static video frame as an image, so it might not explicitly mention enough dynamics, so you will need to infer on these yourself.
You may confidently assume that the object/event requested by the query exist in the video.
Remember always give your decision in the json format provided above."""

                conv1 = [{"role": "system", "content": propose_prompt}, {"role": "user", "content": "Now let's begin!"}]
                final_timestamps = []
                for _ in range(30):
                    if len(conv1) > 60:
                        break
                    resp1 = call_llm(conv1, log_list=log_entry.setdefault("localize_llm_calls", []))
                    conv1.append({"role": "assistant", "content": resp1})
                    try:
                        j1 = extract_dict_from_response(resp1)
                        if j1.get("decision") == "end":
                            final_timestamps = j1.get("timestamps", [])
                            break
                        elif j1.get("decision") == "tool":
                            req_ts = round(j1["parameter"])
                            pt = self.mapping_v2p_singletimestamp(int(req_ts))
                            cap = self.get_caption_at_physical(pt)
                            conv1.append({"role": "user", "content": cap})
                        else:
                            break
                    except:
                        break
                if not final_timestamps:
                    final_timestamps = [int(x) for x in np.linspace(0, floor(v_dur), 7, dtype=int)[1:6]]
                final_timestamps = sorted(list(set([e for e in final_timestamps if e != -1])))

                # === Stage 2: Select best timestamp from candidates ===
                proposals_caps = {f'frame caption at {ts}s': self.get_caption_at_physical(self.mapping_v2p_singletimestamp(ts)) for ts in final_timestamps}
                select_prompt = f"""You are a smart assistant to select one timestamp from a given list of video timestamps that is the most suitable one to contain the content of a natrual language query as accurate and informative as possible. Here, the query is: "{query}". The list of timestamps for you to choose from is: {final_timestamps}. The timestamps are all in the unit of second.

Also, for you to start, here are the video frame captions of the frames at these timestamps: {str(proposals_caps)}.

In addition, for you to acquire additional video frame captions at other timestamps to have a better understanding of the video content, a tool (python function) will be helping you to get the frame caption of at a certain timestamp (in the unit of second). Whenever you need to call this tool, send a message in this json format: {{"decision": "tool", "parameter": [timestamp you need, as an integer]}}. For example, if you want the caption at 3s of the video, return {{"decision": "tool", "parameter": 3}}, then it will be returned to you.
Whenever you think you are confident enough to confirm the selection of the best timestamp, return {{"decision": "end", "timestamp": [your result timestamp as an integer]}}.

Also, the frame rate of this video is {round(self.frame_rate, 2)} frames per second, and the total duration is {floor(v_dur)}.

Note that the frame captions are captions of the static video frame as an image, so it might not explicitly mention enough dynamics, so you will need to infer on these yourself.
Remember always give your decision in the json format provided above."""

                conv2 = [{"role": "system", "content": select_prompt}, {"role": "user", "content": "Now let's begin!"}]
                final_choice = -1
                for _ in range(30):
                    if len(conv2) > 60:
                        break
                    resp2 = call_llm(conv2, log_list=log_entry.setdefault("localize_llm_calls", []))
                    conv2.append({"role": "assistant", "content": resp2})
                    try:
                        j2 = extract_dict_from_response(resp2)
                        if j2.get("decision") == "end":
                            final_choice = j2.get("timestamp", -1)
                            break
                        elif j2.get("decision") == "tool":
                            req_ts = round(j2["parameter"])
                            pt = self.mapping_v2p_singletimestamp(int(req_ts))
                            cap = self.get_caption_at_physical(pt)
                            conv2.append({"role": "user", "content": cap})
                        else:
                            break
                    except:
                        break
                if not isinstance(final_choice, int):
                    final_choice = -1

                # === Stage 3: Spread from anchor to find timestamp range ===
                if final_choice != -1:
                    spread_prompt = f"""You are a smart assistant to carefully find (locate) a timestamp range of a video that corresponds to a natrual language query as accurate and informative as possible, given an anchor timestamp which is very likely to be located within the timestamp range you should return.
Here, the query is: "{query}", and the anchor timestamp provided to you is: {final_choice}.
Again, this anchor timestamp should be within the timestamp range you return. You may start from this anchor timestamp and gradually expand forwards and backwards to determine the start and end as the boundaries.

A tool (python function) will be helping you to get the frame caption at a certain timestamp (in the unit of second). Whenever you need to call this tool, send a message in this json format: {{"decision": "tool", "parameter": [timestamp you need, as an integer]}}. For example, if you want the caption at 5s of the video, return {{"decision": "tool", "parameter": 5}}. The corresponding caption will be returned to you.
Whenever you think you are confident enough to provide the timestamp, return {{"decision": "end", "timestamps": [your result timestamps, in the form of a two-layered list]}}. For example, if you decide to preserve the sub-clip from 4s to 18s, then return {{"decision": "end", "timestamps": [[4, 18]]}}.

Also, the frame rate of this video is {round(self.frame_rate, 2)} frames per second, and the total duration is {floor(v_dur)}.

Note that the frame captions are captions of the static video frame as an image, so it might not explicitly mention enough dynamics, so you will need to infer on these dynamic actions yourself."""
                else:
                    spread_prompt = f"""You are a smart assistant to carefully find (locate) a timestamp range of a video that corresponds to a natrual language query as accurate and informative as possible. Here, the query is: "{query}".
Before we formally begin, here is a set of original captions with their timestamps provided for you to have an overall rough understanding of the video: {str_initial_captions}.

A tool (python function) will be helping you to get the frame caption at a certain timestamp (in the unit of second). Whenever you need to call this tool, send a message in this json format: {{"decision": "tool", "parameter": [timestamp you need, as an integer]}}. For example, if you want the caption at 5s of the video, return {{"decision": "tool", "parameter": 5}}. The corresponding caption will be returned to you.
Whenever you think you are confident enough to provide the timestamp, return {{"decision": "end", "timestamps": [your result timestamps, in the form of a two-layered list]}}. For example, if you decide to preserve the sub-clip from 4s to 18s, then return {{"decision": "end", "timestamps": [[4, 18]]}}.

Also, the frame rate of this video is {round(self.frame_rate, 2)} frames per second, and the total duration is {floor(v_dur)}.

Note that the frame captions are captions of the static video frame as an image, so it might not explicitly mention enough dynamics, so you will need to infer on these yourself."""

                conv3 = [{"role": "system", "content": spread_prompt}, {"role": "user", "content": "Now let's begin!"}]
                spread_timestamps = [[0, floor(v_dur)]]
                for _ in range(30):
                    if len(conv3) > 60:
                        break
                    resp3 = call_llm(conv3, log_list=log_entry.setdefault("localize_llm_calls", []))
                    conv3.append({"role": "assistant", "content": resp3})
                    try:
                        j3 = extract_dict_from_response(resp3)
                        if j3.get("decision") == "end":
                            ts_result = j3.get("timestamps", [[0, floor(v_dur)]])
                            if ts_result and isinstance(ts_result, list) and isinstance(ts_result[0], list):
                                spread_timestamps = ts_result
                            break
                        elif j3.get("decision") == "tool":
                            req_ts = round(j3["parameter"])
                            pt = self.mapping_v2p_singletimestamp(int(req_ts))
                            cap = self.get_caption_at_physical(pt)
                            conv3.append({"role": "user", "content": cap})
                        else:
                            break
                    except:
                        break

                result = str(spread_timestamps)
            else:
                result = f"Unknown tool call: {tool_call_str}"
                log_entry["parsed"] = {"function": "unknown"}

        except Exception as e:
            result = f"Tool execution error: {e}"
            log_entry["parsed"] = {"function": "error", "error": str(e)}

        log_entry["result"] = str(result)[:2000]
        tool_log_list.append(log_entry)
        return result


# ============ Main Processing ============

def process_dataset(dataset_name):
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name} (Full Baseline)")
    print(f"{'='*60}")

    dataset_dir = f"{DATA_BASE}/{dataset_name}"
    caption_cache_dir = f"{dataset_dir}/captions_1fps"
    os.makedirs(caption_cache_dir, exist_ok=True)

    output_path = os.path.join(OUTPUT_BASE, f"full_{dataset_name}.json")
    log_path = os.path.join(OUTPUT_BASE, f"full_{dataset_name}_log.json")

    from emc_utils.utils import load_test_split
    try:
        input_data = load_test_split(dataset_dir)
    except FileNotFoundError as e:
        print(f"  Dataset files not found in {dataset_dir}: {e}, skipping.")
        return

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

            item = input_data[key]
            video_path = item['video_path']
            question = item['question']
            video_id = os.path.splitext(os.path.basename(video_path))[0]

            sample_log = {
                "sample_id": key,
                "dataset": dataset_name,
                "video_path": video_path,
                "video_id": video_id,
                "original_question": question,
                "start_time": datetime.now().isoformat(),
                "model": CORE_MODEL,
                "caption_model": "gpt-4o",
                "pipeline": "full_baseline",
                "result": None,
                "error": None,
                "trajectory": None,
            }

            try:
                vid_duration, frame_rate = get_video_info(video_path)
                if vid_duration is None:
                    raise Exception(f"Cannot open video: {video_path}")

                sample_log["vid_duration"] = vid_duration
                sample_log["frame_rate"] = frame_rate

                emc = FullBaselineEMC(
                    video_path, video_id, question,
                    vid_duration, frame_rate, caption_cache_dir
                )
                emc.run()

                final_timestamps = emc.current_cut_segment
                final_question = emc.current_question
                status = "succeeded"

                # Validate
                if not final_timestamps or final_timestamps == [[0, vid_duration]]:
                    if emc.terminate and not emc.success_history:
                        status = "no_screening_possible"

                sample_log["result"] = {
                    "screened_timestamps": final_timestamps,
                    "screened_question": final_question,
                    "status": status,
                }
                sample_log["trajectory"] = emc.log

            except Exception as e:
                sample_log["error"] = str(e)
                final_timestamps = [[0, vid_duration if 'vid_duration' in dir() and vid_duration else 180]]
                final_question = question
                status = "failed"
                sample_log["result"] = {
                    "screened_timestamps": final_timestamps,
                    "screened_question": final_question,
                    "status": status,
                }

            sample_log["end_time"] = datetime.now().isoformat()

            with output_lock:
                outputs[key] = sample_log["result"]
                all_logs[key] = sample_log
                try:
                    _atomic_json_write(output_path, outputs)
                    _atomic_json_write(log_path, all_logs)
                except:
                    pass

            pbar.update(1)

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


if __name__ == "__main__":
    print(f"EMC Full Baseline (ReSimplifyIt)")
    print(f"Model: {CORE_MODEL}")
    print(f"Caption model: gpt-4o")
    print(f"Datasets: {datasets_to_run}")
    print(f"Threads: {args.num_threads}")
    print()

    for ds in datasets_to_run:
        process_dataset(ds)

    print("\nAll datasets processed!")
