import cv2
import torch
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import time
import openai
from openai import OpenAI
import math
import random
import time
import json
import threading

from math import floor

from git_ignore import *
from tvs_utils.utils import *
from tvs_utils.tools import load_video_raw, scatter_frame_indices_to_timestamp_ranges

from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor, CLIPProcessor, CLIPModel

client = OpenAI(api_key=openai_api_key)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()


class ReSimplifyIt_blind_Datapoint(TVS_Datapoint):
    def __init__(self, question, video_fname, videocaptionpath, vid_frame_rate, vid_duration, iframes_times, iframes_feats, vid_path):
        super().__init__(question, video_fname, videocaptionpath, vid_frame_rate, vid_duration, iframes_times, iframes_feats)
        self.vid_path = vid_path

    def build(self, question, video_fname, videocaptionpath, vid_frame_rate, vid_duration, iframes_times, iframes_feats, vid_path):
        self.question = question
        self.resolution = None # this work is TVS not TSVS. Spatial counterpart leave to future work.
        self.video_fname = video_fname
        self.videocaptionpath = videocaptionpath
        self.vid_frame_rate = vid_frame_rate
        self.vid_duration = vid_duration
        self.iframes_times = iframes_times
        self.iframes_feats = iframes_feats
        self.vid_path = vid_path
        self.raw_video = None


datapoint_ref = ReSimplifyIt_blind_Datapoint(None, None, None, None, None, None, None, None)


def get_duration(datapoint=datapoint_ref) -> float:
    '''
    return the duration (unit: second) of the video.
    '''
    return datapoint.vid_duration


def get_resolution(datapoint=datapoint_ref) -> tuple:
    '''
    return the resolution of the video, in the form of a tuple.
    '''
    return datapoint.resolution


def get_total_frame_num(datapoint=datapoint_ref) -> int:
    '''
    return the total number of frames of the video.
    '''
    return floor(datapoint.vid_duration * datapoint.vid_frame_rate)


def grounding_select(query:str, concerned_indices_input:list=None, datapoint:ReSimplifyIt_blind_Datapoint=datapoint_ref, threshold:float=20, clip_model=clip_model) -> list:
    '''
    return the indices of all frames whose CLIP score between the given obj_name is higher than a threshold.
    '''

    if datapoint.raw_video is None:
        datapoint.raw_video = load_video_raw(datapoint.vid_path)
    video_input = datapoint.raw_video
    return_indices = []

    video_input_indexed = [(i, pil) for (i, pil) in enumerate(video_input)]
    if concerned_indices_input is not None:
        video_input_indexed = [ele for ele in enumerate(video_input) if ele[0] in concerned_indices_input]

    i = 0
    batch_size, skip_size = 200, 20
    while i < len(video_input_indexed) - 1:
        batch_end_idx = i + batch_size - 1
        if i + batch_size > len(video_input_indexed):
            batch_end_idx = len(video_input_indexed) - 1
        batch_enum = video_input_indexed[i: batch_end_idx + 1: skip_size]
        # print(batch_enum)
        batch_imgs = [ele[1] for ele in batch_enum]
        inputs = processor(text=[query], images=batch_imgs, return_tensors="pt", padding=True)
        logits = clip_model(**inputs).logits_per_image
        logits_lst = logits.reshape(logits.shape[0]).tolist()
        for ii, score in enumerate(logits_lst):
            if score > threshold:
                return_indices += list(range(batch_enum[ii][0], min(batch_enum[ii][0] + skip_size, len(video_input_indexed))))

        i += batch_size
    
    return return_indices


def indices_list_intersect(frame_indices_list_1:list, frame_indices_list_2:list) -> list:
    '''
    return the frame indices that are contained in both the two input indices lists.
    '''
    if isinstance(frame_indices_list_1, range):
        frame_indices_list_1 = list(frame_indices_list_1)
    if isinstance(frame_indices_list_2, range):
        frame_indices_list_2 = list(frame_indices_list_2)
    return sorted(list(set(frame_indices_list_1) & set(frame_indices_list_2)))


def indices_list_union(frame_indices_list_1:list, frame_indices_list_2:list) -> list:
    '''
    return the frame indices that are contained in both the two input indices lists.
    '''
    if isinstance(frame_indices_list_1, range):
        frame_indices_list_1 = list(frame_indices_list_1)
    if isinstance(frame_indices_list_2, range):
        frame_indices_list_2 = list(frame_indices_list_2)
    return sorted(list(set(frame_indices_list_1 + frame_indices_list_2)))


def indices_concat_and_fill(frame_indices_list_1:list, frame_indices_list_2:list) -> list:
    '''
    return a list of all indices within the range of the min value and max value appeared in the argument lists.
    '''
    if isinstance(frame_indices_list_1, range):
        frame_indices_list_1 = list(frame_indices_list_1)
    if isinstance(frame_indices_list_2, range):
        frame_indices_list_2 = list(frame_indices_list_2)
    min_idx = min(min(frame_indices_list_1), min(frame_indices_list_2))
    max_idx = max(max(frame_indices_list_1), max(frame_indices_list_2))
    return list(range(min_idx, max_idx + 1))


def indices_concat(frame_indices_list_1:list, frame_indices_list_2:list) -> list:
    if isinstance(frame_indices_list_1, range):
        frame_indices_list_1 = list(frame_indices_list_1)
    if isinstance(frame_indices_list_2, range):
        frame_indices_list_2 = list(frame_indices_list_2)
    return sorted(list(set(frame_indices_list_1 + frame_indices_list_2)))


def timestamp_to_single_index(timestamp:float, datapoint=datapoint_ref) -> list:
    '''
    return (in the form of a list with only one element) the index of the frame locating at the given timestamp of the original video.
    '''
    duration = datapoint.vid_duration
    num_total = floor(datapoint.vid_duration * datapoint.vid_frame_rate)
    return [max(min(round(num_total * (timestamp / duration)) - 1, num_total - 1), 0)]

def single_timestamp_to_index_range(timestamp:float, datapoint=datapoint_ref) -> list:
    '''
    return the indices of the frame locating at the given timestamp of the original video.
    '''
    duration = datapoint.vid_duration
    num_total = floor(datapoint.vid_duration * datapoint.vid_frame_rate)
    min_index = max(min(round(num_total * (timestamp / duration)) - 30, num_total - 1), 0)
    max_index = max(min(round(num_total * (timestamp / duration)) + 30, num_total - 1), 0)
    return list(range(min_index, max_index+1))

def range_timestamp_to_index_range(start:float, end:float, datapoint=datapoint_ref) -> list:
    '''
    return indices of all frames between the two timestamps given.
    '''
    duration = datapoint.vid_duration
    num_total = floor(datapoint.vid_duration * datapoint.vid_frame_rate)
    min_index = max(min(round(num_total * (start / duration)) - 1, num_total - 1), 0)
    max_index = max(min(round(num_total * (end / duration)) - 1, num_total - 1), 0)
    return list(range(min_index, max_index+1))


api_key = openai_api_key
core_model = "gpt-4o"
force_redo = False

def get_response_not_modify_conversation(conversation, model=core_model):
    completion = client.chat.completions.create(
                model=model,
                messages=conversation,
                max_tokens=1024,
                temperature=0.2
            )
    response = completion.choices[0].message.content
    print(response, '\n')
    return response


if os.path.exists(TVS_save_path_blind) and not force_redo:
    to_write = json.load(open(TVS_save_path_blind))
else:
    to_write = dict()

yc2tvs = json.load(open(dataset_json_path))
all_keys = list(yc2tvs.keys())


for key in tqdm(all_keys):

    screened_timestamps = None
    screened_question = None
    outcome = 'failed'

    if key in to_write and not force_redo:
        continue

    dp = yc2tvs[key]

    question = dp["question"]
    video_fname = dp["vid_fname"]
    videocaptionpath = os.path.join(dataset_captions_folder, dp['vid_name']+'.json')
    vid_frame_rate = dp["vid_frame_rate"]
    vid_duration = dp["vid_duration"]

    vid_feat = torch.load(os.path.join(dataset_vid_1fps_feature_folder, video_fname.split('.')[0]+'.pt'))
    vid_iframe_times = json.load(open(iframes_times_save_path))[video_fname]
    vid_iframe_feats = vid_feat[vid_iframe_times]

    datapoint_ref.build(question, video_fname, videocaptionpath, vid_frame_rate, vid_duration, None, None, os.path.join(video_folder_resimplifyit_blind, video_fname))
    print(datapoint_ref.question)
    print(datapoint_ref.vid_path)

    err_count = 0

    while err_count < retry_tolerance_for_resimplifyit_blind:
        try:
            sys_prompt = """
            You are a assistant for the video question answering process, in which a candidate is presented with a video and a question for them to answer.\
            Your objective is to help the candidate so that he will be able to give the answer with watching the shortest possible sub-clip(s) of the video. \
            Your task is to cut the video to acquire this sub-clip(s) and also to modify the question, so that the candidate directly answering your modified question with presented only this sub-clip(s) of the video would be equivalent to answering the original question with presented the original whole uncut video. \
            For example, when answering the question "what did the person do after putting down the dog", you will need to provide the sub-clip starting from approximately the last frame of the person put down the dog till the end, and the corresponding revised question would be "what did the person do in this video". \
            In this way, if the the man in the video went preparing food after putting down the dog, then the answer "he went prepare food" would be simultaneously correct for both the original and revised questions. \
            
            You will be provided with a list of tools to process the video, and the original question to be answered by the candidate based on which to select the frames. Here is the list of tools you have access to, with the description (content in the brackets are the arguments needed):
            [1]: get_duration(): return the duration of the video as a floating point value.
            [2]: get_resolution(): return the resolution of the video, as a tuple.
            [3]: get_total_frame_num(): return total number of the frames of the video, as an integer.
            [4]: grounding_select(obj_name, concerned_indices_input): return, in the form of a list of integers, the indices of all frames containing the object given by obj_name, after taking the intersection of indices provided by the argument 'concerned_indices_input'. 'concerned_indices_input' is also a list of indices, and will be set to indices of all frames in the video if 'None' is passed.
            [5]: indices_list_intersect(frame_indices_list_1, frame_indices_list_2): return, in the form of a list of integers, the intersection of the two arguments as list. Both arugment 'frame_indices_list_1' and 'frame_indices_list_2' are a list of indices.
            [6]: indices_list_union(frame_indices_list_1, frame_indices_list_2): return, in the form of a list of integers, the union of the two arguments as list. Both arugment 'frame_indices_list_1' and 'frame_indices_list_2' are a list of indices.
            [7]: indices_concat_and_fill(frame_indices_list_1, frame_indices_list_2):  first take the sorted union of the two lists given by the arguments, and then fill in all the missing values so that every two adjacent element only differ by 1. Both arugment 'frame_indices_list_1' and 'frame_indices_list_2' are a list of indices.
            [8]: indices_concat(frame_indices_list_1, frame_indices_list_2): return, in the form of a list, the concatenation of the two lists provided by the arguments.
            [9]: timestamp_to_single_index(timestamp): return a list with a single integer, which integer is the index of the frame at the given timestamp. The argument timestamp is a floating point value, whose unit is second.
            [10]: single_timestamp_to_index_range(timestamp): return, in the form of a list, the indices of 60 consecutive frames, the midpoint of which is at the given timestamp. The argument timestamp is a floating point value, whose unit is second.
            [11]: range_timestamp_to_index_range(start, end): return, in the form of a list, the indices of all frames which are between the two timestamps which are provided by the arguments. The argument start and end are both floating point values, whose unit are both second.

            Above are all the tools to have access to. Please note that selecting frames out of all the frames of the original video is being cut and clipped, therefore you will also need to modify the aforementioned prompt, to make it align well with the reduced video frames.
            """

            usr_prompt = f"""
            Now the original question is: '{question}' Having access to the information of all the tools mentioned above, provide me the python code which could achieve the selection of frames. You may define variables to store intermediate result, and determine the value of some arguments when necessary, but you should not require the downstream task operator to replace any of your assumption on arguments, as no more information but the original video is provided to the downstream task. Please use the variable name 'final_frames' to store your final list of frame index. Please only provide the code and revised question in this format: 【"Code:[your whole paragraph of code] Revised question:[your revised prompt]"】, where [your whole paragraph of code] should be an empty string if you think no tools need to be called and the whole original video should be passed to the downstream task.
            """

            conversation = list()

            conversation.append({"role": "system", "content": sys_prompt})
            conversation.append({"role": "user", "content": usr_prompt})

            response = get_response_not_modify_conversation(conversation)
            response = response.replace('【', '').replace('】', '')

            assert "Code:" in response and "Revised question:" in response, "The response format is not correct!"
            
            code = response.split("Code:")[1].split("Revised question:")[0].strip()
            code = code.replace('python', '')
            code = code.replace('`', '')
            revised_question = response.split("Revised question:")[1].strip()

            exec(code, globals())
            final_frames = final_frames

            screened_timestamps = scatter_frame_indices_to_timestamp_ranges(final_frames, datapoint_ref.vid_frame_rate, datapoint_ref.vid_duration)
            screened_question = revised_question

            if screened_timestamps not in (None, [], [[]]):
                outcome = 'succeeded'
                break
                
        except:
            print("Error in the response format! retrying......")
            err_count += 1
            continue

        # break # del this line when not debugging
        

    to_write[key] = {
        "screened_timestamps": screened_timestamps,
        "screened_question": screened_question,
        "outcome": outcome,
    }

    with open(TVS_save_path_blind, 'w') as f:
        json.dump(to_write, f)
