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

from config import *
from emc_utils.utils import isodata_clustering, find_closest_to_centers, dprint

dataset = 'yc2emc' # 'yc2emc', 'activitynetqa', 'nextqa', 'nextoe'
force_redo_this_dataset = True
num_threads = 90

all_caps_folder = dataset_captions_folders[dataset]
data_path = dataset_json_paths[dataset]

with open(data_path) as f:
    input_data = json.load(f)

client = OpenAI(api_key=openai_api_key)
core_model = 'gpt-4o'

def get_response_and_modify_conversation(message, conversation, model=core_model):
    conversation.append({"role": "user", "content": message})
    completion = client.chat.completions.create(
                model=model,
                messages=conversation,
                max_tokens=1024,
                temperature=0.2
            )
    response = completion.choices[0].message.content
    conversation.append({"role": "assistant", "content": response})
    dprint(response, '\n')
    return response

def get_response_not_modify_conversation(conversation, model=core_model):
    completion = client.chat.completions.create(
                model=model,
                messages=conversation,
                max_tokens=1024,
                temperature=0.2
            )
    response = completion.choices[0].message.content
    dprint(response, '\n')
    return response


def pickout_and_jsonize(s):
    temp = s.split('【')[-1]
    temp = temp.split('】')[0]
    try:
        temp = json.loads(temp)
    except:
        dprint('load json failed: -------------------------', temp, '------------------------')
        temp = json.loads(temp)
    return temp

if not os.path.exists(EMC_save_path_simple):
    outputs = dict()
    outputs[dataset] = dict()
else:
    with open(EMC_save_path_simple, 'r') as f:
        outputs = json.load(f)
    if dataset not in outputs:
        outputs[dataset] = dict()

if force_redo_this_dataset:
    all_keys = list(input_data.keys())
else:
    all_keys = list(set(list(input_data.keys()))-set(list(outputs[dataset].keys())))

pbar = tqdm(total=len(all_keys))

def thread_run():
    while True:
        try:
            key = all_keys.pop(0)
        except:
            return # list is empty
        fault_count = 5

        while True:
            try:
                vid_name = input_data[key]['vid_name']
                
                question = input_data[key]['question']

                vid_fname = input_data[key]['vid_fname']
                frame_rate = input_data[key]['vid_frame_rate']
                vid_duration = input_data[key]['vid_duration']

                try:
                    vid_allcap = json.load(open(os.path.join(all_caps_folder, vid_name+'.json')))
                except:
                    vid_allcap = json.load(open(os.path.join(all_caps_folder, vid_fname.split('.')[0]+'.json')))


                try:
                    all_iframe_times = json.load(open(iframes_times_save_path))
                    vid_iframe_times = all_iframe_times[vid_fname]

                    vid_feat = torch.load(os.path.join(dataset_vid_1fps_feature_folder, vid_fname.split('.')[0]+'.pt'))
                    vid_iframe_feats = vid_feat[vid_iframe_times]

                    k = 12  # initialization of number of clusters
                    assignments, centers = isodata_clustering(vid_iframe_feats, k, min_intra_cossim_otherwise_split=0.85, max_inter_cossim_otherwise_merge=0.98, min_elements_per_cluster=max(1, round(len(vid_iframe_feats)/50)), max_n_clusters=15, min_n_clusters=min(10, len(vid_iframe_feats)))
                    # min_intra_cossim_otherwise_split: larger value more likely split
                    # max_inter_cossim_otherwise_merge: larger value less likely merge

                    indices_of_selected_key_iframes_in_iframes_tensor_unsort = find_closest_to_centers(vid_iframe_feats, centers, assignments)
                    indices_of_selected_key_iframes_in_iframes_tensor_sorted = indices_of_selected_key_iframes_in_iframes_tensor_unsort.sort().values
                    center_times = torch.tensor(list(map(lambda x: round(x, 2), vid_iframe_times)))[indices_of_selected_key_iframes_in_iframes_tensor_sorted].tolist()

                    center_times = list(set(map(int, center_times)))
                except:
                    k = 10
                    center_times = list(range(len(vid_allcap)))[::round(len(vid_allcap)/k)]


                batched_initial_captions = str({str(k): vid_allcap[str(k)] for k in center_times})

                initial_prompt_template = '''
                You are a assistant for the video question answering process, in which a candidate is presented with a video and a question for them to answer.\
                Your objective is to help the candidate so that he will be able to give the answer with watching the shortest posible sub-clip(s) of the video. \
                Your task is to cut the video to acquire this sub-clip(s) and also to modify the question, so that the candidate directly answering your modified question with presented only this sub-clip(s) of the video would be equivalent to answering the original question with presented the original whole uncut video. \
                For example, when answering the question "what did the person do after putting down the dog", you will need to provide the sub-clip starting from approximately the last frame of the person put down the dog till the end, and the corresponding revised question would be "what did the person do in this video". \
                In this way, if the the man in the video went preparing food after putting down the dog, then the answer "he went prepare food" would be simultaneously correct for both the original and revised questions. \
                You will need to cut the video in the form of providing me the timestamps, which is a list of [start, end] unit clips in the unit of second. \
                A tool (python function) will be helping you to get the frame caption of at a certain timestamp (in the unit of second). Whenever you need to call this tool, send a message in this json format: 【{"decision": "tool", "parameter": [timestamp you need]}】. For example, if you want the caption at 3.5s of the video, return 【{"decision": "tool", "parameter": 3.5}】. The corresponding caption will be returned to you. \
                Whenever you think you are confident enough to provide the timestamp, return 【{"decision": "end", "timestamps": [your result timestamps], "revised_question": [your revised question]}】. For example, if you decide to preserve the sub-clip from 1.4s to 4.8s with the revised prompt being "what did the person do in this video", then return 【{"decision": "end", "timestamps": [[1.4, 4.8]], "revised_question": "what did the person do in this video"}】.

                Before we formally begin, here is a set of original captions with their timestamps provided for you to have an overall rough understanding of the video: [initial_captions_flag].
                Also, the frame rate of this video is [frame_rate_flag] frames per second, and the total duration is [duration_flag].

                [sys_usr_split]Now let's begin! and the original question is "[original_question_flag]".
                '''

                initial_captions = batched_initial_captions

                initial_prompt = initial_prompt_template.replace("[initial_captions_flag]", initial_captions).replace("[frame_rate_flag]", str(round(frame_rate, 2))).replace("[duration_flag]", str(math.floor(vid_duration))).replace("[original_question_flag]", question)
                initial_prompt_sys = initial_prompt.split('[sys_usr_split]')[0]
                initial_prompt_usr = initial_prompt.split('[sys_usr_split]')[1]

                conversation = list()
                response = '【{}】'

                conversation.append({"role": "system", "content": initial_prompt_sys})
                conversation.append({"role": "user", "content": initial_prompt_usr})

                dprint('starting multi-turn conv with gpt-4o......')
                while True:
                    if len(conversation) > 30:
                        final_timestamps = [[]]
                        break
                    err_count = 0
                    while True:
                        response = get_response_not_modify_conversation(conversation)
                        yes = True
                        try:
                            j = pickout_and_jsonize(response)
                            yes = yes and 'decision' in j
                            if j['decision'] == 'tool':
                                yes = yes and j["parameter"] >= 0
                            elif j['decision'] == 'end':
                                yes = yes and "timestamps" in j and "revised_question" in j
                                for ele in j["timestamps"]:
                                    yes = yes and ele[0]>=0 and ele[1]>=0
                        except Exception as e:
                            err_count += 1
                            dprint(e, 'retrying......')
                            yes = False
                            if err_count >= 10:
                                j = {'decision': 'end', 'timestamps': [[]], 'revised_question': None}
                                break
                        if yes:
                            conversation.append({"role": "assistant", "content": response})
                            break

                    if j['decision'] == 'end':
                        final_timestamps = j["timestamps"]
                        revised_question = j['revised_question']
                        status = 'succeeded'
                        break
                    else:
                        request = round(j["parameter"])
                        cap = vid_allcap[str(int(request))]
                        conversation.append({"role": "user", "content": cap})
                dprint('multi-turn conv with gpt-4o ended.')

                if final_timestamps in ([[]], [], None) or type(final_timestamps) != list:
                    raise Exception('final_timestamps is empty or not a list')
                
                break
                
            except:
                fault_count -= 1
                if fault_count <= 0:
                    final_timestamps = [[0, vid_duration]]
                    revised_question = question
                    status = 'failed'
                    break
                else:
                    continue

        output = dict()
        output['screened_timestamps'] = final_timestamps
        output['screened_question'] = revised_question
        output['status'] = status

        while True:
            # handle "RuntimeError: dictionary changed size during iteration"
            try:
                outputs[dataset][key] = output
                with open(EMC_save_path_simple, 'w') as f:
                    json.dump(outputs, f)
                break
            except:
                pass

        pbar.update(1)


if num_threads > 1:
    import threading
    threads = list()
    for i in range(num_threads):
        threads.append(threading.Thread(target=thread_run))
    for i in range(num_threads):
        threads[i].start()
        time.sleep(0.01)
    for i in range(num_threads):
        threads[i].join()
else:
    thread_run()


pbar.close()