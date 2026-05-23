import openai
from openai import OpenAI
import json
import torch
import numpy as np
from tqdm import tqdm
import re
import os
from math import floor
import threading
import time
import sys
from emc_utils.utils import *

from config import openai_api_key

from config import *



api_key = openai_api_key
core_model = "gpt-4o"
tool_model = "gpt-3.5-turbo"

num_threads = 1

client = OpenAI(api_key=api_key)

prompt_launcher = """
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

""" # [quesion_flag] [failure_history_flag] [success_history_flag]
# no need to seperate sys_prompt and usr_prompt in prompt_launcher, because launcher is realized by a SINGLE-turn conv with got

prompt_validator = """
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
[your message]: if you choose "succeeded" as your decision, then it should be the two-layer list as mentioned before, as the video edit result of the plan. If you choose "failed", this should be a brief reason on the failure (e.g. requested timestamp exceeds video length, video doesn’t have the object/event needed, etc.). If you choose "view", this should be the question to ask the viewer about the video content.

Hint: it is not always necessary to invoke the viewer. For example, if the instruction is "keep the video clip from 10s to 20s", you can directly return "succeeded" with [[10, 20]] as the message, if the video length is not shorter than 20s.

For your ease of decision, here are some initial frame captions and their timestamps for you (in the form of key value pairs, where the value is the frame caption and the key is its corresponding timestamp, in the unit of second): [initial_captions_flag].
[sys_usr_split]

Now let's start!

""" # [plan_flag] [video_length_flag] [frame_rate_flag] [initial_captions_flag] [sys_usr_split] 


prompt_viewer = """
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

""" # [sys_usr_split] [validator_request_flag] [video_length_flag]
# omitted tool: 2. get_snippet_cap(start, end): parameter "start" and "end" are both integers. Return the caption of the video snippet (clip) between the given timestamps. (only suitable for very rough scan. the longer the snippet is, the less detailed and confident the returned caption will be).

prompt_get_cap_snippet = """
You are a helpful assistant that can infer the content of a video snippet by looking at language descriptions of some of its frames. in the following content, the "key" is the timestamp in the unit of second, and "value" is the image caption of the frame at the corresponding second. 

[scatter_captions_flag] 
[sys_usr_split]
Now infer the content of the video snippet and give me a coherent caption of it. Please directly return the snippet caption, containing as much information as possible.
""" # [scatter_captions_flag] [sys_usr_split]



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
    dprint_gpt_response(response, '\n')
    return response

def get_response_not_modify_conversation(conversation, model=core_model):
    completion = client.chat.completions.create(
                model=model,
                messages=conversation,
                max_tokens=1024,
                temperature=0.2
            )
    response = completion.choices[0].message.content
    dprint_gpt_response(response, '\n')
    return response

def extract_dict_from_response(untrimmed_response):
    pattern = re.compile(r'\{(.*?)\}', re.DOTALL)
    match = pattern.search(untrimmed_response)
    content = match.group(1)
    result_dict = eval('{' + content + '}')
    return result_dict



class ReSimplifyIt(EMC):
    def __init__(self, datapoint):

        super().__init__(datapoint=datapoint)

        self.failure_history = FailureHistory(self)
        self.success_history = SuccessHistory(self)

        self.current_cut_segment = [[0, datapoint.vid_duration]]
        self.current_videolength = sum([ele[1] - ele[0] for ele in self.current_cut_segment])

        self.loaded_video = None

        self.launch_num_threshold = 2

        self.terminate = False

        self.all_frame_captions = json.load(open(self.videocaptionpath))


    def load_video(self, vid_fpath):
        pass

    def set_current_segment_and_update_length(self, segment):
        self.current_cut_segment = segment
        self.current_videolength = sum([ele[1] - ele[0] for ele in segment])


    def mapping_v2p_singletimestamp(self, virtual_timestamp:float):
        '''
        v2p: virtual to physical
        '''
        cut_segments = self.current_cut_segment
        accumulated_time = 0

        for start, end in cut_segments:
            segment_duration = end - start  

            if accumulated_time <= virtual_timestamp < accumulated_time + segment_duration:
                return start + (virtual_timestamp - accumulated_time)

            accumulated_time += segment_duration

        return -1
    
    def mapping_p2v_singletimestamp(self, physical_timestamp:float):
        '''
        p2v: physical to virtual
        '''
        
        accumulated_time = 0
        cut_segments = self.current_cut_segment

        for seg in cut_segments:
            if physical_timestamp > seg[1] + 0.5:
                accumulated_time += seg[1] - seg[0]
            else:
                return accumulated_time + physical_timestamp - seg[0]

        return -1
    
    def mapping_v2p_timestampssegments(self, virtual_timestamps:list):
        '''
        v2p: virtual to physical
        '''
        cut_segments = self.current_cut_segment
        original_intervals = []

        for virtual_start, virtual_end in virtual_timestamps:
            original_interval = []
            accumulated_time = 0

            for start, end in cut_segments:
                segment_duration = end - start
                
                if accumulated_time < virtual_end and accumulated_time + segment_duration > virtual_start:
                    segment_start = max(virtual_start - accumulated_time, 0) + start
                    segment_end = min(virtual_end - accumulated_time, segment_duration) + start
                    
                    original_interval.append([segment_start, segment_end])
                accumulated_time += segment_duration
            original_intervals.extend(original_interval)

        return original_intervals
    
    def double_layer_in(p_timestamp, segments):
        for start, end in segments:
            if start <= p_timestamp <= end:
                return True
        return False
    
    def run(self):
        while not self.terminate:
            round = EMC_round(self)
            round.run()

class EMC_round():
    def __init__(self, umbrella_emc:ReSimplifyIt):
        self.umbrella_emc = umbrella_emc

        self.init_cut_segment = umbrella_emc.current_cut_segment

        self.launcher = Launcher(self)
        self.validator = Validator(self)
        self.viewer = Viewer(self)

        self.input_question = umbrella_emc.current_question

        self.latest_trial_modified_question_from_launcher = None
        self.latest_trial_editing_plan_from_launcher = None
        self.latest_decision_from_validator = None
        self.latest_message_from_validator = None

        self.trial_max = 2
        self.trial_count = 0
        

    def run(self):
        while True:
            if self.trial_count > self.trial_max:
                self.umbrella_emc.terminate = True
                break

            self.launcher.run()

            if self.launcher.trial_decision == "terminate":
                self.umbrella_emc.terminate = True
                break

            elif self.launcher.trial_decision == "process":
                self.latest_trial_modified_question_from_launcher = self.launcher.trial_modified_question
                self.latest_trial_editing_plan_from_launcher = self.launcher.trial_editing_plan
                self.latest_decision_from_validator = self.launcher.validator_decision
                self.latest_message_from_validator = self.launcher.validator_message

                if self.latest_decision_from_validator == "failed":
                    lesson = {
                        "failed_modified_question": self.launcher.trial_modified_question,
                        "corresponding_failed_plan": self.launcher.trial_editing_plan,
                        "reason": self.latest_message_from_validator
                    }
                    self.umbrella_emc.failure_history.learn_lesson(lesson)
                    self.trial_count += 1
                elif self.latest_decision_from_validator == "succeeded":
                    physical_segment = self.umbrella_emc.mapping_v2p_timestampssegments(list(self.validator.latest_message))
                    self.umbrella_emc.set_current_segment_and_update_length(physical_segment)
                    self.umbrella_emc.current_question = self.launcher.trial_modified_question
                    breakthrough = {
                        "succeeded_modified_question": self.launcher.trial_modified_question,
                        "corresponding_succeeded_plan": self.launcher.trial_editing_plan
                    }
                    self.umbrella_emc.success_history.claim_breakthrough(breakthrough)
                    self.umbrella_emc.failure_history.refresh()
                    break
                else:
                    raise ValueError("something is wrong.")


        
class Launcher():
    def __init__(self, umbrella_emc_round:EMC_round):
        self.umbrella_emc_round = umbrella_emc_round

        self.trial_decision = None
        self.trial_modified_question = None
        self.trial_editing_plan = None

        self.validator_decision = None
        self.validator_message = None

    def run(self):
        # [quesion_flag] [failure_history_flag]
        ready_launcher_prompt = prompt_launcher.replace("[quesion_flag]", self.umbrella_emc_round.input_question).replace("[failure_history_flag]", self.umbrella_emc_round.umbrella_emc.failure_history.to_str()).replace("[success_history_flag]", self.umbrella_emc_round.umbrella_emc.success_history.to_str())
        dprint_prompt(ready_launcher_prompt)

        conversation = list()

        conversation.append({"role": "system", "content": "You are a smart and helpful assistant"})
        conversation.append({"role": "user", "content": ready_launcher_prompt})

        response = get_response_not_modify_conversation(conversation)

        self.trial_decision = extract_dict_from_response(response)["decision"]
        self.trial_modified_question = extract_dict_from_response(response)["modified_question"]
        self.trial_editing_plan = extract_dict_from_response(response)["description"]

        if self.trial_decision == "terminate":
            return
        elif self.trial_decision == "process":
            self.umbrella_emc_round.validator.run()
            self.validator_decision = self.umbrella_emc_round.validator.latest_decision
            self.validator_message = self.umbrella_emc_round.validator.latest_message
        else:
            raise ValueError("launcher's decision out of control!")



class Validator():
    def __init__(self, umbrella_emc_round:EMC_round):
        self.umbrella_emc_round = umbrella_emc_round

        self.candidate_plan = None
        self.video_length = self.umbrella_emc_round.umbrella_emc.current_videolength

        self.init_prompt_mode = "isodata"

        self.latest_decision = None
        self.latest_message = None


    def run(self):
        self.candidate_plan = self.umbrella_emc_round.launcher.trial_editing_plan
        # [plan_flag] [video_length_flag] [sys_usr_split]
        ready_validator_prompt = prompt_validator.replace("[plan_flag]", self.candidate_plan).replace("[video_length_flag]", str(floor(self.video_length))).replace("[frame_rate_flag]", str(floor(self.umbrella_emc_round.umbrella_emc.frame_rate))).replace('[initial_captions_flag]', self.get_initial_captions())
        ready_validator_prompt_sys, ready_validator_prompt_usr = ready_validator_prompt.split("[sys_usr_split]")
        dprint_prompt(ready_validator_prompt)
        newest_validator_prompt_usr = ready_validator_prompt_usr

        conversation = list()
        conversation.append({"role": "system", "content": ready_validator_prompt_sys})

        while True:
            response = get_response_and_modify_conversation(newest_validator_prompt_usr, conversation)
            self.latest_decision = extract_dict_from_response(response)["decision"]
            self.latest_message = extract_dict_from_response(response)["message"]

            if self.latest_decision == "succeeded":
                break
            elif self.latest_decision == "failed":
                break
            elif self.latest_decision == "view":
                self.umbrella_emc_round.viewer.run()
                newest_validator_prompt_usr = self.umbrella_emc_round.viewer.latest_message
            else:
                raise ValueError("validator's decision out of control!")
                

    def get_initial_captions(self):
        num = 10
        v_duration = self.umbrella_emc_round.umbrella_emc.current_videolength
        if self.init_prompt_mode == 'uniform':
            sampled_vtimestamps = list(range(floor(v_duration)))[::floor(v_duration/num)][:num]
            sampled_vtimestamps = list(map(lambda x: x+floor(v_duration/(num*2)), sampled_vtimestamps))
            sampled_ptimestamps = list(map(lambda x: self.umbrella_emc_round.umbrella_emc.mapping_v2p_singletimestamp(x), sampled_vtimestamps))
        elif self.init_prompt_mode == 'isodata':
            fault_tolerance = 3
            while fault_tolerance > 0:
                try:
                    frame_feats = self.umbrella_emc_round.umbrella_emc.video_1fps_feat.float()
                    iframe_feats = self.umbrella_emc_round.umbrella_emc.video_iframe_feats.float()
                    iframe_times = self.umbrella_emc_round.umbrella_emc.video_iframes_times
                    v_iframe_p_times = [t for t in iframe_times if any(start <= t <= end for start, end in self.umbrella_emc_round.umbrella_emc.current_cut_segment)]
                    v_iframe_times = [self.umbrella_emc_round.umbrella_emc.mapping_p2v_singletimestamp(e) for e in v_iframe_p_times]
                    v_iframe_feats = frame_feats[[floor(e) for e in iframe_times]]
                    assignments, centers = isodata_clustering(v_iframe_feats, num, min_intra_cossim_otherwise_split=0.85, max_inter_cossim_otherwise_merge=0.98, min_elements_per_cluster=max(1, round(len(v_iframe_feats)/50)), max_n_clusters=15, min_n_clusters=min(10, len(v_iframe_feats)))
                    # min_intra_cossim_otherwise_split: larger value more likely split
                    # max_inter_cossim_otherwise_merge: larger value less likely merge
                    indices_of_selected_key_iframes_in_iframes_tensor_unsort = find_closest_to_centers(v_iframe_feats, centers, assignments)
                    indices_of_selected_key_iframes_in_iframes_tensor_sorted = indices_of_selected_key_iframes_in_iframes_tensor_unsort.sort().values
                    center_times = torch.tensor(list(map(lambda x: round(x, 2), v_iframe_times)))[indices_of_selected_key_iframes_in_iframes_tensor_sorted].tolist()
                    center_times = list(set(map(floor, center_times)))
                    sampled_vtimestamps = center_times
                    sampled_ptimestamps = list(map(lambda x: self.umbrella_emc_round.umbrella_emc.mapping_v2p_singletimestamp(x), sampled_vtimestamps))
                    if len(sampled_vtimestamps) < 3:
                        raise ValueError("Too few frames in clustering result.")
                    break
                except Exception as e:
                    print(e)
                    sys.exit(0)
                    fault_tolerance -= 1
                    if fault_tolerance <= 0:
                        sampled_vtimestamps = list(range(floor(v_duration)))[::floor(v_duration/num)][:num]
                        sampled_vtimestamps = list(map(lambda x: x+floor(v_duration/(num*2)), sampled_vtimestamps))
                        sampled_ptimestamps = list(map(lambda x: self.umbrella_emc_round.umbrella_emc.mapping_v2p_singletimestamp(x), sampled_vtimestamps))
                        break
        else:
            raise ValueError("fuck")
        
        captions = {str(sampled_vtimestamps[i]): self.umbrella_emc_round.umbrella_emc.all_frame_captions[str(sampled_ptimestamps[i])] for i in range(len(sampled_vtimestamps))}
        captions_str = str(captions)

        return captions_str


class Viewer():
    def __init__(self, umbrella_emc_round:EMC_round):
        self.umbrella_emc_round = umbrella_emc_round

        self.latest_decision = None
        self.latest_message = None
        self.latest_tool_output = None

        self.video_length = self.umbrella_emc_round.umbrella_emc.current_videolength


    def run(self):
        self.input_request = self.umbrella_emc_round.validator.latest_message

        assert self.input_request is not None
        # [sys_usr_split] [validator_request_flag]
        ready_viewer_prompt = prompt_viewer.replace("[validator_request_flag]", self.input_request).replace('[video_length_flag]', str(self.video_length))
        ready_viewer_prompt_sys, ready_viewer_prompt_usr = ready_viewer_prompt.split('[sys_usr_split]')
        dprint_prompt(ready_viewer_prompt)

        newest_viewer_prompt_usr = ready_viewer_prompt_usr
        conversation = list()
        conversation.append({"role": "system", "content": ready_viewer_prompt_sys})

        while True:
            response = get_response_and_modify_conversation(newest_viewer_prompt_usr, conversation)
            self.latest_decision = extract_dict_from_response(response)["decision"]
            self.latest_message = extract_dict_from_response(response)["message"]

            if self.latest_decision == "respond":
                break # self.latest_message is now the viewer's response to validator
            elif self.latest_decision == "tool":
                exec('self.'+self.latest_message)
                newest_viewer_prompt_usr = self.latest_tool_output
            else:
                raise ValueError("viewer's decision out of control!")

    def get_image_cap(self, virtual_timestamp:int):
        physical_timestamp = self.umbrella_emc_round.umbrella_emc.mapping_v2p_singletimestamp(virtual_timestamp)
        if physical_timestamp  > self.umbrella_emc_round.umbrella_emc.vid_physical_duration + 2:
            return "Error. This timestamp exceeds the video length."
        key = str(round(max(0, min(physical_timestamp, self.umbrella_emc_round.umbrella_emc.vid_physical_duration - 1))))
        caption = self.umbrella_emc_round.umbrella_emc.all_frame_captions[key]
        self.latest_tool_output = caption
        return caption
        

    def get_snippet_cap(self, virtual_start:int, virtual_end:int):
        physical_start = self.umbrella_emc_round.umbrella_emc.mapping_v2p_singletimestamp(virtual_start)
        physical_end = self.umbrella_emc_round.umbrella_emc.mapping_v2p_singletimestamp(virtual_end)
        if physical_start  > self.umbrella_emc_round.umbrella_emc.vid_physical_duration + 2:
            self.latest_tool_output = "Error. The starting timestamp exceeds the video length. Please check your input."
            return
        caps = ""
        ext = ""
        skip = max(1, floor((physical_end - physical_start) / 20))
        for v_timestamp in range(int(virtual_start), int(virtual_end + 1), skip):
            candidate = str(self.get_image_cap(v_timestamp))
            if "Error" in candidate:
                ext = "Note that the ending timestamp provided exceeds the video length. Snippet caption from start to the end of the video is provided."
                break
            caps += str(v_timestamp) + ": "
            caps += candidate
            caps += "\n"
        caps += "\n"

        ready_sys_prompt, ready_usr_prompt = prompt_get_cap_snippet.replace("[scatter_captions_flag]", caps).split("[sys_usr_split]")
        conversation = list()
        conversation.append({"role": "system", "content": ready_sys_prompt})
        conversation.append({"role": "user", "content": ready_usr_prompt})

        self.latest_tool_output = get_response_not_modify_conversation(conversation, model=tool_model)+ "\n" + ext

    def scan(self, virtual_start:int, virtual_end:int):
        print("Scanning video snippet from " + str(virtual_start) + "s to " + str(virtual_end) + "s..., current length is " + str(self.umbrella_emc_round.umbrella_emc.current_videolength) + "s")
        self.get_snippet_cap(virtual_start=virtual_start, virtual_end=virtual_end)
        print(self.latest_tool_output)
        
    def localize(self, query):

        question = query

        vid_allcap = self.umbrella_emc_round.umbrella_emc.all_frame_captions

        vid_fname = self.umbrella_emc_round.umbrella_emc.video_fname

        k = init_n_clusters  # initialization of number of clusters, set it to appropriate value (e.g. 10)

        try:
            all_iframes_times = self.umbrella_emc_round.umbrella_emc.video_iframes_times
            all_iframes_features = self.umbrella_emc_round.umbrella_emc.video_iframes_feats

            filterd_p_iframe_times = [t for t in all_iframes_times if self.umbrella_emc_round.umbrella_emc.double_layer_in(t, self.umbrella_emc.current_cut_segment) and t < all_iframes_features.shape[0]]
            filterd_iframe_features = all_iframes_features[[all_iframes_times.index(t) for t in filterd_p_iframe_times]]
            
            assignments, centers = isodata_clustering(filterd_iframe_features, k, min_intra_cossim_otherwise_split=0.85, max_inter_cossim_otherwise_merge=0.98, min_elements_per_cluster=max(1, round(len(all_iframes_features)/50)), max_n_clusters=max_num_clusters, min_n_clusters=min(min_num_clusters, len(all_iframes_features)))
            # min_intra_cossim_otherwise_split: larger value more likely split
            # max_inter_cossim_otherwise_merge: lerger value less likely merge

            indices_of_selected_key_iframes_in_iframes_tensor_sorted = find_closest_to_centers(filterd_iframe_features, centers, assignments).sort().values
            center_p_times = torch.tensor(filterd_p_iframe_times)[indices_of_selected_key_iframes_in_iframes_tensor_sorted].tolist()
            center_p_times = sorted(list(set(map(int, center_p_times))))

        except:
            center_v_times = list(range(floor(self.umbrella_emc_round.umbrella_emc.current_videolength)))[::round(floor(self.umbrella_emc_round.umbrella_emc.current_videolength)/k)]
            center_p_times = list(map(lambda x: self.umbrella_emc_round.umbrella_emc.mapping_v2p_singletimestamp(x), center_v_times))

        d = {'frame caption at '+str(self.umbrella_emc_round.umbrella_emc.mapping_p2v_singletimestamp(k))+'s': vid_allcap[str(k)] for k in center_p_times}

        str_initial_captions = str(d)

        initial_prompt_template = '''
        You are a smart assistant to find some timestamps of a video that corresponds to a natrual language query as accurate and informative as possible. Here, the query is: "[original_question_flag]".

        In fact, you will need to propose five of these timestamps.

        A tool (python function) will be helping you to get the frame caption of at a certain timestamp (in the unit of second). Whenever you need to call this tool, send a message in this json format: 【{"decision": "tool", "parameter": [timestamp you need, as an integer]}】. For example, if you need the caption at 3s of the video, return 【{"decision": "tool", "parameter": 3}】, then it will be returned to you. \
        Whenever you think you are confident enough to provide the timestamps, return 【{"decision": "end", "timestamps": [your result five timestamps]}】. For example, if you decide that timestamp 5s, 28s, 97s, 112s, and 343s are the most possible timestamps to contain the content of the query, then return 【{"decision": "end", "timestamp": [5, 28, 97, 112, 343]}】.

        Before we formally begin, here is a set of original captions with their timestamps provided for you to have an overall rough understanding of the video (in the form of key-value pairs, where the key is the timestamp in the unit of second, and its value is the corresponding frame caption at this timestamp): [initial_captions_flag]. (so you don't have to call tool to get captions of these timestamps, as they are already provided here.)
        Also, the frame rate of this video is [frame_rate_flag] frames per second, and the total duration is [duration_flag].

        Note that the frame captions are captions of the static video frame as an image, so it might not explicitly mention enough dynamics, so you will need to infer on these yourself.

        You may confidently assume that the object/event requested by the query exist in the video, while the captions are based on static images so you need to reasonably infer the dynamics and actions yourself.
        Remember always give your decision in the json format provided above. If you are very confident that the requested object/event does not appear in the video (which is really unlikely), return 【{"decision": "end", "timestamp": [-1, -1, -1, -1, -1]}】.

        [sys_usr_split]Now let's begin!
        '''

        initial_captions = str_initial_captions


        initial_prompt = initial_prompt_template.replace("[initial_captions_flag]", initial_captions).replace("[frame_rate_flag]", str(round(self.umbrella_emc_round.umbrella_emc.frame_rate, 2))).replace("[duration_flag]", str(floor(self.umbrella_emc_round.umbrella_emc.current_videolength))).replace("[original_question_flag]", question)
        initial_prompt_sys = initial_prompt.split('[sys_usr_split]')[0]
        initial_prompt_usr = initial_prompt.split('[sys_usr_split]')[1]

        dprint_viewer_subprompt(initial_prompt)

        conversation = list()
        response = '【{}】'

        conversation.append({"role": "system", "content": initial_prompt_sys})
        conversation.append({"role": "user", "content": initial_prompt_usr})

        while True:
            if len(conversation) > 60:
                final_timestamps = []
                break
            while True:
                err_count = 0
                response = get_response_not_modify_conversation(conversation)
                yes = True
                try:
                    j = extract_dict_from_response(response)
                    yes = yes and 'decision' in j
                    if j['decision'] == 'tool':
                        yes = yes and j["parameter"] >= 0
                    elif j['decision'] == 'end':
                        yes = yes and "timestamps" in j
                except Exception as e:
                    dprint(e, response, 'retrying......')
                    err_count += 1
                    yes = False
                    if err_count >= 10:
                        j = {'decision': 'end', 'timestamps': np.linspace(0, floor(self.umbrella_emc_round.umbrella_emc.current_videolength), 11, dtype=int)[1:10:2].tolist()}
                        break
                if yes:
                    conversation.append({"role": "assistant", "content": response})
                    break

            if j['decision'] == 'end':
                final_timestamps = j["timestamps"]
                break
            else:
                request = round(j["parameter"])
                cap = vid_allcap[str(self.umbrella_emc_round.umbrella_emc.mapping_v2p_singletimestamp(int(request)))]
                conversation.append({"role": "user", "content": cap})
        

        final_timestamps = sorted(list(set([ele for ele in final_timestamps if ele != -1])))
        

        if len(final_timestamps) == 0:
            final_timestamps = np.linspace(0, floor(self.umbrella_emc_round.umbrella_emc.current_videolength), 11, dtype=int)[1:10:2].tolist()

        selection_prompt_template = """
        You are a smart assistant to select one timestamp from a given list of video timestamps that is the most suitable one to contain the content of a natrual language query as accurate and informative as possible. Here, the query is: "[original_question_flag]". The list of timestamps for you to choose from is: [choices_flag]. The timestamps are all in the unit of second.

        Also, for you to start, here are the video frame captions of the frames at these timestamps: [proposals_frame_captions_flag]. 
        
        In addition, for you to acquire additional video frame captions at other timestamps to have a better understanding of the video content, a tool (python function) will be helping you to get the frame caption of at a certain timestamp (in the unit of second). Whenever you need to call this tool, send a message in this json format: 【{"decision": "tool", "parameter": [timestamp you need, as an integer]}】. For example, if you want the caption at 3s of the video, return 【{"decision": "tool", "parameter": 3}】, then it will be returned to you. 
        Whenever you think you are confident enough to confirm the selection of the best timestamp, return 【{"decision": "end", "timestamp": [your result timestamp as an integer]}】. For example, if the list given to you is [5, 28, 97, 112, 343] and you think that the video frame at 97 second is most likely to contain the content of the query, then return 【{"decision": "end", "timestamp": 97}】.

        Also, the frame rate of this video is [frame_rate_flag] frames per second, and the total duration is [duration_flag].

        Note that the frame captions are captions of the static video frame as an image, so it might not explicitly mention enough dynamics, so you will need to imagine as vividly as possible and infer on these yourself.

        Remember always give your decision in the json format provided above. 

        [sys_usr_split]Now let's begin!
        """  # [original_question_flag] [proposals_frame_captions_flag] [choices_flag] [frame_rate_flag] [duration_flag]

        proposals_frame_captions = dict()

        for ts in final_timestamps:
            proposals_frame_captions['frame caption at '+str(ts)+'s'] = vid_allcap[str(self.umbrella_emc_round.umbrella_emc.mapping_v2p_singletimestamp(ts))]

        proposals_frame_captions = str(proposals_frame_captions)

        selection_prompt = selection_prompt_template.replace("[frame_rate_flag]", str(round(self.umbrella_emc_round.umbrella_emc.frame_rate, 2))).replace("[proposals_frame_captions_flag]", proposals_frame_captions).replace("[duration_flag]", str(floor(self.umbrella_emc_round.umbrella_emc.current_videolength))).replace("[original_question_flag]", question).replace('[choices_flag]', str(final_timestamps))
        selection_prompt_sys = selection_prompt.split('[sys_usr_split]')[0]
        selection_prompt_usr = selection_prompt.split('[sys_usr_split]')[1]

        dprint_viewer_subprompt(selection_prompt)

        conversation = list()
        response = '【{}】'

        conversation.append({"role": "system", "content": selection_prompt_sys})
        conversation.append({"role": "user", "content": selection_prompt_usr})

        while True:
            if len(conversation) > 60:
                final_choice = -1
                break
            while True:
                err_count = 0
                response = get_response_not_modify_conversation(conversation)
                yes = True
                try:
                    j = extract_dict_from_response(response)
                    yes = yes and 'decision' in j
                    if j['decision'] == 'tool':
                        yes = yes and j["parameter"] >= 0
                    elif j['decision'] == 'end':
                        yes = yes and "timestamp" in j
                except Exception as e:
                    print(e, response, 'retrying......')
                    err_count += 1
                    yes = False
                    if err_count >= 10:
                        j = {'decision': 'end', 'timestamp': -1}
                        break
                if yes:
                    conversation.append({"role": "assistant", "content": response})
                    break

            if j['decision'] == 'end':
                final_choice = j["timestamp"]
                break
            else:
                request = round(j["parameter"])
                cap = vid_allcap[str(self.umbrella_emc_round.umbrella_emc.mapping_v2p_singletimestamp(int(request)))]
                conversation.append({"role": "user", "content": cap})
        

        if type(final_choice) != int:
            final_choice = -1

        if final_choice != -1:

            spread_prompt_template = '''
                You are a smart assistant to carefully find (locate) a timestamp range of a video that corresponds to a natrual language query as accurate and informative as possible, given an anchor timestamp which is very likely to be located within the timestamp range you should return. 
                Here, the query is: "[original_question_flag]", and the anchor timestamp provided to you is: [anchor_timestamp_flag].
                Again, this anchor timestamp should be within the timestamp range you return. You may start from this anchor timestamp and gradually expand forwards and backwards to determine the start and end as the boundaries.

                A tool (python function) will be helping you to get the frame caption at a certain timestamp (in the unit of second). Whenever you need to call this tool, send a message in this json format: 【{"decision": "tool", "parameter": [timestamp you need, as an integer]}】. For example, if you want the caption at 5s of the video, return 【{"decision": "tool", "parameter": 5}】. The corresponding caption will be returned to you. \
                Whenever you think you are confident enough to provide the timestamp, return 【{"decision": "end", "timestamps": [your result timestamps, in the form of a two-layered list]}】. For example, if you decide to preserve the sub-clip from 4s to 18s, then return 【{"decision": "end", "timestamps": [[4, 18]]}】.

                Also, the frame rate of this video is [frame_rate_flag] frames per second, and the total duration is [duration_flag].

                Note that the frame captions are captions of the static video frame as an image, so it might not explicitly mention enough dynamics, so you will need to infer on these dynamic actions yourself.

                [sys_usr_split]Now let's begin!.
            ''' # [original_question_flag] [anchor_timestamp_flag] [frame_rate_flag] [duration_flag]

            spread_prompt = spread_prompt_template.replace("[frame_rate_flag]", str(round(self.umbrella_emc_round.umbrella_emc.frame_rate, 2))).replace("[duration_flag]", str(floor(self.umbrella_emc_round.umbrella_emc.current_videolength))).replace("[original_question_flag]", question).replace('[anchor_timestamp_flag]', str(final_choice))
            spread_prompt_sys = spread_prompt.split('[sys_usr_split]')[0]
            spread_prompt_usr = spread_prompt.split('[sys_usr_split]')[1]
        else:
            spread_prompt_template = '''
                You are a smart assistant to carefully find (locate) a timestamp range of a video that corresponds to a natrual language query as accurate and informative as possible. Here, the query is: "[original_question_flag]".
                Before we formally begin, here is a set of original captions with their timestamps provided for you to have an overall rough understanding of the video (in the form of key-value pairs, where the key is the timestamp in the unit of second, and its value is the corresponding frame caption at this timestamp): [initial_captions_flag].

                A tool (python function) will be helping you to get the frame caption at a certain timestamp (in the unit of second). Whenever you need to call this tool, send a message in this json format: 【{"decision": "tool", "parameter": [timestamp you need, as an integer]}】. For example, if you want the caption at 5s of the video, return 【{"decision": "tool", "parameter": 5}】. The corresponding caption will be returned to you. \
                Whenever you think you are confident enough to provide the timestamp, return 【{"decision": "end", "timestamps": [your result timestamps, in the form of a two-layered list]}】. For example, if you decide to preserve the sub-clip from 4s to 18s, then return 【{"decision": "end", "timestamps": [[4, 18]]}】.

                Also, the frame rate of this video is [frame_rate_flag] frames per second, and the total duration is [duration_flag].

                Note that the frame captions are captions of the static video frame as an image, so it might not explicitly mention enough dynamics, so you will need to infer on these yourself.

                [sys_usr_split]Now let's begin!
            ''' # [original_question_flag] [initial_captions_flag] [frame_rate_flag] [duration_flag]

            spread_prompt = spread_prompt_template.replace("[frame_rate_flag]", str(round(self.umbrella_emc_round.umbrella_emc.frame_rate, 2))).replace("[duration_flag]", str(floor(self.umbrella_emc_round.umbrella_emc.frame_rate))).replace("[original_question_flag]", question).replace('[initial_captions_flag]', str(initial_captions))
            spread_prompt_sys = spread_prompt.split('[sys_usr_split]')[0]
            spread_prompt_usr = spread_prompt.split('[sys_usr_split]')[1]

        dprint_viewer_subprompt(spread_prompt)
        conversation = list()
        response = '【{}】'

        conversation.append({"role": "system", "content": spread_prompt_sys})
        conversation.append({"role": "user", "content": spread_prompt_usr})

        while True:
            err_count = 0
            if len(conversation) > 60:
                final_timestamps = [[]]
                break
            while True:
                response = get_response_not_modify_conversation(conversation)
                yes = True
                try:
                    j = extract_dict_from_response(response)
                    yes = yes and 'decision' in j
                    if j['decision'] == 'tool':
                        yes = yes and j["parameter"] >= 0
                    elif j['decision'] == 'end':
                        yes = yes and "timestamps" in j
                        for ele in j["timestamps"]:
                            yes = yes and ele[0]>=0 and ele[1]>=0
                except Exception as e:
                    print(e, response, 'retrying......')
                    err_count += 1
                    yes = False
                    if err_count >= 10:
                        j = {'decision': 'end', 'timestamps': [[0, floor(self.umbrella_emc_round.umbrella_emc.current_videolength)]]}
                        break
                if yes:
                    conversation.append({"role": "assistant", "content": response})
                    break

            if j['decision'] == 'end':
                final_timestamps = j["timestamps"]
                break
            else:
                request = round(j["parameter"])
                cap = vid_allcap[str(self.umbrella_emc_round.umbrella_emc.mapping_v2p_singletimestamp(int(request)))]
                conversation.append({"role": "user", "content": cap})
        
        message = final_timestamps
        if message in ([[]], [], None) or type(final_timestamps) != list:
            message = 'localization failed.'
            final_timestamps = [[0, self.umbrella_emc_round.umbrella_emc.current_videolength]]
        
        self.latest_tool_output = str(message)
        return message


class FailureHistory():
    def __init__(self, umbrella_emc_round:EMC_round):
        self.failure_history = list()

    def to_str(self):
        return json.dumps(self.failure_history)
    
    def learn_lesson(self, lesson:dict):
        self.failure_history.append(lesson)

    def refresh(self):
        self.failure_history = list()


class SuccessHistory():
    def __init__(self, umbrella_emc_round:EMC_round):
        self.success_history = list()

    def to_str(self):
        return json.dumps(self.success_history)

    def claim_breakthrough(self, breakthrough:dict):
        self.success_history.append(breakthrough)

    def refresh(self):
        self.success_history = list()




api_key = openai_api_key
core_model = "gpt-4o"
tool_model = "gpt-3.5-turbo"
force_redo = False


if os.path.exists(EMC_save_path) and not force_redo:
    to_write = json.load(open(EMC_save_path))
else:
    to_write = dict()

for dataset_name in ('yc2emc', ):

    if dataset_name in to_write and not force_redo:
        to_write_dataset = to_write[dataset_name]
    else:
        to_write_dataset = dict()

    dataset = json.load(open(dataset_json_paths[dataset_name]))
    all_keys = list(dataset.keys())
    pbar = tqdm(total=len(all_keys))

    # class EMC_Datapoint():
    #     def __init__(self, question, video_fname, videocaptionpath, vid_frame_rate, vid_duration, iframes_times, iframes_feats):
    #         self.question = question
    #         self.video_fname = video_fname
    #         self.videocaptionpath = videocaptionpath
    #         self.vid_frame_rate = vid_frame_rate
    #         self.vid_duration = vid_duration
    #         self.iframes_times = iframes_times
    #         self.iframes_feats = iframes_feats

    def thread_run():

        while True:
            try:
                key = all_keys.pop(0)
            except:
                return
            if key in to_write_dataset and not force_redo:
                pbar.update(1)
                continue

            dp = dataset[key]

            question = dp["question"]
            video_fname = dp["vid_fname"]
            videocaptionpath = os.path.join(dataset_captions_folders[dataset_name], dp['vid_name']+'.json')
            vid_frame_rate = dp["vid_frame_rate"]
            vid_duration = dp["vid_duration"]

            vid_1fps_feat = load_video_1fps_feat(os.path.join(dataset_vid_1fps_feature_folder, video_fname.split('.')[0]+'.pt'))
            vid_iframe_times = json.load(open(iframes_times_save_path))[video_fname]
            vid_iframe_feats = vid_1fps_feat[[floor(e) for e in vid_iframe_times]]

            datapoint = EMC_Datapoint(question, video_fname, videocaptionpath, vid_frame_rate, vid_duration, vid_iframe_times, vid_iframe_feats, vid_1fps_feat)

            emc = ReSimplifyIt(datapoint)
            emc.run()
            modified_question, output_timestamps = emc.current_question, emc.current_cut_segment

            dprint(key, 'gt: ', dataset[key]['question'], dataset[key]['gt_timestamp'], dataset[key]["vid_duration"], dataset[key]['type'])
            dprint(key, modified_question, output_timestamps, '\n')

            to_write_dataset[key] = {
                "screened_timestamps": output_timestamps,
                "screened_question": modified_question
            }

            while True:
                try:
                    to_write[dataset_name] = to_write_dataset
                    break
                except Exception as e:
                    time.sleep(0.01)
                    pass

            while True:
                try:
                    with open(EMC_save_path, 'w') as f:
                        json.dump(to_write, f)
                    break
                except Exception as e:
                    time.sleep(0.01)
                    pass

            pbar.update(1)


    
    if num_threads != 1:
        threads = list()

        for i in range(num_threads):
            threads.append(threading.Thread(target=thread_run))
        # for i in range(num_threads):
        #     threads[i].setDaemon(True)
        for i in range(num_threads):
            threads[i].start()
            time.sleep(0.01)
        for i in range(num_threads):
            threads[i].join()

    else:
        thread_run()

    pbar.close()






