from openai import OpenAI
from typing import Dict, Optional, List
import json
import base64
import os
from git_ignore import *
import sys
import time
from tqdm import tqdm

class ProPrietaryModelInterface:
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the GPT-4 API client. The API key is read from the environment
        variable OPENAI_API_KEY if not provided.
        """
        self.api_key = openai_api_key # imported from git_ignore.py
        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model
        if not self.api_key:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")

    def _build_messages(self, system_message: str, user_content: List) -> List[Dict]:
        """
        Build the messages list required by the OpenAI API.
        """
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    def inference_qa(
        self,
        text_only_raw_question: str,
        all_16_frames: List[str],
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Perform multiple-choice inference using GPT-4 API.
        
        Args:
            question: The question to answer.
            options: Multiple-choice options as a string.
            frames: Optional visual context.
        
        Returns:
            The selected option (e.g., A, B, C, D).
        """
        system_message = "You are a helpful assistant that answers a question based on the provided video frames."
        query = f"<video> The images are uniformly sampled frames from the video. Answer the following question: {text_only_raw_question}\n"
        # print(f"Query: {query}")
        return self.inference_with_frames(
            query=query,
            all_16_frames=all_16_frames,
            num_frames_sampled=8,  # Default to 8 frames
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def inference_with_frames(
        self,
        query: str, # e.g. "<video> According to the video, what is the main topic discussed?"
        all_16_frames: List[str],
        num_frames_sampled: int=8, # Number of frames to sample, choose from 16, 8, 4, 2, or 1
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        A unified inference interface supporting mixed text and image inputs.
        The query may include <image> tags.
        """
        query = query.replace("<video>", "|||<video>|||")
        parts = query.split("|||")
        user_content = []
        frames = all_16_frames[::int(16/num_frames_sampled)]
        for i, part in enumerate(parts):
            if part != '<video>':
                if part:
                    user_content.append({"type": "text", "text": part.strip()})
            else:
                for frame_base64 in frames:
                    visual_context = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}",
                            "detail": "low"
                        }
                    }
                    user_content.append(visual_context)

        messages = self._build_messages(system_message, user_content)
        # for i in messages:
        #     print(i)
        try:
            # # return
            # response = self.client.chat.completions.create(
            #     model=self.model_name,
            #     messages=messages,
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            # )
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=1,
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"



model_names = ['gpt-4.1-mini', 'gpt-4o', 'gpt-4-turbo', 'o4-mini']
dataset_names = ['yc2tvs', 'activitynetqa', 'nextoe']
tvs_oneforall = json.load(open(TVS_save_path, 'r', encoding='utf-8'))


for model_name in model_names:
    interface = ProPrietaryModelInterface(model=model_name)
    if os.path.exists(proprietary_inference_save_paths[model_name]):
        to_write = json.load(open(proprietary_inference_save_paths[model_name], 'r', encoding='utf-8'))
    else:
        to_write = dict()

    for dataset_name in dataset_names:
        if dataset_name not in to_write:
            to_write[dataset_name] = dict()
        dataset_json_path = dataset_json_paths[dataset_name]
        dataset = json.load(open(dataset_json_path, 'r', encoding='utf-8'))
        dataset_tvs = tvs_oneforall[dataset_name]
        
        for key in tqdm(dataset):
            if key in to_write[dataset_name]:
                continue
            ori_dp = dataset[key]
            tvs_dp = dataset_tvs[key]
            ori_question = ori_dp['question']
            screened_question = tvs_dp['screened_question']
            base64_screened = json.load(open(os.path.join(base64_screened_image_folder, dataset_name, f"{key}.json"), 'r', encoding='utf-8'))
            base64_screened_16_frames = [base64_screened[k] for k in base64_screened]
            base64_uniform = json.load(open(os.path.join(base64_uniform_image_folder, dataset_name, f"{key}.json"), 'r', encoding='utf-8'))
            base64_uniform_16_frames = [base64_uniform[k] for k in base64_uniform]

            response_control = interface.inference_qa(
                text_only_raw_question=ori_question,
                all_16_frames=base64_uniform_16_frames
            )
            print(response_control)
            if tvs_dp["status"] == 'failed' or 'error' in str(base64_screened_16_frames):
                response_treatment = response_control
                cut_success = 'failed'
            else:
                response_treatment = interface.inference_qa(
                    text_only_raw_question=screened_question,
                    all_16_frames=base64_screened_16_frames
                )
                cut_success = 'succeeded'
            print(response_treatment)

            to_write[dataset_name][key] = {
                'cut_success': cut_success,
                'response_control': response_control,
                'response_treatment': response_treatment
            }

            with open(proprietary_inference_save_paths[model_name], 'w', encoding='utf-8') as f:
                json.dump(to_write, f)
            






