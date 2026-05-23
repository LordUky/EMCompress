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


def load_video_raw(vis_path):

    cap = cv2.VideoCapture(vis_path)
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        full_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target_h, target_w = 224, 224

        if full_frame.shape[-3] != target_h or full_frame.shape[-2] != target_w:
            temp = torch.from_numpy(full_frame).permute(2, 0, 1).float()
            temp = temp.unsqueeze(0)
            temp = torch.nn.functional.interpolate(temp, size=(target_h, target_w))
            temp = temp.squeeze(0)
            ready = temp.permute(1, 2, 0)
        else:
            ready = full_frame

        all_frames.append(Image.fromarray(ready.to(torch.uint8).numpy()))

    return all_frames



def scatter_frame_indices_to_timestamp_ranges(final_frames, frame_rate, video_duration):
    """
    Convert a list of frame indices into a list of timestamp ranges.

    Args:
        final_frames (list): List of selected frame indices.
        frame_rate (float): Frame rate of the video.
        video_duration (float): Total duration of the video in seconds.

    Returns:
        list: A list of tuples, where each tuple represents a start and end timestamp.
    """
    screened_timestamps = []
    start_idx = final_frames[0]
    for i in range(1, len(final_frames)):
        if final_frames[i] != final_frames[i - 1] + 1:
            end_idx = final_frames[i - 1]
            screened_timestamps.append((
                max(0, round(start_idx / frame_rate, 2)),
                min(round(end_idx / frame_rate, 2), video_duration)
            ))
            start_idx = final_frames[i]
    # Add the last segment
    screened_timestamps.append((
        round(start_idx / frame_rate, 2),
        min(round(final_frames[-1] / frame_rate, 2), video_duration)
    ))
    
    return screened_timestamps