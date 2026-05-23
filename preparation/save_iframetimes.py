# this file stores all i-frame times (rounded to integer seconds) of all videos in a given folder

import json
import subprocess
import os
from tqdm import tqdm
from math import floor

from config import iframes_times_save_path, video_folder_for_computing_iframe_times

vid_folder = video_folder_for_computing_iframe_times
save_path = iframes_times_save_path

def get_iframes_times(video_path):
    assert os.path.exists(video_path)
    # Use ffprobe to get video stream information
    probe_command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=pts_time,pict_type', '-of', 'csv',
        video_path
    ]
    result = subprocess.run(probe_command, capture_output=True, text=True, check=True)
    # print(result)

    # find I-frame indices
    iframe_times = []
    for line in result.stdout.splitlines():
        # print(line)
        parts = line.split(',')
        # print(parts)
        if len(parts) == 3 and parts[2].strip() == 'I':
            iframe_times.append(float(parts[1]))

    duration_command = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    duration_result = subprocess.run(duration_command, capture_output=True, text=True, check=True)
    duration = float(duration_result.stdout.strip())

    # print(video_path, iframe_times, duration)

    iframe_times = list(set(
        round(t) for t in iframe_times
        if 0 <= round(t) <= floor(duration)
    ))
    iframe_times.sort()
    
    if len(iframe_times) == 0:
        raise ValueError("fuck")
    
    return iframe_times


all_vids = os.listdir(vid_folder)

for fname in tqdm(all_vids):
    if os.path.exists(save_path):
        to_write = json.load(open(save_path))
    else:
        to_write = dict()
    if fname in to_write:
        continue
    fpath = os.path.join(vid_folder, fname)
    iframe_times = get_iframes_times(fpath)

    to_write[fname] = iframe_times
    
    with open(save_path, 'w') as f:
        json.dump(to_write, f)