import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from decord import VideoReader, cpu
import random
import numpy as np
import os
from tqdm import tqdm
import time

debug_normal = False
debug_prompt = True
debug_viewer_subprompt = False
debug_gptresponse = True

def dprint(*args, **kwargs):
    if debug_normal:
        print('--------------------------------------------------------------')
        print(*args, **kwargs)
        print('--------------------------------------------------------------')

def dprint_prompt(*args, **kwargs):
    if debug_prompt:
        print('--------------------------------------------------------------')
        print(*args, **kwargs)
        print('--------------------------------------------------------------')

def dprint_viewer_subprompt(*args, **kwargs):
    if debug_viewer_subprompt:
        print('--------------------------------------------------------------')
        print(*args, **kwargs)
        print('--------------------------------------------------------------')

def dprint_gpt_response(*args, **kwargs):
    if debug_gptresponse:
        print('--------------------------------------------------------------')
        print(*args, **kwargs)
        print('--------------------------------------------------------------')

def load_video_raw(vis_path):
    assert os.path.exists(vis_path)

    dprint('loading raw video from '+vis_path+'......')

    cap = cv2.VideoCapture(vis_path)
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        full_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        all_frames.append(full_frame)

    dprint('raw video loaded.')
    return torch.tensor(np.array(all_frames))


def extract_iframes_as_tensor(loaded_raw, iframes_times, meta):
    dprint('extracting iframes as tensors......')
    total_frame_num = len(loaded_raw)
    duration = meta['duration']
    indices = torch.tensor([round(total_frame_num * ele/duration) for ele in iframes_times])
    # print(indices)
    dprint('iframes as tensors loaded')
    return loaded_raw[indices]


def double_layer_IoU(listlist1, listlist2):
    MIN = min(min(min(sublist) for sublist in listlist1), min(min(sublist) for sublist in listlist2))
    MAX = max(max(max(sublist) for sublist in listlist1), max(max(sublist) for sublist in listlist2))
    U = MAX - MIN

    I = 0
    for lst1 in listlist1:
        for lst2 in listlist2:
            I += max(0, min(lst1[1], lst2[1]) - max(lst1[0], lst2[0]))

    return I/U


def load_video_pt(vis_path):

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
            ready = torch.from_numpy(full_frame)

        all_frames.append(ready.to(torch.uint8).unsqueeze(0))
        # print(len(all_frames), end=' ')

    return torch.cat(all_frames)

def load_video_1fps_feat(feature_path):
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file {feature_path} does not exist.")
    
    with open(feature_path, 'rb') as f:
        data = torch.load(f).cpu()

    return data


def isodata_clustering(data, k_init, max_iterations=100, min_intra_cossim_otherwise_split=0.95, max_inter_cossim_otherwise_merge=0.98, max_n_clusters=15, min_n_clusters=1, max_center_shift=0.01, min_elements_per_cluster=1):
    dprint('ISODATA clustering started......')
    # Normalize data for cosine similarity
    data_normalized = torch.nn.functional.normalize(data, p=2, dim=1)

    # Initialize cluster centers randomly
    centers = data[torch.randperm(len(data))[:k_init]].to(data.device)

    for _ in range(max_iterations):

        dprint('ite:', _, end = ': ')

        # Calculate cosine similarities
        centers_normalized = torch.nn.functional.normalize(centers, p=2, dim=1)
        similarities = torch.mm(data_normalized, centers_normalized.T)
        assignments = torch.argmax(similarities, dim=1)

        # Update cluster centers
        mask = assignments.unsqueeze(1) == torch.arange(k_init, device=data.device).unsqueeze(0)
        cluster_sums = torch.matmul(mask.float().T, data)
        cluster_counts = mask.sum(dim=0).float()

        prev_centers = centers
        new_centers = torch.where(cluster_counts.unsqueeze(1) > 0, cluster_sums / cluster_counts.unsqueeze(1), centers)
        centers = new_centers

        merged = False
        split = False

        # Splitting clusters
        mask_split = cluster_counts > 0
        if mask_split.any():
            avg_similarities = torch.full_like(mask_split, float('inf'), dtype=torch.float).to(data.device)
            for i in range(k_init):
                if mask_split[i]:
                    cluster_points = data_normalized[assignments == i]
                    avg_similarities[i] = torch.mean(torch.mm(cluster_points, centers_normalized[i].unsqueeze(0).T))

            split_indices = avg_similarities < min_intra_cossim_otherwise_split
            if split_indices.sum().item() >= 1:
                trues = torch.nonzero(split_indices).squeeze()
                if len(trues.shape) == 0:
                    index = int(trues)
                else:
                    index = random.choice(trues.tolist())
            if split_indices.any() and (k_init + 1 <= max_n_clusters):
                delta = 0.1 * torch.randn(1, data.shape[1], device=data.device)
                new_center_1 = centers[index] + delta
                new_center_2 = centers[index] - delta
                centers = torch.cat((centers[:index], new_center_1, new_center_2, centers[index+1:]))
                k_init += 1
                similarities = torch.mm(data_normalized, torch.nn.functional.normalize(centers, p=2, dim=1).T)
                assignments = torch.argmax(similarities, dim=1)
                split = True
                dprint('split to:', len(centers))

        # Merging clusters based on inter-cluster similarity
        if not split and k_init - 1 >= min_n_clusters:
            similarities_matrix = torch.mm(centers_normalized, centers_normalized.T)
            similarities_matrix.fill_diagonal_(float('-inf'))
            max_sim, max_indices = torch.max(similarities_matrix, dim=1)

            merge_indices = torch.nonzero(max_sim > max_inter_cossim_otherwise_merge)
            if len(merge_indices) > 0:
                i = merge_indices[random.randint(0, len(merge_indices) - 1)].item()
                j = max_indices[i].item()

                centers[i] = (centers[i] + centers[j]) / 2
                centers = torch.cat((centers[:j], centers[j+1:]))
                k_init -= 1
                similarities = torch.mm(data_normalized, torch.nn.functional.normalize(centers, p=2, dim=1).T)
                assignments = torch.argmax(similarities, dim=1)
                merged = True
                dprint('merged_sim to:', len(centers))

        # Merging clusters based on violation of minimum number of elements per cluster
        if k_init - 1 >= 1: # Force clustering
            no_small_clusters = False
            while not no_small_clusters and k_init - 1 >= 1:
                no_small_clusters = True
                small_cluster_index = None
                for i in range(k_init):
                    if sum(assignments == i) < min_elements_per_cluster:
                        no_small_clusters = False
                        small_cluster_index = i
                        break
                if not no_small_clusters:
                    centers = torch.cat((centers[:small_cluster_index], centers[small_cluster_index+1:]))
                    k_init -= 1
                    similarities = torch.mm(data_normalized, torch.nn.functional.normalize(centers, p=2, dim=1).T)
                    assignments = torch.argmax(similarities, dim=1)
                    merged = True
                    dprint('merged_min to:', len(centers))


        # Check for convergence
        if not split and not merged:
            center_shift = torch.mean(torch.sqrt(torch.sum((new_centers - prev_centers) ** 2, dim=1)))
            if center_shift < max_center_shift:
                dprint('converged.')
                break
        if not split and not merged:
            dprint('nothing special.')

    dprint('ISODATA clustering done.')
    return assignments.float(), centers.float()


def find_closest_to_centers(data, centers, assignments):
    dprint('finding original frames closest to center......')
    # Normalize data and centers for cosine similarity
    data_normalized = torch.nn.functional.normalize(data, p=2, dim=1)
    centers_normalized = torch.nn.functional.normalize(centers, p=2, dim=1)

    # Initialize a tensor to store the closest indices
    closest_indices = torch.empty(len(centers), dtype=torch.long, device=data.device)

    for cluster_id in range(len(centers)):
        # Get points belonging to the current cluster
        cluster_points = data_normalized[assignments == cluster_id]
        original_indices = torch.nonzero(assignments == cluster_id).squeeze()

        if len(original_indices.shape) == 0:
            original_indices = torch.unsqueeze(original_indices, dim=-1)

        # Compute similarities with the cluster center
        similarities = torch.mm(cluster_points, centers_normalized[cluster_id].unsqueeze(0).T).squeeze()
        # print(similarities.shape)
        # Find the index of the point with the maximum similarity
        closest_point_idx = torch.argmax(similarities).item()

        # Map back to the original data index
        closest_indices[cluster_id] = original_indices[closest_point_idx]

    dprint('indices of original frames closest to centers returned.')
    # the returned values are not sorted yet
    return closest_indices

class TVS_Datapoint():
    def __init__(self, question, video_fname, videocaptionpath, vid_frame_rate, vid_duration, iframes_times, iframes_feats, vid_1fps_feat):
        self.question = question
        self.video_fname = video_fname
        self.videocaptionpath = videocaptionpath
        self.vid_frame_rate = vid_frame_rate
        self.vid_duration = vid_duration
        self.iframes_times = iframes_times
        self.iframes_feats = iframes_feats
        self.vid_1fps_feat = vid_1fps_feat

class TVS():
    def __init__(self, datapoint:TVS_Datapoint):
        self.current_question = datapoint.question
        self.video_fname = datapoint.video_fname
        self.videocaptionpath = datapoint.videocaptionpath
        self.frame_rate = datapoint.vid_frame_rate
        self.vid_physical_duration = datapoint.vid_duration

        self.video_iframes_times = datapoint.iframes_times
        self.video_iframe_feats = datapoint.iframes_feats
        self.video_1fps_feat = datapoint.vid_1fps_feat