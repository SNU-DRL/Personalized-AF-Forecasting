import os
import pickle
import random
import sys
from typing import List
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dataset import ecg_class
from src.utils import preprocess_recording, is_low_signal_window
from src.augmentation import transform

sys.modules['ecg_class'] = ecg_class
RR_INTERVAL_WIDTH = 150

class PAFDataModule:
    def __init__(
        self,
        train_episodes: List=None,
        eval_episodes: List=None,
        sampling_rate: int=250,
        sampler: bool=False,
        include_af_episodes: bool=False,
        use_standardization: bool=True,
        use_augmentation: bool=False,
        mean_dict: dict=None,
    ):
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        
        self.sampler = sampler
        self.train_set, self.eval_set = None, None
        self.mean_dict = mean_dict
        
        if train_episodes is not None:
            self.train_windows = []
            for train_episode in train_episodes:
                if train_episode.label == 2:
                    if include_af_episodes:
                        replace_episode_label(train_episode, 2, 1)
                    else:
                        continue
                for window in train_episode.windows.values():
                    if not is_low_signal_window(window.recording, sampling_rate): # filter out low signal windows
                        self.train_windows.append(window)
            if use_augmentation:
                self.train_set = PAFWindowDataset(self.train_windows, sampling_rate, shuffle=True, use_standardization=use_standardization, features_mean_dict=self.mean_dict, transform=transform(0.3))
            else:
                self.train_set = PAFWindowDataset(self.train_windows, sampling_rate, shuffle=True, use_standardization=use_standardization, features_mean_dict=self.mean_dict)
            if self.mean_dict is None:
                self.mean_dict = self.train_set.features_mean_dict
            print(f"Loaded dataset for training: {len(self.train_set)}")
        
        if eval_episodes is not None:
            self.eval_windows = []
            for eval_episode in eval_episodes:
                if eval_episode.label == 2:
                    continue
                self.eval_windows += list(eval_episode.windows.values())
            self.eval_set = PAFWindowDataset(self.eval_windows, sampling_rate, shuffle=False, use_standardization=use_standardization, features_mean_dict=self.mean_dict)
            print(f"Loaded dataset for evaluation: {len(self.eval_set)}")

    @classmethod
    def from_datapath(
        cls,
        train_data_path: str=None,
        eval_data_path: str=None,
        sampling_rate: int=250,
        sampler: bool=False,
        include_af_episodes: bool=False,
        use_standardization: bool=True,
        use_augmentation: bool=False,
        mean_dict: dict=None,
    ):
        train_episodes = None
        eval_episodes = None
        
        if train_data_path is not None:
            with open(train_data_path, "rb") as fp:
                serialized = fp.read()
            train_episodes = pickle.loads(serialized)
        
        if eval_data_path is not None:
            with open(eval_data_path, "rb") as fp:
                serialized = fp.read()
            eval_episodes = pickle.loads(serialized)
        
        return cls(train_episodes, eval_episodes, sampling_rate, sampler, include_af_episodes, use_standardization, use_augmentation, mean_dict)
    
    def train_dataloader(self, batch_size=32):
        if self.train_set is None:
            return None
        if self.sampler:
            class_sample_count = np.array(
                [len(np.where(self.train_set.labels == t)[0]) for t in np.unique(self.train_set.labels)]
            )
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in self.train_set.labels])

            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            sampler = None

        return DataLoader(self.train_set, pin_memory=True, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)
    
    def eval_dataloader(self, batch_size=1):
        if self.eval_set is None:
            return None
        return DataLoader(self.eval_set, pin_memory=True, batch_size=batch_size, shuffle=False)

class PAFWindowDataset(Dataset): # unit: window
    def __init__(self, windows: List, sampling_rate: int, shuffle: bool=True, use_standardization: bool=True, features_mean_dict: dict=None, transform=lambda x: x):
        self.windows = windows
        self.transform = transform
        
        # window info
        self.patient_dirs = []
        self.episode_strs = []
        self.window_strs = []
        
        # window data
        self.labels = []
        self.recordings = None
        self.features_dict = { # raw features
            "heart_rate": [], "rmssd": [], "rr_intervals": [], "pac_burden": []
        }
        self.p_features_dict = { # preprocessed features / used for training and evaluation
            "heart_rate": [], "rmssd": [], "rr_intervals": [], "pac_burden": []
        }
        
        self.features_mean_dict = {
            "heart_rate": None, "rmssd": None, "rr_intervals": None, "pac_burden": None
        }
        
        if shuffle:
            random.shuffle(self.windows)
                
        recording_list = []
        for window in self.windows:
            self.patient_dirs.append(window.patient_dir)
            self.episode_strs.append(window.episode_str)            
            self.window_strs.append(f"{window.episode_str}|{window.window_str}")
            
            self.labels.append(window.label)
            recording_list.append(window.recording)
            for feature in self.features_dict.keys():
                self.features_dict[feature].append(window.features[feature])
        
        if features_mean_dict is None:
            self.calculate_mean_dict()
        else:
            self.features_mean_dict = features_mean_dict
        
        self.preprocess_features()
        
        recordings = preprocess_recording(np.array(recording_list), fs=sampling_rate, standardize=use_standardization)
        recordings = torch.tensor(recordings, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.recordings = recordings
        
    def __len__(self):
        return len(self.window_strs)

    def __getitem__(self, idx):
        return self.window_strs[idx], \
                self.transform(self.recordings[idx]), \
                self.labels[idx], \
                torch.tensor([self.p_features_dict["heart_rate"][idx], self.p_features_dict["rmssd"][idx], self.p_features_dict["pac_burden"][idx]], dtype=torch.float32), \
                torch.tensor(self.p_features_dict["rr_intervals"][idx], dtype=torch.float32), \

    def iter_episodes(self):
        pass
    
    def calculate_mean_dict(self):
        for feature in self.features_dict.keys():
            if feature == "rr_intervals":
                self.features_mean_dict[feature] = np.nanmean(np.concatenate(self.features_dict[feature]))
            else:
                self.features_mean_dict[feature] = np.nanmean(self.features_dict[feature])
    
    def preprocess_features(self):
        for feature in self.features_dict.keys():
            if feature == "rr_intervals":
                rr_intervals = self.features_dict[feature]
                for rr_interval in rr_intervals:
                    rr_interval = (rr_interval - self.features_mean_dict[feature]) / 100
                    if len(rr_interval) > RR_INTERVAL_WIDTH:
                        rr_interval = rr_interval[-RR_INTERVAL_WIDTH:]
                    pad_width = RR_INTERVAL_WIDTH - len(rr_interval)
                    padded_rr_interval = np.pad(rr_interval, (pad_width, 0))
                    self.p_features_dict["rr_intervals"].append(padded_rr_interval)
            elif feature in ["heart_rate", "rmssd"]:
                for v in self.features_dict[feature]:
                    if math.isnan(v):
                        self.p_features_dict[feature].append(0)
                    else:
                        self.p_features_dict[feature].append((v - self.features_mean_dict[feature]) / 100)
            else: # pac burden
                for v in self.features_dict[feature]:
                    if math.isnan(v):
                        self.p_features_dict[feature].append(0)
                    else:
                        self.p_features_dict[feature].append(v)

def replace_episode_label(episode, src_label, dst_label):
    if episode.label == src_label:
        episode.label = dst_label
        for key, window in episode.windows.items():
            window.label = dst_label
            window.window_str = "b_af"