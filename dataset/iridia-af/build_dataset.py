import os
import random
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from ecg_class import ECGEpisode

from utils import (deserialize_list_of_tuples, filter_span, read_beat_df,
                   read_recording, sampling_from_spans, save_episodes,
                   split_span_list)

random.seed(1)
dir_path = os.path.dirname(os.path.realpath(__file__))

def main(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    SPAN_LIST_PATH = os.path.join(cfg["DATA_BUILD_PATH"], "span_list.csv")
    PATIENT_SPLIT_PATH = os.path.join(cfg["DATA_BUILD_PATH"], "patient_split.csv")
    
    span_df = pd.read_csv(SPAN_LIST_PATH)
    patient_df = pd.read_csv(PATIENT_SPLIT_PATH)
    test_df = patient_df[patient_df["split"] == "test"]
    
    test_p_tr, test_p_te = build_dataset_from_df_test(test_df, span_df, cfg)
    save_episodes(cfg, test_p_tr, "test_p_tr")
    save_episodes(cfg, test_p_te, "test_p_te")
    
    episode_stats_df = pd.DataFrame(columns=["patient_dir", "split", "label_0", "label_1", "label_2"])
    
    for split, episodes in OrderedDict({
        "test_p_tr": test_p_tr,
        "test_p_te": test_p_te,
    }).items():
        print(split)
        labels = list(map(lambda x: x.label, episodes))
        values, counts = np.unique(labels, return_counts=True)
        print(dict(zip(values, counts)))
        
        # for episode statistics
        split_stats_dict = {}
        patients = set(map(lambda x: x.patient_dir, episodes))
        for patient in patients:
            patient_episodes = list(filter(lambda x: x.patient_dir == patient, episodes))
            patient_labels = list(map(lambda x: x.label, patient_episodes))
            values, counts = np.unique(patient_labels, return_counts=True)
            num_label_0 = counts[np.where(values==0)[0]]
            if len(num_label_0) == 0:
                num_label_0 = np.array([0])
            num_label_1 = counts[np.where(values==1)[0]]
            if len(num_label_1) == 0:
                num_label_1 = np.array([0])
            num_label_2 = counts[np.where(values==2)[0]]
            if len(num_label_2) == 0:
                num_label_2 = np.array([0])
            split_stats_dict[patient] = [num_label_0.item(), num_label_1.item(), num_label_2.item(), split]
        
        split_stats_df = pd.DataFrame.from_dict(split_stats_dict, orient="index", columns=["label_0", "label_1", "label_2", "split"]).reset_index(names=["patient_dir"])
        episode_stats_df = pd.concat([episode_stats_df, split_stats_df], ignore_index=True)
        
    episode_stats_df = episode_stats_df.sort_values(by=["split", "patient_dir"])
    episode_stats_df.to_csv(f"{cfg['DATA_BUILD_PATH']}/episode_stats.csv", index=False)

def build_dataset_from_df(split_df, span_df, cfg):
    episode_list = []
    
    for idx, row in split_df.iterrows():
        # for each patient
        patient_group_id, patient_id = row["patient_group_id"], row["patient_id"]
        patient_dir = os.path.join(cfg["DATA_BASE_PATH"], patient_group_id, patient_id)
        print(patient_dir)
        selected_df = span_df[(span_df["patient_group_id"] == patient_group_id) & (span_df["patient_id"] == patient_id)]
        try:
            label_0_spans = deserialize_list_of_tuples(selected_df[selected_df["label"] == 0]["spans"].iloc[0])
        except IndexError:
            label_0_spans = []
        try:
            label_1_spans = deserialize_list_of_tuples(selected_df[selected_df["label"] == 1]["spans"].iloc[0])
        except IndexError:
            label_1_spans = []
        try:
            label_2_spans = deserialize_list_of_tuples(selected_df[selected_df["label"] == 2]["spans"].iloc[0])
        except IndexError:
            label_2_spans = []
        
        try:
            beat_df = read_beat_df(cfg, patient_dir)
        except FileNotFoundError: # beat information not available
            beat_df = pd.DataFrame(columns=["BeatClass", "BeatTime", "BeatRow"])
        recording = read_recording(cfg, patient_dir)
        
        rs, ts, ls, _ = sampling_from_spans(label_0_spans, label_1_spans, label_2_spans, recording, cfg)
        for r, t, l in zip(rs, ts, ls):
            episode_beat_df = beat_df[(beat_df.BeatRow >= t[1]) & (beat_df.BeatRow < t[2])]
            episode = ECGEpisode(patient_dir, r, t, l, cfg["SAMPLING_RATE"], episode_beat_df)
            episode.build_windows(cfg['WINDOW_SEC'], cfg['STRIDE_SEC'])
            episode_list.append(episode)
            
    return episode_list

def build_dataset_from_df_test(split_df, span_df, cfg):
    P_TRAINING_TIME = cfg["PERSONALIZED_TRAINING_DAYS"] * 24 * 60 * 60 * cfg["SAMPLING_RATE"]

    p_tr_episode_list = []
    p_te_episode_list = []
    
    for idx, row in split_df.iterrows():
        # for each patient
        patient_group_id, patient_id = row["patient_group_id"], row["patient_id"]
        patient_dir = os.path.join(cfg["DATA_BASE_PATH"], patient_group_id, patient_id)
        print(patient_dir)
        selected_df = span_df[(span_df["patient_group_id"] == patient_group_id) & (span_df["patient_id"] == patient_id)]
        
        label_0_spans = deserialize_list_of_tuples(selected_df[selected_df["label"] == 0]["spans"].iloc[0])
        if len(label_0_spans) == 0:
            print("Warning: no label 0 spans detected")
        p_training_label_0_spans = split_span_list(label_0_spans, P_TRAINING_TIME, True)
        p_test_label_0_spans = split_span_list(label_0_spans, P_TRAINING_TIME, False)
        
        label_1_spans = deserialize_list_of_tuples(selected_df[selected_df["label"] == 1]["spans"].iloc[0])
        p_training_label_1_spans = filter_span(label_1_spans, P_TRAINING_TIME, True)
        p_test_label_1_spans = filter_span(label_1_spans, P_TRAINING_TIME, False)

        label_2_spans = deserialize_list_of_tuples(selected_df[selected_df["label"] == 2]["spans"].iloc[0])
        p_training_label_2_spans = filter_span(label_2_spans, P_TRAINING_TIME, True)
        p_test_label_2_spans = filter_span(label_2_spans, P_TRAINING_TIME, False)

        beat_df = read_beat_df(cfg, patient_dir)
        recording = read_recording(cfg, patient_dir)
        
        tr_rs, tr_ts, tr_ls, _ = sampling_from_spans(p_training_label_0_spans, p_training_label_1_spans, p_training_label_2_spans, recording, cfg)
        for r, t, l in zip(tr_rs, tr_ts, tr_ls):
            episode_beat_df = beat_df[(beat_df.BeatRow >= t[1]) & (beat_df.BeatRow < t[2])]
            episode = ECGEpisode(patient_dir, r, t, l, cfg["SAMPLING_RATE"], episode_beat_df)
            episode.build_windows(cfg['WINDOW_SEC'], cfg['STRIDE_SEC'])
            p_tr_episode_list.append(episode)
        
        te_rs, te_ts, te_ls, _ = sampling_from_spans(p_test_label_0_spans, p_test_label_1_spans, p_test_label_2_spans, recording, cfg)
        for r, t, l in zip(te_rs, te_ts, te_ls):
            episode_beat_df = beat_df[(beat_df.BeatRow >= t[1]) & (beat_df.BeatRow < t[2])]
            episode = ECGEpisode(patient_dir, r, t, l, cfg["SAMPLING_RATE"], episode_beat_df)
            episode.build_windows(cfg['WINDOW_SEC'], cfg['STRIDE_SEC'])
            p_te_episode_list.append(episode)

    return p_tr_episode_list, p_te_episode_list

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Needs one argument: configuration name (e.g. python select_af_samples.py config)")
        sys.exit()
    main(sys.argv[1])