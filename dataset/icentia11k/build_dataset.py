import os
import random
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import wfdb
import yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from ecg_class import ECGEpisode

from utils import (deserialize_list_of_tuples, sampling_from_spans,
                   save_episodes)

SEED = 42

random.seed(SEED)
dir_path = os.path.dirname(os.path.realpath(__file__))

def main(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    SPAN_LIST_PATH = os.path.join(cfg["DATA_BUILD_PATH"], "span_list.csv")
    PATIENT_SPLIT_PATH = os.path.join(cfg["DATA_BUILD_PATH"], "patient_split.csv")
    
    span_df = pd.read_csv(SPAN_LIST_PATH)
    patient_df = pd.read_csv(PATIENT_SPLIT_PATH)
    
    training_df = patient_df[patient_df["split"] == "training"]
    validation_df = patient_df[patient_df["split"] == "validation"]
    test_df = patient_df[patient_df["split"] == "test"]
    
    training = build_dataset_from_df(training_df, span_df, cfg)
    save_episodes(cfg, training, "training")
    validation = build_dataset_from_df(validation_df, span_df, cfg)
    save_episodes(cfg, validation, "validation")
    test_p_tr, test_p_te = build_dataset_from_df_test(test_df, span_df, cfg)
    save_episodes(cfg, test_p_tr, "test_p_tr")
    save_episodes(cfg, test_p_te, "test_p_te")
    
    episode_stats_df = pd.DataFrame(columns=["patient_dir", "split", "label_0", "label_1", "label_2"])
    
    for split, episodes in OrderedDict({
        "training": training,
        "validation": validation,
        "test_p_tr": test_p_tr,
        "test_p_te": test_p_te,
    }).items():
        print(split)
        labels = list(map(lambda x: x.label, episodes))
        values, counts = np.unique(labels, return_counts=True)
        print(dict(zip(values, counts)))
        
        # for episode statistics
        split_stats_dict = {}
        patients = set(map(lambda x: os.path.split(x.patient_dir)[0], episodes))
        for patient in patients:
            patient_episodes = list(filter(lambda x: x.patient_dir.startswith(patient), episodes))
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
        patient_df = span_df[(span_df["patient_group_id"] == patient_group_id) & (span_df["patient_id"] == patient_id)]
        
        label_0_spans = []
        label_1_spans = []
        label_2_spans = []
        recording_dict = {}

        for segment_id in set(patient_df["segment_id"].values):            
            # checks if recording is broken
            recording, is_broken = read_recording(cfg, patient_group_id, patient_id, segment_id)
            if is_broken:
                continue
            
            recording_dict[segment_id] = recording

            try:
                s_label_0_spans = deserialize_list_of_tuples(patient_df[(patient_df["segment_id"] == segment_id) & (patient_df["label"] == 0)]["spans"].iloc[0])
            except IndexError:
                s_label_0_spans = []
            try:
                s_label_1_spans = deserialize_list_of_tuples(patient_df[(patient_df["segment_id"] == segment_id) & (patient_df["label"] == 1)]["spans"].iloc[0])
            except IndexError:
                s_label_1_spans = []
            try:
                s_label_2_spans = deserialize_list_of_tuples(patient_df[(patient_df["segment_id"] == segment_id) & (patient_df["label"] == 2)]["spans"].iloc[0])
            except IndexError:
                s_label_2_spans = []
            
            label_0_spans.extend(build_span_id(segment_id, s_label_0_spans))
            label_1_spans.extend(build_span_id(segment_id, s_label_1_spans))
            label_2_spans.extend(build_span_id(segment_id, s_label_2_spans))

        rs, ts, ls, ids = sampling_from_spans(label_0_spans, label_1_spans, label_2_spans, recording_dict, cfg)

        for r, t, l, i in zip(rs, ts, ls, ids):
            segment_id, _ = i.split("_")
            segment_dir = os.path.join(patient_dir, f"{patient_id}_{segment_id}")

            try:
                beat_df, rhythm_df = read_beat_rhythm(cfg, patient_group_id, patient_id, segment_id)
            except FileNotFoundError: # beat information not available
                beat_df, rhythm_df = pd.DataFrame(columns=["BeatClass", "BeatTime", "BeatRow"]), None

            episode_beat_df = beat_df[(beat_df.BeatRow >= t[1]) & (beat_df.BeatRow < t[2])]
            episode = ECGEpisode(segment_dir, r, t, l, cfg["SAMPLING_RATE"], episode_beat_df)
            episode.build_windows(cfg['WINDOW_SEC'], cfg['STRIDE_SEC'])
            episode_list.append(episode)
    
    return episode_list

def build_dataset_from_df_test(split_df, span_df, cfg):
    PERSONALIZED_TRAINING_SEGMENTS = cfg["PERSONALIZED_TRAINING_SEGMENTS"]

    p_tr_episode_list = []
    p_te_episode_list = []
    
    for idx, row in split_df.iterrows():
        # for each patient
        patient_group_id, patient_id = row["patient_group_id"], row["patient_id"]
        patient_dir = os.path.join(cfg["DATA_BASE_PATH"], patient_group_id, patient_id)
        print(patient_dir)
        patient_df = span_df[(span_df["patient_group_id"] == patient_group_id) & (span_df["patient_id"] == patient_id)]
        
        p_tr = {"label_0_spans": [], "label_1_spans": [], "label_2_spans": []}
        p_te = {"label_0_spans": [], "label_1_spans": [], "label_2_spans": []}
        recording_dict = {}

        for segment_id in set(patient_df["segment_id"].values):            
            recording, is_broken = read_recording(cfg, patient_group_id, patient_id, segment_id)
            if is_broken:
                continue
            
            recording_dict[segment_id] = recording

            try:
                label_0_spans = deserialize_list_of_tuples(patient_df[(patient_df["segment_id"] == segment_id) & (patient_df["label"] == 0)]["spans"].iloc[0])
            except IndexError:
                label_0_spans = []
            try:
                label_1_spans = deserialize_list_of_tuples(patient_df[(patient_df["segment_id"] == segment_id) & (patient_df["label"] == 1)]["spans"].iloc[0])
            except IndexError:
                label_1_spans = []
            try:
                label_2_spans = deserialize_list_of_tuples(patient_df[(patient_df["segment_id"] == segment_id) & (patient_df["label"] == 2)]["spans"].iloc[0])
            except IndexError:
                label_2_spans = []

            if segment_id < f"s{PERSONALIZED_TRAINING_SEGMENTS}":
                target = p_tr
            else:
                target = p_te
            
            target["label_0_spans"].extend(build_span_id(segment_id, label_0_spans))
            target["label_1_spans"].extend(build_span_id(segment_id, label_1_spans))
            target["label_2_spans"].extend(build_span_id(segment_id, label_2_spans))

        tr_rs, tr_ts, tr_ls, tr_ids = sampling_from_spans(p_tr["label_0_spans"], p_tr["label_1_spans"], p_tr["label_2_spans"], recording_dict, cfg)
        
        for r, t, l, i in zip(tr_rs, tr_ts, tr_ls, tr_ids):
            segment_id, _ = i.split("_")
            segment_dir = os.path.join(patient_dir, f"{patient_id}_{segment_id}")

            try:
                beat_df, rhythm_df = read_beat_rhythm(cfg, patient_group_id, patient_id, segment_id)
            except FileNotFoundError: # beat information not available
                beat_df, rhythm_df = pd.DataFrame(columns=["BeatClass", "BeatTime", "BeatRow"]), None
            
            episode_beat_df = beat_df[(beat_df.BeatRow >= t[1]) & (beat_df.BeatRow < t[2])]
            episode = ECGEpisode(segment_dir, r, t, l, cfg["SAMPLING_RATE"], episode_beat_df)
            episode.build_windows(cfg['WINDOW_SEC'], cfg['STRIDE_SEC'])
            p_tr_episode_list.append(episode)
        
        te_rs, te_ts, te_ls, te_ids = sampling_from_spans(p_te["label_0_spans"], p_te["label_1_spans"], p_te["label_2_spans"], recording_dict, cfg)
        for r, t, l, i in zip(te_rs, te_ts, te_ls, te_ids):
            segment_id, _ = i.split("_")
            segment_dir = os.path.join(patient_dir, f"{patient_id}_{segment_id}")

            try:
                beat_df, rhythm_df = read_beat_rhythm(cfg, patient_group_id, patient_id, segment_id)
            except FileNotFoundError: # beat information not available
                beat_df, rhythm_df = pd.DataFrame(columns=["BeatClass", "BeatTime", "BeatRow"]), None

            episode_beat_df = beat_df[(beat_df.BeatRow >= t[1]) & (beat_df.BeatRow < t[2])]
            episode = ECGEpisode(segment_dir, r, t, l, cfg["SAMPLING_RATE"], episode_beat_df)
            episode.build_windows(cfg['WINDOW_SEC'], cfg['STRIDE_SEC'])
            p_te_episode_list.append(episode)

    return p_tr_episode_list, p_te_episode_list

def read_beat_rhythm(cfg, patient_group_id, patient_id, segment_id):
    record_path = os.path.join(cfg["DATA_BASE_PATH"], patient_group_id, patient_id, f"{patient_id}_{segment_id}")
    
    try:
        ann = wfdb.rdann(record_path,  "atr")
        beat_df = pd.DataFrame([ann.symbol, ann.sample]).T
        beat_df.columns = ["BeatClass", "BeatRow"]
        beat_df = beat_df[beat_df["BeatClass"] != "+"]
        beat_df['BeatTime'] = beat_df.BeatRow / cfg["SAMPLING_RATE"]
        
        rhythm_episodes = []
        rhythm_ano_mask = (np.array(ann.aux_note)!='None')
        sample_with_rhythm_ano = ann.sample[rhythm_ano_mask]
        aux_note_with_rhythm_ano = np.array(ann.aux_note)[rhythm_ano_mask]
        for beat, ann in (it := zip(sample_with_rhythm_ano, aux_note_with_rhythm_ano)):
            if ann[0] == "(":
                if ann[-1] != ")":
                    next_beat, next_ann = next(it)
                    assert next_ann == ")"
                    rhythm_episodes.append((beat/cfg["SAMPLING_RATE"], next_beat/cfg["SAMPLING_RATE"], ann[1:]))
                else:
                    rhythm_episodes.append((beat/cfg["SAMPLING_RATE"], beat/cfg["SAMPLING_RATE"], ann[1:-1]))
                    
        rhythm_df = pd.DataFrame(rhythm_episodes, columns = ["startT", "endT", "Label"])
        
    except FileNotFoundError as e:
        beat_df = pd.DataFrame(columns=["BeatClass", "BeatTime", "BeatRow"])
        rhythm_df = pd.DataFrame(columns=["startT", "endT", "Label"])

    return beat_df, rhythm_df    

def read_recording(cfg, patient_group_id, patient_id, segment_id):
    record_path = os.path.join(cfg["DATA_BASE_PATH"], patient_group_id, patient_id, f"{patient_id}_{segment_id}")
    is_broken = False
    
    try:
        recordig_raw = wfdb.rdrecord(record_path)
        recording = recordig_raw.p_signal
        if len(recording) <= 1:
            # broken data
            raise FileNotFoundError
        recording = recording.squeeze()
    
    except FileNotFoundError as e:
        print(f"file not found : {record_path}")
        recording = None
        is_broken = True
        
    return recording, is_broken

def build_span_id(segment_id, span_list):
    return [(f"{segment_id}_{span_idx}", start, end) for span_idx, start, end in span_list]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Needs one argument: configuration name (e.g. python select_af_samples.py config)")
        sys.exit()
    main(sys.argv[1])