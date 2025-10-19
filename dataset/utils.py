import math
import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from ecg_class import preprocess_recording

SEED = 42

random.seed(SEED)

def is_valid_span(wider, narrower):
    if narrower is None:
        return False
    start1, end1 = wider
    start2, end2 = narrower
    
    # Check if episode is completely within span
    return start2 < end2 and start1 <= start2 and end2 <= end1


def find_non_overlapping_spans(spans, start, end):
    # Sort the input spans by their start values
    spans.sort(key=lambda x: x[0])
    
    non_overlapping_spans = []
    current_start = start
    
    for span in spans:
        span_start, span_end = span
        # convert to integer values
        span_start = math.floor(span_start)
        span_end = math.ceil(span_end)
        
        # If the current start is less than the span's start, add a non-overlapping span
        if current_start < span_start:
            non_overlapping_spans.append((current_start, span_start)) # [inclusive, exclusive)
        
        # Update the current start to be one past the current span's end
        current_start = span_end + 1
    
    # If there are remaining spans beyond the given 'end', add a final non-overlapping span
    if current_start <= end:
        non_overlapping_spans.append((current_start, end))
    
    return non_overlapping_spans

def serialize_list_of_tuples(input):
    res = ''
    for span_idx, start, end in input:
        res = res + str(span_idx) + "@" + str(start) + "#" + str(end) + "|"
    res = res[:-1] # remove last "|"
    return res

def deserialize_list_of_tuples(input):
    res = []
    if not pd.isna(input):
        for line in input.split("|"):
            span_idx, span_str = line.split("@")
            start, end = span_str.split("#")
            res.append((int(span_idx), int(start), int(end)))
    return res

# https://www.geeksforgeeks.org/merging-intervals/
def merge_af_intervals(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(by=['start_point'])
    df = df.reset_index(drop=True)
    index = 0
    for i in range(1, len(df)):
        if df.loc[index, 'end_point'] >= df.loc[i, 'start_point']:
            df.loc[index, 'end_point'] = max(df.loc[index, 'end_point'], df.loc[i, 'end_point'])
        else:
            index += 1
            df.loc[index] = df.loc[i]
    df = df.truncate(after=index)
    
    return df
    
def read_rhythm_df(cfg, patient_dir):
    if cfg["DATASET"] == "MobiCARE":
        rhythm_df = pd.read_csv(os.path.join(patient_dir, 'updatedRhythmClass.csv'), header=0)
        rhythm_df['startT'] = rhythm_df['startT'].apply(lambda x: x * cfg["SAMPLING_RATE"])
        rhythm_df['endT'] = rhythm_df['endT'].apply(lambda x: x * cfg["SAMPLING_RATE"])
        rhythm_df.columns = ["start_point", "end_point", "Label"]
    elif cfg["DATASET"] == "IRIDIA-AF":
        rhythm_df = pd.read_csv(os.path.join(patient_dir, f'rhythmClassP{cfg["DATA_STRING"]}.csv'), header=0)
    else:
        raise ValueError
    return rhythm_df

def read_beat_df(cfg, patient_dir):
    if cfg["DATASET"] == "MobiCARE":
        beat_df = pd.read_csv(os.path.join(patient_dir, 'updatedBeatClass.csv'), header=0)
    elif cfg["DATASET"] == "IRIDIA-AF":
        beat_df = pd.DataFrame(columns=["BeatClass", "BeatTime", "BeatRow"])
    else:
        raise ValueError
    return beat_df

def read_sample_beat_df(beat_df, sample_start_point, sample_end_point): # MobiCARE only
    if (beat_df is not None) and (len(beat_df) != 0):
        sample_df = beat_df[(beat_df.BeatRow>=sample_start_point) & (beat_df.BeatRow<=sample_end_point)]
    else:
        sample_df = None
    return sample_df


def read_sample_rhythm_df(rhythm_df, sample_start_point, sample_end_point):
    if (rhythm_df is not None) and (len(rhythm_df) != 0):
        sample_df = rhythm_df[(rhythm_df.start_point>=sample_start_point) & (rhythm_df.end_point<=sample_end_point)]
    else:
        sample_df = None
    return sample_df


def read_recording(cfg, patient_dir: str) -> np.ndarray:
    if cfg["DATASET"] == "MobiCARE":
        recording = np.load(os.path.join(patient_dir, "ecgSignal.npy"))
    elif cfg["DATASET"] == "IRIDIA-AF":
        recording = np.load(os.path.join(patient_dir, f"ecgSignal{cfg['DATA_STRING']}.npy"))
    else:
        raise ValueError
    return np.squeeze(recording)

def filter_span(x: List, time_criterion: int, before: bool = True) -> List:
    if before:
        return [item for item in x if item[1] < time_criterion] # if borderline -> training
    else:
        return [item for item in x if item[1] >= time_criterion]

def split_span_list(span_list: List[tuple], time_criterion: int, before: bool = True):
    if before:
        res = [t for t in span_list if t[1] < time_criterion]
        if len(res) == 0:
            return []
        last = res.pop()
        if last[2] > time_criterion:
            last = (last[0], last[1], time_criterion)
        res.append(last)
    else:
        res = [t for t in span_list if t[2] > time_criterion]
        if len(res) == 0:
            return []
        first = res.pop(0)
        if first[1] < time_criterion:
            first = (first[0], time_criterion, first[2])
        res.insert(0, first)
    return res

def save_episodes(cfg, episodes, split):
    serialized = pickle.dumps(episodes, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{cfg['DATA_BUILD_PATH']}/{split}.pickle", 'wb') as fp:
        fp.write(serialized)

def has_flat_segments(recording: np.ndarray) -> bool:
    flat_segments_length = find_flat_segments_length(recording, int(recording.shape[0]/300), threshold=0) # episode: 300 seconds
    if flat_segments_length >= int(recording.shape[0] / 10): # noise
        return True
    else:
        return False

# https://stackoverflow.com/questions/78154458/how-to-remove-constant-part-of-a-signal-in-python
def find_flat_segments_length(data, min_length, threshold=0):
    """
    Find flat segments from data within a threshold and with a minimum length.

    Parameters:
    data (ndarray): Input data array.
    threshold (float): Threshold for flatness.
    min_length (int): Minimum length of flat segment to be found.

    Returns:
    ndarray: flat segments
    """
    flat_segments = []
    start_idx = None
    
    for i in range(len(data)):
        if start_idx is None:
            start_idx = i
        elif abs(data[i] - data[i-1]) > threshold:
            if i - start_idx >= min_length:
                flat_segments.append((start_idx, i-1))
            start_idx = None
    
    if start_idx is not None and len(data) - start_idx >= min_length:
        flat_segments.append((start_idx, len(data)-1))
    
    if len(flat_segments) > 0:
        flat_indices = np.concatenate([np.arange(start, end+1) for start, end in flat_segments])
        return flat_indices.shape[0]
    else:
        return 0
    
def is_noisy_episode(recording: np.ndarray, fs: int) -> bool: # is_low_signal_episode or is_high_voltage_episode
    preprocessed_recording = preprocess_recording(recording, fs)
    abs_recording = np.absolute(preprocessed_recording)
    if np.any(abs_recording > 5) or np.all(abs_recording < 0.03):
        return True
    else:
        return False

def is_low_signal_episode(recording: np.ndarray, fs: int) -> bool:
    preprocessed_recording = preprocess_recording(recording, fs)
    abs_recording = np.absolute(preprocessed_recording)
    if np.all(abs_recording < 0.03):
        return True
    else:
        return False

def is_high_voltage_episode(recording: np.ndarray, fs: int) -> bool:
    preprocessed_recording = preprocess_recording(recording, fs)
    abs_recording = np.absolute(preprocessed_recording)
    if np.any(abs_recording > 5):
        return True
    else:
        return False

def sampling_from_spans(label_0_spans: List[Tuple[int, int]],
                        label_1_spans: List[Tuple[int, int]],
                        label_2_spans: List[Tuple[int, int]],
                        recording, cfg) -> Tuple[List, List, List]:
    episode_length = cfg["EPISODE_LENGTH_SEC"] * cfg["SAMPLING_RATE"]
    random.shuffle(label_0_spans)
    random.shuffle(label_1_spans)
    random.shuffle(label_2_spans)
    label_list = []
    
    # label 2 episodes: sort label_2_spans by the length of span
    label_2_spans = sorted(label_2_spans, key=lambda x: x[2] - x[1], reverse=True)
    label_2_span_ids = list(map(lambda x: x[0], label_2_spans))
    label_2 = []
    for label_2_span in label_2_spans:
        span_id, span_start, span_end = label_2_span
        
        if cfg["DATASET"] == "ICENTIA11K":
            segment_id, _ = span_id.split("_")
            span_recording = recording[segment_id][span_start:span_end]
        else:
            span_recording = recording[span_start:span_end]
            
        if has_flat_segments(span_recording) or is_low_signal_episode(span_recording, cfg["SAMPLING_RATE"]): # filtering out noise recordings
            continue
        else:
            label_2.append((span_recording, label_2_span, span_id)) # r, t, i
        if len(label_2) >= cfg["MAX_NUM_LABEL_1"]:
            break
    
    # label 1 episodes: assign priority to spans which have long label 2 episodes
    label_1_spans_w_label_2 = []
    label_2_span_ids = list(map(lambda x: x[2], label_2))
    for span_id in label_2_span_ids:
        target_span = list(filter(lambda x: x[0] == span_id, label_1_spans))
        if len(target_span) == 1:
            label_1_spans_w_label_2.append(target_span[0])
    label_1_spans_wo_label_2 = list(filter(lambda x: not (x[0] in label_2_span_ids), label_1_spans))
    label_1_spans = label_1_spans_w_label_2 + label_1_spans_wo_label_2
        
    label_1 = []
    label_1_noise = []
    for label_1_span in label_1_spans:
        span_id, span_start, span_end = label_1_span
        
        if cfg["DATASET"] == "ICENTIA11K":
            segment_id, _ = span_id.split("_")
            span_recording = recording[segment_id][span_start:span_end]
        else:
            span_recording = recording[span_start:span_end]
            
        if has_flat_segments(span_recording) or is_low_signal_episode(span_recording, cfg["SAMPLING_RATE"]): # filtering out noise recordings
            continue
        else:
            if is_high_voltage_episode(span_recording, cfg["SAMPLING_RATE"]):
                label_1_noise.append((span_recording, label_1_span, span_id))
            else:
                label_1.append((span_recording, label_1_span, span_id))
        if len(label_1) >= cfg["MAX_NUM_LABEL_1"]:
            break
    if len(label_1) < cfg["MAX_NUM_LABEL_1"]:
        label_1.extend(label_1_noise[:(cfg["MAX_NUM_LABEL_1"] - len(label_1))])
    num_label_1 = len(label_1)
    
    target_num_label_2 = num_label_1 * cfg["LABEL_2_FACTOR"]
    label_2 = label_2[:target_num_label_2]
    num_label_2 = len(label_2)
    
    # label 0 episodes
    target_num_label_0 = max(num_label_1 * cfg["LABEL_0_FACTOR"], cfg["MIN_NUM_LABEL_0"])
    label_0_time_info_candidates = []
    for label_0_span in label_0_spans:
        span_id, span_start, span_end = label_0_span
        sample_points = list(range(span_start, span_end+1, episode_length))
        for sample_point in sample_points[:-1]:
            label_0_time_info_candidates.append((span_id, sample_point, sample_point + episode_length))
    
    random.shuffle(label_0_time_info_candidates)
    label_0 = []
    label_0_noise = []
    for label_0_span in label_0_time_info_candidates:
        span_id, span_start, span_end = label_0_span
        
        if cfg["DATASET"] == "ICENTIA11K":
            segment_id, _ = span_id.split("_")
            span_recording = recording[segment_id][span_start:span_end]
        else:
            span_recording = recording[span_start:span_end]
            
        if has_flat_segments(span_recording) or is_low_signal_episode(span_recording, cfg["SAMPLING_RATE"]): # filtering out noise recordings
            continue
        else:
            if is_high_voltage_episode(span_recording, cfg["SAMPLING_RATE"]):
                label_0_noise.append((span_recording, label_0_span, span_id))
            else:
                label_0.append((span_recording, label_0_span, span_id))
        if len(label_0) >= target_num_label_0:
            break
    if len(label_0) < target_num_label_0:
        label_0.extend(label_0_noise[:(target_num_label_0 - len(label_0))])
    num_label_0 = len(label_0)
    
    recording_list = list(map(lambda x: x[0], label_0 + label_1 + label_2))
    time_info_list = list(map(lambda x: x[1], label_0 + label_1 + label_2))
    label_list = [0 for _ in range(num_label_0)] + [1 for _ in range(num_label_1)] + [2 for _ in range(num_label_2)]
    id_list = list(map(lambda x: x[2], label_0 + label_1 + label_2)) # only used for ICENTIA11K
    
    return recording_list, time_info_list, label_list, id_list