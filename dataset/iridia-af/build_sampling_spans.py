"""
    Build indices for spans that satisfies specific condition.
"""

import csv
import os
import sys

import pandas as pd
import yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils import (find_non_overlapping_spans, is_valid_span,
                   merge_af_intervals, read_rhythm_df,
                   serialize_list_of_tuples)

# label 0
LABEL0_SELECTION = {
    "start": "(span_start + cfg['NON_AF_DISTANCE_SEC'] * sampling_rate, span_end - cfg['NON_AF_DISTANCE_SEC'] * sampling_rate)",
    "between": "(span_start + cfg['NON_AF_DISTANCE_SEC'] * sampling_rate, span_end - cfg['NON_AF_DISTANCE_SEC'] * sampling_rate)",
    "end": "(span_start + cfg['NON_AF_DISTANCE_SEC'] * sampling_rate, span_end - cfg['NON_AF_DISTANCE_SEC'] * sampling_rate)",
}

# label 1
LABEL1_SELECTION = {
    "start": "(span_end - (cfg['AF_DISTANCE_SEC'] + cfg['EPISODE_LENGTH_SEC']) * sampling_rate, span_end - cfg['AF_DISTANCE_SEC'] * sampling_rate)", # cfg['AF_DISTANCE_SEC']-second gap for pre-AF episodes
    "between": "(span_end - (cfg['AF_DISTANCE_SEC'] + cfg['EPISODE_LENGTH_SEC']) * sampling_rate, span_end - cfg['AF_DISTANCE_SEC'] * sampling_rate)",
    "end": "None",
}

# label 2
AF_SELECTION = {
    "start": "(span_end - (cfg['AF_DISTANCE_SEC'] + cfg['WINDOW_SEC'] - cfg['STRIDE_SEC']) * sampling_rate, min(span_end - (cfg['AF_DISTANCE_SEC'] + cfg['WINDOW_SEC'] - cfg['STRIDE_SEC'] - cfg['EPISODE_LENGTH_SEC']) * sampling_rate, next_span_start))",
    "between": "(span_end - (cfg['AF_DISTANCE_SEC'] + cfg['WINDOW_SEC'] - cfg['STRIDE_SEC']) * sampling_rate, min(span_end - (cfg['AF_DISTANCE_SEC'] + cfg['WINDOW_SEC'] - cfg['STRIDE_SEC'] - cfg['EPISODE_LENGTH_SEC']) * sampling_rate, next_span_start))",
    "end": "None"
}

dir_path = os.path.dirname(os.path.realpath(__file__))

def main(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        
    AF_BURDEN_LIST_PATH = os.path.join(cfg["DATA_BUILD_PATH"], "af_burden_list.csv")
    SPAN_LIST_PATH = os.path.join(cfg["DATA_BUILD_PATH"], "span_list.csv")
        
    f = open(SPAN_LIST_PATH, "w")
    writer = csv.writer(f)
    writer.writerow(["patient_group_id", "patient_id", "sample_length", "spans", "label"])
    sampling_rate = cfg["SAMPLING_RATE"]
    
    af_burden_df = pd.read_csv(AF_BURDEN_LIST_PATH)

    for idx, row in af_burden_df.iterrows():
        patient_group_id, patient_id, af_burden = row["patient_group_id"], row["patient_id"], row["af_burden"]
        
        if af_burden == 0 or af_burden > cfg["AF_BURDEN_THRESHOLD"]:
            # skip this sample
            continue
        
        patient_dir = os.path.join(cfg["DATA_BASE_PATH"], patient_group_id, patient_id)
        print(f"Processing... {patient_dir}")
        
        with open(os.path.join(patient_dir, f"sample_length{cfg['DATA_STRING']}.txt"), "r") as g:
            len_recording = int(g.readline()) # count number of datapoints
        
        rhythm_df = read_rhythm_df(cfg, patient_dir)

        af_rhythm_df = rhythm_df[rhythm_df.Label == 'AFIB']
        af_rhythm_df = merge_af_intervals(af_rhythm_df)
        af_start_points = af_rhythm_df['start_point'].values
        af_end_points = af_rhythm_df['end_point'].values
        
        af_episodes = list(zip(af_start_points, af_end_points))
        outside_af_episodes = find_non_overlapping_spans(af_episodes, 0, len_recording)
        
        target_spans_af = []
        for label, SELECTION in enumerate([LABEL0_SELECTION, LABEL1_SELECTION]):
            target_spans = []
            for span_idx, span in enumerate(outside_af_episodes):
                span_start, span_end = span
                if span_idx == (len(outside_af_episodes) - 1): # last span
                    next_span_start = len_recording
                else:
                    next_span_start = outside_af_episodes[span_idx+1][0]
                
                valid_span_min = span_start
                valid_span_max = next_span_start

                if span_start == 0: # start
                    target_span = eval(SELECTION["start"])
                elif span_end == len_recording: # end
                    target_span = eval(SELECTION["end"])
                else: # in between
                    target_span = eval(SELECTION["between"])
                
                len_target_span = target_span[1] - target_span[0] if (target_span is not None) else 0
                
                if is_valid_span((valid_span_min, valid_span_max), target_span) and (len_target_span >= cfg["WINDOW_SEC"] * sampling_rate):
                    target_spans.append((span_idx, *target_span))
                    if label == 1:
                        # Valid label 1 episode -> process AF selection
                        if span_start == 0: # start
                            target_span = eval(AF_SELECTION["start"])
                        elif span_end == len_recording: # end
                            target_span = eval(AF_SELECTION["end"])
                        else: # in between
                            target_span = eval(AF_SELECTION["between"])
                        
                        len_target_span = target_span[1] - target_span[0] if (target_span is not None) else 0

                        if is_valid_span((valid_span_min, valid_span_max), target_span) and (len_target_span >= cfg["WINDOW_SEC"] * sampling_rate):
                            target_spans_af.append((span_idx, *target_span))
                        
            writer.writerow([patient_group_id, patient_id, len_recording, serialize_list_of_tuples(target_spans), label])
            f.flush()
        
        writer.writerow([patient_group_id, patient_id, len_recording, serialize_list_of_tuples(target_spans_af), 2])
        f.flush()   
            
    f.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Needs one argument: configuration name (e.g. python build_sampling_spans.py config)")
        sys.exit()
    main(sys.argv[1])