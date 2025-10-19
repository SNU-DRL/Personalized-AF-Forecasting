"""
    Build indices for spans that satisfies specific condition.
"""

import csv
import os
import sys

import numpy as np
import pandas as pd
import wfdb
import yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils import (find_non_overlapping_spans, is_valid_span,
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
    writer.writerow(["patient_group_id", "patient_id", "segment_id", "sample_length", "spans", "label"])
    sampling_rate = cfg["SAMPLING_RATE"]
    len_recording = cfg["LEN_RECORDING"] # count number of datapoints
    
    af_burden_df = pd.read_csv(AF_BURDEN_LIST_PATH)

    # Process each patients
    for (patient_group_id, patient_id), group in af_burden_df.groupby(['patient_group_id', 'patient_id']):        
        patient_path = os.path.join(cfg["DATA_BASE_PATH"], patient_group_id, patient_id)
        print(f"Processing... {patient_path}")
        
        with open(os.path.join(patient_path, "RECORDS")) as g:
            record_list = g.readlines()
        segment_ids = [x.split("_")[-1].strip() for x in record_list]
        non_af_segments = [x for x in segment_ids if x not in set(group['segment_id'])]
        
        # Calculate AF burden for each patient
        patient_af_burdens = group.af_burden.tolist()
        for _ in range(len(non_af_segments)):
            patient_af_burdens.append(0)
        patient_af_burden = np.mean(patient_af_burdens)

        if patient_af_burden >= cfg["AF_BURDEN_THRESHOLD"]:
            print(f"AF burden too high; skip this patient - {patient_path}")
            continue
        
        # Process each segment
        # ICENTIA11K: 50 (or less) recordings for each patient        
        for idx, row in group.iterrows():
            segment_id, af_burden = row['segment_id'], row['af_burden']
            record_name = os.path.join(patient_path, f"{patient_id}_{segment_id}")

            ann = wfdb.rdann(record_name, "atr")
            aux_note_unique = np.unique(ann.aux_note)
            if ("(AFIB" in aux_note_unique) or ("AFIB" in aux_note_unique):
                rhythm_ano_mask = (np.array(ann.aux_note)!='None')
                sample_with_rhythm_ano = ann.sample[rhythm_ano_mask]
                aux_note_with_rhythm_ano = np.array(ann.aux_note)[rhythm_ano_mask]
                
                af_episodes = []
                for s, a in (it := zip(sample_with_rhythm_ano, aux_note_with_rhythm_ano)):
                    if a == "(AFIB":
                        next_s, next_a = next(it)
                        assert next_a == ")"
                        af_episodes.append((s, next_s))
                    elif a == "(AFIB)":
                        af_episodes.append((s, s+1))

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
                    
                    if is_valid_span((valid_span_min, valid_span_max), target_span) and (len_target_span >= cfg["EPISODE_LENGTH_SEC"] * sampling_rate):
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
                            
                writer.writerow([patient_group_id, patient_id, segment_id, len_recording, serialize_list_of_tuples(target_spans), label])
                f.flush()
            
            writer.writerow([patient_group_id, patient_id, segment_id, len_recording, serialize_list_of_tuples(target_spans_af), 2])
            f.flush()
        
        # Label 0 selection in non_af_segments of AF patients
        for segment_id in non_af_segments:
            span_start = 0
            span_end = len_recording
            target_span = eval(LABEL0_SELECTION["between"])            
            len_target_span = target_span[1] - target_span[0] if (target_span is not None) else 0
            if (len_target_span >= cfg["EPISODE_LENGTH_SEC"] * sampling_rate):
                writer.writerow([patient_group_id, patient_id, segment_id, len_recording, serialize_list_of_tuples([(0, *target_span)]), 0])
                f.flush()
    
    f.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Needs one argument: configuration name (e.g. python build_sampling_spans.py config)")
        sys.exit()
    main(sys.argv[1])