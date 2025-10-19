import os
import random
import sys

import pandas as pd
import yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils import deserialize_list_of_tuples

SEED = 42

random.seed(SEED)
dir_path = os.path.dirname(os.path.realpath(__file__))

def main(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    
    PERSONALIZED_MIN_NUM_SEGMENTS = cfg["PERSONALIZED_MIN_NUM_SEGMENTS"]
    PERSONALIZED_TRAINING_SEGMENTS = cfg["PERSONALIZED_TRAINING_SEGMENTS"]
    PERSONALIZED_MIN_EPISODES = cfg["PERSONALIZED_MIN_EPISODES"]
    SPAN_LIST_PATH = os.path.join(cfg["DATA_BUILD_PATH"], "span_list.csv")
    PATIENT_SPLIT_PATH = os.path.join(cfg["DATA_BUILD_PATH"], "patient_split.csv")
    
    span_list = pd.read_csv(SPAN_LIST_PATH, index_col=False, dtype={"spans": "string"})

    span_df = pd.DataFrame(data=[],
                           columns=["patient_group_id", "patient_id", "num_segments", "label_0_p_training", "label_0_p_test", "label_1_p_training", "label_1_p_test"])
    span_df = span_df.astype(dtype={"patient_group_id": "string", "patient_id": "string", "num_segments": "int64", "label_0_p_training": "int64", "label_0_p_test": "int64", "label_1_p_training": "int64", "label_1_p_test": "int64"})

    for key, group in span_list.groupby(["patient_id"]):
        num_segments = len(set(group["segment_id"].values))
        patient_dict = {"patient_group_id": group["patient_group_id"].iloc[0], "patient_id": key[0], "num_segments": num_segments}
        for label in [0,1]:
            label_group = group[group["label"]==label]
            label_group_p_training = label_group[label_group["segment_id"] < f"s{PERSONALIZED_TRAINING_SEGMENTS}"]
            label_group_p_test = label_group[label_group["segment_id"] >= f"s{PERSONALIZED_TRAINING_SEGMENTS}"]

            p_training_spans = []
            for segment_spans in label_group_p_training.spans:
                spans = deserialize_list_of_tuples(segment_spans)
                p_training_spans.extend(spans)
            patient_dict[f"label_{label}_p_training"] = len(p_training_spans)
            
            p_test_spans = []
            for segment_spans in label_group_p_test.spans:
                spans = deserialize_list_of_tuples(segment_spans)
                p_test_spans.extend(spans)
            patient_dict[f"label_{label}_p_test"] = len(p_test_spans)
        
        span_df = span_df._append(patient_dict, ignore_index=True)

    # exclude patients with no label 0 or label 1 spans
    span_df = span_df[~( (span_df["label_0_p_training"]==0)&(span_df["label_0_p_test"]==0) )]
    span_df = span_df[~( (span_df["label_1_p_training"]==0)&(span_df["label_1_p_test"]==0) )]

    test_candidates = span_df[
        (span_df["num_segments"] >= PERSONALIZED_MIN_NUM_SEGMENTS) &
        (span_df["label_0_p_training"] >= 1) &
        (span_df["label_0_p_test"] >= 1) &
        (span_df["label_1_p_training"] >= PERSONALIZED_MIN_EPISODES) &
        (span_df["label_1_p_test"] >= 1)
    ]

    test_df = test_candidates.sample(n=cfg["NUM_TEST"], random_state=SEED).sort_values(by="patient_id")
    remaining_df = span_df[~span_df.index.isin(test_df.index)]
    validation_df = remaining_df.sample(n=cfg["NUM_VALIDATION"], random_state=SEED).sort_values(by="patient_id")
    training_df = remaining_df[~remaining_df.index.isin(validation_df.index)].sort_values(by="patient_id")

    training_df["split"] = "training"
    validation_df["split"] = "validation"
    test_df["split"] = "test"
    
    all_df = pd.concat([training_df, validation_df, test_df])
    all_df.to_csv(PATIENT_SPLIT_PATH, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Needs one argument: configuration name (e.g. python split_patients.py config)")
        sys.exit()
    main(sys.argv[1])