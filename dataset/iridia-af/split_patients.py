import os
import random
import sys

import pandas as pd
import yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils import deserialize_list_of_tuples, filter_span

random.seed(1)
dir_path = os.path.dirname(os.path.realpath(__file__))

def main(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    
    PERSONALIZED_MIN_LENGTH_POINTS = cfg["PERSONALIZED_MIN_LENGTH_POINTS"]
    PERSONALIZED_TRAINING_TIME = cfg["PERSONALIZED_TRAINING_DAYS"] * 24 * 60 * 60 * cfg["SAMPLING_RATE"]
    PERSONALIZED_MIN_EPISODES = cfg["PERSONALIZED_MIN_EPISODES"]
    SPAN_LIST_PATH = os.path.join(cfg["DATA_BUILD_PATH"], "span_list.csv")
    PATIENT_SPLIT_PATH = os.path.join(cfg["DATA_BUILD_PATH"], "patient_split.csv")
    
    span_df = pd.read_csv(SPAN_LIST_PATH, index_col=False, dtype={"spans": "string"})
    
    span_list = span_df["spans"].apply(lambda x: deserialize_list_of_tuples(x))
    span_list_p_training = span_list.apply(lambda x: filter_span(x, PERSONALIZED_TRAINING_TIME, True))
    span_list_p_test = span_list.apply(lambda x: filter_span(x, PERSONALIZED_TRAINING_TIME, False))
    span_df["span_counts_p_training"] = span_list_p_training.apply(lambda x : len(x))
    span_df["span_counts_p_test"] = span_list_p_test.apply(lambda x : len(x))
        
    all_patients = set(span_df["patient_id"].values)
    test_candidates = set()
    for patient_id in all_patients:
        patient_df = span_df[span_df["patient_id"] == patient_id]
        if patient_df.iloc[0]["sample_length"] < PERSONALIZED_MIN_LENGTH_POINTS:
            continue
        row_label_0 = patient_df[patient_df["label"]==0].iloc[0]
        flag_label_0 = (row_label_0["span_counts_p_training"] >= 1) and (row_label_0["span_counts_p_test"] >= 1)
        row_label_1 = patient_df[patient_df["label"]==1].iloc[0]
        flag_label_1 = (row_label_1["span_counts_p_training"] >= PERSONALIZED_MIN_EPISODES) and (row_label_1["span_counts_p_test"] >= 1)
        
        if flag_label_0 and flag_label_1:
            test_candidates.add(patient_id)
    
    training_patients = set()
    if cfg["NUM_TEST"] == -1: # use all candidates as test patients
        test_patients = list(test_candidates)
    else:
        test_patients = sorted(random.sample(test_candidates, cfg["NUM_TEST"]))
    validation_candidates = all_patients - set(test_patients) - training_patients
    validation_patients = sorted(random.sample(validation_candidates, cfg["NUM_VALIDATION"]))
    training_patients = sorted(list(training_patients | (validation_candidates - set(validation_patients))))
    
    label_1_df = span_df[span_df["label"] == 1]
    training_df, validation_df, test_df = label_1_df[label_1_df["patient_id"].isin(training_patients)], label_1_df[label_1_df["patient_id"].isin(validation_patients)], label_1_df[label_1_df["patient_id"].isin(test_patients)]
    training_df["split"] = "training"
    validation_df["split"] = "validation"
    test_df["split"] = "test"
    
    all_df = pd.concat([training_df, validation_df, test_df]).drop(columns=["spans", "label"])
    all_df.to_csv(PATIENT_SPLIT_PATH, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Needs one argument: configuration name (e.g. python split_patients.py config)")
        sys.exit()
    main(sys.argv[1])