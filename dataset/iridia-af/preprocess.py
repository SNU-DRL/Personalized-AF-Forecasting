import os

import numpy as np
import pandas as pd
from record import create_record
from scipy import signal
from tqdm import tqdm

RECORDS_PATH = "/data1/iridia-af/iridia-af-records-v1.0.1"
METADATA_PATH = "./build_dataset/iridia-af/metadata.csv"

ORIGINAL_SAMPLING_RATE = 200
TARGET_SAMPLING_RATE_1 = 250 # first channel; lead I

config_list = [
    {"lead": 1, "sampling_rate": TARGET_SAMPLING_RATE_1},
]

def convert_format():
    metadata_df = pd.read_csv(METADATA_PATH)
    
    for record_id in tqdm(metadata_df["record_id"].unique()):
        save_dir = os.path.join(RECORDS_PATH, record_id)
        
        record = create_record(record_id, metadata_df, RECORDS_PATH)
        record.load_ecg()
        recording = np.concatenate(record.ecg, dtype=np.float32, axis=0)
        
        for lead_idx, cfg in enumerate(config_list):
            lead_recording = recording[:, lead_idx]

            # 2) resample recording
            recording_length = len(lead_recording)
            target_recording_length = int(np.round(recording_length * cfg["sampling_rate"] / ORIGINAL_SAMPLING_RATE))
            resampled_recording = signal.resample(lead_recording, target_recording_length) # np.float32
            sample_length = len(resampled_recording)

            # 3) build af episode dataframe
            df = pd.DataFrame(columns=["start_point" , "end_point" , "Label"]) # points should be recalculated after resampling
            for i, row in record.ecg_labels_df.iterrows():
                start_day = row['start_file_index']
                end_day = row['end_file_index']
                
                start_point = row['start_qrs_index']
                for day in range(start_day):
                    start_point += record.ecg[day].shape[0]
                start_point = int(np.round(start_point * cfg["sampling_rate"] / ORIGINAL_SAMPLING_RATE))
                
                end_point = row['end_qrs_index']
                for day in range(end_day):
                    end_point += record.ecg[day].shape[0]
                end_point = int(np.round(end_point * cfg["sampling_rate"] / ORIGINAL_SAMPLING_RATE))
                
                Label = "AFIB"
                df = pd.concat([df, pd.DataFrame({"start_point": [start_point], "end_point": [end_point], "Label": [Label]})], ignore_index=True)
            
            # 4) save resulting files
            with open(os.path.join(save_dir, f"ecgSignal{cfg['lead']}.npy"), "wb") as f:
                np.save(f, resampled_recording)
            
            df.to_csv(os.path.join(save_dir, f"rhythmClassP{cfg['lead']}.csv"), index=False)
            
            with open(os.path.join(save_dir, f"sample_length{cfg['lead']}.txt"), "w") as f:
                f.write(str(sample_length))

if __name__ == "__main__":
    convert_format()