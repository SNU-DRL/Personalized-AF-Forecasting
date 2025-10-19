import csv
import os
import sys
from glob import glob

import yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils import merge_af_intervals, read_rhythm_df

dir_path = os.path.dirname(os.path.realpath(__file__))

def main(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    
    os.makedirs(cfg["DATA_BUILD_PATH"], exist_ok=True)
    f = open(os.path.join(cfg["DATA_BUILD_PATH"], "af_burden_list.csv"), "w")
    writer = csv.writer(f)
    writer.writerow(["patient_group_id", "patient_id", "af_burden"])

    for DATA_PATH in cfg["DATA_PATH_LIST"]:
        patient_group_id = os.path.split(DATA_PATH)[-1]
        patient_dirs = sorted(glob(f"{DATA_PATH}/*"))
                
        for patient_dir in patient_dirs:
            print(patient_dir)
            patient_id = os.path.split(patient_dir)[-1]

            # Read rhythm label file
            try:
                rhythm_df = read_rhythm_df(cfg, patient_dir)
            except FileNotFoundError as e:
                rhythm_df = None
            
            # AFib burden == 0% -> No-AF group
            if (rhythm_df is None) or (len(rhythm_df) == 0) or ('AFIB' not in rhythm_df['Label'].values):
                print(f"[No-AF] AF burden: 0%")
                af_burden = 0
            else:
                afib_rhythm_df = rhythm_df[rhythm_df.Label == 'AFIB']
                afib_rhythm_df = merge_af_intervals(afib_rhythm_df)          
                afib_duration = (afib_rhythm_df['end_point'] - afib_rhythm_df['start_point']).values
                afib_duration_total= sum(afib_duration) # sec * sampling_rate
                
                with open(os.path.join(patient_dir, f"sample_length{cfg['DATA_STRING']}.txt"), "r") as g:            
                    len_recording = int(g.readline())
        
                af_burden = afib_duration_total / len_recording
                print(f"[AF] AF burden: {af_burden*100:.3f}%")
                
            writer.writerow([patient_group_id, patient_id, af_burden])
            
            f.flush()

    f.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Needs one argument: configuration name (e.g. python select_af_samples.py config)")
        sys.exit()
    main(sys.argv[1])