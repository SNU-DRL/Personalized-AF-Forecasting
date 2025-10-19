import csv
import os
import sys
from functools import reduce
from glob import glob

import numpy as np
import wfdb
import yaml

dir_path = os.path.dirname(os.path.realpath(__file__))

def main(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    
    os.makedirs(cfg["DATA_BUILD_PATH"], exist_ok=True)
    f = open(os.path.join(cfg["DATA_BUILD_PATH"], "af_burden_list.csv"), "w")
    writer = csv.writer(f)
    writer.writerow(["patient_group_id", "patient_id", "segment_id", "af_burden"])

    for DATA_PATH in cfg["DATA_PATH_LIST"]:
        patient_group_id = os.path.split(DATA_PATH)[-1]
        patient_dirs = sorted(glob(f"{DATA_PATH}/*"))
        
        for patient_dir in patient_dirs:
            print(patient_dir)
            patient_id = os.path.split(patient_dir)[-1]

            atr_files = sorted(glob(f"{patient_dir}/*.atr"))
            
            for atr_file in atr_files:
                record_name = os.path.splitext(atr_file)[0]
                _, patient_segment = os.path.split(record_name)
                patient_id, segment_id = patient_segment.split("_")
                
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
                    
                    af_burden = reduce(lambda acc, cur: acc + (cur[1] - cur[0]), af_episodes, 0) / cfg["LEN_RECORDING"] # length of each recording
                    print(f"[AF] AF burden ({segment_id}/{len(atr_files)}): {af_burden*100:.3f}%")
                    writer.writerow([patient_group_id, patient_id, segment_id, af_burden])
            f.flush()
    f.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Needs one argument: configuration name (e.g. python select_af_samples.py config)")
        sys.exit()
    main(sys.argv[1])