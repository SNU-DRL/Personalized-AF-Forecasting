'''
Process evaluation results from different random seeds - calculate results for each patient.
'''
import os
import sys
from glob import glob

import pandas as pd

if len(sys.argv) != 2:
    print("Usage (needs two arguments): python process_metrics_patient.py ./results_personalized_241104_af/resnet18_15_bs128_lr5e-3_wd1e-4_ep3__bs32_lr1e-3_wd1e-4_ep5")
    sys.exit()

RESULTS_BASE_PATH = sys.argv[1]
print(f"Processing: {RESULTS_BASE_PATH}")
result_dirs = glob(f"{RESULTS_BASE_PATH}_seed*")
patient_list = [d for d in os.listdir(result_dirs[0]) if os.path.isdir(os.path.join(result_dirs[0], d))]
patient_eval_metrics_list = []

for patient in patient_list:
    eval_metrics = pd.read_csv(os.path.join(result_dirs[0], patient, "eval_metrics.csv"))
    last_epoch = eval_metrics["epoch"].max()
    eval_metrics = eval_metrics[eval_metrics['epoch'] == last_epoch]
    
    for result_dir in result_dirs[1:]:
        _eval_metrics = pd.read_csv(os.path.join(result_dir, patient, "eval_metrics.csv"))
        _eval_metrics = _eval_metrics[_eval_metrics['epoch'] == last_epoch]
        eval_metrics = pd.concat([eval_metrics, _eval_metrics], ignore_index=True)

    # for each patients
    eval_metrics = eval_metrics.drop(['epoch','train_loss','val_loss','tn','fp','fn','tp'], axis=1, errors='ignore')
    eval_metrics_by_blocks = eval_metrics.groupby(['block'], sort=False, as_index=False).agg(['mean'])
    eval_metrics_by_blocks.columns = eval_metrics_by_blocks.columns.map(lambda x: x[0])
    # eval_metrics_by_blocks = eval_metrics_by_blocks.round(3)
    eval_metrics_by_blocks['epoch'] = last_epoch
    eval_metrics_by_blocks['patient'] = patient
    # change column order
    cols = eval_metrics_by_blocks.columns.tolist()
    cols = ['patient'] + [col for col in cols if col!="patient"]
    eval_metrics_by_blocks = eval_metrics_by_blocks[cols]
        
    patient_eval_metrics_list.append(eval_metrics_by_blocks)

all_patients_eval_metrics = pd.concat(patient_eval_metrics_list, ignore_index=True)
dirname, filename = os.path.split(RESULTS_BASE_PATH)
all_patients_eval_metrics.to_csv(os.path.join(dirname, f"{filename}_patient.csv"))
