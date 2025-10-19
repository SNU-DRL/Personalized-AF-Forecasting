'''
Process evaluation results from different random seeds.
'''
import os
import sys
from glob import glob

import pandas as pd

if len(sys.argv) != 2:
    print("Usage (needs two arguments): python process_metrics.py ./results_global/resnet18_15_bs32_lr1e-5_wd1e-4_ep5")
    sys.exit()

RESULTS_BASE_PATH = sys.argv[1]
print(f"Processing: {RESULTS_BASE_PATH}")
result_dirs = glob(f"{RESULTS_BASE_PATH}_seed*")

eval_metrics = pd.read_csv(os.path.join(result_dirs[0], "eval_metrics.csv"))
last_epoch = eval_metrics["epoch"].max()
eval_metrics = eval_metrics[eval_metrics['epoch'] == last_epoch]

for result_dir in result_dirs[1:]:
    _eval_metrics = pd.read_csv(os.path.join(result_dir, "eval_metrics.csv"))
    _eval_metrics = _eval_metrics[_eval_metrics['epoch'] == last_epoch]
    eval_metrics = pd.concat([eval_metrics, _eval_metrics], ignore_index=True)

eval_metrics = eval_metrics.drop(['epoch','train_loss','val_loss','auprc_baseline','tn','fp','fn','tp','threshold'], axis=1, errors='ignore')
eval_metrics_by_blocks = eval_metrics.groupby(['block'], sort=False, as_index=False).agg(['mean', 'std'])
eval_metrics_by_blocks.columns = eval_metrics_by_blocks.columns.map("_".join)
eval_metrics_by_blocks = eval_metrics_by_blocks.round(3)

dirname, filename = os.path.split(RESULTS_BASE_PATH)
eval_metrics_by_blocks.to_csv(os.path.join(dirname, f"{filename}.csv"))
