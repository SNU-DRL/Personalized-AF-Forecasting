import argparse
import copy
import json
import os
import pickle
import random

import pandas as pd
import torch

from src.dataset import PAFDataModule
from src.models.model_wrapper import ModelWrapper
from src.setup import setup
from src.trainer import Trainer


def main(args):
    # device
    device = setup(args)

    # Load datasets
    with open(args.train_data_path, "rb") as fp:
        serialized = fp.read()
    train_episodes = pickle.loads(serialized)

    with open(args.eval_data_path, "rb") as fp:
        serialized = fp.read()
    eval_episodes = pickle.loads(serialized)
    
    # model
    mean_dict = None
    if args.model_load_path is not None:
        model = torch.load(args.model_load_path)
        if args.load_mean_dict:
            mean_dict = model.mean_dict
        print(f"Successfully loaded model: {args.model_load_path}")
    else:
        model = ModelWrapper(args.arch, args.pretrained_backbone_path, args.use_feature)
        if args.load_mean_dict:
            print("Warning: `load_mean_dict` was set to true, but no model was loaded. The mean_dict will be calculated using the training set.")
    
    if args.freeze_layers_upto >= 0:
        layer_num = 0
        # ResNet18: 4(seq1), 5(seq2), 6(seq3), 7(seq4)
        for child in model.backbone.children():
            if layer_num <= args.freeze_layers_upto:
                for param in child.parameters():
                    param.requires_grad = False
                layer_num += 1
    
    # Patient-wise training & evaluation
    if "icentia11k" in args.train_data_path:
        patient_dirs = {os.path.split(episode.patient_dir)[0] for episode in train_episodes} # Set
    else:
        patient_dirs = {episode.patient_dir for episode in train_episodes} # Set
    print(f"Number of patients used for personalization: {len(patient_dirs)}")
    
    patient_result_dir_list = []
    for patient_dir in patient_dirs:
        patient_phase, patient_id = patient_dir.split("/")[-2], patient_dir.split("/")[-1]
        patient_dirname = f"{patient_phase}@{patient_id}"
        patient_result_dir = f"{args.result_dir}/{patient_dirname}"
        
        print(f"Processing: {patient_dirname}")
        if "icentia11k" in args.train_data_path:
            p_train_episodes = list(filter(lambda x: x.patient_dir.startswith(patient_dir), train_episodes))
            p_eval_episodes = list(filter(lambda x: x.patient_dir.startswith(patient_dir), eval_episodes))
        else:
            p_train_episodes = list(filter(lambda x: x.patient_dir == patient_dir, train_episodes))
            p_eval_episodes = list(filter(lambda x: x.patient_dir == patient_dir, eval_episodes))
        
        if len(p_eval_episodes) == 0:
            print("No evaluation episodes found: skip this patient data")
            continue
        elif args.force_num_episodes > 0:
            # only use label 0 and 1 episodes in the training episodes
            _p_train_episodes_0 = list(filter(lambda x: x.label == 0, p_train_episodes))
            _p_train_episodes_1 = list(filter(lambda x: x.label == 1, p_train_episodes))
            _p_train_episodes_2 = list(filter(lambda x: x.label == 2, p_train_episodes))
            
            if len(_p_train_episodes_0 + _p_train_episodes_1) < 30:
                print(f"Analyzing the effect of the number of training samples: skip this recording ({len(_p_train_episodes_0 + _p_train_episodes_1)} training episodes)")
                continue
            else:
                num_episodes_1 = args.force_num_episodes // 3
                num_episodes_0 = args.force_num_episodes - num_episodes_1
                
                p_train_episodes_0 = random.sample(_p_train_episodes_0, num_episodes_0)
                p_train_episodes_1 = random.sample(_p_train_episodes_1, num_episodes_1)
                p_train_episodes_1_span_indices = list(map(lambda x: x.span_idx, p_train_episodes_1))
                p_train_episodes_2 = list(filter(lambda x: x.span_idx in p_train_episodes_1_span_indices, _p_train_episodes_2))
                
                p_train_episodes = p_train_episodes_0 + p_train_episodes_1 + p_train_episodes_2
        
        os.makedirs(patient_result_dir, exist_ok=True)
        patient_result_dir_list.append(patient_result_dir)
        
        # Dataset
        p_data_module = PAFDataModule(p_train_episodes, p_eval_episodes, args.sampling_rate, args.sampler, args.include_af_episodes, args.use_augmentation, mean_dict)
        p_train_loader = p_data_module.train_dataloader(args.batch_size)
        p_eval_loader = p_data_module.eval_dataloader(args.batch_size)
        
        # additional paths
        model_kwargs = {
            "result_dir": patient_result_dir,
            "model_save_dir": f"{patient_result_dir}/checkpoints",
            "eval_metrics_path": f"{patient_result_dir}/eval_metrics.csv",
            "eval_preds_path": f"{patient_result_dir}/eval_preds.csv",
        }
        
        # Training setup & Hparams
        p_model = copy.deepcopy(model)
        p_model.mean_dict = p_data_module.mean_dict
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(p_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
        p_trainer = Trainer(p_model, criterion, optimizer, scheduler, model_kwargs, device)
        
        if args.mode in ["fit"]:
            os.makedirs(model_kwargs["model_save_dir"], exist_ok=True)
            p_trainer.fit(p_train_loader, p_eval_loader, args.epochs)
        if args.mode in ["evaluation"]:
            p_trainer.evaluate(p_eval_loader)

    eval_metrics = None
    last_epoch = None
    for patient_result_dir in patient_result_dir_list:
        try:
            _eval_metrics = pd.read_csv(os.path.join(patient_result_dir, "eval_metrics.csv"))
        except FileNotFoundError:
            continue
        
        last_epoch = _eval_metrics["epoch"].max()
        _eval_metrics = _eval_metrics[_eval_metrics['epoch'] == last_epoch]

        if eval_metrics is None:
            eval_metrics = _eval_metrics
        else:
            eval_metrics = pd.concat([eval_metrics, _eval_metrics], ignore_index=True)
            
    # Similar to analysis/process_metrics.py
    if eval_metrics is None:
        print("No patient-level eval_metrics.csv was found: skip aggregation")
        return
    eval_metrics = eval_metrics.drop(['epoch','train_loss','val_loss','auprc_baseline','tn','fp','fn','tp','threshold'], axis=1, errors='ignore')
    eval_metrics_by_blocks = eval_metrics.groupby(['block'], sort=False, as_index=False).agg(['mean']) # ignoring NaNs
    eval_metrics_by_blocks.columns = eval_metrics_by_blocks.columns.map(lambda x: x[0])
    eval_metrics_by_blocks['epoch'] = last_epoch
    eval_metrics_by_blocks.to_csv(os.path.join(args.result_dir, f"eval_metrics.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Argument naming convention:
    # - file: ..._path
    #### If load/save operations may exist for the same data
    #   - load: ..._load_path
    #   - save: ..._save_path 
    # - directory: ..._dir

    # Dataset
    parser.add_argument('--train_data_path', type=str, default='/data4/ecg_data/icentia11k/dataset_final/test_p_tr.pickle')
    parser.add_argument('--eval_data_path', type=str, default='/data4/ecg_data/icentia11k/dataset_final/test_p_te.pickle')
    parser.add_argument('--sampling_rate', type=int, default='250')
    parser.add_argument('--load_mean_dict', action='store_true', help="use pre-computed mean values for features")

    # Model
    parser.add_argument('--arch', type=str, default='resnet18_15')
    parser.add_argument('--model_load_path', type=str, default=None) # For testing trained model
    parser.add_argument('--pretrained_backbone_path', type=str, default=None) # For loading pre-trained backbone model
    parser.add_argument('--use_feature', action='store_true')
    
    # Hyperparameters for training
    parser.add_argument('--batch_size', '-bs', type=int, default='32')
    parser.add_argument('--learning_rate', '-lr', type=float, default="1e-4")
    parser.add_argument('--weight_decay', '-wd', type=float, default="1e-4")
    parser.add_argument('--epochs', '-ep', type=int, default=2)
    parser.add_argument('--sampler', action='store_true')

    # Settings
    parser.add_argument('--mode', type=str, choices=["fit", "evaluation"], default="fit", help="fit: training(+evaluation)")
    parser.add_argument('--gpu_num', type=str, default='0')
    parser.add_argument('--seed', type=int, default='42')
    parser.add_argument('--include_af_episodes', action='store_true')
    parser.add_argument('--use_augmentation', action='store_true')
    parser.add_argument('--freeze_layers_upto', type=int, default=-1, help='freeze layers upto layer (0 ~ #layers-1). fine-tune all when -1.')
    parser.add_argument('--force_num_episodes', type=int, default=-1, help='sensitivity study: number of samples used in training')
    
    # Result
    parser.add_argument('--result_dir', type=str, default='./results_demo_personalized')

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)    
    print(json.dumps(vars(args), indent=4))

    # Verify arguments
    if args.mode in ["evaluation"]:
        assert os.path.isfile(args.model_load_path) == True

    main(args)
