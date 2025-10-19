import argparse
import json
import os
import pickle

import torch

from src.attribution import ATTRIBUTION_METHODS
from src.dataset import PAFDataModule
from src.setup import setup
from src.trainer import Trainer


def main(args):
    # device
    device = setup(args)

    with open(args.eval_data_path, "rb") as fp:
        serialized = fp.read()
    eval_episodes = pickle.loads(serialized)
    
    # Patient-wise training & evaluation
    if "icentia11k" in args.eval_data_path:
        patient_dirs = {os.path.split(episode.patient_dir)[0] for episode in eval_episodes} # Set
    else:
        patient_dirs = {episode.patient_dir for episode in eval_episodes} # Set
    print(f"Number of patients used for personalization: {len(patient_dirs)}")
    
    patient_result_dir_list = []
    for patient_dir in patient_dirs:
        patient_phase, patient_id = patient_dir.split("/")[-2], patient_dir.split("/")[-1]
        patient_dirname = f"{patient_phase}@{patient_id}"
        patient_result_dir = f"{args.result_dir}/{patient_dirname}"
        os.makedirs(patient_result_dir, exist_ok=True)
        patient_result_dir_list.append(patient_result_dir)
        print(f"Processing: {patient_dirname}")
        
        # load proper models here...        
        mean_dict = None
        model_load_path = f"{args.model_load_dir}/{patient_dirname}/checkpoints/model_last.pt"
        p_model = torch.load(model_load_path)
        if args.load_mean_dict:
            mean_dict = p_model.mean_dict
        print(f"Successfully loaded model: {model_load_path}")
                
        if "icentia11k" in args.eval_data_path:
            p_eval_episodes = list(filter(lambda x: x.patient_dir.startswith(patient_dir), eval_episodes))
        else:
            p_eval_episodes = list(filter(lambda x: x.patient_dir == patient_dir, eval_episodes))
        
        if len(p_eval_episodes) == 0:
            print("No evaluation episodes found: skip this patient data")
            continue
        
        # Dataset
        p_data_module = PAFDataModule(None, p_eval_episodes, args.sampling_rate, use_standardization=args.use_standardization, mean_dict=mean_dict)
        p_eval_loader = p_data_module.eval_dataloader(args.batch_size) # evaluation only
        
        # additional paths
        model_kwargs = {
            "result_dir": patient_result_dir,
        }
        
        # Training setup & Hparams
        p_trainer = Trainer(p_model, None, None, None, model_kwargs, device)
        
        if args.mode == "attribution":
            p_trainer.attribute(p_eval_loader, args.attr_method)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Argument naming convention:
    # - file: ..._path
    #### If load/save operations may exist for the same data
    #   - load: ..._load_path
    #   - save: ..._save_path 
    # - directory: ..._dir

    # Dataset
    parser.add_argument('--eval_data_path', type=str, default='/data4/ecg_data/icentia11k/dataset_final/test_p_te.pickle')
    parser.add_argument('--sampling_rate', type=int, default='250')
    parser.add_argument('--load_mean_dict', action='store_true', help="use pre-computed mean values for features")

    # Model
    parser.add_argument('--model_load_dir', type=str, default=None) # For testing trained model
    parser.add_argument('--use_feature', action='store_true')
    
    # Settings
    parser.add_argument('--attr_method', default="guided_gradcam", type=str, choices=ATTRIBUTION_METHODS.keys())
    parser.add_argument('--gpu_num', type=str, default='0')
    parser.add_argument('--seed', type=int, default='42')
    parser.add_argument('--use_standardization', action='store_true') # not used

    # Result
    parser.add_argument('--result_dir', type=str, default='./results_demo_personalized_attribution')

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print(json.dumps(vars(args), indent=4))

    # Verify arguments
    assert os.path.isdir(args.model_load_dir) == True
    args.batch_size = 1 # set batch size to 1 for running attribution methods

    main(args)
