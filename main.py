import argparse
import json
import os

import torch

from src.dataset import PAFDataModule
from src.models.model_wrapper import ModelWrapper
from src.setup import setup
from src.trainer import Trainer


def main(args):
    # device
    device = setup(args)

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

    # dataloader
    data_module = PAFDataModule.from_datapath(args.train_data_path, args.eval_data_path, args.sampling_rate, args.sampler, args.include_af_episodes, args.use_augmentation, mean_dict)
    model.mean_dict = data_module.mean_dict
    train_loader = data_module.train_dataloader(args.batch_size)
    eval_loader = data_module.eval_dataloader(args.batch_size)

    # additional paths
    model_kwargs = {
        "result_dir": args.result_dir,
        "eval_metrics_path": args.eval_metrics_path,
        "eval_preds_path": args.eval_preds_path,
        "model_save_dir": args.model_save_dir,
    }

    # hparams
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
    
    trainer = Trainer(model, criterion, optimizer, scheduler, model_kwargs, device)
    if args.mode in ["fit"]:
        trainer.fit(train_loader, eval_loader, args.epochs)
    if args.mode in ["evaluation"]:
        trainer.evaluate(eval_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Argument naming convention:
    # - file: ..._path
    #### If load/save operations may exist for the same data
    #   - load: ..._load_path
    #   - save: ..._save_path 
    # - directory: ..._dir

    # Dataset
    parser.add_argument('--train_data_path', type=str, default='/data4/ecg_data/icentia11k/dataset_final/training.pickle')
    parser.add_argument('--eval_data_path', type=str, default='/data4/ecg_data/icentia11k/dataset_final/validation.pickle')
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
    
    # Result
    parser.add_argument('--result_dir', type=str, default='./results_demo')

    args = parser.parse_args()

    # Resulting directories
    args.model_save_dir = f"{args.result_dir}/checkpoints"
    args.eval_metrics_path = f"{args.result_dir}/eval_metrics.csv" # evaluation metrics (each epoch)
    args.eval_preds_path = f"{args.result_dir}/eval_preds.csv" # prediction results of a final model

    os.makedirs(args.result_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)    
    print(json.dumps(vars(args), indent=4))

    # Verify arguments
    if args.mode in ["evaluation"]:
        assert os.path.isfile(args.model_load_path) == True
    if args.mode in ["fit"]:
        os.makedirs(args.model_save_dir, exist_ok=True)
    
    main(args)
