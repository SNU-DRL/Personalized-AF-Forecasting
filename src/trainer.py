# Reference: https://xylambda.github.io/blog/python/pytorch/machine-learning/2021/01/04/pytorch_trainer.html

import os
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.attribution import Attribution, plot_attribution
from src.utils import compute_metrics

block_to_window_dict = {}
for i in range(25):
    block_to_window_dict[f"b{i}"] = f"w{i+1}"

class Trainer:
    """
    Parameters
    ----------
    model : torch.Module
        The model to train.
    criterion : torch.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.
    scheduler : torch.optim.lr_scheduler
        Scheduler to perform the learning rate scheduler.
    model_kwards : dict
        Dictionary for paths, model selecction methods
    
    """
    def __init__(self, model, criterion, optimizer, scheduler, model_kwargs, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_kwargs = model_kwargs
        self.device = device
        
        self.model.to(self.device)
        self.best_model = None
        self.eval_metrics_df = pd.DataFrame(columns=["epoch","train_loss","block","val_loss","auroc","auprc","auprc_baseline","accuracy","sensitivity","specificity","precision","npv","f1_score","threshold","tn","fp","fn","tp"])
        
    def fit(self, train_loader, eval_loader, epochs):
        """
        Parameters
        ----------
        train_loader : 
        eval_loader : 
        epochs : int
            Number of training epochs.
        
        """
        # track total training time
        total_start_time = time.time()
        # best_loss = float("inf")
        # best_auroc = 0
        preds_result = None
        
        # ---- train process ----
        for epoch in range(1, epochs+1):

            # train
            # train_loss = 0.0
            train_loss = self._train(train_loader)
            print(f"- [{epoch:03d}/{epochs:03d}] Train loss: {train_loss:.4f}, learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            if eval_loader is not None:
                # validation
                val_loss, val_metrics_dict, preds_result = self._evaluate(eval_loader, use_threshold=False, return_preds=(epoch==epochs))
                print(f"- [{epoch:03d}/{epochs:03d}] Validation loss: {val_loss:.4f}")
                epoch_metrics_dict = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
                
                blocks = sorted(val_metrics_dict.keys(), key=(lambda x: int(x[1:]) if x[1:].isnumeric() else 0))
                for block in blocks:
                    metrics_dict_block = val_metrics_dict[block]
                    updated_metrics_dict_block = metrics_dict_block | epoch_metrics_dict | {"block": block}
                    self._log(updated_metrics_dict_block, self.model_kwargs["eval_metrics_path"])
                    
            self.scheduler.step()

            # self._save_model(self.model_kwargs["model_save_dir"], "best")
            # print("*"*30)

        # last epoch: save prediction results
        if preds_result is not None:
            preds_result_df = pd.DataFrame.from_records(list(zip(*preds_result)), columns=["id", "label", "prob", "pred", "block"])
            preds_result_df.to_csv(self.model_kwargs["eval_preds_path"])
        total_time = time.time() - total_start_time

        # final message
        print(f"End of training. Total time: {round(total_time, 5)} seconds")
        self._save_model(self.model_kwargs["model_save_dir"], "last")

    def evaluate(self, eval_loader):
        eval_loss, eval_metrics_dict, preds_result = self._evaluate(eval_loader, use_threshold=False, return_preds=True) # use_threshold choice?        
        print(f"- Evaluation loss: {eval_loss:.4f}")
        epoch_metrics_dict = {"epoch": -1, "train_loss": -1, "val_loss": eval_loss}

        blocks = sorted(eval_metrics_dict.keys(), key=(lambda x: int(x[1:]) if x[1:].isnumeric() else 0))
        for block in blocks:
            metrics_dict_block = eval_metrics_dict[block]
            updated_metrics_dict_block = metrics_dict_block | epoch_metrics_dict | {"block": block}
            self._log(updated_metrics_dict_block, self.model_kwargs["eval_metrics_path"])
            
        eval_result_df = pd.DataFrame.from_records(list(zip(*preds_result)), columns=["id", "label", "prob", "pred", "block"])
        eval_result_df.to_csv(self.model_kwargs["eval_preds_path"])

    def attribute(self, loader, attr_method):
        attribution = Attribution(self.model, attr_method)

        for idx_batch, data_batch in enumerate(pbar := tqdm(loader)):
            xid, x, y, feature, rr = data_batch
            if y[0].item() == 0: # skip label 0 examples
                continue
            x, y, feature, rr = x.to(self.device), y.to(self.device), feature.to(self.device), rr.to(self.device)
            x.requires_grad = True
            y_hat = self.model(x, feature, rr)
            _probs = F.softmax(y_hat, dim=1)
            probs = _probs[:, 1]
            prob = probs[0].item()
            
            # apply feature attribution only if prob >= 0.7
            if prob < 0.7:
                continue
            
            # batch_size is set to 1
            attr_x = attribution.apply(x, y)
            
            xid_str = os.path.split(xid[0])[-1]
            episode_str, window_str = xid_str.split("|")
            window_id = f"{y.item()}@" + episode_str + '$'+ block_to_window_dict[window_str]
            
            fig_path = f"{self.model_kwargs['result_dir']}/{window_id}.jpeg"
            x = x.squeeze(0).detach().cpu().numpy()
            
            # guided GC - use absolute value
            if attr_method == "guided_gradcam":
                attr_x = np.abs(attr_x)
            
            plot_attribution(x, y.item(), prob, attr_x, path=fig_path)

    def _save_model(self, model_save_dir, desc):
        """
            - desc: model description (ex. 10epoch, best, last, ...)
        """
        torch.save(self.model, os.path.join(model_save_dir, f"model_{desc}.pt"))
    
    def _log(self, metrics_dict, metrics_save_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.eval_metrics_df = pd.concat([self.eval_metrics_df, pd.Series(metrics_dict).to_frame().T], ignore_index=True)
        numeric_columns = self.eval_metrics_df.columns.tolist()
        numeric_columns.remove("block")
        self.eval_metrics_df[numeric_columns] = self.eval_metrics_df[numeric_columns].apply(pd.to_numeric)
        # self.eval_metrics_df.round(4).to_csv(metrics_save_path, index=False)
        self.eval_metrics_df.to_csv(metrics_save_path, index=False)

    def _train(self, loader):
        self.model.train()
        total_loss = 0.0

        for idx_batch, data_batch in enumerate(pbar := tqdm(loader)):
            xid, x, y, feature, rr = data_batch
            x, y, feature, rr = x.to(self.device), y.to(self.device), feature.to(self.device), rr.to(self.device)
            y_hat = self.model(x, feature, rr)

            self.optimizer.zero_grad()
            loss = self._compute_loss(y_hat, y)
            loss.backward()
            self.optimizer.step()

            total_loss += y_hat.shape[0] * loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        total_loss /= len(loader.dataset)
        return total_loss
    
    def _evaluate(self, loader, use_threshold=False, return_preds=False):
        self.model.eval()
        total_loss = 0.0

        if use_threshold:
            threshold = self.model.threshold
        else:
            threshold = None

        xids = []
        probs = []
        labels = []
        eval_results_dict = OrderedDict() # includes blockwise evaluation results
        
        # for episode_evaluation
        episode_probs = []
        episode_labels = []
        
        with torch.no_grad():
            for idx_batch, data_batch in enumerate(pbar := tqdm(loader)):
                xid, x, y, feature, rr = data_batch
                
                x, y, feature, rr = x.to(self.device), y.to(self.device), feature.to(self.device), rr.to(self.device)
                y_hat = self.model(x, feature, rr)
                _probs = F.softmax(y_hat, dim=1)

                loss = self._compute_loss(y_hat, y)
                total_loss += y_hat.shape[0] * loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                xids.extend(xid)
                labels.extend(y)
                probs.extend(_probs[:, 1])

            total_loss /= len(xids)

        labels = torch.tensor(labels).detach().cpu().numpy()
        probs = torch.tensor(probs).detach().cpu().numpy()
        
        # annotate windows with episode, block info
        windows = []
        for idx, xid in enumerate(xids):
            episode, block = xid.split("|")
            windows.append({
                "episode": episode,
                "block": block, # (block == window_str in ECGWindow)
                "label": labels[idx],
                "prob": probs[idx]
            })
        
        metrics_dict_all = compute_metrics(labels, probs, threshold)
        eval_results_dict["all"] = metrics_dict_all
        
        # Episode evaluation
        if self.model_kwargs["episode_eval"]:
            episode_labels = []
            episode_probs = []
            
            all_episodes = list(map(lambda x: x["episode"], windows))
            for episode in np.unique(all_episodes):
                episode_windows = list(filter(lambda x: x["episode"] == episode, windows))
                episode_label = episode_windows[0]["label"].item()
                episode_window_probs = list(map(lambda x: x["prob"], episode_windows))
                if self.model_kwargs["episode_eval_strategy"] == "max":
                    episode_prob = max(episode_window_probs)
                elif self.model_kwargs["episode_eval_strategy"] == "mean":
                    episode_prob = np.mean(episode_window_probs)
                else:
                    raise NotImplementedError("episode_eval_strategy not implemented")
                episode_labels.append(episode_label)
                episode_probs.append(episode_prob)    
            
            metrics_dict_episode = compute_metrics(np.array(episode_labels), np.array(episode_probs), threshold)
            eval_results_dict["episode"] = metrics_dict_episode
        
        # Block-wise evaluation
        all_blocks = list(map(lambda x: x["block"], windows))
        
        for block in np.unique(all_blocks):
            block_windows = list(filter(lambda x: x["block"] == block, windows))
            block_labels = list(map(lambda x: x["label"], block_windows))
            block_probs = list(map(lambda x: x["prob"], block_windows))
            metrics_dict_block = compute_metrics(np.array(block_labels), np.array(block_probs), threshold)
            eval_results_dict[f"{block}"] = metrics_dict_block
        
        if threshold is None: # update threshold
            threshold = metrics_dict_all["threshold"] # threshold for all windows
            self.model.threshold = threshold
            
        if return_preds:
            preds = (probs > threshold).astype(int)
            preds_result = (xids, labels.tolist(), probs.tolist(), preds.tolist(), all_blocks)
            return total_loss, eval_results_dict, preds_result
        else:
            return total_loss, eval_results_dict, None

    def _compute_loss(self, y_hat, y):
        loss = self.criterion(y_hat, y)

        # apply regularization if any
        # loss += penalty.item()
            
        return loss
