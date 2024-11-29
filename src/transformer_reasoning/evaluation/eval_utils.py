import pandas as pd
import glob
import re
from torch.nn import CrossEntropyLoss
from pathlib import Path
from transformer_reasoning.utils import get_project_root
import torch
from tqdm import tqdm
import numpy as np

def tokenwise_loss(inputs, logits):
    """Calculate per-token loss."""
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def get_checkpoints(min_order, max_order, N, num_parameters, wd, relations=None, hop_ratio=None, layers=4):
    relations_str = f'_r{relations}' if relations is not None else ''
    hop_ratio_str = f'_hr{hop_ratio}' if hop_ratio is not None else ''
    file_pattern = f'./results/n{N}_p{num_parameters}_omin{min_order}_omax{max_order}_wd{wd}_l{layers}_lr0.001_beta10.99_sf{relations_str}{hop_ratio_str}/*'
    files = glob.glob(file_pattern)
    return files


def load_eval_results():
    files = glob.glob('./results/n*_p*_omin1_omax*_wd0.1_l*_lr0.001_beta10.99_sf*/eval_results.csv')

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        
        # Extract parameters from path
        params = re.search(r'n(\d+)_p(\d+)_omin(\d+)_omax(\d+)_wd([\d\.]+)_l(\d+)_lr([\d\.]+)_beta1([\d\.]+)_(sf|adamw|adamw-linear)(_r\d+)?(_hr\d+)?', f)
        n_profiles = int(params.group(1))
        n_params = int(params.group(2))
        min_train_hops = int(params.group(3))
        max_train_hops = int(params.group(4))
        weight_decay = float(params.group(5))
        layers = int(params.group(6))
        lr = float(params.group(7))
        beta1 = float(params.group(8))
        optimizer = params.group(9)
        relations = int(params.group(10).lstrip('_r')) if params.group(10) else np.nan
        hop_ratio = int(params.group(11).lstrip('_hr')) if params.group(11) else np.nan
        
        # Add columns
        df['hops'] = [1,2] * (len(df)//2)
        df['lr'] = lr
        df['layers'] = layers
        df['weight_decay'] = weight_decay
        df['optimizer'] = optimizer
        df['beta1'] = beta1
        df['relations'] = relations
        df['hop_ratio'] = hop_ratio
        df['N_profiles'] = n_profiles
        df['n_params'] = n_params
        df['min_train_hops'] = min_train_hops
        df['max_train_hops'] = max_train_hops

        dfs.append(df)

    df = pd.concat(dfs)
    return df

def evaluate_model_histograms(model, onehop_loader, twohop_loader):
    """Evaluate model and return individual losses for each question."""
    onehop_losses = []
    twohop_losses = []
    
    with torch.no_grad():
        for loader, losses in [(onehop_loader, onehop_losses), (twohop_loader, twohop_losses)]:
            for batch in tqdm(loader):
                labels = batch['labels']
                last_neg = labels[:,-1] == -100
                if not last_neg.any():
                    continue
                    
                filtered_inputs = batch['input_ids'][last_neg]
                filtered_labels = batch['labels'][last_neg]
                is_pos = filtered_labels >= 0
                
                outputs = model(
                    input_ids=filtered_inputs.to(model.device),
                    labels=filtered_labels.to(model.device)
                )
                
                # Extract individual losses for each question
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = filtered_labels[..., 1:].contiguous().to(shift_logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Reshape and mask to get per-question losses
                loss = loss.view(filtered_labels.shape[0], -1)
                is_answer_start = torch.roll(is_pos, -1)[:, :-1]
                is_answer_end = torch.roll(is_pos, 1)[:, :-1]
                
                for q_loss, q_starts, q_ends in zip(loss, is_answer_start, is_answer_end):
                    accumulate = False
                    current_loss = []
                    
                    for token_loss, is_start, is_end in zip(q_loss, q_starts, q_ends):
                        if is_start:
                            accumulate = True
                            current_loss = []
                        
                        if accumulate:
                            current_loss.append(token_loss.item())
                            
                        if is_end:
                            accumulate = False
                            if current_loss:  # Only append if we collected some losses
                                losses.append(np.sum(current_loss))
                
                if len(losses) >= 1000:
                    break
    
    return onehop_losses, twohop_losses