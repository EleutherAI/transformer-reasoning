import pandas as pd
import glob
import re
from torch.nn import CrossEntropyLoss
from pathlib import Path
from transformer_reasoning.utils import get_project_root

def tokenwise_loss(inputs, logits):
    """Calculate per-token loss."""
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def get_checkpoints(min_order, max_order, N, num_parameters, wd):
    files = glob.glob(f'./results/n{N}_p{num_parameters}_omin{min_order}_omax{max_order}_wd{wd}_l4_lr0.001_beta10.99_sf/*')
    return files


def load_eval_results():
    files = glob.glob('./results/n*_p*_omin1_omax*_wd0.1_l4_lr0.001_beta10.99_sf/eval_results_full.csv')

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        
        # Extract parameters from path
        params = re.search(r'n(\d+)_p(\d+)_omin(\d+)_omax(\d+)_wd([\d\.]+)_l(\d+)_lr([\d\.]+)_beta1([\d\.]+)_(sf|adamw|adamw-linear)', f)
        n_profiles = int(params.group(1))
        n_params = int(params.group(2))
        min_train_hops = int(params.group(3))
        max_train_hops = int(params.group(4))
        weight_decay = float(params.group(5))
        layers = int(params.group(6))
        lr = float(params.group(7))
        beta1 = float(params.group(8))
        optimizer = params.group(9)
        
        # Add columns
        df['hops'] = [1,2] * (len(df)//2)
        df['lr'] = lr
        df['layers'] = layers
        df['weight_decay'] = weight_decay
        df['optimizer'] = optimizer
        df['beta1'] = beta1
        
        dfs.append(df)

    df = pd.concat(dfs)
    return df