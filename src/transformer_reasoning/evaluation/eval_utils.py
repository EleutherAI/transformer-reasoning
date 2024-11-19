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


def filename_schemes(min_order, max_order, N, num_parameters, wd):
    parent_dir = get_project_root() / f"results/n{N}_p{num_parameters}_omin{min_order}_omax{max_order}_wd{wd}_infinite"
    if not parent_dir.exists():
        print(f"Skipping {parent_dir} - path does not exist")
        return None
    return parent_dir

def load_eval_results():
    files = glob.glob('./results/n*_p*_omin1_omax2_wd0.1_l4_lr0.001_beta10.99_sf/eval_results.csv')

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        
        # Extract parameters from path
        params = re.search(r'n(\d+)_p(\d+).*lr(.+)_beta1(.+)_(sf|adamw|adamw-linear)', f)
        n_profiles = int(params.group(1))
        n_params = int(params.group(2))
        lr = float(params.group(3))
        beta1 = float(params.group(4))
        optimizer = params.group(5)
        
        # Add columns
        df['N_profiles'] = n_profiles
        df['n_params'] = n_params
        df['hops'] = [1,2] * (len(df)//2)
        df['lr'] = lr
        df['optimizer'] = optimizer
        df['beta1'] = beta1
        
        dfs.append(df)

    df = pd.concat(dfs)