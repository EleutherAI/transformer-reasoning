import pandas as pd
import glob
import re
from transformer_reasoning.utils import get_project_root
from pathlib import Path
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
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


def get_checkpoints(min_order, max_order, N, num_parameters, wd, commit_str, relations=None, layers=4):
    relations_str = f'_r{str(relations).lstrip("_r")}' if relations is not None else ''
    file_pattern = f'./results/{commit_str}/mup_n{N}_p{num_parameters}_omin{min_order}'\
        f'_omax{max_order}_wd{wd}_l{layers}_lr0.001_beta10.99_sf{relations_str}/*'
    files = glob.glob(file_pattern)
    return files


def load_eval_results(skip_mode=True, commit_hashes=[], subjectwise=False, base_path='.'):

    if commit_hashes:
        files = []
        for commit_hash in commit_hashes:
            if subjectwise:
                files += glob.glob(f'{base_path}/results/{commit_hash}/mup_n*_p*_omin1_omax*_wd0.1_l*_lr0.001_beta10.99_sf*/eval_results_full.csv')
            else:
                files += glob.glob(f'{base_path}/results/{commit_hash}/mup_n*_p*_omin1_omax*_wd0.1_l*_lr0.001_beta10.99_sf*/eval_results.csv')
    else:
        files = glob.glob(f'{base_path}/results/n*_p*_omin1_omax*_wd0.1_l*_lr0.001_beta10.99_sf*/eval_results.csv') + \
            glob.glob(f'{base_path}/results/mup_n*_p*_omin1_omax*_wd0.1_l*_lr0.001_beta10.99_sf*/eval_results.csv')

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if 'mode' in df.columns and skip_mode:
            continue
        if 'curr' in f:
            continue
        
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
        
        # Extract commit hash from path if present
        commit_match = re.search(r'/results/([^/]+)/', f)
        commit_hash = commit_match.group(1) if commit_match else 'unknown'
        
        # Add columns
        if 'mode' not in df.columns:
            df['hops'] = [1,2] * (len(df)//2)
            df['mode'] = df['hops'].apply(lambda x: 'train_onehop' if x == 1 else 'train_twohop')
        else:
            df['hops'] = df['mode'].apply(lambda x: 1 if x == 'train_onehop' else 2)
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
        df['commit_hash'] = commit_hash
        df['currency'] = 'current'
        if 'old' in f:
            df['currency'] = 'old'

        df = df[~df['mode'].astype(str).str.match(r'^\d*\.?\d*$')]
        
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

def get_sizes_and_entropies():
    from transformer_reasoning.evaluation.measure_capacity import dataset_entropy
    files = glob.glob('./results/n*_p*_omin1_omax*_wd0.1_l*_lr0.001_beta10.99_sf*/eval_results.csv') + \
        glob.glob('./results/n*_p*_omin1_omax*_wd0.1_l*_lr0.001_beta10.99_sf*/eval_results_old.csv')

    model_sizes_and_entropies = []

    for f in files:
        df = pd.read_csv(f)
        
        # Extract parameters from path
        params = re.search(r'n(\d+)_p(\d+)_omin(\d+)_omax(\d+)_wd([\d\.]+)_l(\d+)_lr([\d\.]+)_beta1([\d\.]+)_(sf|adamw|adamw-linear)(_r\d+)?(_hr\d+)?', f)
        n_profiles = int(params.group(1))
        n_params = int(params.group(2))
        layers = int(params.group(6))
        relations = int(params.group(10).lstrip('_r')) if params.group(10) else None

        rel_str = f"_r{relations}" if relations is not None else ""

        profiles_dataset = load_dataset(f"EleutherAI/profiles_dataset_{n_profiles}_uniform{rel_str}")['train']        

        dataset_entropy_optimal_1hop, dataset_entropy_optimal_2hop = dataset_entropy(
            profiles_dataset, 
            n_profiles, 
            scheme='optimal',
            selection_scheme='enumerate'
        )

        dataset_entropy_big_hash_1hop, dataset_entropy_big_hash_2hop = dataset_entropy(
            profiles_dataset, 
            n_profiles, 
            scheme='2-hop-big-hash',
            selection_scheme='enumerate'
        )

        model_sizes_and_entropies.append({
            'n_profiles': n_profiles,
            'n_params': n_params,
            'layers': layers,
            'relations': relations,
            'entropy_1hop': dataset_entropy_optimal_1hop['total_capacity'],
            'entropy_optimal_2hop': dataset_entropy_optimal_2hop['total_capacity'],
            'entropy_big_hash_2hop': dataset_entropy_big_hash_2hop['total_capacity'],
            'total_entropy_optimal': dataset_entropy_optimal_1hop['total_capacity'] + dataset_entropy_optimal_2hop['total_capacity'],
            'total_entropy_big_hash': dataset_entropy_big_hash_1hop['total_capacity'] + dataset_entropy_big_hash_2hop['total_capacity']
        })


    outfile = Path(get_project_root()) / 'model_sizes_and_entropies.csv'
    df = pd.DataFrame(model_sizes_and_entropies)
    df.to_csv(outfile, index=False)
    return df