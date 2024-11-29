import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_reasoning.evaluation.eval_utils import (
    get_checkpoints,
    evaluate_model_histograms
)
from transformer_reasoning.evaluation.qa_evaluation import evaluate_qa_loss
from transformer_reasoning.train.train_utils import InfiniteQADataset, evaluate_single_model
from transformer_reasoning.generate_dataset.generate_qa_dataset import ATTRIBUTES, get_available_relations
import pandas as pd
from pandas import DataFrame
from transformer_reasoning.utils import get_project_root
from datasets import load_dataset
import argparse
import os

def get_checkpoint_paths(model_dir: str, num_checkpoints: int = 20) -> List[Path]:
    """Get evenly spaced checkpoint paths from a directory."""
    all_checkpoints = list(model_dir.glob("checkpoint-*"))
    
    if len(all_checkpoints) <= num_checkpoints:
        return all_checkpoints
    
    # Get evenly spaced indices
    indices = np.linspace(0, len(all_checkpoints)-1, num_checkpoints, dtype=int)
    return [all_checkpoints[i] for i in indices]

def evaluate_checkpoints(
    N: int,
    min_order: int,
    max_order: int,
    params: int,
    wd: float,
    compute_histograms: bool = False,
) -> Union[DataFrame, Tuple[DataFrame, Dict[str, List[float]]]]:
    """Evaluate loss for different tasks across checkpoints."""
    # Load existing results if they exist
    results_path = f'./results/n{N}_p{params}_omin{min_order}_omax{max_order}_wd{wd}_l4_lr0.001_beta10.99_sf/eval_results_full.csv'
    existing_results = pd.DataFrame()
    if Path(results_path).exists():
        existing_results = pd.read_csv(results_path)
        last_step = existing_results['global_step'].max() if len(existing_results) > 0 else 0
    else:
        last_step = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoints = get_checkpoints(min_order, max_order, N, params, wd)

    if not checkpoints:
        return existing_results if not existing_results.empty else None

    # Filter checkpoints to only evaluate new ones
    checkpoint_dirs = [x for x in checkpoints if os.path.isdir(x)]
    checkpoint_paths = [
        cp for cp in checkpoint_dirs 
        if int(str(cp).split('-')[-1]) >= last_step
    ]
    
    if not checkpoint_paths:
        return existing_results if not existing_results.empty else None

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llama_multihop_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    profiles_dataset = load_dataset(f"EleutherAI/profiles_dataset_{N}_uniform")['train']
    
    subjects = ATTRIBUTES + get_available_relations(profiles_dataset[0])

    if compute_histograms:
        checkpoint_paths = [max(checkpoint_dirs, key=lambda x: int(str(x).split('-')[-1]))]
    
    all_results = []
    histograms = {
        'loss': [], 
        'n_params': [], 
        'N_profiles': [], 
        'min_train_hops': [], 
        'max_train_hops': [], 
        'wd': [],
        'subject': [],
        'hops': []
    } if compute_histograms else None
   
    for subject in subjects:
        qa_dataset_1 = InfiniteQADataset(
            profiles_dataset, 
            tokenizer, 
            orders=[1], 
            qa_indices=range(len(profiles_dataset)),
            subjects=[subject]
        )
        qa_dataset_2 = InfiniteQADataset(
            profiles_dataset, 
            tokenizer, 
            orders=[2], 
            qa_indices=range(len(profiles_dataset)),
            subjects=[subject]
        )
    
        dataloader_1hop = DataLoader(qa_dataset_1, batch_size=16, shuffle=False, num_workers=15, pin_memory=True)
        dataloader_2hop = DataLoader(qa_dataset_2, batch_size=16, shuffle=False, num_workers=15, pin_memory=True)
        
        results_dicts = []
        
        for checkpoint_path in checkpoint_paths:
            if 'eval_results.csv' not in checkpoint_path:
                print(f"Evaluating checkpoint: {checkpoint_path}, subject: {subject}")
                    
                step = int(checkpoint_path.split('-')[-1])
                
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
                
                if compute_histograms:
                    onehop_and_twohop = evaluate_model_histograms(model, dataloader_1hop, dataloader_2hop)
                    for hops, losses in enumerate(onehop_and_twohop):
                        histograms['loss'].extend(losses)
                        histograms['n_params'].extend([params]*len(losses))
                        histograms['N_profiles'].extend([N]*len(losses))
                        histograms['min_train_hops'].extend([min_order]*len(losses))
                        histograms['max_train_hops'].extend([max_order]*len(losses))
                        histograms['wd'].extend([wd]*len(losses))
                        histograms['subject'].extend([subject]*len(losses))
                        histograms['hops'].extend([hops-1]*len(losses))
                        avg_result = {
                            'subject': subject,
                            'global_step': step,
                            'loss': np.mean(losses),
                            'min_train_hops': min_order,
                            'max_train_hops': max_order,
                            'hops': hops-1
                        }
                        results_dicts.append(avg_result)

                else:
                    results_dict_1hop = evaluate_single_model(model, dataloader_1hop, step, 1)
                    results_dict_2hop = evaluate_single_model(model, dataloader_2hop, step, 2)
                    
                    results_dicts.append(results_dict_1hop)
                    results_dicts.append(results_dict_2hop)
                
                del model
                torch.cuda.empty_cache()
        
        results = pd.DataFrame(results_dicts)
        results['subject'] = subject
        results['n_params'] = params
        results['N_profiles'] = N
        results['min_train_hops'] = min_order
        results['max_train_hops'] = max_order
        results['wd'] = wd
        
        all_results.append(results)

    all_results = pd.concat(all_results)
    
    # Combine with existing results if they exist
    if not existing_results.empty:
        all_results = pd.concat([existing_results, all_results])
    return (all_results, pd.DataFrame(histograms)) if compute_histograms else all_results

def plot_losses(all_results: DataFrame, save_path: Optional[str] = None):
    """Plot losses over training steps in a grid of subplots."""
    orders = [(1,1), (1,2)]
    hops = [1, 2]  # 1-hop and 2-hop
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(orders), len(hops), 
                            figsize=(12, 8), 
                            sharex=True,
                            sharey=True)
    
    # Add space for colorbar
    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    
    # Create log-spaced colormap
    param_range = all_results['model_params'].unique()
    norm = LogNorm(vmin=min(param_range), vmax=max(param_range))
    
    for i, (min_order, max_order) in enumerate(orders):
        for j, hop in enumerate(hops):
            ax = axes[i, j]
            
            # Filter results for this order
            order_results = all_results[all_results['model_min_hops'] == min_order]
            order_results = order_results[order_results['model_max_hops'] == max_order]
            
            # Plot appropriate loss line
            loss_key = f"qa_{hop}hop_loss" if hop > 0 else "bio_loss"
            
            # Create lmplot for each configuration
            for _, group in order_results.groupby(['model_wd', 'model_params']):
                color = plt.cm.viridis(norm(group['model_params'].iloc[0]))
                sns.regplot(
                    data=group,
                    x="steps",
                    y=loss_key,
                    ax=ax,
                    lowess=True,
                    scatter_kws={'color': color},
                    marker="o" if group['dataset_finite'].iloc[0] else "s",
                    line_kws={'color': color, 'alpha': 0.7},
                    label=f"wd={group['model_wd'].iloc[0]}, params={group['model_params'].iloc[0]}"
                )
            
            # Set titles and labels
            if i == 0:
                ax.set_title(f"{hop}-hop QA" if hop > 0 else "Biographical")
            if j == 0:
                ax.set_ylabel(f"Min train hops {min_order}, max train hops {max_order}\nLoss")
            if i == len(orders) - 1:
                ax.set_xlabel("Training Steps")
            
            # Set y-axis to log scale and fix limits
            ax.set_yscale('log')
            
            ax.grid(True)
    
    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    # Add colorbar for parameter count
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    fig.colorbar(sm, cax=cbar_ax, label='Parameter Count')
    
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        all_results.to_csv(str(save_path).replace('.png', '.csv'), index=False)
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_histograms', action='store_true')
    args = parser.parse_args()

    wd = 0.1
    for N in [10000, 15000, 20000, 25000, 30000, 50000]:
        full_results = []
        all_histograms = []

        for params in [435888, 890320, 1166688, 1475824]:
            for min_order in [1, 2]:
                for max_order in range(min_order, 3):
                    result = evaluate_checkpoints(
                        N=N,
                        min_order=min_order,
                        max_order=max_order,
                        params=params,
                        wd=wd,
                        compute_histograms=args.compute_histograms
                    )
                    if result is not None:
                        if args.compute_histograms:
                            df, histograms = result
                            hist_df = pd.DataFrame(histograms)
                            hist_df.to_csv(f'./results/n{N}_p{params}_omin{min_order}_omax{max_order}_wd{wd}_l4_lr0.001_beta10.99_sf/eval_results_histograms.csv')
                            all_histograms.append(hist_df)
                        else:
                            result.to_csv(f'./results/n{N}_p{params}_omin{min_order}_omax{max_order}_wd{wd}_l4_lr0.001_beta10.99_sf/eval_results_full.csv')
                            full_results.append(result)
                        
        # Append new results if any
        if full_results:
            full_results_df = pd.concat(full_results)
            full_results_df.to_csv(get_project_root() / f"results/loss_over_time_{N}_batched.csv", index=False)

        if all_histograms:
            hist_df = pd.concat(all_histograms)
            hist_df.to_csv(get_project_root() / f"results/loss_over_time_histograms_{N}_batched.csv", index=False)

        # if len(full_results_df) > 0:
        #     save_path = get_project_root() / f"results/loss_over_time_{N}_batched.png"
            # plot_losses(full_results_df, save_path)

if __name__ == "__main__":
    main()
