import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import List, Dict, Optional
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_reasoning.evaluation.eval_utils import (
    filename_schemes
)
from transformer_reasoning.evaluation.qa_evaluation import evaluate_qa_loss
from transformer_reasoning.train.train_utils import InfiniteBiosDataset
import pandas as pd
from pandas import DataFrame
from transformer_reasoning.utils import get_project_root
from datasets import load_dataset

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
    finite: bool = False,
    checkpoints: list = [],
    num_samples: int = 8000,
    only_norms: bool = False
) -> Dict[str, List[float]]:
    """Evaluate loss for different tasks across checkpoints."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_dir = filename_schemes(min_order, max_order, N, params, wd)
    if not model_dir:
        return None

    checkpoint_paths = checkpoints
    
    # Only load datasets if we're calculating losses
    if not only_norms:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llama_multihop_tokenizer")
        tokenizer.pad_token = tokenizer.eos_token
        profiles_dataset = load_dataset(f"EleutherAI/profiles_dataset_{N}_uniform")['train']
        qa_dataset_1 = iter(InfiniteBiosDataset(profiles_dataset, tokenizer, orders = [1], qa_prob=1, qa_indices = list(range(len(profiles_dataset)))))
        qa_dataset_2 = iter(InfiniteBiosDataset(profiles_dataset, tokenizer, orders = [2], qa_prob=1, qa_indices = list(range(len(profiles_dataset)))))
        
        qa_data_1hop = []
        qa_data_2hop = []
        counter = 0
        for qa_item_1, qa_item_2 in zip(qa_dataset_1, qa_dataset_2):
            counter += 1
            if counter > num_samples:
                break
            qa_data_1hop.append(qa_item_1)
            qa_data_2hop.append(qa_item_2)
    
    results = {
        "steps": [],
        "l2_norm": [],
        "model_min_hops": [],
        "model_max_hops": [],
        "model_params": [],
        "model_wd": [],
        "dataset_N": [],
        "dataset_finite": [],
    }
    
    if not only_norms:
        results.update({
            "qa_1hop_loss": [],
            "qa_2hop_loss": [],
        })
    
    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating checkpoint: {checkpoint_path}")
        
        step = int(checkpoint_path.stem.split('-')[-1])
        results["steps"].append(step)
        
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
        l2_norm = torch.norm(torch.cat([p.flatten() for p in model.parameters()])).item()
        results["l2_norm"].append(l2_norm)
        
        if not only_norms:
            qa_1hop_loss = evaluate_qa_loss(model, qa_data_1hop, device, num_samples)
            qa_2hop_loss = evaluate_qa_loss(model, qa_data_2hop, device, num_samples)
            results["qa_1hop_loss"].append(qa_1hop_loss)
            results["qa_2hop_loss"].append(qa_2hop_loss)
            print(f"qa_1hop_loss: {qa_1hop_loss}, qa_2hop_loss: {qa_2hop_loss}")
        
        results["model_min_hops"].append(min_order)
        results["model_max_hops"].append(max_order)
        results["model_params"].append(params)
        results["model_wd"].append(wd)
        results["dataset_N"].append(N)
        results["dataset_finite"].append(finite)
        
        del model
        torch.cuda.empty_cache()
    
    return results

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
            for _, group in order_results.groupby(['model_wd', 'dataset_finite', 'model_params']):
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
                    label=f"wd={group['model_wd'].iloc[0]}, params={group['model_params'].iloc[0]}, {'fin' if group['dataset_finite'].iloc[0] else 'inf'}"
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_checkpoints", type=int, default=20,
                       help="Number of checkpoints to evaluate")
    parser.add_argument("--num_samples", type=int, default=2000,
                       help="Number of samples to evaluate")
    parser.add_argument("--recalculate", action="store_true",
                       help="Recalculate results")
    parser.add_argument("--only_norms", action="store_true",
                       help="Only calculate L2 norms")
    
    args = parser.parse_args()
    wd = 0.1
    new_results = []
    for N in [10000, 15000, 20000, 25000, 30000, 50000]:
        csv_path = get_project_root() / f"results/loss_over_time_{N}_uniform.csv"
        results_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        
        for params in [435888, 890320, 1166688, 1475824]:
            for min_order in [1, 2]:
                for max_order in range(min_order, 3):
                    if csv_path.exists() and not args.recalculate:
                        # Load existing results
                        results_df_filtered = results_df[
                            (results_df['model_min_hops'] == min_order) &
                            (results_df['model_max_hops'] == max_order) &
                            (results_df['model_params'] == params) &
                            (results_df['model_wd'] == wd)
                        ]

                        if 'l2_norm' not in results_df_filtered.columns:
                            results_df_filtered['l2_norm'] = np.nan
                        
                        model_dir = filename_schemes(min_order, max_order, N, params, wd)
                        if not model_dir:
                            continue
                            
                        all_checkpoints = sorted(list(model_dir.glob("checkpoint-*")))
                        
                        if args.only_norms:
                            # For norm-only mode, evaluate checkpoints that either:
                            # 1. Don't have l2_norm in results
                            # 2. Have NaN l2_norm values
                            existing_steps = set(results_df_filtered['steps'])
                            new_checkpoints = [cp for cp in all_checkpoints 
                                             if int(cp.stem.split('-')[-1]) in existing_steps]
                        else:
                            # Original logic for full evaluation
                            last_step = results_df_filtered['steps'].max()
                            if np.isnan(last_step):
                                last_step = 0
                            print(f"Evaluating from step {last_step}")
                            new_checkpoints = [cp for cp in all_checkpoints 
                                             if int(cp.stem.split('-')[-1]) > last_step]
                        
                        if len(new_checkpoints) == 0:
                            continue

                        if (len(new_checkpoints) < args.num_checkpoints) or args.only_norms:
                            checkpoints_to_evaluate = new_checkpoints
                        else:
                            cp_steps = [int(cp.stem.split('-')[-1]) for cp in new_checkpoints]
                            num_checkpoints = int(args.num_checkpoints * (max(cp_steps) - last_step)/max(cp_steps))
                            indices = np.linspace(0, len(new_checkpoints)-1, num_checkpoints, dtype=int)
                            checkpoints_to_evaluate = [new_checkpoints[i] for i in indices]

                        result = evaluate_checkpoints(
                            N=N,
                            min_order=min_order,
                            max_order=max_order,
                            params=params,
                            wd=wd,
                            checkpoints=checkpoints_to_evaluate,
                            num_samples=args.num_samples,
                            only_norms=args.only_norms
                        )
                        
                        if result:
                            new_result_df = pd.DataFrame(result)
                            if args.only_norms:
                                # Create merge key columns
                                merge_cols = ['steps', 'model_min_hops', 'model_max_hops', 
                                            'model_params', 'model_wd', 'dataset_N']
                                
                                # Merge new norms with existing results, keeping all loss values
                                results_df = pd.merge(
                                    results_df,
                                    new_result_df[merge_cols + ['l2_norm']],
                                    on=merge_cols,
                                    how='left',
                                    suffixes=('_old', '')
                                )
                                # Update l2_norm column, preferring new values
                                if 'l2_norm_old' in results_df.columns:
                                    results_df['l2_norm'] = results_df['l2_norm'].fillna(results_df['l2_norm_old'])
                                    results_df = results_df.drop(columns=['l2_norm_old'])
                            else:
                                # Original behavior for full evaluations
                                new_results.append(new_result_df)
                        
        # Append new results if any
        if new_results:
            new_results_df = pd.concat(new_results)
            results_df = pd.concat([results_df, new_results_df], ignore_index=True)
        if new_results or args.only_norms:
            results_df[results_df['dataset_N'] == N].to_csv(csv_path, index=False)

        if len(results_df) > 0 and not args.only_norms:
            save_path = get_project_root() / f"results/loss_over_time_{N}_uniform.png"
            plot_losses(results_df[results_df['dataset_N'] == N], save_path)

if __name__ == "__main__":
    main()
