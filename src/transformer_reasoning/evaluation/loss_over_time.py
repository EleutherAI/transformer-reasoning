import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import List, Dict, Optional
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_reasoning.evaluation.eval_utils import (
    evaluate_bio_loss,
    evaluate_qa_loss
)
import pandas as pd
from pandas import DataFrame
from transformer_reasoning.evaluation.measure_capacity import filename_schemes
from transformer_reasoning.utils import get_project_root
from datasets import load_dataset, load_from_disk

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
    old: bool = False,
    num_checkpoints: int = 10,
    num_samples: int = 5000
) -> Dict[str, List[float]]:
    """Evaluate loss for different tasks across checkpoints."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = filename_schemes(min_order, max_order, N, params, wd, old, finite)
    if not model_dir:
        return None
    checkpoint_paths = get_checkpoint_paths(model_dir, num_checkpoints)
    


    # Load datasets
    insert_eos = False
    if params == 1_000_000 and finite:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        if N == 10000:
            bios_dataset = load_dataset("EleutherAI/transformer-reasoning-bios-dataset-10000", revision="a4029b437d3d96cb591d12b89b6c05bade648b9d")['train']
        else:
            bios_dataset = load_dataset(f"EleutherAI/transformer-reasoning-bios-dataset-{N}")['train']
    else:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llama_multihop_tokenizer")
        bios_dataset = load_dataset(f"EleutherAI/transformer-reasoning-bios-dataset-{N}")['train']
        insert_eos = True
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load QA dataset from disk instead of HF hub
    qa_dataset = load_from_disk(str(get_project_root() / f"generated_data/qa_dataset_{N}"))
    
    # Combine samples proportionally from all splits
    splits = ['train', 'validation', 'heldout_profiles']
    qa_data = []
    for split in splits:
        split_data = qa_dataset[split]
        qa_data.extend(split_data)
    
    # Filter by hop/order after combining
    qa_data_1hop = [x for x in qa_data if x['questions.order'] == 1]
    qa_data_2hop = [x for x in qa_data if x['questions.order'] == 2]
    
    results = {
        "steps": [],
        "bio_loss": [],
        "qa_1hop_loss": [],
        "qa_2hop_loss": [],
        "model_min_hops": [],
        "model_max_hops": [],
        "model_params": [],
        "model_wd": [],
        "dataset_N": [],
        "dataset_finite": []
    }
    
    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating checkpoint: {checkpoint_path}")
        
        # Extract step number from checkpoint filename
        step = int(checkpoint_path.stem.split('-')[-1])
        results["steps"].append(step)
        
        # Load model using HuggingFace transformers
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
        # Evaluate on each task
        bio_loss = evaluate_bio_loss(model, tokenizer, bios_dataset, device, num_samples, insert_eos)
        qa_1hop_loss = evaluate_qa_loss(model, tokenizer, qa_data_1hop, device, num_samples, insert_eos)
        qa_2hop_loss = evaluate_qa_loss(model, tokenizer, qa_data_2hop, device, num_samples, insert_eos)
        
        results["bio_loss"].append(bio_loss)
        results["qa_1hop_loss"].append(qa_1hop_loss)
        results["qa_2hop_loss"].append(qa_2hop_loss)
        results["model_min_hops"].append(min_order)
        results["model_max_hops"].append(max_order)
        results["model_params"].append(params)
        results["model_wd"].append(wd)
        results["dataset_N"].append(N)
        results["dataset_finite"].append(finite)
        print(f"bio_loss: {bio_loss}, qa_1hop_loss: {qa_1hop_loss}, qa_2hop_loss: {qa_2hop_loss}")
        # Free up memory
        del model
        torch.cuda.empty_cache()
    
    return results

def plot_losses(all_results: DataFrame, save_path: Optional[str] = None):
    """Plot losses over training steps in a grid of subplots."""
    orders = [(1,1), (1,2), (2,2)]
    hops = [1, 2]  # 1-hop and 2-hop
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(orders), len(hops), 
                            figsize=(12, 8), 
                            sharex=True)
    
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
            ax.set_ylim(5e-3, 5.5)
            
            ax.grid(True)
    
    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    # Add colorbar for parameter count
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    fig.colorbar(sm, cax=cbar_ax, label='Parameter Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        all_results.to_csv(str(save_path).replace('.png', '.csv'), index=False)
    else:
        plt.show()
    
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_checkpoints", type=int, default=10,
                       help="Number of checkpoints to evaluate")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of samples to evaluate")
    parser.add_argument("--old", action="store_true",
                       help="Use old checkpoints with different filename scheme")
    
    args = parser.parse_args()
    
    newness = "old" if args.old else "new"

    for N in [10000, 25000]:
        csv_path = get_project_root() / f"results/loss_over_time_{N}_{newness}.csv"
        
        if csv_path.exists():
            # Load existing results
            results_df = pd.read_csv(csv_path)
        else:
            # Calculate new results
            results = []
            for min_order in [1, 2]:
                for max_order in range(min_order, 3):
                    for params in [200_000, 300_000, 500_000, 800_000, 1_000_000, 1_500_000, 2_000_000, 5_000_000]:
                        for wd in [0.01, 0.1]:
                                results.append(evaluate_checkpoints(
                                    N=N,
                                    min_order=min_order,
                                    max_order=max_order,
                                    params=params,
                                    wd=wd,
                                    num_checkpoints=args.num_checkpoints,
                                    num_samples=args.num_samples,
                                    old=args.old
                                ))
            results_df = pd.concat([pd.DataFrame(r) for r in results])

        if len(results_df) > 0:
            save_path = get_project_root() / f"results/loss_over_time_{N}_{newness}.png"
            plot_losses(results_df, save_path)

if __name__ == "__main__":
    main()
