import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import seaborn as sns
from transformers import AutoTokenizer
from transformer_reasoning.models.llama_mup import LlamaMuPForCausalLM
from transformer_reasoning.train.train_utils import (
    set_model_base_shapes,
    InfiniteQADataset,
    evaluate_single_model
)
from transformer_reasoning.evaluation.eval_utils import (
    get_checkpoints,
    evaluate_model_histograms
)
from transformer_reasoning.evaluation.qa_evaluation import evaluate_qa_loss
from transformer_reasoning.generate_dataset.generate_qa_dataset import ATTRIBUTES, get_available_relations
from transformer_reasoning.evaluation.eval_utils import load_eval_results
import pandas as pd
from pandas import DataFrame
from transformer_reasoning.utils import get_project_root
from datasets import load_dataset
import argparse
import os

def get_checkpoint_paths(model_dir: str, num_checkpoints: int = 20, timesteps_to_keep_path: Optional[str] = None) -> List[Path]:
    """Get evenly spaced checkpoint paths from a directory."""
    all_checkpoints = list(model_dir.glob("checkpoint-*"))
    # If timesteps file exists, filter checkpoints to only those timesteps
    if timesteps_to_keep_path and Path(timesteps_to_keep_path).exists():
        timesteps_df = pd.read_csv(timesteps_to_keep_path)
        valid_steps = set(timesteps_df['global_step'].astype(int))
        all_checkpoints = [
            cp for cp in all_checkpoints 
            if int(str(cp).split('-')[-1]) in valid_steps
        ]
        return all_checkpoints
    
    if len(all_checkpoints) <= num_checkpoints:
        return all_checkpoints
    
    # Get evenly spaced indices
    indices = np.linspace(0, len(all_checkpoints)-1, num_checkpoints, dtype=int)
    return [str(all_checkpoints[i]) for i in indices]

def evaluate_checkpoints(
    N: int,
    min_order: int,
    max_order: int,
    params: int,
    wd: float,
    layer: int,
    compute_histograms: bool = False,
    rel_str: Optional[str] = None,
    latest_only: bool = False,
    commit_hash: Optional[str] = None,
    output_dir: Optional[str] = "."
) -> Union[DataFrame, Tuple[DataFrame, Dict[str, List[float]]]]:
    """Evaluate loss for different tasks across checkpoints."""
    # Load existing results if they exist
    if commit_hash is not None:
        results_path = f'./results/{commit_hash}/mup_n{N}_p{params}_omin{min_order}_omax{max_order}_wd{wd}_l{layer}_lr0.001_beta10.99_sf{rel_str}/eval_results_full.csv'
        save_path = f'./results/{commit_hash}/mup_n{N}_p{params}_omin{min_order}_omax{max_order}_wd{wd}_l{layer}_lr0.001_beta10.99_sf{rel_str}/eval_results_old.csv'
    else:
        results_path = f'./results/mup_n{N}_p{params}_omin{min_order}_omax{max_order}_wd{wd}_l{layer}_lr0.001_beta10.99_sf{rel_str}/eval_results_full.csv'
        save_path = f'./results/mup_n{N}_p{params}_omin{min_order}_omax{max_order}_wd{wd}_l{layer}_lr0.001_beta10.99_sf{rel_str}/eval_results_old.csv'
    existing_results = pd.DataFrame()
    if Path(results_path).exists():
        existing_results = pd.read_csv(results_path)
        last_step = existing_results['global_step'].max() if len(existing_results) > 0 else 0
    else:
        last_step = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Update checkpoint filtering to use timesteps_to_keep.csv if it exists
    timesteps_to_keep_path = Path(output_dir) / f'results/timesteps_to_keep{rel_str}.csv'
    
    checkpoints = get_checkpoints(min_order, max_order, N, params, wd, commit_hash, rel_str, layers=layer)
    checkpoint_dirs = [Path(x) for x in checkpoints if os.path.isdir(x)]
    checkpoint_paths = get_checkpoint_paths(checkpoint_dirs[0].parent, timesteps_to_keep_path=timesteps_to_keep_path)   
    if not checkpoint_paths:
        return existing_results if not existing_results.empty else None

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llama_multihop_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    profiles_dataset = load_dataset(f"EleutherAI/profiles_dataset_{N}_uniform{rel_str}")['train']
    
    subjects = ATTRIBUTES + get_available_relations(profiles_dataset[0])

    if compute_histograms or latest_only:
        sorted_checkpoints = sorted(checkpoint_dirs, key=lambda x: int(str(x).split('-')[-1]), reverse=True)
        for checkpoint in sorted_checkpoints:
            try:
                optimizer_state = torch.load(f"{checkpoint}/optimizer.pt")
                checkpoint_paths = [str(checkpoint)]
                break
            except (RuntimeError, EOFError, FileNotFoundError):
                continue
        else:
            print("No valid checkpoints with optimizer.pt found")
            return None

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
   
    # Load optimizer state to get heldout sets
    optimizer_state = torch.load(f"{checkpoint_paths[0]}/optimizer.pt")
    heldout_sets = optimizer_state.get('heldout_sets')
    
    # Define eval modes based on training orders
    if 2 in range(min_order, max_order + 1):
        eval_modes = [
            "train_onehop",
            "train_twohop",
            "eval_first_people",
            "eval_relations",
            "eval_person_relation_pairs",
            "eval_second_people",
            "eval_second_attributes",
            "eval_second_person_attribute_pairs",
            "eval_complete_two_hop_questions"
        ]
    else:
        eval_modes = ["train_onehop"]
    
    for subject in subjects:
        # Create evaluation datasets for each mode
        eval_datasets = {}
        for mode in eval_modes:
            try:
                eval_datasets[mode] = InfiniteQADataset(
                    profiles_dataset=profiles_dataset,
                    tokenizer=tokenizer,
                    max_seq_len=512,
                    orders=[1] if mode == "train_onehop" else [2],
                    mode=mode.replace("train_onehop", "train").replace("train_twohop", "train"),
                    subjects=[subject],
                    heldout_sets=heldout_sets
                )
            except ValueError as e:
                print(f"Skipping {mode} for subject {subject}: {e}")
                continue
        
        if not eval_datasets:
            print(f"No valid evaluation modes for subject {subject}, skipping")
            continue
        
        eval_loaders = {
            mode: DataLoader(
                dataset,
                batch_size=16,
                shuffle=False,
                num_workers=15,
                pin_memory=True
            ) for mode, dataset in eval_datasets.items()
        }
        
        results_dicts = []
        for checkpoint_path in checkpoint_paths:
            if 'eval_results.csv' not in checkpoint_path:
                print(f"Evaluating checkpoint: {checkpoint_path}, subject: {subject}")
                    
                step = int(checkpoint_path.split('-')[-1])
                
                model = LlamaMuPForCausalLM.from_pretrained(checkpoint_path).to(device)
                set_model_base_shapes(model, layer, tokenizer, restore_from_checkpoint=True)
                
                if compute_histograms:
                    onehop_and_twohop = evaluate_model_histograms(model, eval_loaders['train_onehop'], eval_loaders['train_twohop'])
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
                    # Evaluate all modes
                    for mode, loader in eval_loaders.items():
                        results_dict = evaluate_single_model(model, loader, step, mode)
                        results_dict['subject'] = subject
                        results_dict['n_params'] = params
                        results_dict['N_profiles'] = N
                        results_dict['min_train_hops'] = min_order
                        results_dict['max_train_hops'] = max_order
                        results_dict['wd'] = wd
                        results_dicts.append(results_dict)
                
                del model
                torch.cuda.empty_cache()
        
        results = pd.DataFrame(results_dicts)
        results['subject'] = subject
        results['n_params'] = params
        results['N_profiles'] = N
        results['min_train_hops'] = min_order
        results['max_train_hops'] = max_order
        results['wd'] = wd

        results.to_csv(save_path, index=False)
        
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
    parser.add_argument("--relations", type=int, default=None)
    parser.add_argument("--skip_mode", action="store_true")
    parser.add_argument("--commit_hashes", nargs="+", default=[])
    parser.add_argument('--compute_histograms', action='store_true')
    parser.add_argument('--latest_only', action='store_true')
    parser.add_argument("--no_mup", action="store_true")
    parser.add_argument("--base_path", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    # Load eval results to get model configurations
    eval_results = load_eval_results(skip_mode=args.skip_mode, 
                                     commit_hashes=args.commit_hashes, 
                                     base_path=args.base_path,
                                     no_mup=args.no_mup)
    rel_str = f"_r{args.relations}" if args.relations and args.relations != 4 else ""

    if 'subject' not in eval_results.columns:
        eval_results['subject'] = np.nan

    if args.relations is not None:
        eval_results = eval_results[eval_results['relations'] == args.relations]
    else:
        eval_results = eval_results[eval_results['relations'].isna()]

    # Group by unique model configurations
    model_configs = eval_results.groupby([
        'N_profiles', 
        'n_params', 
        'min_train_hops', 
        'max_train_hops', 
        'weight_decay',
        'layers'
    ]).first().reset_index()

    full_results = []
    all_histograms = []

    # Evaluate each model configuration
    for _, config in model_configs.iterrows():
        result = evaluate_checkpoints(
            N=config['N_profiles'],
            min_order=config['min_train_hops'],
            max_order=config['max_train_hops'],
            params=config['n_params'],
            wd=config['weight_decay'],
            layer=config['layers'],
            commit_hash=config['commit_hash'],
            compute_histograms=args.compute_histograms,
            rel_str=rel_str,
            latest_only=args.latest_only,
            output_dir=args.output_dir
        )
        
        if result is not None:
            if args.compute_histograms:
                df, histograms = result
                hist_df = pd.DataFrame(histograms)
                save_path = f'./results/{config["commit_hash"]}/'\
                    + f'mup_n{config["N_profiles"]}_p{config["n_params"]}_'\
                    + f'omin{config["min_train_hops"]}_omax{config["max_train_hops"]}_'\
                    + f'wd{config["weight_decay"]}_l{config["layers"]}_'\
                    + f'lr0.001_beta10.99_sf{rel_str}'
                Path(save_path).mkdir(parents=True, exist_ok=True)
                hist_df.to_csv(f'{save_path}/eval_results_histograms.csv')
                all_histograms.append(hist_df)
            else:
                save_path = f'./results/{config["commit_hash"]}/'\
                    + f'mup_n{config["N_profiles"]}_p{config["n_params"]}_'\
                    + f'omin{config["min_train_hops"]}_omax{config["max_train_hops"]}_'\
                    + f'wd{config["weight_decay"]}_l{config["layers"]}_'\
                    + f'lr0.001_beta10.99_sf{rel_str}'
                Path(save_path).mkdir(parents=True, exist_ok=True)
                result.to_csv(f'{save_path}/eval_results_full.csv')
                full_results.append(result)

if __name__ == "__main__":
    main()
