import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_reasoning.evaluation.eval_utils import load_eval_results
from transformer_reasoning.utils import get_project_root
from transformer_reasoning.generate_dataset.generate_qa_dataset import RELATIONS
import numpy as np
from datasets import load_dataset

def calculate_uniform_loss(N, relations=None):
    """Calculate average negative log of uniform distribution over answer sets."""
    # Load dataset
    rel_str = f"_r{relations}" if relations else ""
    profiles_dataset = load_dataset(f"EleutherAI/profiles_dataset_{N}_uniform{rel_str}")['train']
    
    # Get unique values for each attribute
    attribute_sizes = {
        'birth_city': len(set(profiles_dataset['birth_city'])),
        'university': len(set(profiles_dataset['university'])),
        'employer': len(set(profiles_dataset['employer'])),
        'birth_date': len(set(profiles_dataset['birth_date'])),
    }
    
    # Add relationship attributes
    for rel in RELATIONS:
        if rel in profiles_dataset.features:
            attribute_sizes[rel] = N 
    
    # Calculate average uniform loss
    uniform_loss = np.mean([np.log(size) for size in attribute_sizes.values()])
    return uniform_loss

def plot_hop_comparison(eval_results, N, n_params, args):
    """Compare 1-hop vs 2-hop trained models' performance."""
    df = eval_results[(eval_results['N_profiles'] == N) & (eval_results['n_params'] == n_params)]

    if len(df) == 0:
        print(f"No data found for N={N}, n_params={n_params}")
        return
    
    layers = df['layers'].unique()[0]
    plt.figure(figsize=(10, 6))
    
    # Plot 1-hop trained models (1-hop evaluation)
    one_hop = df[(df['max_train_hops'] == 1) & (df['hops'] == 1) & (df['currency'] == 'current')].sort_values('global_step')
    plt.loglog(one_hop['global_step'], one_hop['loss'], 
             label=f'1-hop trained (1-hop eval) (layers={layers})', alpha=0.8)
    
    # Plot 2-hop trained models (2-hop evaluation)
    two_hop = df[(df['max_train_hops'] == 2) & (df['hops'] == 2) & (df['currency'] == 'current')].sort_values('global_step')
    plt.loglog(two_hop['global_step'], two_hop['loss'], 
             label=f'2-hop trained (2-hop eval) (layers={layers})', alpha=0.8)
    
    # Add uniform loss baseline
    uniform_loss = calculate_uniform_loss(N, args.relations)
    plt.axhline(y=uniform_loss, color='r', linestyle='--', label='Uniform Loss', alpha=0.5)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'1-hop vs 2-hop Training (N={N}, n_params={n_params})')
    plt.legend()
    plt.savefig(get_project_root() / f'results/plots/hop_comparison_N{N}_p{n_params}.png')
    plt.close()

def plot_one_hop_comparison(eval_results, N, n_params, args):
    """Compare 1-hop evaluation of 1-hop vs 2-hop trained models."""
    df = eval_results[(eval_results['N_profiles'] == N) & (eval_results['n_params'] == n_params)]
    if len(df) == 0:
        print(f"No data found for N={N}, n_params={n_params}")
        return
    

    layers = df['layers'].unique()[0]

    plt.figure(figsize=(10, 6))
    
    # Plot 1-hop trained models (1-hop evaluation)
    one_hop = df[(df['max_train_hops'] == 1) & (df['hops'] == 1)].sort_values('global_step')
    plt.loglog(one_hop['global_step'], one_hop['loss'], 
             label=f'1-hop trained (layers={layers})', alpha=0.8)
    
    # Plot 2-hop trained models (1-hop evaluation)
    two_hop = df[(df['max_train_hops'] == 2) & (df['hops'] == 1)].sort_values('global_step')
    plt.loglog(two_hop['global_step'], two_hop['loss'], 
             label=f'2-hop trained (layers={layers})', alpha=0.8)
    
    # Add uniform loss baseline
    uniform_loss = calculate_uniform_loss(N, args.relations)
    plt.axhline(y=uniform_loss, color='r', linestyle='--', label='Uniform Loss', alpha=0.5)

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'1-hop Evaluation Comparison (N={N}, n_params={n_params})')
    plt.legend()
    plt.savefig(get_project_root() / f'results/plots/one_hop_comparison_N{N}_p{n_params}.png')
    plt.close()

def plot_scale_comparison(eval_results, N, mode, args):
    """Compare different model scales for fixed N and hop count."""
    df = eval_results[(eval_results['N_profiles'] == N) & 
                     (eval_results['mode'] == mode) &
                     (eval_results['max_train_hops'] == 2)]
    if len(df) == 0:
        print(f"No data found for N={N}, mode={mode}")
        return
    
    plt.figure(figsize=(10, 6))
    
    for n_params in sorted(df['n_params'].unique()):
        for currency in df['currency'].unique():
            data = df[(df['n_params'] == n_params) & (df['currency'] == currency)].sort_values('global_step')
            if len(data) == 0:
                continue
            layers = data['layers'].unique()[0]
            plt.loglog(data['global_step'], data['loss'], 
                    label=f'{n_params/1e6:.1f}M params (layers={layers}, currency={currency})', alpha=0.8)
    
    # Add uniform loss baseline
    uniform_loss = calculate_uniform_loss(N, args.relations)
    plt.axhline(y=uniform_loss, color='r', linestyle='--', label='Uniform Loss', alpha=0.5)

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'{mode} Evaluation Across Scales (N={N})')
    plt.legend()
    plt.savefig(get_project_root() / f'results/plots/scale_comparison_N{N}_mode{mode}.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relations", type=int, default=None)
    args = parser.parse_args()

    # Load evaluation results
    eval_results = load_eval_results(skip_mode=False)
    
    if args.relations is not None:
        eval_results = eval_results[eval_results['relations'] == args.relations]
    else:
        eval_results = eval_results[eval_results['relations'].isna()]

    # Get unique N values
    N_sizes = sorted(eval_results['N_profiles'].unique())
    n_params_sizes = sorted(eval_results['n_params'].unique())

    modes = eval_results['mode'].unique()

    for N in N_sizes:
        for n_params in n_params_sizes:
            plot_hop_comparison(eval_results, N, n_params, args)
            plot_one_hop_comparison(eval_results, N, n_params, args)
        for mode in modes:
            plot_scale_comparison(eval_results, N, mode, args)

if __name__ == "__main__":
    main()
