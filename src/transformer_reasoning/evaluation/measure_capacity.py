import argparse
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset

from transformer_reasoning.utils import get_project_root
from transformer_reasoning.evaluation.measure_entropy import calculate_entropy
from transformer_reasoning.evaluation.eval_utils import load_eval_results

import seaborn as sns

def calculate_capacities(total_losses, entropies, name_selection_entropy, birth_date_selection_entropy, N, hops, scheme):
    """Calculate per-attribute and 2nd order QA capacities."""
    # Collect attribute values for entropy calculation    
    num_relations = 4
    qa_multiplier = 1
    qa_capacity = {}

    if scheme == "2-hop-big-hash" and hops == 2:
        qa_multiplier = num_relations

    for attr, entropy in entropies.items():
        label = f"{attr}_capacity"
        qa_loss = total_losses[attr]
        if attr in ['parent', 'child', 'best_friend', 'worst_enemy']:
            if entropy > qa_loss:
                qa_capacity[label] = (entropy - qa_loss) * qa_multiplier * N + name_selection_entropy/num_relations
            else:
                excess_loss = qa_loss - entropy
                qa_capacity[label] = (name_selection_entropy - excess_loss * N)/num_relations
        elif attr == 'birth_date':
            if entropy > qa_loss:
                qa_capacity[label] = (entropy - qa_loss) * qa_multiplier * N + birth_date_selection_entropy
            else:
                excess_loss = qa_loss - entropy
                qa_capacity[label] = birth_date_selection_entropy - excess_loss * N
        else:
            qa_capacity[label] = (entropy - qa_loss) * qa_multiplier * N

    qa_capacity['total_capacity'] = sum(qa_capacity.values())

    return qa_capacity

def get_nearest_param_color(param_count):
    all_param_sizes = np.logspace(np.log10(500_000), np.log10(10_000_000), 20)  # 20 discrete colors
    param_colors = plt.cm.viridis(np.linspace(0, 1, len(all_param_sizes)))
    idx = np.abs(all_param_sizes - param_count).argmin()
    return param_colors[idx]

def normalized_capacity_plot(df, scheme):
    plt.figure(figsize=(10, 6))
    
    order_markers = {1: 'o', 2: '^'}
    
    # Calculate capacity per parameter
    df['capacity_per_param'] = df['capacity'] / df['n_params']
    
    # Group by model configuration
    for _, group in df.groupby(['n_params', 'N_profiles', 'max_train_hops', 'weight_decay']):
        hop1_data = group[group['hops'] == 1]['capacity_per_param'].iloc[0]
        hop2_data = group[group['hops'] == 2]['capacity_per_param'].iloc[0]
        
        plt.scatter(hop1_data, hop2_data,
                   marker=order_markers[group['max_train_hops'].iloc[0]],
                   color=get_nearest_param_color(group['n_params'].iloc[0]),
                   label=f"N={group['N_profiles'].iloc[0]}, params={group['n_params'].iloc[0]}, "
                         f"hops={group['max_train_hops'].iloc[0]}, wd={group['weight_decay'].iloc[0]}")

    # Add reference line y = 2-x
    x = np.linspace(0, 2, 100)
    plt.plot(x, 2-x, '--', color='gray', label='y = 2-x')

    plt.xlabel('1 Hop Capacity / Number of Parameters')
    plt.ylabel('2 hop Capacity / Number of Parameters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('1 Hop vs 2 Hop Capacity')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(get_project_root() / f'results/capacity_plot_{scheme}.png', bbox_inches='tight')
    plt.close()

def get_closest_step_data(df, target, use_normalized=True):
    """For each group, get the row with step count closest to target."""
    step_col = 'normalized_step' if use_normalized else 'global_step'
    return df.iloc[(df[step_col] - target).abs().argsort()].iloc[0]

def cap_vs_params_plot(df, N_sizes, use_normalized_step=True, scheme=None):
    # Get target from N=10000 data
    step_col = 'normalized_step' if use_normalized_step else 'global_step'
    target = df[df['N_profiles'] == 50000][step_col].max()
    
    # Filter to closest steps for each config
    filtered_df = df.groupby(['N_profiles', 'n_params', 'max_train_hops', 'weight_decay', 'hops']).apply(
        lambda x: get_closest_step_data(x, target, use_normalized_step)
    ).reset_index(drop=True)

    # Rest of plotting code remains the same
    for N in N_sizes:
        plt.figure(figsize=(10, 6))
        plot_df = filtered_df[filtered_df['N_profiles'] == N]

        fig, axes = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns
        for max_ord, ax in zip([1, 2], axes):
            data = plot_df[plot_df['max_train_hops'] == max_ord]
            sns.scatterplot(data=data, x='n_params', y='capacity', style='hops', ax=ax)
            sns.lineplot(data=data, x='n_params', y='capacity', style='hops', ax=ax)
            ax.set_title(f'Max Order = {max_ord}')

        plt.savefig(get_project_root() / f'results/capacity_vs_params_N{N}_{scheme}.png', bbox_inches='tight')

def cap_vs_N_plot(df, scheme=None):
    filtered_df = df.groupby(['N_profiles', 'n_params', 'max_train_hops', 'weight_decay', 'hops']).last().reset_index(drop=False)
    filtered_df = filtered_df[filtered_df['global_step'] >= 2500000]
    param_sizes = filtered_df['n_params'].unique()
    
    attributes = ['parent', 'child', 'best_friend', 'worst_enemy', 
                 'birth_date', 'birth_city', 'employer', 'university']
    
    for param_size in param_sizes:
        # Create a tall figure with 9 rows and 2 columns
        fig, axes = plt.subplots(9, 2, figsize=(20, 40), sharey='row')
        plt.subplots_adjust(hspace=0.4)  # Add more space between rows
        
        plot_df = filtered_df[filtered_df['n_params'] == param_size]
        
        # Plot total capacity in first row
        for max_ord, ax in zip([1, 2], axes[0]):
            data = plot_df[plot_df['max_train_hops'] == max_ord]
            sns.scatterplot(data=data, x='N_profiles', y='capacity', style='hops', ax=ax)
            sns.lineplot(data=data, x='N_profiles', y='capacity', style='hops', ax=ax)
            ax.set_title(f'Total Capacity - Max Train Hops = {max_ord}')
            ax.set_xlabel('N profiles')
            ax.set_ylabel('Capacity (bits)')
        
        # Plot individual attributes
        for idx, attr in enumerate(attributes, start=1):
            capacity_col = f'{attr}_capacity'
            for max_ord, ax in zip([1, 2], axes[idx]):
                data = plot_df[plot_df['max_train_hops'] == max_ord]
                sns.scatterplot(data=data, x='N_profiles', y=capacity_col, style='hops', ax=ax)
                sns.lineplot(data=data, x='N_profiles', y=capacity_col, style='hops', ax=ax)
                ax.set_title(f'{attr.replace("_", " ").title()} Capacity - Max Order = {max_ord}')
                ax.set_xlabel('N profiles')
                ax.set_ylabel('Capacity (bits)')
        
        plt.suptitle(f'Capacity vs N (params={param_size})', y=0.995, fontsize=16)
        plt.savefig(get_project_root() / f'results/capacity_vs_N_params{param_size}_{scheme}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

def cap_vs_norm_plot(df, scheme=None):
    """Plot capacity vs parameter norm using all timestep data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1-hop capacity
    sns.scatterplot(
        data=df[df['hops'] == 1],
        x='parameter_norm',
        y='capacity',
        hue='max_train_hops',
        style='N_profiles',
        alpha=0.5,
        ax=ax1
    )
    sns.regplot(
        data=df[df['hops'] == 1],
        x='parameter_norm',
        y='capacity',
        scatter=False,
        ax=ax1,
        color='gray',
        line_kws={'linestyle': '--'}
    )
    ax1.set_title('1-hop Capacity vs Parameter Norm')
    ax1.set_xlabel('Parameter L2 Norm')
    ax1.set_ylabel('1-hop Capacity (bits)')
    
    # Plot 2-hop capacity
    sns.scatterplot(
        data=df[df['hops'] == 2],
        x='parameter_norm',
        y='capacity',
        hue='max_train_hops',
        style='N_profiles',
        alpha=0.5,
        ax=ax2
    )
    sns.regplot(
        data=df[df['hops'] == 2],
        x='parameter_norm',
        y='capacity',
        scatter=False,
        ax=ax2,
        color='gray',
        line_kws={'linestyle': '--'}
    )
    ax2.set_title('2-hop Capacity vs Parameter Norm')
    ax2.set_xlabel('Parameter L2 Norm')
    ax2.set_ylabel('2-hop Capacity (bits)')
    
    plt.tight_layout()
    plt.savefig(get_project_root() / f'results/capacity_vs_norm_{scheme}.png', bbox_inches='tight')
    plt.close()

def capacity_plot(df, N_sizes, entropies_optimal, name_selection_entropy_optimal, birth_date_selection_entropy_optimal):
    order_markers = {1: 'o', 2: '^'}

    for N in N_sizes:
        plt.figure(figsize=(10, 6))

        # Calculate theoretical max capacities using calculate_capacities with zero losses
        zero_losses = [0.0, 1]  # Single loss value for calculate_capacities
        max_capacity = calculate_capacities(zero_losses, entropies_optimal, 
                                         name_selection_entropy_optimal, 
                                         birth_date_selection_entropy_optimal, 
                                         N, 1)['total_capacity']
        
        N_results = df[df['N_profiles'] == N]
        
        # Group by model configuration and plot
        for _, group in N_results.groupby(['n_params', 'max_train_hops', 'weight_decay']):
            hop1_data = group[group['hops'] == 1]['capacity'].iloc[0]
            hop2_data = group[group['hops'] == 2]['capacity'].iloc[0]
            
            plt.scatter(hop1_data, hop2_data,
                       marker=order_markers[group['max_train_hops'].iloc[0]],
                       color=get_nearest_param_color(group['n_params'].iloc[0]),
                       label=f"params={group['n_params'].iloc[0]}, "
                             f"order={group['max_train_hops'].iloc[0]}, "
                             f"wd={group['weight_decay'].iloc[0]}")

        # Add reference lines for max capacities
        plt.axvline(x=max_capacity, color='gray', linestyle='--', alpha=0.5, 
                   label=f'Max Capacity ({max_capacity/1e6:.1f}M)')
        plt.axhline(y=max_capacity, color='gray', linestyle='--', alpha=0.5)

        plt.xlabel('1-hop Capacity (bits)')
        plt.ylabel('2-hop Capacity (bits)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'1-hop vs 2-hop Capacity (N={N})')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(get_project_root() / f'results/capacity_plot_N{N}.png', bbox_inches='tight')
        plt.close()

def process_timestep_data(eval_results, N, entropies, name_selection_entropy, birth_date_selection_entropy, scheme):
    """Calculate capacities for each timestep in the loss data."""
    results = []
    filtered_eval_results = eval_results[eval_results['N_profiles'] == N]
    # Group by model configuration
    for (num_parameters, min_order, max_order, wd, eval_hops, global_step), group in filtered_eval_results.groupby(
        ['n_params', 'min_train_hops', 'max_train_hops', 'weight_decay', 'hops', 'global_step']
    ):
        total_losses = {}
        for _, row in group.iterrows():
            subject = row['subject']
            total_losses[subject] = row['loss'] / np.log(2)

        total_losses['all_questions'] = group['loss'].mean() / np.log(2)
        # Calculate capacities
        capacities = calculate_capacities(
            total_losses, 
            entropies, 
            name_selection_entropy, 
            birth_date_selection_entropy, 
            N,
            eval_hops,
            scheme
        )
        
        result = {
            'n_params': num_parameters,
            'N_profiles': N,
            'min_train_hops': min_order,
            'max_train_hops': max_order,
            'hops': eval_hops,
            'weight_decay': wd,
            'global_step': global_step,
            'parameter_norm': row['parameter_l2'],
            'capacity': capacities['total_capacity'],
        }

        result.update(capacities)

        results.append(result)

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--scheme", type=str, choices=["optimal", "2-hop-big-hash"], default="optimal")
    args = parser.parse_args()

    eval_results = load_eval_results()

    N_sizes = list(set(eval_results['N_profiles']))

    all_timestep_results = []
    final_results = []

    for N in N_sizes:
        
        # Load and process dataset
        profiles_dataset = load_dataset(f"EleutherAI/profiles_dataset_{N}_uniform")['train']

        attribute_values = {
            'name': list(set(profiles_dataset['name'])),
            'birth_date': list(set(profiles_dataset['birth_date'])),
            'birth_city': list(set(profiles_dataset['birth_city'])),
            'university': list(set(profiles_dataset['university'])),
            'employer': list(set(profiles_dataset['employer'])),
            'child': list(set(profile['name'] for profile in profiles_dataset['child'])),
            'parent': list(set(profile['name'] for profile in profiles_dataset['parent'])),
            'best_friend': list(set(profile['name'] for profile in profiles_dataset['best_friend'])),
            'worst_enemy': list(set(profile['name'] for profile in profiles_dataset['worst_enemy']))
        }

        # Calculate entropies
        entropies = {attr: calculate_entropy(values, attr, scheme=args.scheme, n_profiles=N) 
                    for attr, values in attribute_values.items() 
                    if attr != 'name'}
        name_selection_entropy = calculate_entropy(attribute_values['name'], 'name', selection=True, scheme=args.scheme, n_profiles=N)
        birth_date_selection_entropy = calculate_entropy(attribute_values['birth_date'], 'birth_date', selection=True, scheme=args.scheme, n_profiles=N)

        # Process all timesteps
        timestep_results = process_timestep_data(
            eval_results, 
            N, 
            entropies, 
            name_selection_entropy, 
            birth_date_selection_entropy,
            args.scheme
        )
        all_timestep_results.append(timestep_results)
        # Get final timestep results for each configuration
        final_timesteps = timestep_results.groupby(
            ['n_params', 'min_train_hops', 'max_train_hops', 'weight_decay', 'hops']
        ).last().reset_index()
        final_results.append(final_timesteps)


    # Combine results
    all_timestep_df = pd.concat(all_timestep_results, ignore_index=True)
    final_df = pd.concat(final_results, ignore_index=True)
    final_df = final_df[final_df['global_step'] >= 2500000]

    # Save all timestep results
    all_timestep_df.to_csv(get_project_root() / f'results/capacity_results_all_timesteps_{args.scheme}.csv', index=False)
    final_df.to_csv(get_project_root() / f'results/capacity_results_{args.scheme}.csv', index=False)

    max_steps = all_timestep_df.groupby(['n_params', 'min_train_hops', 'max_train_hops', 'weight_decay'])['global_step'].max()
    # Adjust step counts if max steps > 5M
    def adjust_steps(row):
        group_key = (row['n_params'], row['min_train_hops'], row['max_train_hops'], row['weight_decay'])
        steps = row['global_step']
        if max_steps[group_key] > 5_000_000:
            steps = steps / 2
        return steps / N
        
    all_timestep_df['normalized_step'] = all_timestep_df.apply(adjust_steps, axis=1)

    # Generate plots
    # normalized_capacity_plot(final_df, args.scheme)
    # cap_vs_params_plot(all_timestep_df, N_sizes, scheme=args.scheme)
    cap_vs_N_plot(all_timestep_df, scheme=args.scheme)
    
    # Plot capacity vs norm using all timesteps
    # cap_vs_norm_plot(all_timestep_df, scheme=args.scheme)

    


    

if __name__ == "__main__":
    main()
