import argparse
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset

from transformer_reasoning.utils import get_project_root
from transformer_reasoning.evaluation.measure_entropy import calculate_entropy


import seaborn as sns

def calculate_capacities(total_losses, entropies, name_selection_entropy, birth_date_selection_entropy, N):
    """Calculate per-attribute and 2nd order QA capacities."""
    # Collect attribute values for entropy calculation    
    
    avg_entropy_per_profile = (sum(entropies) + name_selection_entropy/N + birth_date_selection_entropy/N)

    qa_loss_2 = total_losses[2][0] / total_losses[2][1]  # Using all_tokens_loss
    qa_loss_1 = total_losses[1][0] / total_losses[1][1]  # Using all_tokens_loss
    qa_capacity_2 = max(0, (avg_entropy_per_profile - qa_loss_2 * len(entropies)) * N)
    qa_capacity_1 = max(0, (avg_entropy_per_profile - qa_loss_1 * len(entropies)) * N)
    
    print("\n2 hop QA capacity:")
    print(f"Average entropy per Q (excluding name): {avg_entropy_per_profile/len(entropies):.4f}")
    print(f"2 hop QA loss: {qa_loss_2:.4f}")
    print(f"1 hop QA loss: {qa_loss_1:.4f}")
    print(f"2 hop QA capacity: {qa_capacity_2:.4f}")
    print(f"1 hop QA capacity: {qa_capacity_1:.4f}")

    return qa_capacity_2, qa_capacity_1

def get_nearest_param_color(param_count):
    all_param_sizes = np.logspace(np.log10(500_000), np.log10(10_000_000), 20)  # 20 discrete colors
    param_colors = plt.cm.viridis(np.linspace(0, 1, len(all_param_sizes)))
    idx = np.abs(all_param_sizes - param_count).argmin()
    return param_colors[idx]

def normalized_capacity_plot(df):
      # Create plot
    plt.figure(figsize=(10, 6))
    
    # Define markers for different orders
    order_markers = {1: 'o', 2: '^'}
    


    
    df['qa_1_per_param_optimal'] = df['qa_capacity_1_optimal'] / df['num_params']
    df['qa_2_per_param_optimal'] = df['qa_capacity_2_optimal'] / df['num_params']
    for _, row in df.iterrows():
        plt.scatter(row['qa_1_per_param_optimal'], row['qa_2_per_param_optimal'],
                    marker=order_markers[row['max_order']], 
                    color=get_nearest_param_color(row['num_params']),
                    label=f"N={row['N']}, params={row['num_params']}, hops={row['max_order']}, wd={row['weight_decay']}")

    # Add reference line y = 2-x
    x = np.linspace(0, 2, 100)
    plt.plot(x, 2-x, '--', color='gray', label='y = 2-x')

    plt.xlabel('1 Hop Capacity / Number of Parameters')
    plt.ylabel('2 hop Capacity / Number of Parameters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('1 Hop vs 2 Hop Capacity')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(get_project_root() / f'results/capacity_plot.png', bbox_inches='tight')
    plt.close()

def get_closest_step_data(df, target, use_normalized=True):
    """For each group, get the row with step count closest to target."""
    step_col = 'normalized_step' if use_normalized else 'steps'
    return df.iloc[(df[step_col] - target).abs().argsort()].iloc[0]

def cap_vs_params_plot(df, N_sizes, use_normalized_step=True):
    # Get target from N=10000 data
    step_col = 'normalized_step' if use_normalized_step else 'steps'
    target = df[df['N'] == 50000][step_col].max()
    
    # Filter to closest steps for each config
    filtered_df = df.groupby(['N', 'num_params', 'max_order', 'weight_decay']).apply(
        lambda x: get_closest_step_data(x, target, use_normalized_step)
    ).reset_index(drop=True)

    # Rest of plotting code remains the same
    for N in N_sizes:
        plt.figure(figsize=(10, 6))
        plot_df = filtered_df[filtered_df['N'] == N]
        plot_df = pd.melt(
            plot_df,
            id_vars=['num_params', 'N', 'min_order', 'max_order', 'weight_decay'],
            value_vars=['qa_capacity_1_optimal',
                        'qa_capacity_2_optimal'],
            var_name='metric',
            value_name='capacity'
        )

        plot_df['hops'] = plot_df['metric'].str.extract('qa_capacity_(\d+)_').astype(int)
        plot_df['scheme'] = plot_df['metric'].str.extract('qa_capacity_\d+_(\w+)')

        fig, axes = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns
        for max_ord, ax in zip([1, 2], axes):
            data = plot_df[plot_df['max_order'] == max_ord]
            sns.scatterplot(data=data, x='num_params', y='capacity', hue='scheme', style='hops', ax=ax)
            sns.lineplot(data=data, x='num_params', y='capacity', hue='scheme', style='hops', ax=ax)
            ax.set_title(f'Max Order = {max_ord}')

        plt.savefig(get_project_root() / f'results/capacity_vs_params_N{N}.png', bbox_inches='tight')

def cap_vs_N_plot(df, use_normalized_step=True):
    # Get target from N=10000 data
    step_col = 'normalized_step' if use_normalized_step else 'steps'
    target = df[df['N'] == 50000][step_col].max()
    
    # Filter to closest steps for each config
    filtered_df = df.groupby(['N', 'num_params', 'max_order', 'weight_decay']).apply(
        lambda x: get_closest_step_data(x, target, use_normalized_step)
    ).reset_index(drop=True)

    param_sizes = filtered_df['num_params'].unique()
    for param_size in param_sizes:
        plt.figure(figsize=(10, 6))
        plot_df = filtered_df[filtered_df['num_params'] == param_size]
        plot_df = pd.melt(
            plot_df,
            id_vars=['num_params', 'N', 'min_order', 'max_order', 'weight_decay'],
            value_vars=['qa_capacity_1_optimal',
                        'qa_capacity_2_optimal',],
            var_name='metric',
            value_name='capacity'
        )

        plot_df['hops'] = plot_df['metric'].str.extract('qa_capacity_(\d+)_').astype(int)
        plot_df['scheme'] = plot_df['metric'].str.extract('qa_capacity_\d+_(\w+)')

        fig, axes = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns
        for max_ord, ax in zip([1, 2], axes):
            data = plot_df[plot_df['max_order'] == max_ord]
            sns.scatterplot(data=data, x='N', y='capacity', hue='scheme', style='hops', ax=ax)
            sns.lineplot(data=data, x='N', y='capacity', hue='scheme', style='hops', ax=ax)
            ax.set_title(f'Max Order = {max_ord}')

        plt.savefig(get_project_root() / f'results/capacity_vs_N_params{param_size}.png', bbox_inches='tight')

def cap_vs_norm_plot(df):
    """Plot capacity vs parameter norm using all timestep data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1-hop capacity
    sns.scatterplot(
        data=df,
        x='parameter_norm',
        y='qa_capacity_1_optimal',
        hue='max_order',
        style='N',
        alpha=0.5,
        ax=ax1
    )
    sns.regplot(
        data=df,
        x='parameter_norm',
        y='qa_capacity_1_optimal',
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
        data=df,
        x='parameter_norm',
        y='qa_capacity_2_optimal',
        hue='max_order',
        style='N',
        alpha=0.5,
        ax=ax2
    )
    sns.regplot(
        data=df,
        x='parameter_norm',
        y='qa_capacity_2_optimal',
        scatter=False,
        ax=ax2,
        color='gray',
        line_kws={'linestyle': '--'}
    )
    ax2.set_title('2-hop Capacity vs Parameter Norm')
    ax2.set_xlabel('Parameter L2 Norm')
    ax2.set_ylabel('2-hop Capacity (bits)')
    
    plt.tight_layout()
    plt.savefig(get_project_root() / 'results/capacity_vs_norm.png', bbox_inches='tight')
    plt.close()

def capacity_plot(df, N_sizes, entropies_optimal, name_selection_entropy_optimal, birth_date_selection_entropy_optimal, all_param_sizes):
    order_markers = {1: 'o', 2: '^'}

    for N in N_sizes:
            plt.figure(figsize=(10, 6))

            # Calculate theoretical max capacities using calculate_capacities with zero losses
            zero_losses = {
                1: [0.0, 1], 
                2: [0.0, 1],
            }
            max_qa_2_optimal, max_qa_1_optimal = calculate_capacities(zero_losses, entropies_optimal, name_selection_entropy_optimal, birth_date_selection_entropy_optimal, N)
            N_results = df[df['N'] == N]

            # Plot unnormalized capacities
            for _, result in N_results.iterrows():
                plt.scatter(result['qa_capacity_1_optimal'], result['qa_capacity_2_optimal'],
                        marker=order_markers[result['max_order']],
                        color=get_nearest_param_color(result['num_params']),
                        label=f"params={result['num_params']}, order={result['max_order']}, wd={result['weight_decay']}")
            
            profile_independent_entropy_optimal = name_selection_entropy_optimal + birth_date_selection_entropy_optimal
            # Add reference lines for max capacities
            plt.axvline(x=max_qa_1_optimal + profile_independent_entropy_optimal, color='gray', linestyle='--', alpha=0.5, 
                    label=f'Max Bio Capacity ({(max_qa_1_optimal + profile_independent_entropy_optimal)/1e6:.1f}M)')
            plt.axhline(y=max_qa_2_optimal, color='gray', linestyle='--', alpha=0.5,
                    label=f'Max QA Capacity ({max_qa_2_optimal/1e6:.1f}M)')
            
            plt.xlabel('Bio Capacity (bits)')
            plt.ylabel('QA Capacity (bits)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Bio Capacity vs QA Capacity (N={N})')
            plt.grid(True)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(get_project_root() / f'results/capacity_plot_N{N}.png', bbox_inches='tight')
            plt.close()

def process_timestep_data(loss_df, N, entropies_optimal, name_selection_entropy_optimal, birth_date_selection_entropy_optimal):
    """Calculate capacities for each timestep in the loss data."""
    results = []
    
    # Group by model configuration
    for (num_parameters, min_order, max_order, wd), group in loss_df.groupby(
        ['model_params', 'model_min_hops', 'model_max_hops', 'model_wd']
    ):
        # For each timestep
        for _, row in group.iterrows():
            # Initialize losses using CSV data
            total_losses = {
                1: [row['qa_1hop_loss'] / np.log(2), 1],  # Convert from nats to bits
                2: [row['qa_2hop_loss'] / np.log(2), 1]
            }
            
            # Calculate capacities
            qa_capacity_2_optimal, qa_capacity_1_optimal = calculate_capacities(
                total_losses, 
                entropies_optimal, 
                name_selection_entropy_optimal, 
                birth_date_selection_entropy_optimal, 
                N
            )
            
            results.append({
                'num_params': num_parameters,
                'N': N,
                'min_order': min_order,
                'max_order': max_order,
                'qa_capacity_2_optimal': qa_capacity_2_optimal,
                'qa_capacity_1_optimal': qa_capacity_1_optimal,
                'weight_decay': wd,
                'steps': row['steps'],
                'parameter_norm': row['l2_norm']
            })
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="latest")
    args = parser.parse_args()

    all_timestep_results = []
    final_results = []
    N_sizes = [10000, 15000, 20000, 25000, 30000, 50000]

    for N in N_sizes:
        # Load loss data from CSV
        csv_path = get_project_root() / f"results/loss_over_time_{N}_batched.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Loss data not found at {csv_path}")
        
        loss_df = pd.read_csv(csv_path)
        
        # Load and process dataset
        profiles_dataset = load_dataset(f"EleutherAI/profiles_dataset_{N}_uniform")['train']

        attribute_values = {
            'name': list(set(profiles_dataset['name'])),
            'birth_date': list(set(profiles_dataset['birth_date'])),
            'birth_city': list(set(profiles_dataset['birth_city'])),
            'university': list(set(profiles_dataset['university'])),
            'employer': list(set(profiles_dataset['employer'])),
            'child': list(set(profile['name'] for profile in profiles_dataset['child'])),
            'best_friend': list(set(profile['name'] for profile in profiles_dataset['best_friend'])),
            'worst_enemy': list(set(profile['name'] for profile in profiles_dataset['worst_enemy']))
        }

        # Calculate entropies
        entropies_optimal = [calculate_entropy(values, attr, scheme="optimal") 
                    for attr, values in attribute_values.items() 
                    if attr != 'name']
        name_selection_entropy_optimal = calculate_entropy(attribute_values['name'], 'name', selection=True, scheme="optimal")
        birth_date_selection_entropy_optimal = calculate_entropy(attribute_values['birth_date'], 'birth_date', selection=True, scheme="optimal")

        # Process all timesteps
        timestep_results = process_timestep_data(
            loss_df, 
            N, 
            entropies_optimal, 
            name_selection_entropy_optimal, 
            birth_date_selection_entropy_optimal
        )
        all_timestep_results.append(timestep_results)
        
        # Get final timestep results for each configuration
        final_timesteps = timestep_results.groupby(
            ['num_params', 'min_order', 'max_order', 'weight_decay']
        ).last().reset_index()
        final_results.append(final_timesteps)

    # Combine results
    all_timestep_df = pd.concat(all_timestep_results, ignore_index=True)
    final_df = pd.concat(final_results, ignore_index=True)

    # Save all timestep results
    all_timestep_df.to_csv(get_project_root() / 'results/capacity_results_all_timesteps.csv', index=False)
    final_df.to_csv(get_project_root() / 'results/capacity_results.csv', index=False)

    max_steps = all_timestep_df.groupby(['num_params', 'min_order', 'max_order', 'weight_decay'])['steps'].max()
    # Adjust step counts if max steps > 5M
    def adjust_steps(row):
        group_key = (row['num_params'], row['min_order'], row['max_order'], row['weight_decay'])
        steps = row['steps']
        if max_steps[group_key] > 5_000_000:
            steps = steps / 2
        return steps / N
        
    all_timestep_df['normalized_step'] = all_timestep_df.apply(adjust_steps, axis=1)

    # Generate plots
    normalized_capacity_plot(final_df)
    cap_vs_params_plot(all_timestep_df, N_sizes)
    cap_vs_N_plot(all_timestep_df)
    
    # Plot capacity vs norm using all timesteps
    cap_vs_norm_plot(all_timestep_df)

    


    

if __name__ == "__main__":
    main()
