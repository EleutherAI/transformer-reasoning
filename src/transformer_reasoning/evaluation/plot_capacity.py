import seaborn as sns
import matplotlib.pyplot as plt
from transformer_reasoning.evaluation.eval_utils import get_project_root
from transformer_reasoning.generate_dataset.generate_profiles import RELATIONSHIP_TYPES
import numpy as np
import pandas as pd



def cap_vs_N_plot(df, scheme=None, selection_scheme=None, rel_str=None, plot_combination=False):
    filtered_df = df.groupby(['N_profiles', 'n_params', 'max_train_hops', 'weight_decay', 'hops']).last().reset_index(drop=False)
    filtered_df = filtered_df[filtered_df['global_step'] >= 700000]
    param_sizes = filtered_df['n_params'].unique()
    
    # Check if we have per-attribute data
    attributes = RELATIONSHIP_TYPES + ['birth_date', 'birth_city', 'employer', 'university']
    attributes = [att for att in attributes if f'{att}_capacity' in df.columns]
    
    if not plot_combination:
        filtered_df = filtered_df[filtered_df['hops'] != 2.1]
    
    for param_size in param_sizes:
        plot_df = filtered_df[filtered_df['n_params'] == param_size]
        max_capacity = 2 * param_size
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        for max_ord, ax in zip([1, 2], axes):
            data = plot_df[plot_df['max_train_hops'] == max_ord]
            sns.scatterplot(data=data, x='N_profiles', y='total_capacity', style='hops', ax=ax)
            sns.lineplot(data=data, x='N_profiles', y='total_capacity', style='hops', ax=ax)
            sns.lineplot(data=data, x='N_profiles', y='baseline_capacity', ax=ax, color='gray', linestyle='--', label='baseline')
            sns.scatterplot(data=data, x='N_profiles', y='baseline_capacity', ax=ax, color='gray', marker='+', label='baseline')
            for hop_val in data['hops'].unique():
                hop_data = data[data['hops'] == hop_val]
                sns.lineplot(data=hop_data, x='N_profiles', y='dataset_entropy', ax=ax, color='red',
                            linestyle='--' if hop_val == 2 else '-',
                            label=f'{hop_val}-hop entropy')
                sns.scatterplot(data=hop_data, x='N_profiles', y='dataset_entropy', ax=ax, color='red',
                                marker='o' if hop_val == 2 else 'x',
                                label=f'{hop_val}-hop entropy')
            ax.axhline(y=max_capacity, color='black', linestyle='--', label='est. max capacity')
            ax.set_ylim(0, max_capacity * 1.5)
            ax.legend()
            ax.set_title(f'Total Capacity - Max Train Hops = {max_ord}')
            ax.set_xlabel('N profiles')
            ax.set_ylabel('Capacity (bits)')
            
        plt.suptitle(f'Capacity vs N (params={param_size})', y=1.02, fontsize=16)
        plt.savefig(get_project_root() / f'results/total_capacity_vs_N_params{param_size}_{scheme}_{selection_scheme}{rel_str}.png',
                    bbox_inches='tight', dpi=300)
        
        plt.close()


def cap_vs_params_plot(df, scheme=None, selection_scheme=None, rel_str=None, hop=2):
    filtered_df = df[df['global_step'] >= 700000]
    
    filtered_df = filtered_df[
        (filtered_df['hops'] == hop) &
        (filtered_df['max_train_hops'] == hop)
    ]
    
    N_sizes = filtered_df['N_profiles'].unique()
    filtered_df['max_capacity'] = 2 * filtered_df['n_params']

    filtered_df = filtered_df.sort_values(by='global_step', ascending=False).groupby(['N_profiles', 'n_params', 'max_train_hops', 'weight_decay', 'hops', 'layers']).first().reset_index(drop=False)
    
    for N in N_sizes:
        plot_df = filtered_df[filtered_df['N_profiles'] == N]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        hop_df = plot_df[plot_df['hops'] == hop]
        sns.scatterplot(data=hop_df, x='n_params', y='total_capacity', hue='layers', style='layers', ax=ax)
        sns.lineplot(data=hop_df, x='n_params', y='total_capacity', hue='layers', ax=ax)
        
        # Create dummy data points for reference lines if needed
        if len(hop_df) == 1:
            param_val = hop_df['n_params'].iloc[0]
            dummy_params = [param_val * 0.8, param_val, param_val * 1.2]
            dummy_df = pd.DataFrame({
                'n_params': dummy_params,
                'baseline_capacity': hop_df['baseline_capacity'].iloc[0],
                'dataset_entropy': hop_df['dataset_entropy'].iloc[0],
                'max_capacity': np.array(dummy_params) * 2
            })
            ref_df = dummy_df
        else:
            ref_df = hop_df
            
        sns.lineplot(data=ref_df, x='n_params', y='baseline_capacity', color='gray', linestyle='--', label='baseline', ax=ax)
        sns.lineplot(data=ref_df, x='n_params', y='dataset_entropy', color='red', linestyle='--', label=f'{hop}-hop entropy', ax=ax)
        sns.lineplot(data=ref_df, x='n_params', y='max_capacity', color='black', linestyle='--', label='est. max capacity', ax=ax)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Parameters')
        ax.set_ylabel('Capacity (bits)')
        ax.set_title(f'{hop}-hop Capacity vs Parameters (N={N})')
        ax.legend(title='Number of Layers')

        fn = get_project_root() / f'results/{hop}hop_capacity_vs_params_N{N}_{scheme}_{selection_scheme}{rel_str}.png'
        plt.savefig(fn,
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
    from transformer_reasoning.evaluation.measure_capacity import calculate_capacities_per_attr

    order_markers = {1: 'o', 2: '^'}

    for N in N_sizes:
        plt.figure(figsize=(10, 6))

        # Calculate theoretical max capacities using calculate_capacities with zero losses
        zero_losses = [0.0, 1]  # Single loss value for calculate_capacities
        max_capacity = calculate_capacities_per_attr(zero_losses, entropies_optimal, 
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

def plot_derivatives(df, scheme=None, selection_scheme=None, rel_str=None):
    """Plot derivatives vs parameters for different N values."""
    plt.figure(figsize=(12, 8))
    df = df[df['step'] > 2e6]
    df = df[~pd.isna(df['smoothed_derivative'])]
    
    # Convert n_params to categorical and create color palette
    df['n_params'] = df['n_params'].astype('category')
    palette = sns.color_palette('deep', n_colors=len(df['n_params'].unique()))
    
    df = df[df['hops'] == df['max_train_hops']]

    # Plot each group with matching colors for points and trend lines
    for idx, ((n_params, N_profiles), group) in enumerate(df.groupby(['n_params', 'N_profiles'])):
        if not np.isfinite(group['smoothed_derivative']).all():
            continue

        sns.regplot(
            data=group,
            x='step',
            y='smoothed_derivative',
            scatter=True,
            scatter_kws={'alpha': 0.3, 'marker': f'${N_profiles}$'},  # Use N_profiles as marker
            line_kws={'linestyle': '--'},
            color=palette[idx % len(palette)],
            label=f'N={N_profiles}, params={n_params}',
            lowess=True
        )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.xlabel('Step (log scale)')
    plt.ylabel('Smoothed Loss Derivative')
    plt.title('Loss Derivative vs Parameters')
    plt.tight_layout()
    
    plt.savefig(
        get_project_root() / f'results/derivatives_plot_{scheme}_{selection_scheme}{rel_str}.png',
        bbox_inches='tight'
    )
    plt.close()

def loss_vs_normalized_capacity_plot(df, scheme=None, selection_scheme=None, rel_str=None):
    """Plot 2-hop eval loss vs normalized 2-hop capacity and loss, separate plot for each N and layer count."""
    filtered_df = df[
        (df['hops'] == 2) &
        (df['total_capacity'] > 0)
    ]
    
    filtered_df['normalized_content'] = (filtered_df['total_capacity']) / (2 * filtered_df['n_params'])
    filtered_df['eval_loss'] = filtered_df['eval_loss']/np.log(2)
    filtered_df['loss'] = filtered_df['loss']/np.log(2)
    
    for N in filtered_df['N_profiles'].unique():
        for layers in filtered_df['layers'].unique():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
            
            plot_df = filtered_df[
                (filtered_df['N_profiles'] == N) & 
                (filtered_df['layers'] == layers)
            ]
            
            if len(plot_df) > 0:
                # First subplot: eval_loss vs normalized capacity
                sns.scatterplot(
                    data=plot_df,
                    x='normalized_content',
                    y='eval_loss',
                    hue='n_params',
                    alpha=0.7,
                    ax=ax1
                )
                ax1.set_xlabel('Normalized Content (content / max capacity)')
                ax1.set_ylabel('2-hop Evaluation Loss')
                ax1.set_title(f'Evaluation Loss vs Normalized Capacity (N={N}, layers={layers})')
                ax1.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Second subplot: eval_loss vs loss
                sns.scatterplot(
                    data=plot_df,
                    x='loss',
                    y='eval_loss',
                    hue='n_params',
                    alpha=0.7,
                    ax=ax2
                )
                ax2.set_xlabel('Training Loss')
                ax2.set_ylabel('2-hop Evaluation Loss')
                ax2.set_title(f'Evaluation Loss vs Training Loss (N={N}, layers={layers})')
                ax2.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                plt.savefig(
                    get_project_root() / f'results/loss_vs_capacity_N{N}_L{layers}_{scheme}_{selection_scheme}{rel_str}.png',
                    bbox_inches='tight',
                    dpi=300
                )
            plt.close()
