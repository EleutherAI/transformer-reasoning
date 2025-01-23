import argparse
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset

from transformer_reasoning.utils import get_project_root
from transformer_reasoning.evaluation.measure_entropy import calculate_entropy, calculate_selection_entropy
from transformer_reasoning.evaluation.eval_utils import load_eval_results
from transformer_reasoning.evaluation.plot_capacity import cap_vs_N_plot, cap_vs_params_plot, plot_derivatives
from transformer_reasoning.generate_dataset.generate_profiles import RELATIONSHIP_TYPES

import seaborn as sns

def invert_loss_two_hop(loss_b2, n=100):

    if loss_b2 > np.log2(n):
        return np.log2(n)

    py = 2**(-loss_b2)

    a = 1
    b = -1/n
    c = 1/n - py

    px = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    return -np.log2(px)

def calculate_capacities_average(
        total_losses_b2, 
        entropies, 
        name_selection_entropy, 
        birth_date_selection_entropy, 
        N, 
        hops, 
        scheme, 
        incremental=False
    ):
    """Calculate average capacities for all attributes."""

    if scheme == "2-hop-double":
        raise ValueError("2-hop-double requires per-attribute capacities")

    num_relations = len([r for r in RELATIONSHIP_TYPES if r in entropies])
    qa_multiplier = 1

    if incremental:
        name_selection_entropy = 0
        birth_date_selection_entropy = 0

    if scheme == "2-hop-big-hash" and hops == 2:
        qa_multiplier = num_relations

    total_entropy = sum(entropies.values())
    avg_loss_b2 = total_losses_b2['all_questions'] 

    if total_entropy > avg_loss_b2 * len(entropies):
        capacity = (total_entropy - avg_loss_b2 * len(entropies)) * \
            qa_multiplier * N + \
            name_selection_entropy + \
            birth_date_selection_entropy
    elif not incremental:
        excess_loss = avg_loss_b2 * len(entropies) - total_entropy
        capacity = name_selection_entropy + birth_date_selection_entropy - \
            excess_loss * N
    else:
        capacity = 0

    return {'total_capacity': capacity}

def calculate_capacities_per_attr(
        per_attribute_losses_b2, 
        entropies, 
        name_selection_entropy, 
        birth_date_selection_entropy, 
        N, 
        hops, 
        scheme, 
        incremental=False
    ):
    """Calculate per-attribute and 2nd order QA capacities."""
    # Collect attribute values for entropy calculation    
    num_relations = len([r for r in RELATIONSHIP_TYPES if r in entropies])
    qa_multiplier = 1
    qa_capacity = {}

    if incremental:
        name_selection_entropy = 0
        birth_date_selection_entropy = 0

    if scheme == "2-hop-big-hash" and hops == 2:
        qa_multiplier = num_relations

    for attr, entropy in entropies.items():
        label = f"{attr}_capacity"
        qa_loss_b2 = per_attribute_losses_b2[attr]

        if scheme == "2-hop-double" and hops == 2:
            attr_n = 2**(entropy)
            qa_loss_b2 = invert_loss_two_hop(qa_loss_b2, attr_n)
            if attr in RELATIONSHIP_TYPES:
                qa_multiplier = 2
            
        if attr in RELATIONSHIP_TYPES:
            if entropy > qa_loss_b2:
                qa_capacity[label] = (entropy - qa_loss_b2) * qa_multiplier * \
                    N + name_selection_entropy/num_relations
            elif not incremental:
                excess_loss = qa_loss_b2 - entropy
                qa_capacity[label] = (name_selection_entropy - excess_loss * N)/num_relations
            else:
                qa_capacity[label] = 0
        elif attr == 'birth_date':
            if entropy > qa_loss_b2:
                qa_capacity[label] = (entropy - qa_loss_b2) * qa_multiplier * \
                    N + birth_date_selection_entropy
            elif not incremental:
                excess_loss = qa_loss_b2 - entropy
                qa_capacity[label] = birth_date_selection_entropy - \
                    excess_loss * N
            else:
                qa_capacity[label] = 0
        else:
            qa_capacity[label] = (entropy - qa_loss_b2) * qa_multiplier * N

    qa_capacity['total_capacity'] = sum(qa_capacity.values())

    return qa_capacity

def calculate_capacities(
        losses_b2, 
        entropies, 
        name_selection_entropy, 
        birth_date_selection_entropy, 
        N, 
        hops, 
        scheme, 
        incremental=False,
        per_attr=False
    ):
    if per_attr:
        return calculate_capacities_per_attr(
            losses_b2, 
            entropies, 
            name_selection_entropy, 
            birth_date_selection_entropy, 
            N, 
            hops, 
            scheme, 
            incremental
        )
    else:
        return calculate_capacities_average(
            losses_b2, 
            entropies, 
            name_selection_entropy, 
            birth_date_selection_entropy, 
            N, 
            hops, 
            scheme
        )


def process_timestep_data(
        eval_results, 
        N, 
        entropies, 
        name_selection_entropy, 
        birth_date_selection_entropy, 
        scheme,
        per_attr=False
    ):
    """Calculate capacities for each timestep in the loss data."""
    results = []
    filtered_eval_results = eval_results[
        (eval_results['N_profiles'] == N) &
        (eval_results['mode']).isin(['train_onehop', 'train_twohop'])
    ]
    # Group by model configuration
    for (num_parameters, min_order, max_order, wd, eval_hops, global_step, layers), group in filtered_eval_results.groupby(
        ['n_params', 'min_train_hops', 'max_train_hops', 'weight_decay', 'hops', 'global_step', 'layers']
    ):
        total_losses_b2 = {}
        for _, row in group.iterrows():
            subject = row['subject']
            total_losses_b2[subject] = row['loss'] / np.log(2)

        total_losses_b2['all_questions'] = group['loss'].mean() / np.log(2)
        matched_entropies = {attr: entropies[attr] for attr in total_losses_b2 if attr in entropies}
        # Calculate capacities
        capacities = calculate_capacities(
            total_losses_b2, 
            matched_entropies, 
            name_selection_entropy, 
            birth_date_selection_entropy, 
            N,
            eval_hops,
            scheme,
            per_attr=per_attr
        )

        dummy_losses = {subject: 0 for subject in total_losses_b2}
        dummy_losses['all_questions'] = 0

        dataset_entropy = calculate_capacities(
            dummy_losses, 
            matched_entropies, 
            name_selection_entropy, 
            birth_date_selection_entropy, 
            N,
            eval_hops,
            scheme,
            per_attr=per_attr
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
            'baseline_capacity': name_selection_entropy + birth_date_selection_entropy,
            'dataset_entropy': dataset_entropy['total_capacity'],
            'layers': layers
        }

        result.update(capacities)

        results.append(result)

        if eval_hops == 2:
            one_hop_losses_b2 = filtered_eval_results.loc[
                (filtered_eval_results['hops'] == 1) & 
                (filtered_eval_results['n_params'] == num_parameters) &
                (filtered_eval_results['min_train_hops'] == min_order) &
                (filtered_eval_results['max_train_hops'] == max_order) &
                (filtered_eval_results['weight_decay'] == wd) &
                (filtered_eval_results['global_step'] == global_step) &
                (filtered_eval_results['subject'].isin(total_losses_b2.keys()))
            ]
            one_hop_losses_b2_dict = {row['subject']: row['loss'] / np.log(2) for _, row in one_hop_losses_b2.iterrows()}
            one_hop_losses_b2_dict['all_questions'] = one_hop_losses_b2['loss'].mean() / np.log(2)

            incr_capacities = calculate_capacities(
                one_hop_losses_b2_dict, 
                matched_entropies, 
                name_selection_entropy, 
                birth_date_selection_entropy, 
                N, 
                1, 
                scheme, 
                incremental=True,
                per_attr=per_attr
            )
            incr_entropies = calculate_capacities(
                dummy_losses, 
                matched_entropies, 
                name_selection_entropy, 
                birth_date_selection_entropy, 
                N,
                1,
                scheme,
                incremental=True,
                per_attr=per_attr
            )

            result = result.copy()
            result['hops'] = 2.1
            result['capacity'] = incr_capacities['total_capacity'] + capacities['total_capacity']
            result['baseline_capacity'] = name_selection_entropy + birth_date_selection_entropy
            result['dataset_entropy'] = incr_entropies['total_capacity']
            capacities = {attr: incr_capacities[attr] + capacities[attr] for attr in capacities}
            result.update(capacities)
            results.append(result)

    return pd.DataFrame(results)

def dataset_component_entropies(
        test_profiles_dataset, 
        N=None, 
        scheme="optimal",
        selection_scheme="optimal"
    ):
    if N is None:
        N = len(test_profiles_dataset)


    relations = [r for r in RELATIONSHIP_TYPES if r in test_profiles_dataset.features]
    attribute_values = {
        'name': list(set(test_profiles_dataset['name'])),
        'birth_date': list(set(test_profiles_dataset['birth_date'])),
        'birth_city': list(set(test_profiles_dataset['birth_city'])),
        'university': list(set(test_profiles_dataset['university'])),
        'employer': list(set(test_profiles_dataset['employer'])),
        **{r: list(set(profile['name'] for profile in test_profiles_dataset[r])) for r in relations}
    }

    entropies = {attr: calculate_entropy(values, attr, scheme=scheme) 
                for attr, values in attribute_values.items() 
                if attr != 'name'}
    name_selection_entropy = calculate_selection_entropy(
        attribute_values['name'], 
        'name', 
        selection_scheme=selection_scheme,
        N=N
    )
    birth_date_selection_entropy = calculate_selection_entropy(
        attribute_values['birth_date'], 
        'birth_date', 
        selection_scheme=selection_scheme,
        N=N
    )

    return entropies, name_selection_entropy, birth_date_selection_entropy

def dataset_entropy(
        test_profiles_dataset, 
        N=None, 
        scheme="optimal",
        selection_scheme="optimal"
    ):
    entropies, name_selection_entropy, birth_date_selection_entropy = dataset_component_entropies(
        test_profiles_dataset, 
        N, 
        scheme,
        selection_scheme
    )

    dummy_losses = {subject: 0 for subject in entropies}
    dummy_losses['all_questions'] = 0

    dataset_entropy_1hop = calculate_capacities(
        dummy_losses, 
        entropies, 
        name_selection_entropy, 
        birth_date_selection_entropy, 
        N,
        1,
        scheme,
        incremental=False
    )

    dataset_entropy_2hop = calculate_capacities(
        dummy_losses, 
        entropies, 
        name_selection_entropy, 
        birth_date_selection_entropy, 
        N,
        2,
        scheme,
        incremental=True
    )

    return dataset_entropy_1hop, dataset_entropy_2hop

def calculate_smoothed_derivative(group, window_size=5, num_points=50):
    """Calculate smoothed derivatives for the last num_points steps."""
    # Sort by step number and get the last entries
    group = group.sort_values('global_step')
    last_entries = group.tail(num_points)
    
    if len(last_entries) < window_size:
        return pd.DataFrame({'smoothed_derivative': [np.nan], 'step': [np.nan]})
    
    results = []
    # Calculate derivatives using sliding window
    for i in range(len(last_entries) - window_size + 1):
        window = last_entries.iloc[i:i+window_size]
        steps = window['global_step'].values
        losses = window['loss'].values
        derivatives = np.gradient(losses, steps)
        middle_step = steps[len(steps)//2]
        results.append({
            'smoothed_derivative': np.mean(derivatives),
            'step': middle_step
        })
    
    return pd.DataFrame(results)

def create_derivative_table(df):
    """Create table of smoothed derivatives for each configuration."""
    results = []
    # Group by configuration
    for (n_params, N_profiles, layers, max_train_hops, weight_decay), group in df.groupby([
        'n_params', 
        'N_profiles', 
        'layers', 
        'max_train_hops',
        'weight_decay'
    ]):
        # Filter for steps > 1e6 and calculate derivatives for this group
        filtered_group = group[group['global_step'] > 1e6]
        derivatives = calculate_smoothed_derivative(filtered_group[filtered_group['hops'] == 2])
        
        # Add configuration parameters to each row
        for _, row in derivatives.iterrows():
            results.append({
                'n_params': n_params,
                'N_profiles': N_profiles,
                'layers': layers,
                'max_train_hops': max_train_hops,
                'weight_decay': weight_decay,
                'smoothed_derivative': row['smoothed_derivative'],
                'step': row['step']
            })
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--scheme", type=str, choices=["optimal", "2-hop-big-hash", "2-hop-double"], default="optimal")
    parser.add_argument("--selection_scheme", type=str, choices=["optimal", "enumerate", "independent"], default="optimal")
    parser.add_argument("--relations", type=int, default=None)
    parser.add_argument("--subjectwise", action="store_true")
    parser.add_argument("--skip_mode", action="store_true")
    parser.add_argument("--commit_hashes", nargs="+", default=[])
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--base_path", type=str, default=".")
    args = parser.parse_args()

    eval_results = load_eval_results(skip_mode=args.skip_mode, commit_hashes=args.commit_hashes, subjectwise=args.subjectwise, base_path=args.base_path)
    rel_str = f"_r{args.relations}" if args.relations else ""

    if 'subject' not in eval_results.columns:
        eval_results['subject'] = np.nan

    if not args.subjectwise:
        eval_results = eval_results[pd.isna(eval_results['subject'])]
    else:
        eval_results = eval_results[~pd.isna(eval_results['subject'])]

    if args.relations is not None:
        eval_results = eval_results[eval_results['relations'] == args.relations]
    else:
        eval_results = eval_results[eval_results['relations'].isna()]

    N_sizes = list(set(eval_results['N_profiles']))

    all_timestep_results = []
    final_results = []
    for N in N_sizes:
        
        # Load and process dataset
        profiles_dataset = load_dataset(f"EleutherAI/profiles_dataset_{N}_uniform{rel_str}")['train']

        # Calculate entropies
        entropies, name_selection_entropy, birth_date_selection_entropy = dataset_component_entropies(
            profiles_dataset, 
            N, 
            args.scheme,
            args.selection_scheme
        )
        
        # Process all timesteps
        timestep_results = process_timestep_data(
            eval_results, 
            N, 
            entropies, 
            name_selection_entropy, 
            birth_date_selection_entropy,
            args.scheme,
            per_attr=args.subjectwise
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
    final_df = final_df[final_df['global_step'] >= 700000]

    # Save all timestep results
    all_timestep_df.to_csv(get_project_root() / f'results/capacity_results_all_timesteps_{args.scheme}_{args.selection_scheme}{rel_str}.csv', index=False)
    final_df.to_csv(get_project_root() / f'results/capacity_results_{args.scheme}_{args.selection_scheme}{rel_str}.csv', index=False)

    # Adjust step counts if max steps > 5M
    def adjust_steps(row):
        group_key = (row['n_params'], row['min_train_hops'], row['max_train_hops'], row['weight_decay'])
        steps = row['global_step']
        return steps / N
        
    all_timestep_df['normalized_step'] = all_timestep_df.apply(adjust_steps, axis=1)

    # Generate plots
    # normalized_capacity_plot(final_df, args.scheme)
    # cap_vs_params_plot(all_timestep_df, N_sizes, scheme=args.scheme)
    cap_vs_N_plot(all_timestep_df, scheme=args.scheme, selection_scheme=args.selection_scheme, rel_str=rel_str)
    cap_vs_params_plot(all_timestep_df, scheme=args.scheme, selection_scheme=args.selection_scheme, rel_str=rel_str)
    
    
    # Plot capacity vs norm using all timesteps
    # cap_vs_norm_plot(all_timestep_df, scheme=args.scheme)

    if not args.subjectwise:
        derivatives_df = create_derivative_table(eval_results)
        derivatives_df.to_csv(
            get_project_root() / f'results/loss_derivatives_{args.scheme}_{args.selection_scheme}{rel_str}.csv',
            index=False
        )
        
        # Plot derivatives
        plot_derivatives(derivatives_df, scheme=args.scheme, selection_scheme=args.selection_scheme, rel_str=rel_str)

if __name__ == "__main__":
    main()
