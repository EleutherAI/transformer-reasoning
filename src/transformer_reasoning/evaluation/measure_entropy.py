from collections import Counter
import math
from datasets import load_from_disk
from scipy.special import comb
from transformer_reasoning.utils import get_project_root, log_double_factorial
from transformer_reasoning.generate_dataset.generate_profiles import RELATIONSHIP_TYPES
import argparse
from datetime import datetime
import numpy as np
from math import ceil
from typing import Literal


def calculate_selection_entropy(
        values: list[str], 
        attr: str, 
        selection_scheme: Literal['optimal', 'enumerate', 'independent'] = 'enumerate',
        N: int = None
    ):
    
    if attr == 'name':
        first_names = set(value.split()[0] for value in values)
        middle_names = set(value.split()[1] for value in values)
        last_names = set(value.split()[2] for value in values)
        N_names = len(first_names)*len(middle_names)*len(last_names)
        n_names = N if N is not None else len(values)
        if selection_scheme == 'optimal':
            name_selection_entropy = n_names * np.log2(N_names) - n_names * np.log2(n_names)
        elif selection_scheme == 'enumerate':
            name_selection_entropy = n_names * np.log2(N_names)
        elif selection_scheme == 'independent':
            name_selection_entropy = 0
        return name_selection_entropy
    elif attr == 'birth_date':
        start_date = datetime(1900, 1, 1)
        end_date = datetime(2099, 12, 31)
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        n = len(values)
        if selection_scheme in ['optimal', 'enumerate']:
            birth_date_selection_entropy = (math.lgamma(days_between_dates + 1) - 
                                         math.lgamma(n + 1) - 
                                         math.lgamma(days_between_dates - n + 1)) / math.log(2)
        elif selection_scheme == 'independent':
            birth_date_selection_entropy = 0
        return birth_date_selection_entropy

def calculate_entropy(
        values: list[str], 
        attr: str, 
        bipartite: bool = False, 
        scheme: Literal['optimal', '2-hop-big-hash', 'independent'] = 'optimal'
    ):
    # Calculate unique values for each component
    # scheme: 'optimal': use optimal encoding for names/birth_dates,
    #         'enumerate': enumerate all name/birth_date values, neglecting symmetries,
    #         'independent': each name/birth_date occurrence is treated as a random selection from the set of all possible values, neglecting re-use
    if bipartite:
        raise NotImplementedError("Bipartite relationships not implemented yet")
    
    if scheme == "independent":
        if attr in RELATIONSHIP_TYPES:
            first_names = set(value.split()[0] for value in values)
            middle_names = set(value.split()[1] for value in values)
            last_names = set(value.split()[2] for value in values)
            N_names = len(first_names)*len(middle_names)*len(last_names)
            return math.log2(N_names)
        if attr == "birth_date":
            start_date = datetime(1900, 1, 1)
            end_date = datetime(2099, 12, 31)
            time_between_dates = end_date - start_date
            days_between_dates = time_between_dates.days
            return math.log2(days_between_dates)

    unique_values = len(set(values))

    # Default case for other attributes
    return math.log2(unique_values)


def analyze_dataset(N = None, bipartite: bool = False):    
    dataset_path = "profiles_dataset"
    rel_type = "bipartite" if bipartite else "uniform"
    if N != 'none':
        dataset_path = f"profiles_dataset_{N}_{rel_type}"
    
    dataset = load_from_disk(str(get_project_root() / "generated_data" / dataset_path))
    
    # Collect all values for each attribute
    attribute_values = {
        'name': [],
        'birth_date': [],
        'birth_city': [],
        'university': [],
        'employer': [],
        'parent': [],
        'child': [],
        'best_friend': [],
        'worst_enemy': []
    }

    for profile in dataset:
        attribute_values['name'].append(profile['name'])
        attribute_values['birth_date'].append(profile['birth_date'])
        attribute_values['birth_city'].append(profile['birth_city'])
        attribute_values['university'].append(profile['university'])
        attribute_values['employer'].append(profile['employer'])
        attribute_values['parent'].append(profile['parent']['name'])
        attribute_values['child'].append(profile['child']['name'])
        attribute_values['best_friend'].append(profile['best_friend']['name'])
        attribute_values['worst_enemy'].append(profile['worst_enemy']['name'])

    # Calculate entropy for each attribute
    entropies = {attr: calculate_entropy(values, attr, bipartite) for attr, values in attribute_values.items()}

    print("Entropies:")
    for attr, entropy in entropies.items():
        print(f"{attr}: {entropy}")
    print(f"Total entropy per profile: {sum(entropies.values())}")
    print(f"Total entropy: {sum(entropies.values()) * len(dataset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=str, choices=['2500', '5000', '10000', '15000', '25000', 'none'], default='none')
    parser.add_argument('--bipartite', action='store_true', help='Use bipartite relationships')
    args = parser.parse_args()
    analyze_dataset(args.N, args.bipartite)
