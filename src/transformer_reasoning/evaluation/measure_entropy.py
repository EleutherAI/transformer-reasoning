from collections import Counter
import math
from datasets import load_from_disk
from scipy.special import gammaln
from transformer_reasoning.utils import get_project_root, log_double_factorial
import argparse
from datetime import datetime
import numpy as np

def calculate_entropy(values, attr, bipartite: bool = False):
    # Calculate unique values for each component
    if bipartite:
        raise NotImplementedError("Bipartite relationships not implemented yet")
    
    if attr == 'name':
        first_names = set(value.split()[0] for value in values)
        middle_names = set(value.split()[1] for value in values)
        last_names = set(value.split()[2] for value in values)
        
        first_name_entropy = math.log2(len(first_names))
        middle_name_entropy = math.log2(len(middle_names))
        last_name_entropy = math.log2(len(last_names))
        
        total_entropy = first_name_entropy + middle_name_entropy + last_name_entropy
        return total_entropy
    
    elif attr == 'birth_date':
        start_date = datetime(1900, 1, 1)
        end_date = datetime(2099, 12, 31)
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days

        return math.log2(days_between_dates)
    
    # Default case for other attributes
    unique_values = len(set(values))
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
