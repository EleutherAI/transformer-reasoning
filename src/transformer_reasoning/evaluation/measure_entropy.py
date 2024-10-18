from collections import Counter
import math
from datasets import load_from_disk
from transformer_reasoning.utils import get_project_root

def calculate_entropy(values, attr):
    counter = Counter(values)
    total = len(values)
    probabilities = [count / total for count in counter.values()]
    ent = -sum(p * math.log2(p) for p in probabilities)
    if attr in ['child', 'parent', 'best_friend', 'worst_enemy']:
        return ent * 0.5
    return ent

def analyze_dataset():
    dataset = load_from_disk(str(get_project_root() / "generated_data/profiles_dataset"))
    
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
    entropies = {attr: calculate_entropy(values, attr) for attr, values in attribute_values.items()}

    print("Entropies:")
    for attr, entropy in entropies.items():
        print(f"{attr}: {entropy}")
    print(f"Total entropy per profile: {sum(entropies.values())}")
    print(f"Total entropy: {sum(entropies.values()) * len(dataset)}")

if __name__ == "__main__":
    analyze_dataset()