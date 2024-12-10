from datasets import load_dataset
from transformers import AutoTokenizer
import random
from collections import defaultdict

from transformer_reasoning.train.dataset import InfiniteQADataset

# Set random seed for reproducibility
random.seed(42)

# Load dataset and tokenizer
profiles_dataset = load_dataset("EleutherAI/profiles_dataset_1000_uniform_r17", keep_in_memory=True)['train']
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llama_multihop_tokenizer")

# Initialize datasets for all modes
modes = [
    "train",
    "eval_random",
    "eval_first_people",
    "eval_relations",
    "eval_person_relation_pairs",
    "eval_second_people",
    "eval_second_attributes",
    "eval_second_person_attribute_pairs"
]

datasets = {
    mode: InfiniteQADataset(
        profiles_dataset=profiles_dataset,
        tokenizer=tokenizer,
        heldout_fraction=0.1,
        mode=mode,
    ) for mode in modes
}

# Helper function to extract components from a question
def parse_question(question_text, profiles_dataset):
    """Extract first person, relation, second person, and attribute from a question"""
    parts = question_text.split("Question: ")[1].split(" Answer: ")[0]
    
    if "What was" not in parts:
        return None
    
    if "'s" not in parts:
        return None
        
    components = parts.split("'s")
    first_person = components[0].replace("What was ", "")
    profile = [p for p in profiles_dataset if p['name'] == first_person][0]
    
    if len(components) == 2:  # 1-hop
        attribute = components[1].strip().strip('?')
        return {
            "first_person": first_person,
            "relation": None,
            "second_person": None,
            "attribute": attribute,
            "hops": 1
        }
    else:  # 2-hop
        relation = components[1].strip()
        attribute = components[2].strip().strip('?')
        return {
            "first_person": first_person,
            "relation": relation,
            "attribute": attribute,
            "second_person": profile[relation.replace(' ', '_')]['name'],
            "hops": 2
        }

# Test each dataset
def test_dataset(dataset, mode, n_samples=100):
    print(f"\nTesting {mode} dataset:")
    samples = defaultdict(list)
    
    # Collect samples
    for _ in range(n_samples):
        sample = dataset.generate_qa()
        components = parse_question(sample, dataset.profiles)
        if components and components['hops'] == 2:
            samples['first_person'].append(components['first_person'])
            samples['relation'].append(components['relation'])
            samples['attribute'].append(components['attribute'])
            samples['second_person'].append(components['second_person'])
    # Verify samples based on mode
    heldout_sets = dataset.heldout_sets
    
    if mode == "train":
        # Check that no held-out combinations appear in training data
        for first_person in samples['first_person']:
            person_idx = next((i for i in range(len(profiles_dataset)) 
                             if profiles_dataset[i]['name'] == first_person), None)
            assert person_idx not in heldout_sets['first_people'], \
                f"Found held-out first person {first_person} in training data"
            
        for first_person, relation in zip(samples['first_person'], samples['relation']):
            person_idx = next((i for i in range(len(profiles_dataset)) 
                             if profiles_dataset[i]['name'] == first_person), None)
            if relation:
                assert (person_idx, relation) not in heldout_sets['person_relation_pairs'], \
                    f"Found held-out person-relation pair {first_person} {relation} in training data"
    
        for relation in samples['relation']:
            if relation:
                assert relation.replace(' ', '_') not in heldout_sets['relations'], \
                    f"Found held-out relation {relation} in training data"
                
        for attribute in samples['attribute']:
            assert attribute not in heldout_sets['second_attributes'], \
                f"Found held-out attribute {attribute} in training data"

        for second_person in samples['second_person']:
            assert second_person not in heldout_sets['second_people'], \
                f"Found held-out second person {second_person} in training data"
            
        for second_person, attribute in zip(samples['second_person'], samples['attribute']):
            assert (second_person, attribute) not in heldout_sets['second_person_attribute_pairs'], \
                f"Found held-out second person-attribute pair {second_person} {attribute} in training data"


    elif mode == "eval_first_people":
        # Check that all first people are from held-out set
        for first_person in samples['first_person']:
            person_idx = next((i for i in range(len(profiles_dataset)) 
                             if profiles_dataset[i]['name'] == first_person), None)
            assert person_idx in heldout_sets['first_people'], \
                f"Found non-held-out first person {first_person} in eval data"
    
    elif mode == "eval_relations":
        # Check that all relations are from held-out set
        for relation in samples['relation']:
            assert relation.replace(' ', '_') in heldout_sets['relations'], \
                f"Found non-held-out relation {relation} in eval data"
    
    elif mode == "eval_second_attributes":
        # Check that all attributes are from held-out set
        for attribute in samples['attribute']:
            assert attribute.replace(' ', '_') in heldout_sets['second_attributes'], \
                f"Found non-held-out attribute {attribute} in eval data"
    

    elif mode == "eval_second_people":
        # Check that all second people are from held-out set
        for second_person in samples['second_person']:
            sp_idx = next((i for i in range(len(profiles_dataset)) 
                             if profiles_dataset[i]['name'] == second_person), None)
            assert sp_idx in heldout_sets['second_people'], \
                f"Found non-held-out second person {second_person} in eval data"

    elif mode == "eval_second_person_attribute_pairs":
        # Check that all second person-attribute pairs are from held-out set
        for second_person, attribute in zip(samples['second_person'], samples['attribute']):
            assert (second_person, attribute.replace(' ', '_')) in heldout_sets['second_person_attribute_pairs'], \
                f"Found non-held-out second person-attribute pair {second_person} {attribute} in eval data"

    elif mode == "eval_person_relation_pairs":
        # Check that all person-relation pairs are from held-out set
        for first_person, relation in zip(samples['first_person'], samples['relation']):
            person_idx = next((i for i in range(len(profiles_dataset)) 
                             if profiles_dataset[i]['name'] == first_person), None)
            if relation:
                assert (person_idx, relation.replace(' ', '_')) in heldout_sets['person_relation_pairs'], \
                    f"Found non-held-out person-relation pair {first_person} {relation} in eval data"

    print(f"Generated {n_samples} valid samples")
    print(f"Sample question: {sample}")

# Run tests
for mode, dataset in datasets.items():
    test_dataset(dataset, mode)