import argparse
from datasets import load_from_disk, DatasetDict, Dataset
import random
from typing import List, Dict, Union
from transformer_reasoning.utils import get_project_root
import math
import datetime
import os

# Question templates
FIRST_ORDER_TEMPLATE = "What was {name}'s {subject}?"
SECOND_ORDER_TEMPLATE = "What was {name}'s {relation}'s {subject}?"
THIRD_ORDER_TEMPLATE = "What was {name}'s {relation1}'s {relation2}'s {subject}?"

# Attributes and relations
ATTRIBUTES = ['birth_date', 'birth_city', 'university', 'employer']
RELATIONS = ['parent', 'child', 'best_friend', 'worst_enemy']
INVERSE_RELATIONS = {'parent': 'child', 'child': 'parent', 'best_friend': 'best_friend', 'worst_enemy': 'worst_enemy'}

def get_available_relations(profile):
    return [rel for rel in RELATIONS if profile.get(rel)['name']]

def generate_question(profile: Dict, profiles: Dataset, order: int, holdout_subjs: List[str], holdout_rels: List[str]) -> Union[Dict, None]:
    name = profile['name']
    available_relations = [rel for rel in get_available_relations(profile) if rel not in holdout_rels]
    available_subjects = [subj for subj in ATTRIBUTES + available_relations if subj not in holdout_subjs]
    chosen_subject = None
    chosen_relations = {0: None,1: None, 2: None}

    if order == 1:
        if not available_subjects:
            return None
        subject = random.choice(available_subjects)
        chosen_subject = subject
        question = FIRST_ORDER_TEMPLATE.format(name=name, subject=subject.replace('_', ' '))
    elif order == 2:
        if not available_relations or not available_subjects:
            return None
        relation = random.choice(available_relations)
        if profile[relation]['index']==-1:
            return None
        related_profile = profiles[profile[relation]['index']]
        subject = random.choice(available_subjects)
        chosen_subject = subject
        chosen_relations[1] = relation
        question = SECOND_ORDER_TEMPLATE.format(name=related_profile['name'], relation=INVERSE_RELATIONS[relation].replace('_', ' '), subject=subject.replace('_', ' '))
        assert profiles[profile[relation]['index']]['name'] == related_profile['name']
        assert profiles[related_profile[INVERSE_RELATIONS[relation]]['index']]['name'] == name
    else:  # order == 3
        if len(available_relations) < 2 or not available_subjects:
            return None
        relation2 = random.choice([INVERSE_RELATIONS[rel] for rel in available_relations])
        # Double inverse is the same relation; circular relations accepted
        relation1 = random.choice([rel for rel in available_relations])
        subject = random.choice(available_subjects)
        related_profile2 = profiles[profile[relation2]['index']]
        related_profile1 = profiles[related_profile2[relation1]['index']]
        if profile[relation2]['index']==-1 or related_profile2[relation1]['index']==-1:
            return None
        question = THIRD_ORDER_TEMPLATE.format(name=related_profile1['name'], relation1=INVERSE_RELATIONS[relation1].replace('_', ' '), 
                                        relation2=INVERSE_RELATIONS[relation2].replace('_', ' '), subject=subject.replace('_', ' '))

        chosen_subject = subject
        chosen_relations[1] = relation1
        chosen_relations[2] = relation2
        assert related_profile1['name'] == related_profile2[relation1]['name']
        assert related_profile2['name'] == related_profile1[INVERSE_RELATIONS[relation1]]['name']
        assert related_profile2['name'] == profile[relation2]['name']
        assert profile['name'] == related_profile2[INVERSE_RELATIONS[relation2]]['name']
    
    if subject in RELATIONS:
        answer = profile[subject]['name']
    else:
        answer = profile[subject]
    
    if isinstance(answer, datetime.date):
        answer = answer.strftime('%Y-%m-%d')

    return {
        "question": question,
        "answer": str(answer),
        "order": order
    }, {
        "chosen_subject": chosen_subject,
        "chosen_relations": chosen_relations
    }

def generate_questions_for_profile(profile, profiles, holdout_subjs, holdout_rels, qs_per_profile, min_order = 1):
    questions = []
    profile = {k: v[0] for k, v in profile.items()}

    for order in range(min_order, 4):
        order_holdout_subjs = holdout_subjs.get(order, []).copy()
        order_holdout_rels = holdout_rels.get(order, []).copy()
        
        for _ in range(min(math.ceil(qs_per_profile), (len(ATTRIBUTES) + len(RELATIONS))**order)):
            question_params = generate_question(profile, profiles, order, order_holdout_subjs, order_holdout_rels)
            if question_params:
                question, params = question_params
                subject = params['chosen_subject']
                relation = params['chosen_relations'][order - 1]
                if subject in ATTRIBUTES:
                    order_holdout_subjs.append(subject)
                elif subject in RELATIONS:
                    order_holdout_rels.append(relation)
                    
                questions.append(question)
    
    return {"questions": questions}

def generate_qa_dataset(profiles, holdout_subjs: Dict[int, List[str]], holdout_rels: Dict[int, List[str]], 
                        qs_per_profile: float, holdout_profile_fraction: float):
    num_profiles = len(profiles)
    num_profiles_main = math.ceil(num_profiles * (1 - holdout_profile_fraction))
    
    # Shuffle and split profiles
    shuffled_profiles = profiles.shuffle(seed=42)
    main_profiles = shuffled_profiles.select(range(num_profiles_main))
    heldout_profiles = shuffled_profiles.select(range(num_profiles_main, num_profiles))
    
    # Generate main dataset
    main_dataset = main_profiles.map(
        lambda profile: generate_questions_for_profile(profile, profiles, holdout_subjs, holdout_rels, qs_per_profile),
        remove_columns=main_profiles.column_names,
        batched=True,
        batch_size=1
    ).flatten()
    
    # Generate heldout subjects dataset
    complement_holdout_subjs = {
        order: list(set(ATTRIBUTES + RELATIONS) - set(subjs)) for order, subjs in holdout_subjs.items()
    }
    heldout_subjects_dataset = main_profiles.map(
        lambda profile: generate_questions_for_profile(profile, profiles, complement_holdout_subjs, holdout_rels, qs_per_profile),
        remove_columns=main_profiles.column_names,
        batched=True,
        batch_size=1
    ).flatten()
    
    # Generate heldout relations dataset
    complement_holdout_rels = {
        order: list(set(RELATIONS) - set(rels)) for order, rels in holdout_rels.items()
    }
    heldout_relations_dataset = main_profiles.map(
        lambda profile: generate_questions_for_profile(profile, profiles, holdout_subjs, complement_holdout_rels, qs_per_profile, min_order = 2),
        remove_columns=main_profiles.column_names,
        batched=True,
        batch_size=1
    ).flatten()
    
    # Generate heldout profiles dataset
    heldout_profiles_dataset = heldout_profiles.map(
        lambda profile: generate_questions_for_profile(profile, profiles, holdout_subjs, holdout_rels, qs_per_profile),
        remove_columns=heldout_profiles.column_names,
        batched=True,
        batch_size=1
    ).flatten()
    
    return main_dataset, heldout_subjects_dataset, heldout_relations_dataset, heldout_profiles_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate Q&A dataset from profiles")
    parser.add_argument("--N", type=int, required=True, help="Number of profiles to use")
    parser.add_argument("--train_split", type=float, default=0.9, help="Train split ratio (default: 0.9)")
    parser.add_argument("--holdout_1_subjs", nargs="*", default=[], help="Subjects to hold out for first-order questions")
    parser.add_argument("--holdout_2_subjs", nargs="*", default=[], help="Subjects to hold out for second-order questions")
    parser.add_argument("--holdout_3_subjs", nargs="*", default=[], help="Subjects to hold out for third-order questions")
    parser.add_argument("--holdout_2_rels", nargs="*", default=[], help="Relations to hold out for second-order questions")
    parser.add_argument("--holdout_3_rels", nargs="*", default=[], help="Relations to hold out for third-order questions")
    parser.add_argument("--holdout_profile_fraction", type=float, default=0.1, help="Fraction of profiles to hold out")
    parser.add_argument("--qs_per_profile", type=float, default=10, help="Number of questions per profile (can be fractional)")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the QA dataset to the hub")
    args = parser.parse_args()

    # Load the profiles dataset
    profile_path = str(get_project_root() / f"generated_data/profiles_dataset_{args.N}")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Profile dataset for N={args.N} does not exist. Please generate it first.")
    profiles = load_from_disk(profile_path)

    # Prepare holdout subjects and relations
    holdout_subjs = {
        1: args.holdout_1_subjs,
        2: args.holdout_2_subjs,
        3: args.holdout_3_subjs
    }
    holdout_rels = {
        2: args.holdout_2_rels,
        3: args.holdout_3_rels
    }

    # Generate Q&A datasets
    main_dataset, heldout_subjects_dataset, heldout_relations_dataset, heldout_profiles_dataset = generate_qa_dataset(
        profiles, holdout_subjs, holdout_rels, args.qs_per_profile, args.holdout_profile_fraction
    )

    # Split the main dataset into train and validation
    split_dataset = main_dataset.train_test_split(train_size=args.train_split)

    aux_datasets = dict()

    if len(heldout_profiles_dataset) > 0:
        aux_datasets['heldout_profiles'] = heldout_profiles_dataset

    if len(heldout_subjects_dataset) > 0:
        aux_datasets['heldout_subjects'] = heldout_subjects_dataset

    if len(heldout_relations_dataset) > 0:
        aux_datasets['heldout_relations'] = heldout_relations_dataset

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test'],
        **aux_datasets
    })

    # Save the datasets
    output_path = str(get_project_root() / f"generated_data/qa_dataset_{args.N}")
    dataset_dict.save_to_disk(output_path)

    print(f"Generated datasets:")
    print(f"  Train: {len(split_dataset['train'])} samples")
    print(f"  Validation: {len(split_dataset['test'])} samples")
    print(f"  Held-out Subjects: {len(heldout_subjects_dataset)} samples")
    print(f"  Held-out Relations: {len(heldout_relations_dataset)} samples")
    print(f"  Held-out Profiles: {len(heldout_profiles_dataset)} samples")
    print(f"Dataset saved in {output_path}")
    print(f"Dataset saved in {get_project_root() / 'generated_data/qa_dataset/'}")

    if args.push_to_hub:
        dataset_dict.push_to_hub(f"EleutherAI/transformer-reasoning-qa-dataset-{args.N}")

if __name__ == "__main__":
    main()
