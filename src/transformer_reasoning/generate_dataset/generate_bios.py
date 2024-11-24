import random
from datasets import load_from_disk
import os
import argparse
from datetime import datetime

from ngrams_across_time.transformer_reasoning.src.transformer_reasoning.utils import get_project_root

RELATIONS = [
    'best_friend',
    'parent',
    'child',
    'worst_enemy'
]

def load_templates(template_dir):
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith("-templates.txt"):
            attribute = filename.split("-")[0]
            with open(os.path.join(template_dir, filename), 'r') as f:
                attribute_templates = f.read().splitlines()

            quantile_20 = int(0.20 * len(attribute_templates))
            templates[attribute] = attribute_templates[:quantile_20]

    return templates

def generate_bio(profile, templates):
    bio = []
    
    profile = {k: v['name'] if k in RELATIONS else v for k, v in profile.items()}
    profile = {k: v.strftime('%Y-%m-%d') if isinstance(v, datetime) else v for k, v in profile.items()}
    profile = {k: v for k, v in profile.items() if (isinstance(v, str) and len(v)>0)}
    
    # Start with a name template
    name_template = random.choice(templates['name'])
    second_attribute = [attr for attr in profile.keys() if f"{{{attr}}}" in name_template and attr != "name"][0]
    bio.append(name_template.format(**profile))
    
    # Select templates for other attributes, avoiding the second attribute from the name template
    available_attributes = list(set(profile.keys()) - {'name', second_attribute})
    random.shuffle(available_attributes)  # Shuffle the remaining attributes
    
    for attr in available_attributes:
        template = random.choice(templates[attr])
        bio.append(template.format(**profile))
    
    return " ".join(bio)

def generate_bios_for_profile(profile, templates, num_bios):
    bios = []
    profile = {k: v[0] for k, v in profile.items()}
    for _ in range(num_bios):
        bio = generate_bio(profile, templates)
        bios.append(bio)
    
    return {key: [profile[key]] * num_bios for key in profile.keys()} | {"bio": bios}

def main():
    parser = argparse.ArgumentParser(description="Generate bios dataset")
    parser.add_argument("--num_bios", type=int, default=1000, help="Number of bios per person")
    parser.add_argument("--N", type=int, required=True, help="Number of profiles to use")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the bios dataset to the hub")
    args = parser.parse_args()
    # Load the profiles dataset
    profile_path = str(get_project_root() / f"generated_data/profiles_dataset_{args.N}")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Profile dataset for N={args.N} does not exist. Please generate it first.")
    profiles = load_from_disk(profile_path)

    # Load the templates
    templates = load_templates(get_project_root() / "generated_data/templates")
    
    # Generate biographies using map
    bios_dataset = profiles.map(
        lambda profile: generate_bios_for_profile(profile, templates, args.num_bios),
        remove_columns=profiles.column_names,
        batched=True,
        batch_size=1
    ).flatten()

    # Save the biographies
    output_path = str(get_project_root() / f"generated_data/bios/bios_dataset_{args.N}_shuffled")
    bios_dataset.save_to_disk(output_path)

    print(f"Dataset saved in {output_path}")

    if args.push_to_hub:
        bios_dataset.push_to_hub(f"EleutherAI/transformer-reasoning-bios-dataset-{args.N}_shuffled")

if __name__ == "__main__":
    main()
