import random
from datasets import load_from_disk
import os
import argparse
from datetime import datetime

from transformer_reasoning.utils import get_project_root

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
                templates[attribute] = f.read().splitlines()
    return templates

def generate_bio(profile, templates):
    bio = []
    
    profile = {k: v['name'] if k in RELATIONS else v for k, v in profile.items()}
    profile = {k: v.strftime('%Y-%m-%d') if isinstance(v, datetime) else v for k, v in profile.items()}

    # Start with a name template
    name_template = random.choice(templates['name'])
    second_attribute = [attr for attr in profile.keys() if f"{{{attr}}}" in name_template and attr != "name"][0]
    bio.append(name_template.format(**profile))
    
    # Select templates for other attributes, avoiding the second attribute from the name template
    available_attributes = set(templates.keys()) - {'name', second_attribute}
    
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
    parser.add_argument("--num_people", type=int, default=None, help="Number of people to process")
    args = parser.parse_args()

    # Load the profiles dataset
    profiles = load_from_disk(str(get_project_root() / "generated_data/profiles_dataset"))
    
    if args.num_people:
        profiles = profiles.select(range(min(args.num_people, len(profiles))))

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
    bios_dataset.save_to_disk(str(get_project_root() / "generated_data/bios/bios_dataset"))

if __name__ == "__main__":
    main()
