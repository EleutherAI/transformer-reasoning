import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import Dict, List, Tuple
import random
from transformer_reasoning.utils import get_project_root
import argparse
import os

def load_templates(template_dir) -> Dict[str, List[str]]:
    """Load all template files and return a dictionary of templates."""
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith("-templates.txt"):
            attribute = filename.split("-")[0]
            with open(os.path.join(template_dir, filename), 'r') as f:
                templates[attribute] = f.read().splitlines()
    return templates

def find_template_matches(bio: str, templates: Dict[str, List[str]]) -> List[Tuple[str, str, int, int]]:
    """
    Find all template matches in the bio and return their variable positions.
    Returns: List of (attribute, value, start_idx, end_idx) tuples
    """
    matches = []
    
    for attribute, attribute_templates in templates.items():
        for template in attribute_templates:
            # Convert template to regex pattern
            pattern = re.escape(template).replace(f'\\{{{attribute}\\}}', '(.*?)')
            match = re.search(pattern, bio)
            if match:
                value = match.group(1)
                # Find the position of the variable in the original text
                template_parts = template.split(f'{{{attribute}}}')
                start_idx = bio.find(template_parts[0]) + len(template_parts[0])
                end_idx = start_idx + len(value)
                matches.append((attribute, value, start_idx, end_idx))
                break  # Move to next attribute once we find a match
                
    return matches

def get_token_ranges(text: str, tokenizer, matches: List[Tuple[str, str, int, int]]) -> List[Tuple[str, List[int]]]:
    """
    Convert character ranges to token ranges.
    Returns: List of (attribute, token_indices) tuples
    """
    # Tokenize the full text
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_ranges = []
    
    for attribute, value, start_idx, end_idx in matches:
        # Tokenize the text up to the start of the variable
        prefix_tokens = tokenizer.encode(text[:start_idx], add_special_tokens=True)
        # Tokenize the text up to the end of the variable
        full_tokens = tokenizer.encode(text[:end_idx], add_special_tokens=True)
        
        # The variable tokens are between these lengths
        var_token_indices = list(range(len(prefix_tokens)-1, len(full_tokens)-1))
        token_ranges.append((attribute, var_token_indices))
    
    return token_ranges

def calculate_losses(model, tokenizer, text: str, token_ranges: List[Tuple[str, List[int]]]) -> Dict[str, Tuple[float, float]]:
    """
    Calculate losses for each variable's tokens.
    Returns: Dict mapping attributes to (first_token_loss, all_tokens_loss)
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # Get per-token losses (batch_size=1 so we take [0])
        token_losses = outputs.loss * inputs["input_ids"].shape[1]  # Multiply by seq length to get sum instead of mean
        token_losses = token_losses.cpu()
    
    attribute_losses = {}
    for attribute, token_indices in token_ranges:
        if token_indices:  # Check if we found any tokens
            first_token_loss = float(token_losses[token_indices[0]])
            all_tokens_loss = float(sum(token_losses[i] for i in token_indices))
            attribute_losses[attribute] = (first_token_loss, all_tokens_loss)
    
    return attribute_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--num-parameters", type=int, default=1_000_000)
    args = parser.parse_args()

    # Load model and tokenizer
    model_path = get_project_root() / f"results/n{args.N}_p{args.num_parameters}_o{args.order}_continued_2/checkpoint-{args.checkpoint}"
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

    # Load datasets
    bios_dataset = load_from_disk(str(get_project_root() / f"generated_data/bios/bios_dataset_{args.N}"))
    templates = load_templates(get_project_root() / "generated_data/templates")

    # Sample random subset
    sample_indices = random.sample(range(len(bios_dataset)), args.sample_size)
    sample_bios = bios_dataset.select(sample_indices)

    # Initialize aggregated losses
    total_losses = {}  # attribute -> (sum_first_token_loss, sum_all_tokens_loss, count)

    # Process each bio
    for bio in sample_bios["bio"]:
        # Find variable positions
        matches = find_template_matches(bio, templates)
        if not matches:
            continue

        # Get token ranges for variables
        token_ranges = get_token_ranges(bio, tokenizer, matches)
        
        # Calculate losses
        losses = calculate_losses(model, tokenizer, bio, token_ranges)
        
        # Aggregate losses
        for attr, (first_loss, all_loss) in losses.items():
            if attr not in total_losses:
                total_losses[attr] = [0.0, 0.0, 0]
            total_losses[attr][0] += first_loss
            total_losses[attr][1] += all_loss
            total_losses[attr][2] += 1

    # Print results
    print("\nAverage losses per attribute:")
    print(f"{'Attribute':<15} {'First Token':<12} {'All Tokens':<12} {'Count':<8}")
    print("-" * 47)
    for attr, (first_sum, all_sum, count) in total_losses.items():
        if count > 0:
            print(f"{attr:<15} {first_sum/count:<12.4f} {all_sum/count:<12.4f} {count:<8}")

if __name__ == "__main__":
    main()

