import torch
from torch.nn import CrossEntropyLoss
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import Dict, List, Tuple
import random
from transformer_reasoning.utils import get_project_root
import argparse
import os
from tqdm import tqdm
import numpy as np
from transformer_reasoning.evaluation.measure_entropy import calculate_entropy
import matplotlib.pyplot as plt

def tokenwise_loss(inputs, logits):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

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
        # Get the attribute value

        for template in attribute_templates:
            # Replace all variables in template with their values
            pattern = template
            values = {}  # Store all values used in this template
            
            # Find all variables in template
            vars_in_template = re.findall(r'(?<=\{)([^{}}]+)(?=\})', template)
            for var in vars_in_template:
                if var in ['best_friend', 'worst_enemy', 'parent', 'child']:
                    var_value = bio[f"{var}.name"]
                elif var == 'birth_date':
                    var_value = bio['birth_date'].strftime('%Y-%m-%d')
                else:
                    var_value = bio[var]
                values[var] = var_value
                pattern = pattern.replace(f'{{{var}}}', var_value)
                
            # Check if this template matches the bio
            if pattern in bio['bio']:
                # Find positions for each variable in the matched text
                text_pos = bio['bio'].find(pattern)
                for var, value in values.items():
                    var_pos = pattern.find(value)
                    start_idx = text_pos + var_pos
                    end_idx = start_idx + len(value)
                    matches.append((var, value, start_idx, end_idx))
                break  # Move to next attribute once we find a match
                
    return matches

def get_token_ranges(text: str, tokenizer, matches: List[Tuple[str, str, int, int]]) -> List[Tuple[str, List[int]]]:
    """
    Convert character ranges to token ranges.
    Returns: List of (attribute, token_indices) tuples
    """

    token_ranges = []
    
    for attribute, value, start_idx, end_idx in matches:
        # Tokenize the text up to the start of the variable
        prefix_tokens = tokenizer.encode(text[:start_idx], add_special_tokens=True)
        # Tokenize the text up to the end of the variable
        full_tokens = tokenizer.encode(text[:end_idx], add_special_tokens=True)
        
        # Check if tokenizing the prefix has added or changed a token
        if len(prefix_tokens) > 1 and prefix_tokens[-1] != full_tokens[len(prefix_tokens)-1]:
            prefix_tokens = prefix_tokens[:-1]

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
        token_losses = tokenwise_loss(inputs["input_ids"], outputs.logits)
        token_losses = token_losses.cpu()
    
    attribute_losses = {}
    for attribute, token_indices in token_ranges:
        if token_indices:  # Check if we found any tokens
            first_token_loss = np.log2(np.exp(float(token_losses[token_indices[0]])))
            all_tokens_loss = np.log2(np.exp(float(sum(token_losses[i] for i in token_indices))))
            attribute_losses[attribute] = (first_token_loss, all_tokens_loss)
    return attribute_losses

def get_qa_token_ranges(text: str, tokenizer) -> List[int]:
    """Get token indices for the answer portion of a QA text."""
    # Split on "Answer: " to get the answer portion
    parts = text.split("Answer: ")
    if len(parts) != 2:
        return []
        
    # Get position of answer in original text
    answer_start = len(parts[0]) + len("Answer: ")
    
    # Tokenize text up to answer start and full text
    prefix_tokens = tokenizer.encode(text[:answer_start], add_special_tokens=True)
    full_tokens = tokenizer.encode(text, add_special_tokens=True)

    # Check if tokenizing the prefix has added or changed a token
    if len(prefix_tokens) > 1 and prefix_tokens[-1] != full_tokens[len(prefix_tokens)-1]:
        prefix_tokens = prefix_tokens[:-1]

    # Answer tokens are between these lengths
    return list(range(len(prefix_tokens)-1, len(full_tokens)-1))

def calculate_capacities(total_losses, dataset, N):
    """Calculate per-attribute and 2nd order QA capacities."""
    # Collect attribute values for entropy calculation
    attribute_values = {
        'name': [], 'birth_date': [], 'birth_city': [],
        'university': [], 'employer': [],
        'child': [], 'best_friend': [], 'worst_enemy': []
    }
    
    for profile in tqdm(dataset, desc="Collecting attribute values"):
        for attr in attribute_values:
            if attr in ['parent', 'child', 'best_friend', 'worst_enemy']:
                attribute_values[attr].append(profile[attr]['name'])
            else:
                attribute_values[attr].append(profile[attr])

    # Calculate entropies and capacities
    capacities = {}
    total_capacity = 0
    
    print("\nCapacity calculations:")
    print(f"{'Attribute':<15} {'Entropy':<12} {'Loss':<12} {'Capacity':<12}")
    print("-" * 51)
    
    for attr, values in tqdm(attribute_values.items(), desc="Calculating capacities"):
        if attr not in total_losses:
            continue
            
        entropy = calculate_entropy(values, attr)
        avg_loss = total_losses[attr][1] / total_losses[attr][2]  # Using all_tokens_loss
        capacity = (entropy - avg_loss) * N
        
        capacities[attr] = capacity
        total_capacity += capacity
        
        print(f"{attr:<15} {entropy:<12.4f} {avg_loss:<12.4f} {capacity:<12.4f}")
        print(f"Total bio capacity: {total_capacity:.4f}")

    # Calculate 2nd order QA capacity
    if 2 in total_losses:  # Check if we have 2nd order QA losses
        # Average entropy excluding name
        entropies = [calculate_entropy(values, attr) 
                    for attr, values in attribute_values.items() 
                    if attr != 'name']
        avg_entropy = sum(entropies) / len(entropies)
        
        qa_loss_2 = total_losses[2][1] / total_losses[2][2]  # Using all_tokens_loss
        qa_loss_1 = total_losses[1][1] / total_losses[1][2]  # Using all_tokens_loss
        qa_capacity_2 = max(0, (avg_entropy - qa_loss_2) * N * len(entropies))
        qa_capacity_1 = max(0, (avg_entropy - qa_loss_1) * N * len(entropies))
        
        print("\n2 hop QA capacity:")
        print(f"Average entropy (excluding name): {avg_entropy:.4f}")
        print(f"2 hop QA loss: {qa_loss_2:.4f}")
        print(f"1 hop QA loss: {qa_loss_1:.4f}")
        print(f"2 hop QA capacity: {qa_capacity_2:.4f}")
        print(f"1 hop QA capacity: {qa_capacity_1:.4f}")

    print(f"1 hop QA capacity + name capacity: {qa_capacity_1 + capacities['name']:.4f}")

    return capacities, qa_capacity_2, qa_capacity_1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=int, default=650000)
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    results = []
    param_sizes = [2_000_000, 1_000_000]
    N_sizes = [10000, 25000]
    orders = [1, 2]
    wds = [0.1, 0.01]

    for wd in wds:
        for num_parameters in param_sizes:
            for N in N_sizes:
                for order in orders:
                    print(f"\nEvaluating model with {num_parameters} parameters, N={N}, order={order}, weight decay={wd}")
                    
                    continued_number = "continued_2" if num_parameters == 1_000_000 else "continued_3"
                    if wd == 0.01:
                        continued_number = 'light_wd'
                    # Load model and tokenizer
                    model_path = get_project_root() / f"results/n{N}_p{num_parameters}_o{order}_{continued_number}/checkpoint-{args.checkpoint}"
                    if not model_path.exists():
                        print(f"Skipping {model_path} - path does not exist")
                        continue
                    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
                    if num_parameters == 1_000_000:
                        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
                    else:
                        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

                    # Load datasets
                    if N == 10000 and num_parameters == 1_000_000:
                        bios_dataset = load_dataset("EleutherAI/transformer-reasoning-bios-dataset-10000", revision="a4029b437d3d96cb591d12b89b6c05bade648b9d")['train']
                    else:
                        bios_dataset = load_dataset(f"EleutherAI/transformer-reasoning-bios-dataset-{N}")['train']
                    templates = load_templates(get_project_root() / "generated_data/templates")
                    qa_dataset = load_from_disk(str(get_project_root() / f"generated_data/qa_dataset_{N}")).filter(lambda x: x['questions.order'] <= 2)
                    profiles_dataset = load_from_disk(str(get_project_root() / f"generated_data/profiles_dataset_{N}"))

                    # Sample random subset
                    sample_indices = random.sample(range(len(bios_dataset)), args.sample_size)
                    sample_bios = bios_dataset.select(sample_indices)

                    # Sample proportionally from each split
                    splits = ['train', 'validation', 'heldout_profiles']
                    split_sizes = {split: len(qa_dataset[split]) for split in splits}
                    total_size = sum(split_sizes.values())
                    
                    sample_qa = []
                    for split in splits:
                        split_sample_size = int((split_sizes[split] / total_size) * args.sample_size)
                        split_indices = random.sample(range(len(qa_dataset[split])), split_sample_size)
                        sample_qa.extend(qa_dataset[split].select(split_indices))
                    
                    # Shuffle the combined samples
                    random.shuffle(sample_qa)

                    # Initialize aggregated losses
                    total_losses = {}  # attribute/order -> (sum_first_token_loss, sum_all_tokens_loss, count)

                    # Process each bio
                    for bio in tqdm(sample_bios, desc="Processing bios"):
                        # Find variable positions
                        matches = find_template_matches(bio, templates)
                        if not matches:
                            continue

                        # Get token ranges for variables
                        token_ranges = get_token_ranges(bio['bio'], tokenizer, matches)
                        
                        # Calculate losses
                        losses = calculate_losses(model, tokenizer, bio['bio'], token_ranges)
                        
                        # Aggregate losses
                        for attr, (first_loss, all_loss) in losses.items():
                            if attr not in total_losses:
                                total_losses[attr] = [0.0, 0.0, 0]
                            total_losses[attr][0] += first_loss
                            total_losses[attr][1] += all_loss
                            total_losses[attr][2] += 1

                    # Process QA samples
                    for qa in tqdm(sample_qa, desc="Processing QA"):
                        # Format QA text
                        text = f"Question: {qa['questions.question']} Answer: {qa['questions.answer']}"
                        
                        # Get token ranges for answer
                        token_indices = get_qa_token_ranges(text, tokenizer)
                        if not token_indices:
                            continue
                            
                        # Calculate losses
                        inputs = tokenizer(text, return_tensors="pt").to(model.device)
                        with torch.no_grad():
                            outputs = model(**inputs, labels=inputs["input_ids"])
                            token_losses = tokenwise_loss(inputs["input_ids"], outputs.logits)
                            token_losses = token_losses.cpu()
                        
                        # Calculate and store losses
                        question_order = qa['questions.order']
                        first_loss = np.log2(np.exp(float(token_losses[token_indices[0]])))
                        all_loss = np.log2(np.exp(float(sum(token_losses[i] for i in token_indices))))
                        
                        if question_order not in total_losses:
                            total_losses[question_order] = [0.0, 0.0, 0]
                        total_losses[question_order][0] += first_loss
                        total_losses[question_order][1] += all_loss
                        total_losses[question_order][2] += 1

                    # Calculate capacities
                    capacities, qa_capacity_2, qa_capacity_1 = calculate_capacities(total_losses, profiles_dataset, N)
                    results.append({
                        'num_params': num_parameters,
                        'N': N,
                        'order': order,
                        'name_capacity': capacities['name'],
                        'other_bio_capacity': sum(capacities[attr] for attr in capacities if attr != 'name'),
                        'qa_capacity_2': qa_capacity_2,
                        'qa_capacity_1': qa_capacity_1,
                        '1hop_capacity': qa_capacity_1 + capacities['name'],
                        'weight_decay': wd
                    })
                    
                    # Free up GPU memory
                    del model
                    torch.cuda.empty_cache()

    # Create plot
    plt.figure(figsize=(10, 6))
    for result in results:
        bio_per_param = result['1hop_capacity'] / result['num_params']
        qa_per_param = result['qa_capacity_2'] / result['num_params']
        plt.scatter(bio_per_param, qa_per_param, 
                   label=f"N={result['N']}, params={result['num_params']}, order={result['order']}, wd={result['weight_decay']}")

    # Add reference line y = 2-x
    x = np.linspace(0, 2, 100)
    plt.plot(x, 2-x, '--', color='gray', label='y = 2-x')

    plt.xlabel('Bio Capacity / Number of Parameters')
    plt.ylabel('QA Capacity / Number of Parameters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Normalized Bio Capacity vs QA Capacity')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(get_project_root() / 'results/capacity_plot.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
