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
from transformer_reasoning.evaluation.eval_utils import get_qa_token_ranges, get_token_ranges, tokenwise_loss
import matplotlib.pyplot as plt
import pandas as pd

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

def find_template_matches(bio: Dict, bio_text: str, templates: Dict[str, List[str]]) -> List[Tuple[str, str, int, int]]:
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
                text_pos = bio_text.find(pattern)
                for var, value in values.items():
                    var_pos = pattern.find(value)
                    start_idx = text_pos + var_pos
                    end_idx = start_idx + len(value)
                    matches.append((var, value, start_idx, end_idx))
                break  # Move to next attribute once we find a match
                
    return matches



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

def filename_schemes(order, N, num_parameters, wd, finite):

    if finite:
        continued_str = "continued_3" if (num_parameters != 1_000_000) and (wd == 0.1) else "continued_2"
        parent_dir = get_project_root() / f"results/n{N}_p{num_parameters}_o{order}_wd{wd}_{continued_str}"
    else:
        if wd == 0.1:
            parent_dir = get_project_root() / f"results/n{N}_p{num_parameters}_o{order}_wd{wd}_infinite"
            if not parent_dir.exists():
                parent_dir = get_project_root() / f"results/n{N}_p{num_parameters}_o{order}_infinite"
            parent_dir = get_project_root() / f"results/n{N}_p{num_parameters}_o{order}_wd{wd}_infinite"
        else:
            parent_dir = get_project_root() / f"results/n{N}_p{num_parameters}_o{order}_wd{wd}_infinite"
    if not parent_dir.exists():
        print(f"Skipping {parent_dir} - path does not exist")
        return None
    return parent_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--finite", action="store_true")
    args = parser.parse_args()

    results = []
    param_sizes = [5_000_000, 2_000_000, 1_500_000, 1_000_000, 500_000]
    N_sizes = [10000, 25000]
    orders = [1, 2]
    wds = [0.1, 0.01]

    for wd in wds:
        for num_parameters in param_sizes:
            for N in N_sizes:
                for order in orders:
                    print(f"\nEvaluating model with {num_parameters} parameters, N={N}, order={order}, weight decay={wd}")
                    
                    # Load model and tokenizer
                    parent_dir = filename_schemes(order, N, num_parameters, wd, args.finite)
                    if not parent_dir:
                        continue

                    if args.checkpoint == "latest":
                        # Find all checkpoint directories
                        checkpoints = [d for d in parent_dir.glob("checkpoint-*") if d.is_dir()]
                        if not checkpoints:
                            print(f"Skipping {parent_dir} - no checkpoints found")
                            continue
                        # Sort by checkpoint number and get the latest
                        latest = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
                        model_path = latest
                    else:
                        model_path = parent_dir / f"checkpoint-{args.checkpoint}"
                    
                    assert model_path.exists(), f"Model path does not exist: {model_path}"

                    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
                    # Load datasets
                    if num_parameters == 1_000_000 and args.finite:
                        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
                        if N == 10000:
                            bios_dataset = load_dataset("EleutherAI/transformer-reasoning-bios-dataset-10000", revision="a4029b437d3d96cb591d12b89b6c05bade648b9d")['train']
                        else:
                            bios_dataset = load_dataset(f"EleutherAI/transformer-reasoning-bios-dataset-{N}")['train']
                    else:
                        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
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
                        bio_text = bio['bio']
                        if not args.finite:
                            bio_text = '<|endoftext|>' + bio_text

                        matches = find_template_matches(bio, bio_text, templates)
                        if not matches:
                            continue
                        # Get token ranges for variables
                        token_ranges = get_token_ranges(bio_text, tokenizer, matches)
                        
                        # Calculate losses
                        losses = calculate_losses(model, tokenizer, bio_text, token_ranges)
                        
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
                        if not args.finite:
                            text = "<|endoftext|>" + text
                        
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
    
    # Define markers for different orders
    order_markers = {1: 'o', 2: '^'}
    
    # Create fixed color map for parameter counts from 500k to 10M, log-spaced
    all_param_sizes = np.logspace(np.log10(500_000), np.log10(10_000_000), 20)  # 20 discrete colors
    param_colors = plt.cm.viridis(np.linspace(0, 1, len(all_param_sizes)))
    
    # Function to find closest predefined parameter size
    def get_nearest_param_color(param_count):
        idx = np.abs(all_param_sizes - param_count).argmin()
        return param_colors[idx]
    
    for result in results:
        bio_per_param = result['1hop_capacity'] / result['num_params']
        qa_per_param = result['qa_capacity_2'] / result['num_params']
        plt.scatter(bio_per_param, qa_per_param, 
                   marker=order_markers[result['order']],
                   color=get_nearest_param_color(result['num_params']),
                   label=f"N={result['N']}, params={result['num_params']}, order={result['order']}, wd={result['weight_decay']}")

    # Add reference line y = 2-x
    x = np.linspace(0, 2, 100)
    plt.plot(x, 2-x, '--', color='gray', label='y = 2-x')

    plt.xlabel('1 Hop Capacity / Number of Parameters')
    plt.ylabel('2 hop Capacity / Number of Parameters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('1 Hop vs 2 Hop Capacity')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(get_project_root() / f'results/capacity_plot_{args.finite}.png', bbox_inches='tight')
    plt.close()

    # Create separate plots for each N value
    for N in N_sizes:
        plt.figure(figsize=(10, 6))
        
        # Filter results for this N
        N_results = [r for r in results if r['N'] == N]
        
        # Calculate theoretical max capacities using calculate_capacities with zero losses
        zero_losses = {
            'name': [0.0, 0.0, 1],  # [first_token_loss, all_tokens_loss, count]
            'birth_date': [0.0, 0.0, 1],
            'birth_city': [0.0, 0.0, 1],
            'university': [0.0, 0.0, 1],
            'employer': [0.0, 0.0, 1],
            'child': [0.0, 0.0, 1],
            'best_friend': [0.0, 0.0, 1],
            'worst_enemy': [0.0, 0.0, 1],
            1: [0.0, 0.0, 1],  # For 1-hop QA
            2: [0.0, 0.0, 1],  # For 2-hop QA
        }
        max_capacities, max_qa_2, max_qa_1 = calculate_capacities(zero_losses, profiles_dataset, N)
        df = pd.DataFrame(results)
        N_results = df[df['N'] == N]

        # Plot unnormalized capacities
        for _, result in N_results.iterrows():
            plt.scatter(result['1hop_capacity'], result['qa_capacity_2'],
                       marker=order_markers[result['order']],
                       color=get_nearest_param_color(result['num_params']),
                       label=f"params={result['num_params']}, order={result['order']}, wd={result['weight_decay']}")
        
        # Add reference lines for max capacities
        plt.axvline(x=max_qa_1 + max_capacities['name'], color='gray', linestyle='--', alpha=0.5, 
                   label=f'Max Bio Capacity ({(max_qa_1 + max_capacities["name"])/1e6:.1f}M)')
        plt.axhline(y=max_qa_2, color='gray', linestyle='--', alpha=0.5,
                   label=f'Max QA Capacity ({max_qa_2/1e6:.1f}M)')
        
        plt.xlabel('Bio Capacity (bits)')
        plt.ylabel('QA Capacity (bits)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        inf_str = "infinite" if not args.finite else "finite"
        plt.title(f'Bio Capacity vs QA Capacity (N={N}, {inf_str} dataset)')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(get_project_root() / f'results/capacity_plot_N{N}_{args.finite}.png', bbox_inches='tight')
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(get_project_root() / f'results/capacity_results_{args.finite}.csv', index=False)

if __name__ == "__main__":
    main()
