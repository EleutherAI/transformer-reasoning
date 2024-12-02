import torch
from typing import List, Tuple, Dict
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import re
from datasets import Dataset
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

def find_template_matches(bio: Dict, bio_text: str, templates: Dict[str, List[str]]) -> List[Tuple[str, str, int, int]]:
    """
    Find all template matches in the bio and return their variable positions.
    Returns: List of (attribute, value, start_idx, end_idx) tuples
    """
    matches = []
    for attribute, attribute_templates in templates.items():
        for template in attribute_templates:
            pattern = template
            values = {}  
            
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
                
            if pattern in bio['bio']:
                text_pos = bio_text.find(pattern)
                for var, value in values.items():
                    var_pos = pattern.find(value)
                    start_idx = text_pos + var_pos
                    end_idx = start_idx + len(value)
                    matches.append((var, value, start_idx, end_idx))
                break
                
    return matches

def get_token_ranges(text: str, tokenizer, matches: List[Tuple[str, str, int, int]]) -> List[Tuple[str, List[int]]]:
    """
    Convert character ranges to token ranges.
    Returns: List of (attribute, token_indices) tuples
    """
    token_ranges = []
    for attribute, value, start_idx, end_idx in matches:
        prefix_tokens = tokenizer.encode(text[:start_idx], add_special_tokens=True)
        full_tokens = tokenizer.encode(text[:end_idx], add_special_tokens=True)
        
        if len(prefix_tokens) > 1 and prefix_tokens[-1] != full_tokens[len(prefix_tokens)-1]:
            prefix_tokens = prefix_tokens[:-1]

        var_token_indices = list(range(len(prefix_tokens)-1, len(full_tokens)-1))
        token_ranges.append((attribute, var_token_indices))
    return token_ranges

def evaluate_bio_loss(model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: Dataset, device: str = "cuda", num_samples: int = 10000) -> float:
    """Evaluate average loss on biographical data."""
    from transformer_reasoning.evaluation.eval_utils import tokenwise_loss
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for item in tqdm(dataset.select(range(num_samples)), desc="Evaluating bio loss"):
        text = item['bio']
        inputs = tokenizer(text, return_tensors="pt").to(device)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses = tokenwise_loss(inputs["input_ids"], outputs.logits)
            total_loss += losses.sum().item()
            total_tokens += losses.numel()
    
    return total_loss / total_tokens