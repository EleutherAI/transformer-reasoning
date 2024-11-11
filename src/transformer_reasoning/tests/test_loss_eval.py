import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from datasets import load_from_disk, Dataset
from transformer_reasoning.utils import get_project_root
from transformer_reasoning.evaluation.measure_capacity import (
    get_qa_token_ranges, tokenwise_loss, filename_schemes
)
from transformer_reasoning.evaluation.eval_utils import evaluate_qa_loss
import numpy as np
from tqdm import tqdm
import random

def measure_capacity_method(model, tokenizer, qa_data, num_samples=1000, insert_eos=True):
    """Replicate measure_capacity.py's loss calculation"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for qa in tqdm(qa_data, desc="Processing QA (measure_capacity method)"):
        text = f"Question: {qa['questions.question']} Answer: {qa['questions.answer']}"
        if insert_eos:
            text = "<|endoftext|>" + text
            
        token_indices = get_qa_token_ranges(text, tokenizer)
        if not token_indices:
            continue
            
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            token_losses = tokenwise_loss(inputs["input_ids"], outputs.logits)
            token_losses = token_losses.cpu()
        
        losses = [float(token_losses[i]) for i in token_indices]
        total_loss += sum(losses)
        total_tokens += len(losses)
    
    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def loss_over_time_method(model, tokenizer, qa_data):
    """Replicate loss_over_time.py's loss calculation"""
    return evaluate_qa_loss(model, qa_data, model.device, len(qa_data))

def test_loss_calculation_methods():
    # Test parameters
    N = 10000
    order = 1 
    params = 1_000_000
    wd = 0.1
    finite = True
    checkpoint = "10000"
    num_samples = 100  # Reduced for faster testing

    # Load model and data
    parent_dir = filename_schemes(order, N, params, wd)
    assert parent_dir is not None, "Model directory not found"
    
    model_path = parent_dir / f"checkpoint-{checkpoint}"
    assert model_path.exists(), f"Checkpoint not found: {model_path}"

    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    # Load and prepare dataset
    qa_dataset = load_from_disk(str(get_project_root() / f"generated_data/qa_dataset_{N}"))
    qa_data_combined = []
    for split in ['train', 'validation', 'heldout_profiles']:
        qa_data_combined.extend(qa_dataset[split])
    qa_data_1hop = [x for x in qa_data_combined if x['questions.order'] == 1]
    
    random.seed(42)
    sample_indices = random.sample(range(len(qa_data_1hop)), num_samples)
    qa_data_sample = [qa_data_1hop[i] for i in sample_indices]
    qa_data_sample = Dataset.from_list(qa_data_sample)
    
    # Calculate losses using both methods
    mc_loss = measure_capacity_method(model, tokenizer, qa_data_sample, num_samples)
    lot_loss = loss_over_time_method(model, tokenizer, qa_data_sample)
    
    # Assert the results are close
    assert abs(mc_loss - lot_loss) < 0.1, f"Loss difference too large: {abs(mc_loss - lot_loss):.4f}"
