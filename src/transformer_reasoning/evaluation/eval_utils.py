import torch
from torch.nn import CrossEntropyLoss
from typing import List, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import random

def tokenwise_loss(inputs, logits):
    """Calculate per-token loss."""
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

def evaluate_bio_loss(model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: Dataset, device: str = "cuda", num_samples: int = 10000, insert_eos: bool = True) -> float:
    """Evaluate average loss on biographical data."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for item in tqdm(dataset.select(range(num_samples)), desc="Evaluating bio loss"):
        text = item['bio']
        if insert_eos:
            text = f"<|endoftext|>{text}"
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses = tokenwise_loss(inputs["input_ids"], outputs.logits)
            total_loss += losses.sum().item()
            total_tokens += losses.numel()
    
    return total_loss / total_tokens

def evaluate_qa_loss(model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: Dataset, device: str = "cuda", num_samples: int = 10000, insert_eos: bool = True) -> float:
    """Evaluate average loss on QA data."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    # Use random.sample instead of torch.randperm
    indices = random.sample(range(len(dataset)), num_samples)
    
    
    for item in tqdm([dataset[i] for i in indices], desc="Evaluating QA loss"):
        text = f"Question: {item['questions.question']} Answer: {item['questions.answer']}"
        if insert_eos:
            text = f"<|endoftext|>{text}"
        
        token_indices = get_qa_token_ranges(text, tokenizer)
        if not token_indices:
            continue

        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses = tokenwise_loss(inputs["input_ids"], outputs.logits)
            token_loss = losses.cpu()

        losses = [float(token_loss[i]) for i in token_indices]
        total_loss += sum(losses)
        total_tokens += len(losses)

    return total_loss / total_tokens 

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