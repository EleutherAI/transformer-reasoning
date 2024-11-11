import torch
from tqdm import tqdm
from typing import List, Dict
import random
import numpy as np


def evaluate_qa_loss(model: torch.nn.Module, dataset: List[Dict], device: str = "cuda", num_samples: int = 10000) -> float:
    """Evaluate average loss on QA data."""
    model.eval()
    total_loss = 0
    total_questions = 0
    

    for inputs in tqdm(dataset, desc="Evaluating QA loss"):
        labels = torch.tensor(inputs['labels'])
        last_neg = -1
        if labels[-1] != -100:
            last_neg = (labels == -100).nonzero()[-1].item() + 1
        
        labels = labels[:last_neg] 
        input_ids = torch.tensor(inputs['input_ids'][:last_neg])

        with torch.no_grad():
            outputs = model(input_ids=input_ids.view(1, -1).to(device), labels=labels.view(1, -1).to(device))

        is_pos = labels >= 0
        block_starts = ((is_pos) & (~torch.roll(is_pos, 1))).sum()
        
        total_loss += outputs.loss.item()
        total_questions += block_starts

    return (total_loss * is_pos.sum().item() / total_questions).item()
