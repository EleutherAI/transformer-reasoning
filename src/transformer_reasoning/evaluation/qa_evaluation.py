import torch
from tqdm import tqdm
from typing import List, Dict
from datasets import Dataset
import numpy as np
from torch.utils.data import DataLoader


def evaluate_qa_loss(model: torch.nn.Module, dataset: List[Dict], device: str = "cuda", num_samples: int = 10000) -> float:
    """Evaluate average loss on QA data."""
    model.eval()
    total_loss = 0
    total_questions = 0
    
    eval_dataset = Dataset.from_list(dataset)

    dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    for inputs in tqdm(dataloader, desc="Evaluating QA loss"):
        labels = torch.stack(inputs['labels']).T
        last_neg = -1

        # Pick only chunks where last label is -100
        last_neg = labels[:,-1] == -100

        labels = labels[last_neg] 
        input_ids = torch.stack(inputs['input_ids']).T[last_neg]

        with torch.no_grad():
            outputs = model(input_ids=input_ids.to(device), labels=labels.to(device))

        is_pos = labels >= 0
        block_starts = ((is_pos) & (~torch.roll(is_pos, 1))).sum()
        
        total_loss += outputs.loss.item() * is_pos.sum().item()
        total_questions += block_starts

    return (total_loss / total_questions).item()
