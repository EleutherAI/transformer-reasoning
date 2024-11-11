import torch
from torch.nn import CrossEntropyLoss
from pathlib import Path
from transformer_reasoning.utils import get_project_root

def tokenwise_loss(inputs, logits):
    """Calculate per-token loss."""
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def filename_schemes(min_order, max_order, N, num_parameters, wd):
    parent_dir = get_project_root() / f"results/n{N}_p{num_parameters}_omin{min_order}_omax{max_order}_wd{wd}_infinite"
    if not parent_dir.exists():
        print(f"Skipping {parent_dir} - path does not exist")
        return None
    return parent_dir