import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from datasets import load_from_disk, Dataset
from transformer_reasoning.utils import get_project_root
from tqdm import tqdm
from transformers import LlamaForCausalLM
from torch.optim import AdamW
from schedulefree import AdamWScheduleFree
from transformers import TrainingArguments, Trainer

import torch.nn as nn

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

def test_checkpoint_eval_consistency():
    # 1. Load model and create sample data
    checkpoint_path = "results/n30000_p1166688_omin1_omax2_wd0.1_infinite_schedule_free_l4/checkpoint-260000"
    model = LlamaForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Create sample input
    sample_text = "Question: Who is Alice? Answer:"
    inputs = tokenizer(sample_text, return_tensors="pt")
    del inputs['token_type_ids']
    with torch.no_grad():
        saved_output = model(**inputs.to(model.device)).logits

    # 2. Set up training arguments matching the checkpoint
    training_args = TrainingArguments(
        output_dir="dummy",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        optim="schedule_free_adamw",
        lr_scheduler_type="constant",
        learning_rate=1e-3,
        weight_decay=0.1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
    )
    
    optimizer = trainer.create_optimizer()
    optimizer_state = torch.load(f"{checkpoint_path}/optimizer.pt")
    optimizer.load_state_dict(optimizer_state)
    
    for group in optimizer.param_groups:
        assert group['train_mode'] == False, "Optimizer not properly set to eval mode"
    model.eval()
    optimizer.eval()

    with torch.no_grad():
        eval_output = model(**inputs.to(model.device)).logits

    model.train()
    optimizer.train()

    with torch.no_grad():
        train_output = model(**inputs.to(model.device)).logits


    abs_train_eval_max_diff = (train_output.cpu() - eval_output.cpu()).abs().max()
    abs_saved_eval_max_diff = (saved_output.cpu() - eval_output.cpu()).abs().max()
    print(f"Max diff between train and eval: {abs_train_eval_max_diff}")
    print(f"Max diff between saved and eval: {abs_saved_eval_max_diff}")

    assert torch.allclose(saved_output.cpu(), eval_output.cpu(), rtol=1e-3),\
        "Model outputs differ between train and eval modes"

