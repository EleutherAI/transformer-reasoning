import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from datasets import load_from_disk, Dataset, load_dataset
from tqdm import tqdm
from transformers import LlamaForCausalLM
from torch.optim import AdamW
from schedulefree import AdamWScheduleFree
from transformers import TrainingArguments, Trainer
import glob
import os

from transformer_reasoning.utils import get_project_root
from transformer_reasoning.train.train_utils import create_model_and_tokenizer, InfiniteQADataset, train_parallel_models


def test_checkpoint_eval_consistency():
    # 1. Create a small model and dataset
    model, tokenizer, num_params = create_model_and_tokenizer(1000000, num_layers=4)
    profiles = load_dataset("EleutherAI/profiles_dataset_10000_uniform")['train']
    
    train_dataset = InfiniteQADataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=[1],
        qa_prob=1,
        qa_indices=list(range(1000))  # Only use first 100 profiles for QA
    )
    
    # Create dummy eval datasets (required by train_parallel_models)
    onehop_dataset = InfiniteQADataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=[1],
        qa_prob=1,
        qa_indices=list(range(1000))
    )
    
    twohop_dataset = InfiniteQADataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=[2],
        qa_prob=1,
        qa_indices=list(range(1000))
    )
    
    # Set up models dict
    models_dict = {
        'model': [model],
        'num_params': [num_params],
        'num_layers': [2],
        'N_profiles': [1000],
        'orders': [[1]],
        'wd': [0.1],
        'lr': [1e-3],
        'beta1': [0.9],
    }
    
    # Set up minimal training args
    class Args:
        def __init__(self):
            self.num_epochs = 1
            self.train_batch_size = 32
            self.eval_batch_size = 32
            self.gpus = [0]
            self.output_dir = "test_checkpoint"
            self.resume_from_checkpoint = None
            self.lr = 0.001
            self.beta1 = 0.9
            self.wd = 0.1
    args = Args()
    
    # Train for 1 epoch
    train_parallel_models(models_dict, train_dataset, onehop_dataset, twohop_dataset, args, ["test_checkpoint"], dl_workers=20)
    
    # Create sample input and get output from freshly trained model
    sample_text = "Question: Who is Alice? Answer:"
    inputs = tokenizer(sample_text, return_tensors="pt")
    del inputs['token_type_ids']


    # Load model and optimizer from checkpoint
    checkpoints = glob.glob(os.path.join("test_checkpoint", "checkpoint-*"))
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
    loaded_model = LlamaForCausalLM.from_pretrained(latest_checkpoint)
    
    with torch.no_grad():
        saved_output = loaded_model(**inputs.to(loaded_model.device)).logits

    optimizer_state = torch.load(f"{latest_checkpoint}/optimizer.pt")
    loaded_optimizer = AdamWScheduleFree(
        loaded_model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.1,
        warmup_steps=1000
    )
    loaded_optimizer.load_state_dict(optimizer_state['optimizer_state_dict'])
    
    # Test in eval mode
    loaded_model.eval()
    loaded_optimizer.eval()
    with torch.no_grad():
        eval_output = loaded_model(**inputs.to(loaded_model.device)).logits
    
    # Test in train mode
    loaded_model.train()
    loaded_optimizer.train()
    with torch.no_grad():
        train_output = loaded_model(**inputs.to(loaded_model.device)).logits
    
    # Clean up
    import shutil
    shutil.rmtree("test_checkpoint")
    
    # Compare outputs
    abs_train_eval_max_diff = (train_output.cpu() - eval_output.cpu()).abs().max()
    abs_saved_eval_max_diff = (saved_output.cpu() - eval_output.cpu()).abs().max()
    print(f"Max diff between train and eval: {abs_train_eval_max_diff}")
    print(f"Max diff between saved and eval: {abs_saved_eval_max_diff}")
    
    assert torch.allclose(saved_output.cpu(), eval_output.cpu(), rtol=1e-3),\
        "Model outputs differ between saved and loaded versions"

