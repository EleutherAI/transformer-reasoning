import math
from transformers import (
    LlamaForCausalLM, LlamaConfig, 
    AutoTokenizer, TrainerCallback, 
)
import torch
import os
import glob
import numpy as np
import pandas as pd
import json
from datetime import datetime

from typing import TypeVar, Union, List
from datasets import Dataset, DatasetDict
from multiprocessing import cpu_count
T = TypeVar("T", bound=Union[Dataset, DatasetDict])


from torch.utils.data import DataLoader
import random
from schedulefree import AdamWScheduleFree
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.optim import AdamW as TorchAdamW

from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from transformer_reasoning.train.dataset import InfiniteQADataset, load_and_prepare_datasets


def train_single_model(
        model,
        tokenizer,
        args,
        output_dir=None,
        curriculum=False,
        n_steps=None,
        debug=False
        ):
    early_saves = set([int(math.exp(i)) for i in torch.linspace(math.log(100), math.log(100000), 10).tolist()])
    
    # Get rank for distributed training
    is_main_process = True
    if torch.cuda.device_count() > 1 and 'WORLD_SIZE' in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        local_rank = dist.get_rank()
        is_main_process = local_rank == 0
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])
        dist.barrier()
    
    # Create or load optimizer
    start_step = 0
    latest_checkpoint = None
    if args.resume_from_checkpoint and output_dir:
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if checkpoints:
            if args.checkpoint_number:
                latest_checkpoint = [c for c in checkpoints if f"checkpoint-{args.checkpoint_number}" in c][0]
            else:
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
    optimizer, scheduler, start_step, heldout_sets = create_or_load_optimizer(model, args, latest_checkpoint)
    
    train_dataset = load_and_prepare_datasets(
        tokenizer,
        args.N, 
        orders=args.orders, 
        relations=args.relations, 
        hop_ratio=args.hop_ratio, 
        heldout_sets=heldout_sets,
        debug=debug
    )

    if args.curriculum:
        # Start with only 1-hop questions
        train_dataset.order_weights = [1.0, 0.0]
        print("Starting curriculum learning with 1-hop only")

    global_step = start_step
    
    if dist.is_initialized():
        args.train_batch_size = args.train_batch_size // dist.get_world_size()
        args.eval_batch_size = args.eval_batch_size // dist.get_world_size()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers if not debug else 0,
        pin_memory=True
    )
    # Create evaluation datasets using the same held-out sets as train_dataset
    if not debug:
        eval_modes = [
            "train_onehop",
            "train_twohop",
            "eval_first_people",
            "eval_relations",
            "eval_person_relation_pairs",
            "eval_second_people",
            "eval_second_attributes",
            "eval_second_person_attribute_pairs"
        ]
    else:
        eval_modes = []
    
    eval_datasets = {
        mode: InfiniteQADataset(
            profiles_dataset=train_dataset.profiles,
            tokenizer=train_dataset.tokenizer,
            max_seq_len=args.max_seq_length,
            orders=[1] if mode == "train_onehop" else [2],
            mode=mode.replace("train_onehop", "train").replace("train_twohop", "train"),
            heldout_sets=train_dataset.heldout_sets
        ) for mode in eval_modes
    }
    
    eval_loaders = {
        mode: DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=True
        ) for mode, dataset in eval_datasets.items()
    }
    
    results_dicts = []
    curriculum_switched = not curriculum
    
    # Initialize logging only on main process
    log_file = None
    if is_main_process:
        log_file = os.path.join('./logs', f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    # Store eval results at the start
    eval_losses = {}
    epoch = 0
    if is_main_process:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader

    # Training loop
    last_batch = None
    for epoch in range(args.num_epochs):
        # Set epoch for dataset
        train_dataset.set_epoch(epoch)
        
        model.train()
        if hasattr(optimizer, 'train'):
            optimizer.train()

        for i, batch in enumerate(pbar):
            last_batch = batch
            batch = {k: v.to(model.device) if hasattr(v, 'to') else v 
                    for k, v in batch.items()}
            del batch['text']
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            optimizer.zero_grad()
            

            global_step += 1

            should_save = (global_step < 100000 and global_step in early_saves) or \
                (global_step >= 100000 and global_step % 100000 == 0)
            
            # Update progress bar with both train and stored eval losses
            postfix = {
                'step': global_step,
                **{f'{mode}_loss': eval_losses.get(mode, 0) for mode in eval_modes}
            }
            if is_main_process:
                pbar.set_postfix(postfix)


            if should_save:
                if dist.is_initialized():
                    dist.barrier()  # Sync before eval
                
                # All ranks participate in evaluation
                model.eval()
                if hasattr(optimizer, 'eval'):
                    optimizer.eval()
                
                results = {}
                for mode, loader in eval_loaders.items():
                    results[mode] = evaluate_single_model(model, loader, global_step, mode)
                
                # Only rank 0 saves results and updates progress
                if is_main_process:
                    results_dicts.extend(results.values())
                    
                    # Save checkpoint
                    if output_dir:
                        model_path = f"{output_dir}/checkpoint-{global_step}"
                        if isinstance(model, DDP):
                            model.module.save_pretrained(model_path)
                        else:
                            model.save_pretrained(model_path)
                        
                        optimizer_state = {
                            'optimizer_state_dict': optimizer.state_dict(),
                            'step': global_step,
                            'heldout_sets': train_dataset.heldout_sets,
                            **{f'{mode}_loss': results[mode]['loss'] for mode in eval_modes}
                        }
                        if scheduler:
                            optimizer_state['scheduler_state_dict'] = scheduler.state_dict()
                        torch.save(optimizer_state, f"{model_path}/optimizer.pt")
                    
                    # Log results
                    if not debug:
                        results_df = pd.DataFrame(results_dicts)
                        if os.path.exists(f"{output_dir}/eval_results.csv"):
                            results_df = results_df[results_df['global_step'] == global_step]
                            results_df.to_csv(f"{output_dir}/eval_results.csv", mode='a', header=False, index=False)
                        else:
                            results_df.to_csv(f"{output_dir}/eval_results.csv", index=False)
                    
                    if log_file:
                        log_entry = {
                            'step': global_step,
                            'epoch': epoch,
                            'timestamp': datetime.now().isoformat(),
                            **{f'{mode}_loss': results[mode]['loss'] for mode in eval_modes}
                        }
                        with open(log_file, 'a') as f:
                            f.write(json.dumps(log_entry) + '\n')
                    
                    # Update progress bar
                    pbar.set_postfix({
                        **{f'{mode}_loss': results[mode]['loss'] for mode in eval_modes},
                        'step': global_step,
                    })
                
                if dist.is_initialized():
                    dist.barrier()  # Sync after saving
                
                # Back to training mode
                model.train()
                if hasattr(optimizer, 'train'):
                    optimizer.train()
            
                # Check curriculum condition (using train_twohop loss)
                if curriculum and not curriculum_switched:
                    train_twohop_loss = next(r['loss'] for r in results_dicts[-len(eval_modes):] 
                                           if r['mode'] == 'train_twohop')
                    if train_twohop_loss < 1.0:
                        print("\nSwitching to full curriculum - enabling 2-hop questions")
                        train_dataset.order_weights = [0.1, 1.0]
                        curriculum_switched = True


            if n_steps and global_step >= n_steps:
                break

    if debug:
        return optimizer, last_batch

def evaluate_single_model(model, eval_loader, global_step, mode):
    """Evaluate model on a specific evaluation mode"""
    results_dict = {
        'loss': 0,
        'global_step': global_step,
        'mode': mode,
        'parameter_l2': 0
    }

    with torch.no_grad():
        eval_questions = 0
        pbar = tqdm(eval_loader, desc=f"Evaluating {mode}")
        for eval_batch in pbar:
            labels = eval_batch['labels']
            last_neg = labels[:,-1] == -100
            if not last_neg.any():
                continue
            filtered_inputs = eval_batch['input_ids'][last_neg]
            filtered_labels = eval_batch['labels'][last_neg]
            is_pos = filtered_labels >= 0
            block_starts = ((is_pos) & (~torch.roll(is_pos, 1))).sum()
            eval_questions += block_starts.item()

            outputs = model(
                input_ids=filtered_inputs.to(model.device),
                labels=filtered_labels.to(model.device)
            )
            results_dict['loss'] += outputs.loss.item() * is_pos.sum().item()
            
            all_params = torch.cat([p.flatten() for p in model.parameters()])
            results_dict['parameter_l2'] = torch.norm(all_params).item() / np.sqrt(all_params.numel())
            
            if eval_questions >= 1000:
                break
    
    results_dict['loss'] = results_dict['loss'] / eval_questions
    print(f"Step {global_step}: {mode} Loss = {results_dict['loss']}")
    return results_dict

def get_qa_token_ranges(text: str, tokenizer) -> List[int]:
    """Get token indices for the answer portion of a QA text."""
    parts = text.split("Answer:")
    if len(parts) != 2:
        return []
        
    answer_start = len(parts[0]) + len("Answer:")
    
    prefix_tokens = tokenizer.encode(text[:answer_start], add_special_tokens=True)
    full_tokens = tokenizer.encode(text, add_special_tokens=True)

    if len(prefix_tokens) > 1 and prefix_tokens[-1] != full_tokens[len(prefix_tokens)-1]:
        prefix_tokens = prefix_tokens[:-1]

    return [len(prefix_tokens), len(full_tokens)-1]


class LogConstantCheckpointCallback(TrainerCallback):
    def __init__(self, trainer):
        # Generate log-spaced steps until 20k
        self.early_saves = set([int(math.exp(i)) for i in torch.linspace(math.log(100), math.log(20000), 10).tolist()])
        
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step < 20000 and step in self.early_saves:
            control.should_save = True
            control.should_evaluate = True
        elif step >= 20000 and step % 20000 == 0:
            control.should_save = True
            control.should_evaluate = True
        return control


def calculate_model_size(num_params):
    return num_params * 4 / (1024 * 1024)  # Size in MB

def calculate_architecture(num_params, n_layers=4):
    resid_params = num_params - 10_000*32
    hidden_size = int(math.sqrt(resid_params / (n_layers * 12)))
    hidden_size = max(1, (hidden_size // 16)) * 16  # Round to nearest multiple of 16
    return n_layers, hidden_size


def create_model_and_tokenizer(num_params, num_layers=4):
    n_layers, hidden_size = calculate_architecture(num_params, num_layers)
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llama_multihop_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=n_layers,
        num_attention_heads=hidden_size // 16,
        max_position_embeddings=2048,
    )
    
    model = LlamaForCausalLM(config)
    
    real_num_params = sum(p.numel() for p in model.parameters()) 
    print(f"Model has {real_num_params} parameters")

    return model, tokenizer, real_num_params



def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> list[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names

def create_or_load_optimizer(model, args, checkpoint_path=None):
    """Create a new optimizer or load from checkpoint."""
    if args.optimizer_type == "schedulefree":
        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, 0.999),
            weight_decay=args.wd,
            warmup_steps=1000
        )
        scheduler = None
    else:  # regular AdamW with cosine or linear scheduler
        optimizer = TorchAdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, 0.999),
            weight_decay=args.wd
        )
        if args.optimizer_type == "adamw_cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=1000,
                num_training_steps=args.num_training_steps
            )
        else:  # linear decay
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=1000,
                num_training_steps=args.num_training_steps
            )
    
    start_step = 0
    heldout_sets = None
    if checkpoint_path:
        optimizer_path = f"{checkpoint_path}/optimizer.pt"
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path)
            optimizer.load_state_dict(optimizer_state['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in optimizer_state:
                scheduler.load_state_dict(optimizer_state['scheduler_state_dict'])
            start_step = optimizer_state['step']
            heldout_sets = optimizer_state.get('heldout_sets')
            print(f"Loaded optimizer state from step {start_step}")
    
    return optimizer, scheduler, start_step, heldout_sets   