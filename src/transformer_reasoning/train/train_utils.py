import math
from transformers import (
    LlamaForCausalLM, LlamaConfig, 
    AutoTokenizer, TrainerCallback, 
    PreTrainedTokenizerBase
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


from torch.utils.data import IterableDataset, DataLoader
import random
from schedulefree import AdamWScheduleFree
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW as TorchAdamW

from transformer_reasoning.generate_dataset.generate_bios import generate_bio, load_templates
from transformer_reasoning.generate_dataset.generate_qa_dataset import generate_question
from transformer_reasoning.utils import get_project_root

from tqdm import tqdm


def train_parallel_models(models_dict, train_dataset, onehop_dataset, twohop_dataset, args, output_dirs=None, dl_workers = 25):
    early_saves = set([int(math.exp(i)) for i in torch.linspace(math.log(100), math.log(100000), 10).tolist()])
    
    # Assign each model to a different GPU
    models = [model.to(f'cuda:{cuda_device}') for cuda_device, model in zip(args.gpus, models_dict['model'])]
    
    # Create or load optimizers
    optimizers = []
    schedulers = []
    start_step = 0
    for i, model in enumerate(models):
        latest_checkpoint = None
        if args.resume_from_checkpoint and output_dirs:
            for i, base_dir in enumerate(output_dirs):
                checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        optimizer, scheduler, step = create_or_load_optimizer(model, args, latest_checkpoint)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        if step > start_step:
            start_step = step
    
    global_step = start_step
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=dl_workers,
        pin_memory=True
    )
    
    onehop_loader = DataLoader(
        onehop_dataset,
        batch_size=args.eval_batch_size,
        num_workers=dl_workers,
        pin_memory=True
    )

    twohop_loader = DataLoader(
        twohop_dataset,
        batch_size=args.eval_batch_size,
        num_workers=dl_workers,
        pin_memory=True
    )
    
    results_dicts = []
    # Initialize logging
    log_file = os.path.join('./logs', f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    cuda_streams = {device: torch.cuda.Stream(device=device) for device in args.gpus}

    for epoch in range(args.num_epochs):
        # Training mode
        [model.train() for model in models]
        if hasattr(optimizers[0], 'train'):
            [optimizer.train() for optimizer in optimizers]

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            losses = []
            # Process models in parallel using async CUDA operations
            for cuda_device, (model, optimizer, scheduler) in zip(args.gpus, zip(models, optimizers, schedulers)):
                with torch.cuda.stream(cuda_streams[cuda_device]):
                    optimizer.zero_grad()
                    
                    outputs = model(
                        input_ids=batch['input_ids'].to(f'cuda:{cuda_device}', non_blocking=True),
                        labels=batch['labels'].to(f'cuda:{cuda_device}', non_blocking=True)
                    )
                    
                    loss = outputs.loss
                    losses.append(loss.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()

            # Only synchronize when we need to update the progress bar
            if global_step % 100 == 0:  # Update progress less frequently
                torch.cuda.synchronize()
                avg_loss = sum(losses) / len(losses)
                pbar.set_postfix({'avg_loss': avg_loss, 'step': global_step})
                log_entry = {
                    'step': global_step,
                    'epoch': epoch,
                    'train_loss': avg_loss,
                    'timestamp': datetime.now().isoformat()
                }
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            
                pbar.set_postfix({'loss': avg_loss, 'step': global_step})
            
            global_step += 1

            should_save = (global_step < 100000 and global_step in early_saves) or \
                         (global_step >= 100000 and global_step % 100000 == 0)
            if should_save:
                torch.cuda.synchronize()
                # Evaluation mode
                [model.eval() for model in models]
                if hasattr(optimizers[0], 'eval'):
                    [optimizer.eval() for optimizer in optimizers]
                
                results_dicts.append(evaluate_models(models_dict, onehop_loader, global_step, 1))
                results_dicts.append(evaluate_models(models_dict, twohop_loader, global_step, 2))

                # Save checkpoints using HF interface
                for i, (model, optimizer, scheduler) in enumerate(zip(models, optimizers, schedulers)):
                    model_path = f"{output_dirs[i]}/checkpoint-{global_step}"
                    model.save_pretrained(model_path)
                    
                    # Save optimizer and scheduler state separately
                    optimizer_state = {
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': global_step,
                        'onehop_loss': results_dicts[-2]['loss'][i],
                        'twohop_loss': results_dicts[-1]['loss'][i]
                    }
                    if scheduler:
                        optimizer_state['scheduler_state_dict'] = scheduler.state_dict()
                    torch.save(optimizer_state, f"{model_path}/optimizer.pt")
                
                results_df = pd.DataFrame(results_dicts)
                results_df.to_csv(f"{output_dirs[i]}/eval_results.csv", index=False)
                # Back to training mode
                [model.train() for model in models]
                if hasattr(optimizers[0], 'train'):
                    [optimizer.train() for optimizer in optimizers]
            
                # Add evaluation metrics to log
                log_entry.update({
                    'onehop_loss': results_dicts[-2]['loss'],
                    'twohop_loss': results_dicts[-1]['loss'],
                    'parameter_l2': results_dicts[-1]['parameter_l2']
                })
            

def evaluate_models(models_dict, eval_loader, global_step, hop_count):
    results_dict = {
        'loss': [0, 0, 0, 0],
        'num_params': [],
        'num_layers': [],
        'N_profiles': [],
        'orders': [],
        'wd': [],
        'lr': [],
        'beta1': [],
        'global_step': [],
        'parameter_l2': []
    }

    with torch.no_grad():
        eval_questions = 0
        pbar = tqdm(eval_loader, desc=f"Evaluating {hop_count}-hop")
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

            for i, model in enumerate(models_dict['model']):
                outputs = model(
                    input_ids=filtered_inputs.to(f'cuda:{i}'),
                    labels=filtered_labels.to(f'cuda:{i}')
                )
                
                results_dict['loss'][i] += outputs.loss.item() * is_pos.sum().item()
                
                results_dict.update({
                    k: models_dict[k][i] for k in 
                    ['num_params', 'num_layers', 'N_profiles', 'orders', 'wd', 'lr', 'beta1']
                })
                
                results_dict['global_step'].append(global_step)
                all_params = torch.cat([p.flatten() for p in model.parameters()])
                results_dict['parameter_l2'].append(torch.norm(all_params).item() / np.sqrt(all_params.numel()))
            
            pbar.set_postfix({'eval_questions': eval_questions})
            if eval_questions >= 500:
                break
    
    for i, loss in enumerate(results_dict['loss']):
        results_dict['loss'][i] = loss / eval_questions
    print(f"Step {global_step}: Evaluation Loss = {results_dict['loss']}")
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

    return list(range(len(prefix_tokens), len(full_tokens)))


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

class InfiniteBiosDataset(IterableDataset):
    def __init__(self, profiles_dataset, tokenizer, max_seq_len=512, orders=[1,2], qa_prob=0.5, qa_indices = []):
        self.profiles = profiles_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples_per_yield = math.ceil((max_seq_len//75)*(1-qa_prob) + (max_seq_len//20)*qa_prob)
        self.orders = orders
        self.qa_prob = qa_prob
        self.templates = load_templates(get_project_root() / "generated_data/templates")
        self.qa_indices = qa_indices
        self.order_weights = [10**i for i in range(len(orders))]

    def __len__(self):
        return len(self.qa_indices)

    def __iter__(self):
        steps = 0
        while steps <= len(self.qa_indices):  # Stop after one "epoch"
            steps += 1
            # But the next epoch will yield different samples
            # Generate multiple samples (either bios or QA)
            texts = []
            answer_token_ranges = []
            
            for _ in range(self.samples_per_yield):
                if random.random() < self.qa_prob:
                    if question := self.generate_qa():
                        texts.append(question)
                        # Calculate answer ranges only if tokenizer exists
                        if self.tokenizer:
                            answer_token_ranges.append(get_qa_token_ranges(question, self.tokenizer))
                else:
                    # Generate bio
                    profile_idx = random.randrange(len(self.profiles))
                    profile = self.profiles[profile_idx]
                    bio = generate_bio(profile, self.templates)
                    texts.append(bio)
                    if self.tokenizer:
                        answer_token_ranges.append(list(range(len(bio))))
            
            # If no tokenizer provided, just return the text chunk
            if self.tokenizer is None:
                yield {"text": "\n".join(texts)}
                continue
            
            # Tokenizer-dependent processing
            sep = self.tokenizer.eos_token or "<|endoftext|>"
            joined_text = sep.join(texts)  # Start with separator
            output = self.tokenizer(
                joined_text,
                max_length=self.max_seq_len,
                return_attention_mask=False,
                truncation=True,
                return_tensors="pt"
            )
            # First tokenize each text and create its labels
            individual_labels = []
            for i, at_range in enumerate(answer_token_ranges):
                labels = [-100] * (at_range[-1] + 1)
                labels[at_range[0]:] = [1] * (len(labels) - at_range[0])
                individual_labels.extend(labels)
            
            # Insert sep tokens at max_seq_len intervals
            sep_tokens = self.tokenizer(sep, return_attention_mask=False, add_special_tokens=False)["input_ids"]
            final_labels = []
            for i in range(0, len(individual_labels), self.max_seq_len):
                if i > 0:  # Add separator at the start of each chunk except first
                    final_labels.extend([-100] * len(sep_tokens))
                final_labels.extend(individual_labels[i:i + self.max_seq_len])
            
            # Split into chunks
            chunked_labels = [final_labels[i:i + self.max_seq_len] 
                            for i in range(0, len(final_labels), self.max_seq_len)]
            chunked_labels = chunked_labels[:len(output["input_ids"])]
            for j, (labels_chunk, ids_chunk) in enumerate(zip(chunked_labels, output["input_ids"])):
                for i, (l, id) in enumerate(zip(labels_chunk, ids_chunk)):
                    if l == 1:
                        chunked_labels[j][i] = id
            chunked_labels = torch.tensor(chunked_labels)

            yield {"input_ids": output["input_ids"].squeeze(0), "text": joined_text, "labels": chunked_labels.squeeze(0)}

    def generate_qa(self):
        profile_idx = random.choice(self.qa_indices)
        profile = self.profiles[profile_idx]
        order = random.choices(self.orders, weights=self.order_weights, k=1)[0]
        question, _ = generate_question(profile, self.profiles, order, {}, {})
        if question:
            return f"Question: {question['question']} Answer: {question['answer']}"
        return None


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
    
def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = cpu_count() // 2,
    text_key: str = "text",
    max_seq_len: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> T:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_seq_len` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_seq_len: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.

    Returns:
        The chunked and tokenized dataset.
    """

    def _tokenize_fn(x: dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        sep = tokenizer.eos_token or "<|endoftext|>"
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output.input_ids[0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size]
                for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        return output

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    return data.with_format(format, columns=["input_ids"])



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
    else:  # regular AdamW with cosine scheduler
        optimizer = TorchAdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, 0.999),
            weight_decay=args.wd
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1000,
            num_training_steps=args.num_training_steps
        )
    
    start_step = 0
    if checkpoint_path:
        optimizer_path = f"{checkpoint_path}/optimizer.pt"
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path)
            optimizer.load_state_dict(optimizer_state['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in optimizer_state:
                scheduler.load_state_dict(optimizer_state['scheduler_state_dict'])
            start_step = optimizer_state['step']
            print(f"Loaded optimizer state from step {start_step}")
    
    return optimizer, scheduler, start_step