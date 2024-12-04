import torch
from torch.utils.data import IterableDataset
import random
import jax.numpy as jnp
from typing import Iterator, Dict
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import numpy as np
from typing import Iterator, Dict, Optional
import time
import jax
from functools import partial

from transformer_reasoning.generate_dataset.generate_bios import load_templates
from transformer_reasoning.generate_dataset.generate_qa_dataset import generate_question
from transformer_reasoning.utils import get_project_root


class InfiniteQADataset(IterableDataset):
    def __init__(self, profiles_dataset, tokenizer, max_seq_len=512, orders=[1,2], qa_indices = [], subjects=None, hop_ratio=0.1):
        self.profiles = profiles_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples_per_yield = (max_seq_len//15)
        self.orders = orders
        self.templates = load_templates(get_project_root() / "generated_data/templates")
        self.qa_indices = qa_indices
        self.order_weights = [1/hop_ratio**i for i in range(len(orders))]
        self.worker_id = None
        self.num_workers = None
        self.answer_sep_tokens = tokenizer('Answer:', add_special_tokens=False)['input_ids']
        self.eos_token = tokenizer.eos_token or "<|endoftext|>"
        self.question_sep_tokens = tokenizer(self.eos_token, add_special_tokens=False)['input_ids']
        self.subjects = subjects

    def __len__(self):
        return len(self.qa_indices*10)

    def __iter__(self):
        # Get worker info
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        else:
            self.worker_id = 0
            self.num_workers = 1

        # Calculate this worker's portion of steps
        worker_steps = len(self.qa_indices) * 10 // self.num_workers
        start_step = self.worker_id * worker_steps
        end_step = start_step + worker_steps if self.worker_id < self.num_workers - 1 else len(self.qa_indices) * 10

        steps = start_step
        while steps < end_step:  # Each worker processes its portion
            steps += 1
            # But the next epoch will yield different samples
            # Generate multiple samples (either bios or QA)
            texts = []
            answer_token_ranges = []
            
            for _ in range(self.samples_per_yield):
                question = self.generate_qa()
                texts.append(question)

            # If no tokenizer provided, just return the text chunk
            if self.tokenizer is None:
                yield {"text": "\n".join(texts)}
                continue
            
            # Tokenizer-dependent processing
            sep = self.tokenizer.eos_token or "<|endoftext|>"
            joined_text = sep.join(texts)
            output = self.tokenizer(
                joined_text,
                max_length=self.max_seq_len,
                return_attention_mask=False,
                truncation=True,
                return_tensors="pt"
            )


            labels = self._create_labels_fast(answer_token_ranges, output["input_ids"])


            yield {"input_ids": output["input_ids"].squeeze(0), "text": joined_text, "labels": labels.squeeze(0)}

    def generate_qa(self):
        profile_idx = random.choice(self.qa_indices)
        profile = self.profiles[profile_idx]
        order = random.choices(self.orders, weights=self.order_weights, k=1)[0]
        subject = random.choice(self.subjects) if self.subjects else None
        question, _ = generate_question(profile, self.profiles, order, {}, {}, subject)
        if question:
            return f"Question: {question['question']} Answer: {question['answer']}"
        return None

    def _create_labels_fast(self, answer_token_ranges, input_ids):
        """Create labels by masking everything except answers"""
        # Initialize mask as zeros
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Find starts (Answer:) - need all tokens to match
        starts = torch.all(torch.stack([
            input_ids.roll(-i) == token 
            for i, token in enumerate(self.answer_sep_tokens)
        ]), dim=0)
        
        # Find ends (EOS token)
        ends = input_ids == self.question_sep_tokens[0]
        # Convert to indices
        start_indices = torch.where(starts)[1]
        end_indices = torch.where(ends)[1] + len(self.question_sep_tokens)-1

        if start_indices[0] > end_indices[0]:
            end_indices = end_indices[1:]
        if start_indices[-1] > end_indices[-1]:
            start_indices = start_indices[:-1]
        if len(start_indices) != len(end_indices):
            end_indices = torch.cat([end_indices, torch.tensor([input_ids.shape[1]])])

        # Set mask between each start and its corresponding end
        for start, end in zip(start_indices, end_indices):
            if start < end:  # safety check
                mask[0, start+len(self.answer_sep_tokens):end] = True
        
        # Create labels by copying input_ids and masking
        labels = input_ids.clone()
        labels[~mask] = -100
        
        return labels


class BatchLoader:
    def __init__(
        self,
        dataset: "JAXQADataset",
        batch_size: int,
        num_workers: int = 4,
        prefetch_size: int = 2
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_per_worker = batch_size // num_workers
        self.prefetch_size = prefetch_size
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()

    def _worker(self, worker_id: int, result_dict: dict) -> None:
        """Worker thread that builds a portion of the batch."""
        try:
            iterator = iter(self.dataset)
            while not self.stop_event.is_set():
                worker_inputs = []
                worker_labels = []
                
                for _ in range(self.samples_per_worker):
                    sample = next(iterator)
                    worker_inputs.append(sample["input_ids"])
                    worker_labels.append(sample["labels"])
                
                result_dict[worker_id] = {
                    "input_ids": worker_inputs,
                    "labels": worker_labels
                }
                
                while not self.stop_event.is_set() and worker_id in result_dict:
                    time.sleep(0.001)
                    
        except Exception as e:
            print(f"Worker {worker_id} exception: {e}")
            raise e

    def __iter__(self):
        self.stop_event.clear()
        result_dict = {}
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._worker, i, result_dict)
                for i in range(self.num_workers)
            ]
            
            try:
                while True:
                    while len(result_dict) < self.num_workers:
                        time.sleep(0.001)
                    
                    all_inputs = []
                    all_labels = []
                    for i in range(self.num_workers):
                        worker_result = result_dict[i]
                        all_inputs.extend(worker_result["input_ids"])
                        all_labels.extend(worker_result["labels"])
                    
                    batch = {
                        "input_ids": jnp.stack(all_inputs),
                        "labels": jnp.stack(all_labels)
                    }
                    
                    result_dict.clear()
                    
                    yield batch
                    
            except:
                self.stop_event.set()
                raise
            finally:
                self.stop_event.set()
                for future in futures:
                    future.result()

    def __del__(self):
        self.stop_event.set()


@partial(jax.jit, static_argnums=(2,))
def _create_labels_jax(input_ids, answer_sep_tokens, question_sep_tokens, sep_len):
    """JIT-compiled version of label creation"""

    mask = jnp.zeros_like(input_ids, dtype=bool)
    
    starts = jnp.all(jnp.stack([
        jnp.roll(input_ids, -i, axis=1) == token
        for i, token in enumerate(answer_sep_tokens)
    ]), axis=0)
    
    ends = input_ids == question_sep_tokens[0]
    
    start_indices = jnp.where(starts)[1]
    end_indices = jnp.where(ends)[1] + sep_len - 1
    
    for i in range(len(start_indices)):
        start = start_indices[i] + len(answer_sep_tokens)
        end = end_indices[i]
        if start < end:
            mask = mask.at[0, start:end].set(True)

    labels = jnp.where(mask, input_ids, -100)
    
    return labels


class JAXQADataset:
    def __init__(self, profiles_dataset, tokenizer, max_seq_len=512, orders=[1,2], qa_indices=[], subjects=None, hop_ratio=0.1):
        self.profiles = profiles_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples_per_yield = (max_seq_len//15)
        self.orders = orders
        self.qa_indices = qa_indices
        self.order_weights = [1/hop_ratio**i for i in range(len(orders))]
        self.subjects = subjects
        self.answer_sep_tokens = tokenizer('Answer:', add_special_tokens=False)['input_ids']
        self.eos_token = tokenizer.eos_token or "<|endoftext|>"
        self.question_sep_tokens = tokenizer(self.eos_token, add_special_tokens=False)['input_ids']

        self._create_labels_jit = partial(
            _create_labels_jax,
            answer_sep_tokens=jnp.array(self.answer_sep_tokens),
            question_sep_tokens=jnp.array(self.question_sep_tokens),
            sep_len=len(self.question_sep_tokens)
        )

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        while True:
            texts = []
            for _ in range(self.samples_per_yield):
                question = self.generate_qa()
                texts.append(question)
            
            sep = self.tokenizer.eos_token or "<|endoftext|>"
            joined_text = sep.join(texts)
            output = self.tokenizer(
                joined_text,
                max_length=self.max_seq_len,
                return_attention_mask=False,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = jnp.array(output["input_ids"])
            
            labels = self._create_labels_jit(input_ids)
            
            yield {
                "input_ids": input_ids,
                "labels": labels
            }

    def generate_qa(self):
        profile_idx = random.choice(self.qa_indices)
        profile = self.profiles[profile_idx]
        order = random.choices(self.orders, weights=self.order_weights, k=1)[0]
        subject = random.choice(self.subjects) if self.subjects else None
        question, _ = generate_question(profile, self.profiles, order, {}, {}, subject)
        if question:
            return f"Question: {question['question']} Answer: {question['answer']}"
        return None

    def get_loader(self, batch_size: int, num_workers: int = 4) -> BatchLoader:
        return BatchLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers
        )