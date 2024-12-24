import random
from typing import Iterator, Dict
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
from functools import partial

import torch.distributed as dist
import torch
from torch.utils.data import IterableDataset

import jax.numpy as jnp
import jax

from datasets import load_dataset

from transformer_reasoning.generate_dataset.generate_bios import load_templates
from transformer_reasoning.generate_dataset.generate_qa_dataset import maybe_generate_question, get_available_relations, ATTRIBUTES
from transformer_reasoning.utils import get_project_root


def load_and_prepare_datasets(
        tokenizer, 
        N=250000, 
        orders=None, 
        relations=None, 
        hop_ratio=0.1, 
        jax=False, 
        heldout_sets=None, 
        debug=False
    ):

    dataset_class = JAXQADataset if jax else InfiniteQADataset

    relation_str = f'_r{relations}' if relations else ''
    # Load profiles dataset
    profiles = load_dataset(f"EleutherAI/profiles_dataset_{N}_uniform{relation_str}", keep_in_memory=True)['train']
    # Create infinite training dataset
    train_dataset = dataset_class(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=orders or [1,2],
        hop_ratio=hop_ratio,
        heldout_fraction=0.05,
        heldout_sets=heldout_sets,
        debug=debug
    )
    
    return train_dataset


class InfiniteQADataset(IterableDataset):
    def __init__(self, profiles_dataset, tokenizer, max_seq_len=512, orders=[1,2], subjects=None, 
                 hop_ratio=0.1, heldout_fraction=0.05, mode="train", heldout_sets=None, seed_offset=0,
                 debug=False):
        self.profiles = profiles_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples_per_yield = (max_seq_len//15)
        self.orders = orders
        self.templates = load_templates(get_project_root() / "generated_data/templates")
        self.order_weights = [1/hop_ratio**i for i in range(len(orders))]
        self.worker_id = None
        self.qa_indices = range(len(profiles_dataset))
        self.num_workers = None
        self.answer_sep_tokens = tokenizer('Answer:', add_special_tokens=False)['input_ids']
        self.eos_token = tokenizer.eos_token or "<|endoftext|>"
        self.question_sep_tokens = tokenizer(self.eos_token, add_special_tokens=False)['input_ids']
        self.subjects = subjects
        self._batch_size = None  # Will be set during iteration
        self.mode = mode
        self._epoch = 0
        self._seed_offset = seed_offset
        self.debug = debug
        # Either use provided held-out sets or generate new ones
        if heldout_sets is not None:
            self.heldout_sets = heldout_sets
        else:
            self._generate_all_heldout_sets(heldout_fraction)

        if self.debug:
            self.heldout_sets_for_use = self.heldout_sets
        else:
            self.heldout_sets_for_use = {k: frozenset(v) for k, v in self.heldout_sets.items()}

    def set_epoch(self, epoch):
        """Allow external epoch updates"""
        self._epoch = epoch

    def _generate_all_heldout_sets(self, fraction):
        if not dist.is_initialized() or dist.get_rank() == 0:
            n_profiles = len(self.profiles)
            available_relations = get_available_relations(self.profiles[0])
            n_relations = len(available_relations) 
            n_attributes = len(ATTRIBUTES)

            self.heldout_sets = {
                "first_people": sorted(random.sample(range(n_profiles), max(1, int(n_profiles * fraction)))),
                "relations": sorted(random.sample(available_relations, max(1, int(n_relations * fraction)))),
                "person_relation_pairs": sorted(
                    random.sample([(p, r) for p in range(n_profiles) for r in available_relations], 
                                max(1, int(n_profiles * n_relations * fraction)))
                ),
                "second_people": sorted(random.sample(range(n_profiles), max(1, int(n_profiles * fraction)))),
                "second_attributes": sorted(random.sample(ATTRIBUTES + available_relations, max(1, int(n_attributes * fraction)))),
                "second_person_attribute_pairs": sorted(
                    random.sample([(p, a) for p in range(n_profiles) for a in ATTRIBUTES + available_relations],
                                max(1, int(n_profiles * n_attributes * fraction)))
                ),
                "complete_two_hop_questions": sorted(
                    random.sample([(p, r, a) for p in range(n_profiles) for r in available_relations for a in available_relations + ATTRIBUTES],
                                max(1, int(n_profiles * n_relations * n_attributes * fraction)))
                )
            }

        if dist.is_initialized():
            if dist.get_rank() != 0:
                heldout_sets_list = [{}]
            else:
                heldout_sets_list = [self.heldout_sets]
            
            dist.broadcast_object_list(heldout_sets_list, src=0)
            self.heldout_sets = heldout_sets_list[0]

    def __len__(self):
        # Get DDP info
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        
        # Adjust total length based on world_size
        return (len(self.qa_indices) * 10) // world_size

    def __iter__(self):
        # Get worker info for DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
            self._batch_size = 32
        else:
            self.worker_id = 0
            self.num_workers = 1
            self._batch_size = 1

        # Get DDP info
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        # Print epoch, rank and worker info
        if self.worker_id == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"Dataset iterator called with epoch {self._epoch} mode {self.mode}")

        # Set random seed based on epoch, rank, worker_id, and a random offset
        seed = self._seed_offset + 1000 * rank + 1000000 * self.worker_id + self._epoch
        torch.manual_seed(seed)
        random.seed(seed)
        
        # Calculate samples for this worker based on dataset length
        total_samples = len(self)
        worker_samples = (total_samples // self._batch_size) * self._batch_size // self.num_workers
        
        samples_processed = 0
        while samples_processed < worker_samples:
            texts = []
            for _ in range(self.samples_per_yield):
                question = self.generate_qa(self.heldout_sets_for_use)
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

            labels, position_ids = self._create_labels_fast(output["input_ids"])

            yield {
                "input_ids": output["input_ids"].squeeze(0),
                "text": joined_text,
                "labels": labels.squeeze(0),
                "position_ids": position_ids
            }

            samples_processed += 1

    def generate_qa(self, heldout_sets):
        while True:
            relation, attribute = None, None
            if self.mode == "eval_first_people":
                profile_idx = random.choice(self.heldout_sets["first_people"])
            elif self.mode == "eval_complete_two_hop_questions":
                profile_idx, relation, attribute = random.choice(self.heldout_sets["complete_two_hop_questions"])
            else:
                profile_idx = random.choice(self.qa_indices)
            profile = self.profiles[profile_idx]
            
            # Training uses both 1-hop and 2-hop, eval uses only 2-hop
            order = random.choices(self.orders, weights=self.order_weights, k=1)[0] if self.mode == "train" else 2
            
            question = maybe_generate_question(
                profile=profile, 
                profiles=self.profiles, 
                order=order,
                mode=self.mode,
                heldout_sets=heldout_sets,
                relation=relation,
                subject=attribute
            )
            
            if question:
                return f"Question: {question['question']} Answer: {question['answer']}"

    def _create_labels_fast(self, input_ids):
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
        
        return labels, torch.arange(len(labels[0]))


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
        try:
            iterator = iter(self.dataset)
            while not self.stop_event.is_set():
                worker_inputs = []
                worker_labels = []
                worker_positions = []
                
                for _ in range(self.samples_per_worker):
                    sample = next(iterator)
                    worker_inputs.append(sample["input_ids"])
                    worker_labels.append(sample["labels"])
                    worker_positions.append(sample["position_ids"])
                
                result_dict[worker_id] = {
                    "input_ids": worker_inputs,
                    "labels": worker_labels,
                    "position_ids": worker_positions
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
                    all_positions = []
                    for i in range(self.num_workers):
                        worker_result = result_dict[i]
                        all_inputs.extend(worker_result["input_ids"])
                        all_labels.extend(worker_result["labels"])
                        all_positions.extend(worker_result["position_ids"])
                    batch = {
                        "input_ids": jnp.stack(all_inputs),
                        "labels": jnp.stack(all_labels),
                        "position_ids": jnp.stack(all_positions)
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


# @partial(jax.jit, static_argnums=(1, 2))
def _create_labels_and_positions_jax(input_ids, answer_sep_tokens, question_sep_tokens):
    """JIT-compiled version of label creation and position ID generation"""
    batch_size, seq_len = input_ids.shape
    mask = jnp.zeros_like(input_ids, dtype=bool)
    position_ids = jnp.zeros_like(input_ids)
    
    # Find sequence boundaries and answer starts
    ends = input_ids == question_sep_tokens[0]
    # Shape: (batch_size, num_ends)
    end_indices = jnp.array([jnp.where(ends[b])[0] for b in range(batch_size)])
    
    # Find answer starts
    starts = jnp.all(jnp.stack([
        jnp.roll(input_ids, -i, axis=1) == token
        for i, token in enumerate(answer_sep_tokens)
    ]), axis=0)
    # Shape: (batch_size, num_starts)
    start_indices = jnp.array([jnp.where(starts[b])[0] for b in range(batch_size)])
    
    # Process each batch item separately
    for b in range(batch_size):
        last_end = 0
        batch_ends = end_indices[b]
        batch_starts = start_indices[b]
        
        # Create position IDs and labels for this batch item
        for i in range(len(batch_ends)):
            end = batch_ends[i]
            # Set position IDs for this sequence
            seq_length = end - last_end + 1
            position_ids = position_ids.at[b, last_end:end+1].set(jnp.arange(seq_length))
            
            # Find corresponding answer start for this sequence
            if i < len(batch_starts):
                ans_start = batch_starts[i] + len(answer_sep_tokens)
                if ans_start < end:
                    mask = mask.at[b, ans_start:end].set(True)
            
            last_end = end + 1
    
    # Create labels using the mask
    labels = jnp.where(mask, input_ids, -100)
    
    return labels, position_ids


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


        self._create_labels_and_positions_jit = partial(
            _create_labels_and_positions_jax,
            answer_sep_tokens=self.answer_sep_tokens,
            question_sep_tokens=self.question_sep_tokens,
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
            
            labels, position_ids = self._create_labels_and_positions_jit(input_ids)
            
            yield {
                "input_ids": input_ids.squeeze(0),
                "labels": labels.squeeze(0),
                "position_ids": position_ids.squeeze(0)
            }

    def generate_qa(self):
        profile_idx = random.choice(self.qa_indices)
        profile = self.profiles[profile_idx]
        order = random.choices(self.orders, weights=self.order_weights, k=1)[0]
        subject = random.choice(self.subjects) if self.subjects else None
        question, _ = maybe_generate_question(profile, self.profiles, order, {}, {}, subject)
        if question:
            return f"Question: {question['question']} Answer: {question['answer']}"
        return None

    def get_loader(self, batch_size: int, num_workers: int = 4) -> BatchLoader:
        return BatchLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers
        )