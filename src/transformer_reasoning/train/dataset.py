import random

import torch.distributed as dist
import torch
from torch.utils.data import IterableDataset

from datasets import load_dataset

from transformer_reasoning.generate_dataset.generate_bios import load_templates
from transformer_reasoning.generate_dataset.generate_qa_dataset import maybe_generate_question, get_available_relations, ATTRIBUTES
from transformer_reasoning.utils import get_project_root



def load_and_prepare_datasets(
        tokenizer, 
        N=25000, 
        orders=None, 
        relations=None, 
        hop_ratio=0.1, 
        heldout_sets=None, 
        debug=False,
        question_generator=maybe_generate_question
    ):

    relation_str = f'_r{relations}' if relations else ''
    # Load profiles dataset
    profiles = load_dataset(f"EleutherAI/profiles_dataset_{N}_uniform{relation_str}", keep_in_memory=True)['train']
    # Create infinite training dataset
    train_dataset = InfiniteQADataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=orders or [1,2],
        hop_ratio=hop_ratio,
        heldout_fraction=0.1,
        heldout_sets=heldout_sets,
        debug=debug,
        question_generator=question_generator
    )
    
    return train_dataset


class InfiniteQADataset(IterableDataset):
    def __init__(self, profiles_dataset, tokenizer, max_seq_len=512, orders=[1,2], subjects=None, 
                 hop_ratio=0.1, heldout_fraction=0.1, specific_heldout_fraction=0.02, mode="train", heldout_sets=None, seed_offset=0,
                 debug=False, question_generator=maybe_generate_question):
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
        self._batch_size = None  # Will be set during iteration
        self.mode = mode
        self._epoch = 0
        self._seed_offset = seed_offset
        self.debug = debug
        self.question_generator = question_generator
        # Either use provided held-out sets or generate new ones
        if heldout_sets is not None:
            self.heldout_sets = heldout_sets
        else:
            self._generate_all_heldout_sets(heldout_fraction, specific_heldout_fraction)

        if self.debug:
            self.heldout_sets_for_use = self.heldout_sets
        else:
            self.heldout_sets_for_use = {k: frozenset(v) for k, v in self.heldout_sets.items()}

        self.subjects = subjects
        if subjects is not None:
            self._validate_subjects()


    def set_epoch(self, epoch):
        """Allow external epoch updates"""
        self._epoch = epoch

    def _generate_all_heldout_sets(self, fraction, specific_heldout_fraction):
        if self.orders == [1]:
            self.heldout_sets = {}
            return
        if not dist.is_initialized() or dist.get_rank() == 0:
            n_profiles = len(self.profiles)
            available_relations = get_available_relations(self.profiles[0])
            n_relations = len(available_relations) 
            n_attributes = len(ATTRIBUTES)

            # Store original heldout sets for testing
            self.heldout_sets = {
                "first_people": sorted(random.sample(range(n_profiles), max(1, int(n_profiles * specific_heldout_fraction)))),
                "relations": sorted(random.sample(available_relations, max(1, int(n_relations * specific_heldout_fraction)))),
                "person_relation_pairs": sorted(
                    random.sample([(p, r) for p in range(n_profiles) for r in available_relations], 
                                max(1, int(n_profiles * n_relations * specific_heldout_fraction)))
                ),
                "second_people": sorted(random.sample(range(n_profiles), max(1, int(n_profiles * specific_heldout_fraction)))),
                "second_attributes": sorted(random.sample(ATTRIBUTES + available_relations, max(1, int(n_attributes * specific_heldout_fraction)))),
                "second_person_attribute_pairs": sorted(
                    random.sample([(p, a) for p in range(n_profiles) for a in ATTRIBUTES + available_relations],
                                max(1, int(n_profiles * n_attributes * specific_heldout_fraction)))
                )
            }

            self._generate_explicit_heldout_sets(fraction, n_profiles, available_relations, n_attributes)
        
        if dist.is_initialized():
            if dist.get_rank() != 0:
                heldout_sets_list = [{}]
            else:
                heldout_sets_list = [self.heldout_sets]
            
            dist.broadcast_object_list(heldout_sets_list, src=0)
            self.heldout_sets = heldout_sets_list[0]

    def _generate_explicit_heldout_sets(self, fraction, n_profiles, available_relations, n_attributes):

        n_relations = len(available_relations)

        available_first_people = list(set(range(n_profiles)) - set(self.heldout_sets["first_people"]))

        # Caching profile indices to avoid repeated lookups
        print("caching profile indices")
        profile_indices = {(p, r): self.profiles[p][r]['index'] 
                            for p in range(n_profiles) 
                            for r in available_relations}
        print('cached profile indices')

        available_fp_r_a_sp_tuples = set([(p, r, a, profile_indices[(p, r)]) 
                                            for p in available_first_people 
                                            for r in available_relations 
                                            for a in ATTRIBUTES + available_relations])

        # Calculate tuples for each heldout set
        relations_tuples = set([(p, r, a, sp)
                    for (p, r, a, sp) in available_fp_r_a_sp_tuples
                    if r in self.heldout_sets["relations"]])

        available_fp_r_a_sp_tuples = available_fp_r_a_sp_tuples - relations_tuples

        second_attribute_tuples = set([(p, r, a, sp)
                            for (p, r, a, sp) in available_fp_r_a_sp_tuples
                            if a in self.heldout_sets["second_attributes"]])

        available_fp_r_a_sp_tuples = available_fp_r_a_sp_tuples - second_attribute_tuples

        person_relation_tuples = set([(p, r, a, sp)
                                    for (p, r, a, sp) in available_fp_r_a_sp_tuples
                                    if (p, r) in self.heldout_sets["person_relation_pairs"]])

        available_fp_r_a_sp_tuples = available_fp_r_a_sp_tuples - person_relation_tuples

        second_people_tuples = set([(p, r, a, sp)
                                    for (p, r, a, sp) in available_fp_r_a_sp_tuples
                                    if sp in self.heldout_sets["second_people"]])

        available_fp_r_a_sp_tuples = available_fp_r_a_sp_tuples - second_people_tuples

        second_person_attribute_tuples = set([(p, r, a, sp)
                                        for (p, r, a, sp) in available_fp_r_a_sp_tuples
                                        if (sp, a) in self.heldout_sets["second_person_attribute_pairs"]])

        available_fp_r_a_sp_tuples = available_fp_r_a_sp_tuples - second_person_attribute_tuples

        complete_two_hop_tuples = sorted(random.sample(sorted(available_fp_r_a_sp_tuples), 
                                                            max(1, int(n_profiles * n_relations * n_attributes * fraction))))

        available_fp_r_a_sp_tuples = sorted(set(available_fp_r_a_sp_tuples) - set(complete_two_hop_tuples))

        # Update heldout_sets with all tuple sets
        self.heldout_sets.update({
            "person_relation_tuples": sorted(person_relation_tuples),
            "second_people_tuples": sorted(second_people_tuples),
            "second_person_attribute_tuples": sorted(second_person_attribute_tuples),
            "relations_tuples": sorted(relations_tuples),
            "second_attribute_tuples": sorted(second_attribute_tuples),
            "complete_two_hop_tuples": sorted(complete_two_hop_tuples),
            "available_training_tuples": sorted(available_fp_r_a_sp_tuples),
        })

        num_first_people = (len(self.heldout_sets['first_people']) 
                            * (len(ATTRIBUTES) + len(available_relations))
                            * len(available_relations))

        sum_of_lengths = (len(self.heldout_sets['available_training_tuples']) + 
                            len(self.heldout_sets['person_relation_tuples']) + 
                            len(self.heldout_sets['second_people_tuples']) + 
                            len(self.heldout_sets['second_person_attribute_tuples']) + 
                            len(self.heldout_sets['relations_tuples']) + 
                            len(self.heldout_sets['second_attribute_tuples']) + 
                            len(self.heldout_sets['complete_two_hop_tuples']) +
                            num_first_people)

        full_length = len(profile_indices)*(len(ATTRIBUTES) + len(available_relations))

        print(f"Heldout sets computed. Training tuples: {len(self.heldout_sets['available_training_tuples'])}\n\
                Person relation tuples: {len(self.heldout_sets['person_relation_tuples'])}\n\
                Second people tuples: {len(self.heldout_sets['second_people_tuples'])}\n\
                Second person attribute tuples: {len(self.heldout_sets['second_person_attribute_tuples'])}\n\
                Relations tuples: {len(self.heldout_sets['relations_tuples'])}\n\
                Second attribute tuples: {len(self.heldout_sets['second_attribute_tuples'])}\n\
                Complete two hop tuples: {len(self.heldout_sets['complete_two_hop_tuples'])}\n\
                First people tuples: {num_first_people}\n\
                Sum: {sum_of_lengths}\n\
                Full length: {full_length}")

    def _validate_subjects(self):
        if not self.subjects:
            raise ValueError("Empty subjects list provided")
        
        # For first people eval, all subjects are valid
        if self.mode == "eval_first_people":
            available_relations = get_available_relations(self.profiles[0])
            valid_subjects = set(ATTRIBUTES + available_relations)
        else:
            # Get the relevant tuple set based on mode
            mode_to_tuple_set = {
                "train": "available_training_tuples",
                "eval_complete_two_hop_questions": "complete_two_hop_tuples",
                "eval_second_people": "second_people_tuples",
                "eval_second_attributes": "second_attribute_tuples",
                "eval_person_relation_pairs": "person_relation_tuples",
                "eval_relations": "relations_tuples",
                "eval_second_person_attribute_pairs": "second_person_attribute_tuples"
            }
            
            tuple_set_name = mode_to_tuple_set.get(self.mode)
            if tuple_set_name is None:
                raise ValueError(f"Invalid mode: {self.mode}")
            
            # Extract unique attributes from the relevant tuples
            valid_subjects = {attr for _, _, attr, _ in self.heldout_sets[tuple_set_name]}
        
        # Update self.subjects to be intersection with valid_subjects
        self.subjects = set(self.subjects) & valid_subjects
        if not self.subjects:
            raise ValueError(f"No valid subjects for mode {self.mode}. Valid subjects are {valid_subjects}")

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
            if self.orders == [1] or self.mode == "train":
                order = random.choices(self.orders, weights=self.order_weights, k=1)[0]
                if order == 2:
                    profile_idx, relation, attribute, _ = random.choice(self.heldout_sets["available_training_tuples"])
                else:
                    profile_idx = random.choice(self.qa_indices)
                    relation, attribute = None, None
            else:
                order = 2
                if self.mode == "eval_first_people":
                    profile_idx = random.choice(self.heldout_sets["first_people"])
                    relation, attribute = None, None
                elif self.mode == "eval_complete_two_hop_questions":
                    profile_idx, relation, attribute, _ = random.choice(self.heldout_sets["complete_two_hop_tuples"])
                elif self.mode == "eval_second_people":
                    profile_idx, relation, attribute, _ = random.choice(self.heldout_sets["second_people_tuples"])
                elif self.mode == "eval_second_attributes":
                    profile_idx, relation, attribute, _ = random.choice(self.heldout_sets["second_attribute_tuples"])
                elif self.mode == "eval_person_relation_pairs":
                    profile_idx, relation, attribute, _ = random.choice(self.heldout_sets["person_relation_tuples"])
                elif self.mode == "eval_relations":
                    profile_idx, relation, attribute, _ = random.choice(self.heldout_sets["relations_tuples"])
                elif self.mode == "eval_second_person_attribute_pairs":
                    profile_idx, relation, attribute, _ = random.choice(self.heldout_sets["second_person_attribute_tuples"])
                else:
                    raise ValueError(f"Invalid mode: {self.mode}")

            profile = self.profiles[profile_idx]
            
            # Training uses both 1-hop and 2-hop, eval uses only 2-hop

            if self.subjects is not None:
                attribute = random.choice(list(self.subjects))

            
            question = self.question_generator(
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


class MultiDataset(IterableDataset):
    def __init__(self, datasets, weights):
        """
        Args:
            datasets: List of InfiniteQADataset instances
            weights: List of weights for sampling from each dataset
        """
        self.datasets = datasets
        total_weight = sum(weights)
        self.weights = [w/total_weight for w in weights]  # Normalize weights
        self.tokenizer = datasets[0].tokenizer  # Assume all datasets use same tokenizer
        self.max_seq_len = datasets[0].max_seq_len
        self.profiles_dataset_index = 0

    @property
    def heldout_sets(self):
        """Return list of heldout_sets from child datasets"""
        return self.datasets[self.profiles_dataset_index].heldout_sets
    
    @property
    def profiles(self):
        """Return list of profiles from child datasets"""
        return self.datasets[self.profiles_dataset_index].profiles

    def __len__(self):
        return int(min([len(dataset) * sum(self.weights) / self.weights[i] for i, dataset in enumerate(self.datasets)]))

    def __iter__(self):
        iterators = [iter(dataset) for dataset in self.datasets]
        samples_yielded = 0
        total_samples = len(self)
        
        while samples_yielded < total_samples:
            dataset_idx = random.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
            try:
                sample = next(iterators[dataset_idx])
                sample['dataset_idx'] = torch.tensor([dataset_idx], dtype=torch.long)
                yield sample
                samples_yielded += 1
            except StopIteration:
                return  # This will implicitly raise StopIteration

    def set_epoch(self, epoch):
        """Propagate epoch to all datasets"""
        for dataset in self.datasets:
            dataset.set_epoch(epoch)