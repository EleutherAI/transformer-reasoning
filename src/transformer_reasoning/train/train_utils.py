import math
from transformers import (
    LlamaForCausalLM, LlamaConfig, 
    AutoTokenizer, TrainerCallback, 
    PreTrainedTokenizerBase
)
import torch

from typing import TypeVar, Union
from datasets import Dataset, DatasetDict
from multiprocessing import cpu_count
T = TypeVar("T", bound=Union[Dataset, DatasetDict])

from torch.utils.data import IterableDataset, DataLoader
import random
from transformer_reasoning.generate_dataset.generate_bios import generate_bio, load_templates
from transformer_reasoning.generate_dataset.generate_qa_dataset import generate_question
from transformer_reasoning.utils import get_project_root


class LogConstantCheckpointCallback(TrainerCallback):
    def __init__(self, trainer):
        # Generate log-spaced steps until 20k
        self.early_saves = set([int(math.exp(i)) for i in torch.linspace(math.log(100), math.log(20000), 10).tolist()])
        
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step < 20000 and step in self.early_saves:
            control.should_save = True
            breakpoint()
        elif step >= 20000 and step % 20000 == 0:
            control.should_save = True
        return control

class InfiniteBiosDataset(IterableDataset):
    def __init__(self, profiles_dataset, tokenizer, max_seq_len=512, orders=[1,2], qa_prob=0.5, qa_indices = []):
        self.profiles = profiles_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples_per_yield = math.ceil((max_seq_len//75)*(1-qa_prob) + (max_seq_len//10)*qa_prob)
        self.orders = orders
        self.qa_prob = qa_prob
        self.templates = load_templates(get_project_root() / "generated_data/templates")
        self.qa_indices = qa_indices

    def __len__(self):
        return len(self.profiles)

    def __iter__(self):
        while True:  # Infinite iteration
            # Generate multiple samples (either bios or QA)
            texts = []
            for _ in range(self.samples_per_yield):
                if random.random() < self.qa_prob:
                    if question := self.generate_qa():
                        texts.append(question)
                else:
                    # Generate bio
                    profile_idx = random.randrange(len(self.profiles))
                    profile = self.profiles[profile_idx]
                    texts.append(generate_bio(profile, self.templates))
            
            # Join texts with separator token
            sep = self.tokenizer.eos_token or "<|endoftext|>"
            joined_text = sep.join([""] + texts)  # Start with separator
            
            # Tokenize and chunk
            output = self.tokenizer(
                joined_text,
                max_length=self.max_seq_len,
                return_attention_mask=False,
                return_overflowing_tokens=True,
                truncation=True,
            )
            # Handle overflow tokens
            if overflow := output.pop("overflow_to_sample_mapping", None):
                # Yield all chunks except possibly incomplete last one
                for ids in output["input_ids"][:-1]:
                    yield {"input_ids": ids, "text": joined_text}
            else:
                # If no overflow, yield the single chunk if it's full
                if len(output["input_ids"][0]) == self.max_seq_len:
                    yield {"input_ids": output["input_ids"], "text": joined_text}


    def generate_qa(self):
        profile_idx = random.choice(self.qa_indices)
        profile = self.profiles[profile_idx]
        order = random.choice(self.orders)
        question, _ = generate_question(profile, self.profiles, order, {}, {})
        if question:
            return f"Question: {question['question']} Answer: {question['answer']}"
        return None


def calculate_model_size(num_params):
    return num_params * 4 / (1024 * 1024)  # Size in MB

def calculate_architecture(num_params):
    n_layers = int(math.log(num_params / 1e6, 2)) + 4
    hidden_size = int(math.sqrt(num_params / (n_layers * 4)))
    hidden_size = (hidden_size // 64) * 64  # Round to nearest multiple of 64
    return n_layers, hidden_size


def create_model_and_tokenizer(num_params):
    n_layers, hidden_size = calculate_architecture(num_params)
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=n_layers,
        num_attention_heads=hidden_size // 64,
        max_position_embeddings=2048,
    )
    
    model = LlamaForCausalLM(config)
    
    return model, tokenizer
    
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