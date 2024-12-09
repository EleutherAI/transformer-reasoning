from tokenizers.processors import TemplateProcessing
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import argparse


def train_custom_tokenizer(dataset, vocab_size=1000, min_frequency=2, push_to_hub=False):
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Add pre-tokenizer to handle whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    
    # Create a trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<|endoftext|>", "<unk>"]
    )
    
    # Create an iterator that yields texts from your dataset
    def batch_iterator():
        batch_size = 1000
        texts = []
        for item in dataset:
            texts.append(item)
            if len(texts) == batch_size:
                yield texts
                texts = []
        if texts:
            yield texts
    
    # Train the tokenizer
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Add post-processor for adding special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<|endoftext|> $A",
        special_tokens=[
            ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
        ],
    )
    
    # Convert to HuggingFace tokenizer
    from transformers import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>"
    )
    
    if push_to_hub:
        hf_tokenizer.push_to_hub(f"EleutherAI/llama_multihop_tokenizer")

    return hf_tokenizer

def collect_sample_texts(dataset, n_samples=10000):
    """Collect a finite sample from InfiniteBiosDataset"""
    texts = []
    seen = set()  # To avoid duplicates
    pbar = tqdm(total=n_samples, desc="Collecting samples")
    
    for item in dataset:
        text = item["text"]
        if text not in seen:
            texts.append(text)
            seen.add(text)
            pbar.update(1)
            if len(texts) >= n_samples:
                break
    
    return texts

def compare_tokenizers(texts, custom_tokenizer, baseline_tokenizer):
    """Compare tokenization lengths between two tokenizers"""
    length_stats = defaultdict(list)
    
    for text in tqdm(texts[:100], desc="Comparing tokenizations"):  # Sample 100 texts for comparison
        custom_tokens = len(custom_tokenizer.encode(text))
        baseline_tokens = len(baseline_tokenizer.encode(text))
        
        length_stats["custom"].append(custom_tokens)
        length_stats["baseline"].append(baseline_tokens)
        length_stats["ratio"].append(custom_tokens / baseline_tokens)
    
    stats = {
        "custom_mean": np.mean(length_stats["custom"]),
        "custom_std": np.std(length_stats["custom"]),
        "baseline_mean": np.mean(length_stats["baseline"]),
        "baseline_std": np.std(length_stats["baseline"]),
        "ratio_mean": np.mean(length_stats["ratio"]),
        "ratio_std": np.std(length_stats["ratio"])
    }
    
    return stats

if __name__ == "__main__":
    from datasets import load_dataset
    from transformer_reasoning.train.train_utils import InfiniteQADataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10000)
    args = parser.parse_args()
    
    # Load your profiles dataset
    profiles_dataset = load_dataset(f"EleutherAI/profiles_dataset_{args.N}_uniform")["train"]
    
    # Create InfiniteBiosDataset instance
    infinite_dataset = InfiniteQADataset(
        profiles_dataset, 
        None, 
        max_seq_len=512,
        qa_prob=1
    )
    
    # Collect sample texts
    sample_texts = collect_sample_texts(infinite_dataset, n_samples=100000)
    print(f"Collected {len(sample_texts)} unique samples")
    
    # Train custom tokenizer
    custom_tokenizer = train_custom_tokenizer(
        sample_texts, 
        vocab_size=3000, 
        min_frequency=2,
        push_to_hub=True
    )
    holdout_sample_texts = collect_sample_texts(infinite_dataset, n_samples=1000)
    # Load baseline tokenizer
    baseline_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    
    # Compare tokenizers
    stats = compare_tokenizers(holdout_sample_texts, custom_tokenizer, baseline_tokenizer)
    
    print("\nTokenization Statistics:")
    print(f"Custom Tokenizer: {stats['custom_mean']:.1f} ± {stats['custom_std']:.1f} tokens")
    print(f"Baseline Tokenizer: {stats['baseline_mean']:.1f} ± {stats['baseline_std']:.1f} tokens")
    print(f"Ratio (Custom/Baseline): {stats['ratio_mean']:.2f} ± {stats['ratio_std']:.2f}")

