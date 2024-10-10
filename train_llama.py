import os
import math
import argparse
import requests
from tqdm import tqdm
import torch
from pathlib import Path
from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset
import json
import re

from utils import read_data_source_target

class SimpleTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        self.vocab = {token: i for i, token in enumerate(vocab)}
        self.ids_to_tokens = {i: token for token, i in self.vocab.items()}
        super().__init__()
        self.unk_token = "[UNK]"
        self.eos_token = "</a>"
        self.pad_token = "</a>"

    def _tokenize(self, text):
        return re.findall(r'<[^>]+>', text)
    
    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self):
        return dict(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _pad(self, encoded_inputs, max_length=None, padding_strategy="longest", return_attention_mask=True, **kwargs):
        # Handle non-batched input
        if isinstance(encoded_inputs, dict):
            encoded_inputs = [encoded_inputs]
            was_dict = True
        else:
            was_dict = False

        if padding_strategy == "longest":
            max_length = max(len(inputs["input_ids"]) for inputs in encoded_inputs)

        for inputs in encoded_inputs:
            difference = max_length - len(inputs["input_ids"])
            inputs["input_ids"] += [self.pad_token_id] * difference
            if return_attention_mask:
                inputs["attention_mask"] = [1] * len(inputs["input_ids"]) + [0] * difference

        if return_attention_mask:
            for inputs in encoded_inputs:
                if "attention_mask" not in inputs:
                    inputs["attention_mask"] = [1] * len(inputs["input_ids"])

        # If the input was originally a single dict, return a single dict
        if was_dict:
            return encoded_inputs[0]
        return encoded_inputs

def calculate_model_size(num_params):
    # Assuming 4 bytes per parameter
    return num_params * 4 / (1024 * 1024)  # Size in MB

def calculate_architecture(num_params):
    # These are rough estimates and may need fine-tuning
    n_layers = int(math.log(num_params / 1e6, 2)) + 4
    hidden_size = int(math.sqrt(num_params / (n_layers * 4)))
    hidden_size = (hidden_size // 64) * 64  # Round to nearest multiple of 64
    
    return n_layers, hidden_size

def main(args):
    data_dir = Path("data")/f"composition.{args.num_entities}.{args.num_relations}.{args.phi:.1f}"

    # Calculate model architecture
    n_layers, hidden_size = calculate_architecture(args.num_params)
    print(f"Using {n_layers} layers and hidden size of {hidden_size}, intermediate size of {hidden_size * 4}, attention heads of {hidden_size // 64}")
    

    
    config = LlamaConfig(
        num_hidden_layers=n_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=hidden_size // 64,
    )

    model = LlamaForCausalLM(config)

    
    # Load and preprocess the dataset
    train_file = os.path.join(data_dir, "train.json")
    train_df, train_sample_size = read_data_source_target(train_file, return_num=True)
    train_dataset = Dataset.from_pandas(train_df)


    # # Initialize tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)

    vocab_file = os.path.join(data_dir, "vocab.json")
    tokenizer = SimpleTokenizer(vocab_file)
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        # Combine input and target with a separator
        return tokenizer(examples['target_text'], max_length=6, padding="max_length", truncation=True)

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("./llama_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama model with specified parameters")
    parser.add_argument("--num_params", type=int, default=100_000_000, help="Number of parameters for the model")
    parser.add_argument("--num_entities", type=int, default=2000, help="Number of entities")
    parser.add_argument("--num_relations", type=int, default=200, help="Number of relations")
    parser.add_argument("--phi", type=float, default=18.0, help="Phi parameter for the dataset")
    args = parser.parse_args()
    
    model_size_mb = calculate_model_size(args.num_params)
    print(f"Estimated model size: {model_size_mb:.2f} MB")
    
    main(args)
