import math
import argparse
import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk, concatenate_datasets
from transformer_reasoning.utils import get_project_root
from dataclasses import dataclass
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from torch.utils.data import Dataset

import glob
import os

class OnTheFlyTokenizationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoded = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in encoded.items()}


def calculate_model_size(num_params):
    return num_params * 4 / (1024 * 1024)  # Size in MB

def calculate_architecture(num_params):
    n_layers = int(math.log(num_params / 1e6, 2)) + 4
    hidden_size = int(math.sqrt(num_params / (n_layers * 4)))
    hidden_size = (hidden_size // 64) * 64  # Round to nearest multiple of 64
    return n_layers, hidden_size

def load_and_prepare_datasets(tokenizer, subset_size=None, max_order=None, N=250000, qa_ratio=0.1):
    bios_dataset = load_from_disk(str(get_project_root() / f"generated_data/bios/bios_dataset_{N}"))
    qa_dataset = load_from_disk(str(get_project_root() / f"generated_data/qa_dataset_{N}"))

    # Prepare bios dataset
    bios_train = bios_dataset.select_columns(['bio']).rename_column('bio', 'text')

    # Prepare qa dataset
    def format_qa(example):
        return {"text": f"Question: {example['questions.question']} Answer: {example['questions.answer']}"}

    # Filter qa dataset based on max_order
    if max_order is not None:
        qa_dataset = qa_dataset.filter(lambda x: x['questions.order'] <= max_order)

    qa_train = qa_dataset['train'].map(format_qa)
    qa_val = qa_dataset['validation'].map(format_qa)
    qa_heldout = qa_dataset['heldout_profiles'].map(format_qa)

    # Modify the QA dataset expansion logic
    if len(qa_train) < len(bios_train) * qa_ratio:
        print(f"Expanding qa_train by {math.ceil(len(bios_train) * qa_ratio / len(qa_train))}x")
        repetitions = math.ceil(len(bios_train) * qa_ratio / len(qa_train))
        qa_train = qa_train.map(lambda example: example, batched=True, num_proc=4)
        qa_train = qa_train.flatten_indices()
        qa_train = concatenate_datasets([qa_train] * repetitions)
    
    # Combine bios and qa datasets for training
    train_dataset = concatenate_datasets([bios_train, qa_train])

    if subset_size:
        train_dataset = train_dataset.select(range(min(subset_size, len(train_dataset))))
        qa_val = qa_val.select(range(min(subset_size, len(qa_val))))
        qa_heldout = qa_heldout.select(range(min(subset_size, len(qa_heldout))))

    # Wrap training dataset with OnTheFlyTokenizationDataset
    train_dataset = OnTheFlyTokenizationDataset(train_dataset, tokenizer)
    qa_val = OnTheFlyTokenizationDataset(qa_val.select(range(min(10000, len(qa_val)))), tokenizer)
    qa_heldout = OnTheFlyTokenizationDataset(qa_heldout.select(range(min(10000, len(qa_heldout)))), tokenizer)

    return train_dataset, qa_val, qa_heldout

def create_model_and_tokenizer(num_params):
    n_layers, hidden_size = calculate_architecture(num_params)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
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

def find_question_end(text, tokenizer):
    # Find the last occurrence of "? "
    question_end = text.rfind(": ")
    if question_end == -1:
        return None
    
    # Tokenize up to the question mark, including special tokens
    question_tokens = tokenizer.encode(text[:question_end+1], add_special_tokens=True)
    return len(question_tokens)


def main(args):
    model_size_mb = calculate_model_size(args.num_params)
    print(f"Estimated model size: {model_size_mb:.2f} MB")

    # Load and prepare datasets
    if args.resume_from:
        print(f"Loading model from checkpoint: {args.resume_from}")
        model = LlamaForCausalLM.from_pretrained(args.resume_from)
        tokenizer = AutoTokenizer.from_pretrained(args.resume_from)
    else:
        model, tokenizer = create_model_and_tokenizer(args.num_params)
    train_dataset, val_dataset, heldout_dataset = load_and_prepare_datasets(
        tokenizer, args.subset_size, args.max_order, args.N, args.qa_ratio
    )

    if args.resume_from:
        base_checkpoint_dir = f"./results/n{args.N}_p{args.num_params}_o{args.max_order}"
        latest_checkpoint = max(glob.glob(os.path.join(base_checkpoint_dir, "checkpoint-*")), key=os.path.getctime)

    def preprocess_logits_for_metrics(logits, labels):
        batch_size, seq_length, vocab_size = logits.shape
        selected_logits = []
        
        for i in range(batch_size):
            # Decode the full sequence
            text = tokenizer.decode(labels[i], skip_special_tokens=True)
            question_end = find_question_end(text, tokenizer)
            
            if question_end is not None:
                selected_logits.append(logits[i, question_end:, :])
            else:
                selected_logits.append(logits[i, -1:, :])  # Fallback if no question end found
        
        return torch.cat(selected_logits, dim=0)

    epochs = 50_000_000/len(train_dataset)
    print(f"Epochs: {epochs}")
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/n{args.N}_p{args.num_params}_o{args.max_order}",
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        eval_accumulation_steps=1,
        warmup_steps=500,
        weight_decay=0.1,
        logging_dir=f"./logs/n{args.N}_p{args.num_params}_o{args.max_order}",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=10000,
        save_steps=10000,
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        fp16=True,
        tf32=True,
    )

    # Modify training arguments for continued training
    if args.resume_from:
        initial_lr = training_args.learning_rate
        training_args = TrainingArguments(
            output_dir=f"./results/n{args.N}_p{args.num_params}_o{args.max_order}_continued_2",
            learning_rate=initial_lr * 0.1,  # Reduce learning rate for continued training
            warmup_steps=100,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=16,
            eval_accumulation_steps=1,  
            weight_decay=0.1,
            logging_dir=f"./logs/n{args.N}_p{args.num_params}_o{args.max_order}_continued_2",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=10000,
            save_steps=10000,
            load_best_model_at_end=True,
            dataloader_num_workers=4,
            fp16=True,
            tf32=True,
        )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=heldout_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate on validation set
    val_results = trainer.evaluate(val_dataset)
    print("Validation Results:", val_results)

    # Evaluate on heldout profiles
    heldout_results = trainer.evaluate()
    print("Heldout Profiles Results:", heldout_results)

    # Save the model
    model.save_pretrained(f"./final_model_n{args.N}_p{args.num_params}_o{args.max_order}")
    tokenizer.save_pretrained(f"./final_model_n{args.N}_p{args.num_params}_o{args.max_order}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama model with specified parameters")
    parser.add_argument("--num_params", type=int, default=1_000_000, help="Number of parameters for the model")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of examples to use for training (for testing purposes)")
    parser.add_argument("--N", type=int, default=25000, help="Number of profiles to use for QA dataset")
    parser.add_argument("--max_order", type=int, default=None, help="Maximum order of qa dataset")
    parser.add_argument("--resume_from", action="store_true", help="Resume training from most recent checkpoint")
    parser.add_argument("--qa_ratio", type=float, default=0.1,
                       help="Ratio of QA examples to bios examples")
    args = parser.parse_args()
    
    main(args)
