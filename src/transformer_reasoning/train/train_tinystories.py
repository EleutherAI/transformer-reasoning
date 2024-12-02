import argparse
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from transformer_reasoning.train.train_utils import calculate_model_size, create_model_and_tokenizer, chunk_and_tokenize

import glob
import os


def load_and_prepare_datasets():
    tinystories_dataset = load_dataset("EleutherAI/tinystories-pretokenized-pythia")

    train_dataset = tinystories_dataset["train"]
    val_dataset = tinystories_dataset["validation"]

    return train_dataset, val_dataset


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
    train_dataset, val_dataset = load_and_prepare_datasets()

    if args.resume_from:
        base_checkpoint_dir = f"./results/tinystories_{args.num_params}"
        latest_checkpoint = max(glob.glob(os.path.join(base_checkpoint_dir, "checkpoint-*")), key=os.path.getctime)

    epochs = 9
    print(f"Epochs: {epochs}")
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/tinystories_{args.num_params}",
        num_train_epochs=epochs,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        eval_accumulation_steps=1,
        warmup_steps=500,
        weight_decay=0.1,
        logging_dir=f"./logs/tinystories_{args.num_params}",
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
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    if args.resume_from:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()

    # Save the model
    model.save_pretrained(f"./final_model_tinystories_{args.num_params}")
    tokenizer.save_pretrained(f"./final_model_tinystories_{args.num_params}")
    model.push_to_hub(f"EleutherAI/TinyStories-{args.num_params}")
    tokenizer.push_to_hub(f"EleutherAI/TinyStories-{args.num_params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama model with specified parameters on the TinyStories dataset")
    parser.add_argument("--num_params", type=int, default=1_000_000, help="Number of parameters for the model")
    parser.add_argument("--resume_from", action="store_true", help="Resume training from most recent checkpoint")
    args = parser.parse_args()
    
    main(args)
