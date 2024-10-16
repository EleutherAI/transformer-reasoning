import math
import argparse
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk, concatenate_datasets
from transformer_reasoning.utils import get_project_root

def calculate_model_size(num_params):
    return num_params * 4 / (1024 * 1024)  # Size in MB

def calculate_architecture(num_params):
    n_layers = int(math.log(num_params / 1e6, 2)) + 4
    hidden_size = int(math.sqrt(num_params / (n_layers * 4)))
    hidden_size = (hidden_size // 64) * 64  # Round to nearest multiple of 64
    return n_layers, hidden_size

def load_and_prepare_datasets():
    bios_dataset = load_from_disk(str(get_project_root() / "generated_data/bios/bios_dataset"))
    qa_dataset = load_from_disk(str(get_project_root() / "generated_data/qa_dataset"))

    # Prepare bios dataset
    bios_train = bios_dataset.map(lambda x: {"text": f"Bio: {x['bio']}"})

    # Prepare qa dataset
    def format_qa(example):
        return {"text": f"Question: {example['question']} Answer: {example['answer']}"}

    qa_train = qa_dataset['train'].map(format_qa)
    qa_val = qa_dataset['validation'].map(format_qa)
    qa_heldout = qa_dataset['heldout_profiles'].map(format_qa)

    # Combine bios and qa datasets for training
    train_dataset = concatenate_datasets([bios_train, qa_train])

    return train_dataset, qa_val, qa_heldout

def create_model_and_tokenizer(num_params):
    n_layers, hidden_size = calculate_architecture(num_params)
    
    config = LlamaConfig(
        vocab_size=32000,  # Adjust if needed
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=n_layers,
        num_attention_heads=hidden_size // 64,
        max_position_embeddings=2048,
    )
    
    model = LlamaForCausalLM(config)
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    return model, tokenizer

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    
    # Compute accuracy
    correct = (predictions == torch.tensor(labels)).float()
    accuracy = correct.mean().item()
    
    return {"accuracy": accuracy}

def main(args):
    model_size_mb = calculate_model_size(args.num_params)
    print(f"Estimated model size: {model_size_mb:.2f} MB")

    # Load and prepare datasets
    train_dataset, val_dataset, heldout_dataset = load_and_prepare_datasets()

    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(args.num_params)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        load_best_model_at_end=True,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate on validation set
    val_results = trainer.evaluate()
    print("Validation Results:", val_results)

    # Evaluate on heldout profiles
    heldout_results = trainer.evaluate(heldout_dataset)
    print("Heldout Profiles Results:", heldout_results)

    # Save the model
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama model with specified parameters")
    parser.add_argument("--num_params", type=int, default=10_000_000, help="Number of parameters for the model")
    args = parser.parse_args()
    
    main(args)
