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


@dataclass
class LogSpacedCheckpoint(TrainerCallback):
    """Save checkpoints at log-spaced intervals"""

    base: float = 2.0
    next: int = 1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step >= self.next:
            self.next = round(self.next * self.base)

            control.should_evaluate = True
            control.should_save = True

def calculate_model_size(num_params):
    return num_params * 4 / (1024 * 1024)  # Size in MB

def calculate_architecture(num_params):
    n_layers = int(math.log(num_params / 1e6, 2)) + 4
    hidden_size = int(math.sqrt(num_params / (n_layers * 4)))
    hidden_size = (hidden_size // 64) * 64  # Round to nearest multiple of 64
    return n_layers, hidden_size

def load_and_prepare_datasets(tokenizer, subset_size=None):
    bios_dataset = load_from_disk(str(get_project_root() / "generated_data/bios/bios_dataset_revised"))
    qa_dataset = load_from_disk(str(get_project_root() / "generated_data/qa_dataset"))

    # Prepare bios dataset
    bios_train = bios_dataset.select_columns(['bio']).rename_column('bio', 'text')

    # Prepare qa dataset
    def format_qa(example):
        return {"text": f"Question: {example['questions.question']} Answer: {example['questions.answer']}"}

    qa_train = qa_dataset['train'].map(format_qa)
    qa_val = qa_dataset['validation'].map(format_qa)
    qa_heldout = qa_dataset['heldout_profiles'].map(format_qa)

    # Combine bios and qa datasets for training
    train_dataset = concatenate_datasets([bios_train, qa_train])

    if subset_size:
        train_dataset = train_dataset.select(range(min(subset_size, len(train_dataset))))
        qa_val = qa_val.select(range(min(subset_size, len(qa_val))))
        qa_heldout = qa_heldout.select(range(min(subset_size, len(qa_heldout))))

    # Wrap datasets with OnTheFlyTokenizationDataset
    train_dataset = OnTheFlyTokenizationDataset(train_dataset, tokenizer)
    qa_val = OnTheFlyTokenizationDataset(qa_val, tokenizer)
    qa_heldout = OnTheFlyTokenizationDataset(qa_heldout, tokenizer)

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
    model, tokenizer = create_model_and_tokenizer(args.num_params)
    train_dataset, val_dataset, heldout_dataset = load_and_prepare_datasets(tokenizer, args.subset_size)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=200000,
        save_steps=200000,
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        fp16=True,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=heldout_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        compute_metrics=compute_metrics,
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
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama model with specified parameters")
    parser.add_argument("--num_params", type=int, default=10_000_000, help="Number of parameters for the model")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of examples to use for training (for testing purposes)")
    args = parser.parse_args()
    
    main(args)
