import math
import argparse
import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
from transformer_reasoning.utils import get_project_root
from transformer_reasoning.train.train_utils import calculate_model_size, create_model_and_tokenizer, InfiniteBiosDataset
from transformer_reasoning.generate_dataset.generate_qa_dataset import generate_question

from datasets import Dataset
import glob
import os


def load_and_prepare_datasets(tokenizer, subset_size=None, max_order=None, N=250000, qa_ratio=0.1):
    # Load profiles dataset
    profiles = load_from_disk(str(get_project_root() / f"generated_data/profiles_dataset_{N}"))
    
    shuffled_indices = torch.randperm(len(profiles)).tolist()
    heldout_indices = shuffled_indices[:1000]
    retained_indices = shuffled_indices[1000:]

    # Generate QA dataset for evaluation from the heldout profiles
    heldout_profiles = profiles.select(heldout_indices)
    eval_questions = []
    
    for profile in heldout_profiles:
        qa_result = generate_question(profile, profiles, max_order or 3, {}, {})
        if qa_result:
            question, _ = qa_result
            eval_questions.append(question)
    
    eval_dataset = Dataset.from_list(eval_questions)
    
    # Tokenize the evaluation dataset
    def tokenize_qa(example):
        text = f"<|endoftext|>Question: {example['question']} Answer: {example['answer']}"
        return tokenizer(text, padding=True, truncation=True, max_length=512)
    
    eval_dataset = eval_dataset.map(
        tokenize_qa,
        remove_columns=eval_dataset.column_names
    )

    # Create infinite training dataset with consistent retained indices
    train_dataset = InfiniteBiosDataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        max_order=max_order or 3,
        qa_prob=qa_ratio,
        qa_indices=retained_indices
    )

    return train_dataset, eval_dataset


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
        base_checkpoint_dir = str(get_project_root() / f"results/n{args.N}_p{args.num_params}_o{args.max_order}")
        latest_checkpoint = max(glob.glob(os.path.join(base_checkpoint_dir, "checkpoint-*")), key=os.path.getctime)
        print(f"Loading model from checkpoint: {latest_checkpoint}")
        model = LlamaForCausalLM.from_pretrained(latest_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)
    else:
        model, tokenizer = create_model_and_tokenizer(args.num_params)
    train_dataset, heldout_dataset = load_and_prepare_datasets(
        tokenizer, args.subset_size, args.max_order, args.N, args.qa_ratio
    )


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

    # This is the number of times we step through each profiles; the "dataset size" is infinite
    epochs = 5_000*25000/len(train_dataset)
    print(f"Epochs: {epochs}")
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/n{args.N}_p{args.num_params}_o{args.max_order}_wd{args.wd}_infinite",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_accumulation_steps=1,
        warmup_steps=500,
        weight_decay=args.wd,
        logging_dir=f"./logs/n{args.N}_p{args.num_params}_o{args.max_order}_wd{args.wd}_infinite",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=20000,
        save_steps=20000,
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        fp16=True,
        tf32=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=heldout_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if args.resume_from:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()

    # Evaluate on heldout profiles
    heldout_results = trainer.evaluate()
    print("Heldout Profiles Results:", heldout_results)

    # Save the model
    model.save_pretrained(f"./final_model_n{args.N}_p{args.num_params}_o{args.max_order}_wd{args.wd}")
    tokenizer.save_pretrained(f"./final_model_n{args.N}_p{args.num_params}_o{args.max_order}_wd{args.wd}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama model with specified parameters")
    parser.add_argument("--num_params", type=int, default=1_000_000, help="Number of parameters for the model")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of examples to use for training (for testing purposes)")
    parser.add_argument("--N", type=int, default=25000, help="Number of profiles to use for QA dataset")
    parser.add_argument("--max_order", type=int, default=None, help="Maximum order of qa dataset")
    parser.add_argument("--resume_from", action="store_true", help="Resume training from most recent checkpoint")
    parser.add_argument("--qa_ratio", type=float, default=0.5,
                       help="Ratio of QA examples to bios examples")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    args = parser.parse_args()
    
    main(args)
