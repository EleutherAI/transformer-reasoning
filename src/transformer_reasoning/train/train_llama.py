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
from datasets import load_dataset
from transformer_reasoning.train.train_utils import calculate_model_size, create_model_and_tokenizer, InfiniteBiosDataset, LogConstantCheckpointCallback
from transformer_reasoning.generate_dataset.generate_qa_dataset import generate_question

from datasets import Dataset
import glob
import os


def load_and_prepare_datasets(tokenizer, subset_size=None, N=250000, qa_ratio=0.1, orders=None):
    # Load profiles dataset
    profiles = load_dataset(f"EleutherAI/profiles_dataset_{N}")['train']
    
    shuffled_indices = torch.randperm(len(profiles)).tolist()
    heldout_indices = shuffled_indices[:1000]
    retained_indices = shuffled_indices[1000:]

    # Generate QA dataset for evaluation from the heldout profiles
    heldout_profiles = profiles.select(heldout_indices)
    eval_questions = []
    
    for profile in heldout_profiles:
        qa_result = generate_question(profile, profiles, max(orders) if orders else 3, {}, {})
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
        orders=orders or [1,2],
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
        base_dir = args.resume_from
        checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        print(f"Loading model from checkpoint: {latest_checkpoint}")
        model = LlamaForCausalLM.from_pretrained(latest_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)
    else:
        model, tokenizer = create_model_and_tokenizer(args.num_params)
    train_dataset, heldout_dataset = load_and_prepare_datasets(
        tokenizer, args.subset_size, args.N, args.qa_ratio, args.orders
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
    epochs = 4_500*25000/len(train_dataset)
    print(f"Epochs: {epochs}")
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/n{args.N}_p{args.num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_infinite",
        num_train_epochs=epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=16,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=int(max(1, args.train_batch_size//32)),
        warmup_steps=500,
        weight_decay=args.wd,
        logging_dir=f"./logs/n{args.N}_p{args.num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_infinite",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=20000,
        save_steps=20000,
        dataloader_num_workers=0,
        fp16=True,
        tf32=True,
        hub_strategy="every_save",
        hub_model_id=f"EleutherAI/llama_multihop_n{args.N}_p{args.num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}",
        push_to_hub=args.push_to_hub,
        save_strategy="no"
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

    trainer.add_callback(LogConstantCheckpointCallback(trainer))

    if args.resume_from:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()

    # Evaluate on heldout profiles
    heldout_results = trainer.evaluate()
    print("Heldout Profiles Results:", heldout_results)

    # Save the model
    model.save_pretrained(f"./final_model/n{args.N}_p{args.num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}")
    tokenizer.save_pretrained(f"./final_model/n{args.N}_p{args.num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama model with specified parameters")
    parser.add_argument("--num_params", type=int, default=1_000_000, help="Number of parameters for the model")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of examples to use for training (for testing purposes)")
    parser.add_argument("--N", type=int, default=25000, help="Number of profiles to use for QA dataset")
    parser.add_argument("--orders", type=int, nargs="+", default=None, help="Orders to use for QA dataset")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from given checkpoint")
    parser.add_argument("--qa_ratio", type=float, default=0.5,
                       help="Ratio of QA examples to bios examples")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--push_to_hub", action="store_true", help="Push trained model to hf hub")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()
    
    main(args)
