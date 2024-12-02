import math
import argparse
import torch
from transformers import (
    LlamaForCausalLM
)
from datasets import load_dataset
from transformer_reasoning.train.train_utils import calculate_model_size, create_model_and_tokenizer, InfiniteQADataset, train_single_model

import glob
import os


def load_and_prepare_datasets(tokenizer, N=250000, qa_ratio=0.1, orders=None, relations=None, hop_ratio=0.1):
    relation_str = f'_r{relations}' if relations else ''
    # Load profiles dataset
    profiles = load_dataset(f"EleutherAI/profiles_dataset_{N}_uniform{relation_str}", keep_in_memory=True)['train']
    # Create infinite training dataset
    train_dataset = InfiniteQADataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=orders or [1,2],
        qa_indices=list(range(len(profiles))),
        hop_ratio=hop_ratio
    )
    
    onehop_dataset = InfiniteQADataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=[1],
        qa_indices=list(range(len(profiles))),
        hop_ratio=hop_ratio
    )

    twohop_dataset = InfiniteQADataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=[2],
        qa_indices=list(range(len(profiles))),
        hop_ratio=hop_ratio
    )
    return train_dataset, onehop_dataset, twohop_dataset


def find_question_end(text, tokenizer):
    # Find the last occurrence of "? "
    question_end = text.rfind(": ")
    if question_end == -1:
        return None
    
    # Tokenize up to the question mark, including special tokens
    question_tokens = tokenizer.encode(text[:question_end+1], add_special_tokens=True)
    return len(question_tokens)


def main(args):
    rel_str = f'_r{args.relations}' if args.relations else ''
    # Create single model and tokenizer
    model, tokenizer, real_num_params = create_model_and_tokenizer(args.num_params, args.num_layers)
    # Convert model to bfloat16
    sf_str = "sf" if args.optimizer_type == "schedulefree" else "adamw"
    hop_str = f"_hr{args.hop_ratio}" if args.hop_ratio != 0.1 else ""
    output_dir = f"./results/n{args.N}_p{real_num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_l{args.num_layers}_lr{args.lr}_beta1{args.beta1}_{sf_str}{rel_str}{hop_str}"

    if args.resume_from_checkpoint:
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        print(f"Loading model from checkpoint: {latest_checkpoint}")
        model = LlamaForCausalLM.from_pretrained(latest_checkpoint)

    train_dataset, onehop_dataset, twohop_dataset = load_and_prepare_datasets(
        tokenizer, args.N, args.qa_ratio, orders=args.orders, relations=args.relations, hop_ratio=args.hop_ratio
    )

    model_size_mb = calculate_model_size(real_num_params)
    print(f"Estimated model size: {model_size_mb} MB")
    print(f"Epochs: {args.num_epochs}")

    train_single_model(model, train_dataset, onehop_dataset, twohop_dataset, args, output_dir)

    if args.push_to_hub:
        hub_id = f"EleutherAI/llama_multihop_n{args.N}_p{real_num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_l{args.num_layers}_lr{args.lr}_beta1{args.beta1}_{sf_str}{rel_str}{hop_str}"
        model.push_to_hub(hub_id)
        tokenizer.push_to_hub(hub_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama model with specified parameters")
    parser.add_argument("--num_params", type=int, default=500_000, help="Number of parameters for the model")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of examples to use for training (for testing purposes)")
    parser.add_argument("--N", type=int, default=25000, help="Number of profiles to use for QA dataset")
    parser.add_argument("--orders", type=int, nargs="+", default=None, help="Orders to use for QA dataset")
    parser.add_argument("--qa_ratio", type=float, default=0.5,
                       help="Ratio of QA examples to bios examples")
    parser.add_argument("--hop_ratio", type=float, default=0.1, help="Ratio of one-hop to two-hop QA examples")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--push_to_hub", action="store_true", help="Push trained model to hf hub")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--schedule_free", action="store_true", help="Use schedule-free training")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the model")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW optimizer")
    parser.add_argument("--num_epochs", type=int, default=2000, help="Number of epochs to train for")
    parser.add_argument("--optimizer_type", type=str, default="schedulefree", choices=["schedulefree", "adamw_cosine", "adamw_linear"],
                        help="Type of optimizer to use (schedulefree or regular adamw with cosine scheduler)")
    parser.add_argument("--num_training_steps", type=int, default=9_000_000,
                        help="Total number of training steps (required for adamw cosine scheduler)")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--num_workers", type=int, default=15, help="Number of workers for data loading")
    parser.add_argument("--relations", type=str, default=None, help="Number of relations in the QA dataset")
    args = parser.parse_args()
    
    if args.optimizer_type == "adamw" and args.num_training_steps is None:
        raise ValueError("num_training_steps must be specified when using adamw optimizer")
    
    main(args)