import math
import argparse
import torch
from transformers import (
    LlamaForCausalLM
)
from datasets import load_dataset
from transformer_reasoning.train.train_utils import calculate_model_size, create_model_and_tokenizer, InfiniteQADataset, train_parallel_models

import glob
import os


def load_and_prepare_datasets(tokenizer, N=250000, qa_ratio=0.1, orders=None):
    # Load profiles dataset
    profiles = load_dataset(f"EleutherAI/profiles_dataset_{N}_uniform", keep_in_memory=True)['train']
    # Create infinite training dataset
    train_dataset = InfiniteQADataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=orders or [1,2],
        qa_indices=list(range(len(profiles)))
    )
    
    onehop_dataset = InfiniteQADataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=[1],
        qa_indices=list(range(len(profiles)))
    )

    twohop_dataset = InfiniteQADataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=[2],
        qa_indices=list(range(len(profiles)))
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

    models_dict = {
        'model':[],
        'num_params':[],
        'num_layers':[],
        'N_profiles':[],
        'orders':[],
        'wd':[],
        'lr':[],
        'beta1':[],
    }
    output_dirs = []
    sf_str = "schedulefree" if args.schedule_free else "adamw"
    for nominal_param_count in args.num_params:
        model, tokenizer, real_num_params = create_model_and_tokenizer(nominal_param_count, args.num_layers)
        # Convert model to bfloat16
        model = model.to(dtype=torch.bfloat16)
        models_dict['model'].append(model)
        models_dict['num_params'].append(real_num_params)
        models_dict['num_layers'].append(args.num_layers)
        models_dict['N_profiles'].append(args.N)
        models_dict['orders'].append(args.orders)
        models_dict['wd'].append(args.wd)
        models_dict['lr'].append(args.lr)
        models_dict['beta1'].append(args.beta1)
        output_dirs.append(f"./results/n{args.N}_p{real_num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_l{args.num_layers}_lr{args.lr}_beta1{args.beta1}_{sf_str}")

    if args.resume_from:
        for i, base_dir in enumerate(output_dirs):
            checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            print(f"Loading model from checkpoint: {latest_checkpoint}")
            models_dict['model'][i] = LlamaForCausalLM.from_pretrained(latest_checkpoint)

    train_dataset, onehop_dataset, twohop_dataset = load_and_prepare_datasets(
        tokenizer, args.N, args.qa_ratio, args.orders
    )

    model_sizes_mb = [calculate_model_size(real_num_params) for real_num_params in models_dict['num_params']]
    print(f"Estimated model sizes: {model_sizes_mb} MB")

    # This is the number of times we step through each profiles; the "dataset size" is infinite
    print(f"Epochs: {args.num_epochs}")
    # Set up training arguments
    train_parallel_models(models_dict, train_dataset, onehop_dataset, twohop_dataset, args, output_dirs)

    hub_ids = [f"EleutherAI/llama_multihop_n{args.N}_p{real_num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_l{args.num_layers}_lr{args.lr}_beta1{args.beta1}_{sf_str}" 
               for real_num_params in models_dict['num_params']]
    # Save the model
    for i in range(len(args.num_params)):
        models_dict['model'][i].save_pretrained("./" + output_dirs[i].split("/")[-1])
        tokenizer.save_pretrained("./" + output_dirs[i].split("/")[-1])
        if args.push_to_hub:
            models_dict['model'][i].push_to_hub(hub_ids[i])
            tokenizer.push_to_hub(hub_ids[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama model with specified parameters")
    parser.add_argument("--num_params", type=int, nargs="+", default=[500_000, 700_000, 900_000, 1_000_000], help="Number of parameters for the model")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of examples to use for training (for testing purposes)")
    parser.add_argument("--N", type=int, default=25000, help="Number of profiles to use for QA dataset")
    parser.add_argument("--orders", type=int, nargs="+", default=None, help="Orders to use for QA dataset")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from given checkpoint")
    parser.add_argument("--qa_ratio", type=float, default=0.5,
                       help="Ratio of QA examples to bios examples")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--push_to_hub", action="store_true", help="Push trained model to hf hub")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--schedule_free", action="store_true", help="Use schedule-free training")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the model")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW optimizer")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3], help="GPUs to use for training")
    parser.add_argument("--num_epochs", type=int, default=9000, help="Number of epochs to train for")
    parser.add_argument("--optimizer_type", type=str, default="schedulefree", choices=["schedulefree", "adamw"],
                        help="Type of optimizer to use (schedulefree or regular adamw with cosine scheduler)")
    parser.add_argument("--num_training_steps", type=int, default=9_000_000,
                        help="Total number of training steps (required for adamw cosine scheduler)")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()
    
    if args.optimizer_type == "adamw" and args.num_training_steps is None:
        raise ValueError("num_training_steps must be specified when using adamw optimizer")
    
    main(args)