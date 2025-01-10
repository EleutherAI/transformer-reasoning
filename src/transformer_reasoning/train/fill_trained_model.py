import math
import argparse
import torch
from transformers import LlamaForCausalLM
import glob
import os

from transformer_reasoning.train.train_utils import (
    calculate_model_size, 
    create_model_and_tokenizer, 
    train_single_model
)
from transformer_reasoning.train.dataset import load_and_prepare_datasets, MultiDataset


def get_dataset_loader(args):
    def load_multi_dataset(tokenizer, N, orders, relations, hop_ratio, heldout_sets, debug):
        # We need the heldout sets from the original training run
        dataset1 = load_and_prepare_datasets(
            tokenizer,
            args.N, 
            orders=args.orders, 
            relations=args.relations, 
            hop_ratio=args.hop_ratio,
            debug=args.debug,
            heldout_sets=heldout_sets
        )
        
        dataset2 = load_and_prepare_datasets(
            tokenizer,
            args.N2, 
            orders=[1], 
            relations=17, 
            hop_ratio=args.hop_ratio,
            debug=args.debug
        )

        multi_dataset = MultiDataset(
            datasets=[dataset1, dataset2],
            weights=[args.dataset1_weight, 1.0 - args.dataset1_weight]
        )
        return multi_dataset

    return load_multi_dataset

def main(args):
    # args.resume_from_checkpoint needs to be True for the training code to work
    args.resume_from_checkpoint = True

    # Create tokenizer only - model will be loaded from checkpoint
    _, tokenizer, real_num_params = create_model_and_tokenizer(args.num_params, args.num_layers)
    
    rel_str = f'_r{args.relations}' if args.relations and args.relations != '4' else ''
    hop_str = f'_hr{args.hop_ratio}' if args.hop_ratio != 0.1 else ''
    checkpoint_dir = f"./results/synchronized/n{args.N}_p{real_num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_l{args.num_layers}_lr{args.lr}_beta1{args.beta1}_sf{rel_str}{hop_str}"

    # Load checkpoint
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))

    if not args.checkpoint_number:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
    else:
        latest_checkpoint = [c for c in checkpoints if f"checkpoint-{args.checkpoint_number}" in c][0]
    print(f"Loading model from checkpoint: {latest_checkpoint}")
    model = LlamaForCausalLM.from_pretrained(latest_checkpoint)


    # Setup output directory
    curr_str = "_curr" if args.curriculum else ""
    sf_str = "sf" if args.optimizer_type == "schedulefree" else "adamw"
    hop_str = f"_hr{args.hop_ratio}" if args.hop_ratio != 0.1 else ""
    output_dir = f"./results/synchronized/multi_n{args.N}_{args.N2}_w{args.dataset1_weight}_p{real_num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_l{args.num_layers}_lr{args.lr}_beta1{args.beta1}_{sf_str}{rel_str}{hop_str}{curr_str}"

    model_size_mb = calculate_model_size(real_num_params)
    print(f"Estimated model size: {model_size_mb} MB")
    print(f"Epochs: {args.num_epochs}")

    load_multi_dataset = get_dataset_loader(args)

    train_single_model(model, tokenizer, args, checkpoint_dir, output_dir, args.curriculum, load_dataset_function=load_multi_dataset)

    if args.push_to_hub:
        hub_id = f"EleutherAI/llama_multihop_multi_n{args.N}_{args.N2}_w{args.dataset1_weight}_p{real_num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_l{args.num_layers}_lr{args.lr}_beta1{args.beta1}_{sf_str}{rel_str}{hop_str}{curr_str}"
        model.push_to_hub(hub_id)
        tokenizer.push_to_hub(hub_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue training a Llama model with multiple datasets")
    # Required arguments
    parser.add_argument("--N", type=int, required=True, help="Number of profiles to use for main QA dataset")
    parser.add_argument("--N2", type=int, required=True, help="Number of profiles to use for second QA dataset")
    
    # Optional arguments
    parser.add_argument("--checkpoint_number", type=int, help="Specific checkpoint number to use (uses latest if not specified)")
    parser.add_argument("--dataset1_weight", type=float, default=0.9, help="Weight for first dataset sampling (between 0 and 1)")
    parser.add_argument("--dataset2_loss_weight", type=float, default=0.1, help="Weight for second dataset loss")
    parser.add_argument("--num_params", type=int, default=500_000, help="Number of parameters for the model")
    parser.add_argument("--orders", type=int, nargs="+", default=None, help="Orders to use for QA dataset")
    parser.add_argument("--hop_ratio", type=float, default=0.1, help="Ratio of one-hop to two-hop QA examples")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--push_to_hub", action="store_true", help="Push trained model to hf hub")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.99, help="Beta1 for AdamW optimizer")
    parser.add_argument("--num_epochs", type=int, default=4000, help="Number of epochs to train for")
    parser.add_argument("--optimizer_type", type=str, default="schedulefree", 
                       choices=["schedulefree", "adamw_cosine", "adamw_linear"],
                       help="Type of optimizer to use")
    parser.add_argument("--num_training_steps", type=int, default=9_000_000,
                       help="Total number of training steps (required for adamw cosine scheduler)")
    parser.add_argument("--num_workers", type=int, default=15, help="Number of workers for data loading")
    parser.add_argument("--relations", type=str, default=None, help="Number of relations in the QA dataset")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning starting with 1-hop only")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.optimizer_type != "schedulefree" and args.num_training_steps is None:
        raise ValueError("num_training_steps must be specified when using adamw optimizer")
    
    if not (0 <= args.dataset1_weight <= 1):
        raise ValueError("dataset1_weight must be between 0 and 1")
    
    main(args)