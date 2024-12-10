import math
import argparse
import jax
import jax.numpy as jnp
from flax.training import train_state
import glob
import os

from transformers import FlaxLlamaForCausalLM, LlamaConfig, AutoTokenizer

from transformer_reasoning.train.train_utils import calculate_model_size, calculate_architecture
from transformer_reasoning.train.train_utils_jax import train_single_model
from transformer_reasoning.train.dataset import load_and_prepare_datasets


def main(args):
    # Print device info
    print(f"Available devices: {jax.devices()}")

    rel_str = f'_r{args.relations}' if args.relations else ''
    
    # Create model and tokenizer
    model, tokenizer, real_num_params = create_model_and_tokenizer(args.num_params, args.num_layers)
    
    curr_str = "_curr" if args.curriculum else ""
    sf_str = "sf"  # JAX version always uses schedule-free
    hop_str = f"_hr{args.hop_ratio}" if args.hop_ratio != 0.1 else ""
    output_dir = f"/mnt/ssd-1/david/transformer-reasoning/results/jax_n{args.N}_p{real_num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_l{args.num_layers}_lr{args.lr}_beta1{args.beta1}_{sf_str}{rel_str}{hop_str}{curr_str}"
    
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    train_dataset, onehop_dataset, twohop_dataset = load_and_prepare_datasets(
        tokenizer, 
        args.N, 
        orders=args.orders, 
        relations=args.relations, 
        hop_ratio=args.hop_ratio,
        jax=True
    )

    if args.curriculum:
        # Start with only 1-hop questions
        train_dataset.order_weights = [1.0, 0.0]
        print("Starting curriculum learning with 1-hop only")

    model_size_mb = calculate_model_size(real_num_params)
    print(f"Estimated model size: {model_size_mb} MB")
    print(f"Epochs: {args.num_epochs}")

    # Train model
    train_single_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        onehop_dataset=onehop_dataset,
        twohop_dataset=twohop_dataset,
        args=args,
        output_dir=output_dir,
        curriculum=args.curriculum
    )

    if args.push_to_hub:
        hub_id = f"EleutherAI/llama_multihop_jax_n{args.N}_p{real_num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_l{args.num_layers}_lr{args.lr}_beta1{args.beta1}_{sf_str}{rel_str}{hop_str}{curr_str}"
        model.save_pretrained(hub_id)
        tokenizer.push_to_hub(hub_id)


def create_model_and_tokenizer(num_params, num_layers=4):
    """Creates a FlaxLlamaForCausalLM model and tokenizer."""
    n_layers, hidden_size = calculate_architecture(num_params, num_layers)
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llama_multihop_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=n_layers,
        num_attention_heads=hidden_size // 16,
        max_position_embeddings=2048,
    )
    
    model = FlaxLlamaForCausalLM(config)
    
    real_num_params = sum(p.size for p in jax.tree_util.tree_leaves(model.params))
    print(f"Model has {real_num_params} parameters")

    return model.module, tokenizer, real_num_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Llama model with specified parameters using JAX")
    parser.add_argument("--num_params", type=int, default=500_000, help="Number of parameters for the model")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of examples to use for training (for testing purposes)")
    parser.add_argument("--N", type=int, default=25000, help="Number of profiles to use for QA dataset")
    parser.add_argument("--orders", type=int, nargs="+", default=None, help="Orders to use for QA dataset")
    parser.add_argument("--qa_ratio", type=float, default=0.5, help="Ratio of QA examples to bios examples")
    parser.add_argument("--hop_ratio", type=float, default=0.1, help="Ratio of one-hop to two-hop QA examples")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--push_to_hub", action="store_true", help="Push trained model to hf hub")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.99, help="Beta1 for AdamW optimizer")
    parser.add_argument("--num_epochs", type=int, default=4000, help="Number of epochs to train for")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--num_workers", type=int, default=15, help="Number of workers for data loading")
    parser.add_argument("--relations", type=str, default=None, help="Number of relations in the QA dataset")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning starting with 1-hop only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    args = parser.parse_args()
    
    main(args)