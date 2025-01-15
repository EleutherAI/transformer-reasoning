import argparse
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import glob

from transformer_reasoning.train.train_llama import create_model_and_tokenizer
from transformer_reasoning.train.train_utils import create_or_load_optimizer, set_model_base_shapes
from transformer_reasoning.train.dataset import load_and_prepare_datasets
from transformer_reasoning.generate_dataset.generate_qa_dataset import maybe_generate_question

from transformer_reasoning.models.llama_mup import LlamaMuPForCausalLM

def evaluate_detailed_generalization(model, tokenizer, dataset, output_dir):
    """Evaluate model on each complete two-hop tuple."""
    model.eval()
    results = []
    batch_size = 32
    
    # Get all tuples we want to evaluate
    tuples = dataset.heldout_sets["complete_two_hop_tuples"]
    seen_pairs = set()

    # Process tuples in batches
    for i in tqdm(range(0, len(tuples), batch_size), desc="Evaluating generalization"):
        batch_tuples = tuples[i:i + batch_size]
        questions = []
        batch_pairs = []

        # Generate second-order questions for each tuple
        for profile_idx, relation, attribute, second_person_idx in batch_tuples:
            profile = dataset.profiles[profile_idx]
            question = maybe_generate_question(
                profile=profile,
                profiles=dataset.profiles,
                order=2,
                mode="eval_complete_two_hop_questions",
                heldout_sets=dataset.heldout_sets,
                relation=relation,
                subject=attribute
            )
            questions.append(f"Question: {question['question']} Answer: {question['answer']}")

        len_two_hop = len(questions)
        # Add to batch pairs
        for profile_idx, _, attribute, _ in batch_tuples:
            if (profile_idx, attribute) not in seen_pairs:
                seen_pairs.add((profile_idx, attribute))
                profile = dataset.profiles[profile_idx]
                question = maybe_generate_question(
                    profile=profile,
                    profiles=dataset.profiles,
                    order=1,
                    mode="eval",
                    subject=attribute
                )
                questions.append(f"Question: {question['question']} Answer: {question['answer']}")
                seen_pairs.add((profile_idx, attribute))
                batch_pairs.append((profile_idx, attribute))

        # Tokenize batch
        encodings = tokenizer(
            questions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encodings['input_ids']
        
        starts = torch.all(torch.stack([
            input_ids.roll(-i) == token 
            for i, token in enumerate(dataset.answer_sep_tokens)
        ]), dim=0)

        ends = input_ids == dataset.question_sep_tokens[0]
        
        start_indices = torch.where(starts)[1]
        # Get all end indices
        batch_indices, end_pos = torch.where(ends)
        end_pos = end_pos + len(dataset.question_sep_tokens) - 1
        
        # Group by batch index and get second occurrence
        second_end_indices = []
        for i in range(len(input_ids)):
            batch_ends = end_pos[batch_indices == i]
            if len(batch_ends) > 1:
                second_end_indices.append(batch_ends[1].item())
            else:
                second_end_indices.append(0)

        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for i, (start, end) in enumerate(zip(start_indices, second_end_indices)):
            mask[i, start+len(dataset.answer_sep_tokens):end] = True
        
        # Move to device
        input_ids = encodings['input_ids'].to(model.device)
        
        # Get model outputs without reduction
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            
            # Get logits and compute per-example loss manually
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            labels = input_ids.clone()
            labels[~mask] = -100
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_logits.shape[0], -1)
            
            # Reshape and mask losses (only consider non-masked tokens)
            loss = loss.view(shift_labels.shape)
            
            # Calculate mean loss per example
            losses = torch.sum(loss, dim=1)
            # Store results
            for (profile_idx, relation, attribute, second_person_idx), loss in zip(batch_tuples, losses[:len_two_hop]):
                results.append({
                    'first_person': profile_idx,
                    'relation': relation,
                    'second_person': second_person_idx,
                    'attribute': attribute,
                    'loss': loss.item()
                })
            for (profile_idx, attribute), loss in zip(batch_pairs, losses[len_two_hop:]):
                results.append({
                    'first_person': profile_idx,
                    'relation': None,
                    'second_person': None,
                    'attribute': attribute,
                    'loss': loss.item()
                })
    
    # Create and save DataFrame
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/detailed_generalization_data.csv", index=False)
    return df

def main(args):
    # Create model and tokenizer
    dummy_model, tokenizer, real_num_params = create_model_and_tokenizer(args.num_params, args.num_layers)
    
    # Construct experiment name as in train_llama.py
    rel_str = f'_r{args.relations}' if args.relations else ''
    curr_str = "_curr" if args.curriculum else ""
    sf_str = "sf" if args.optimizer_type == "schedulefree" else "adamw"
    hop_str = f"_hr{args.hop_ratio}" if args.hop_ratio != 0.1 else ""
    
    experiment_name = f"mup_n{args.N}_p{real_num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_l{args.num_layers}_lr{args.lr}_beta1{args.beta1}_{sf_str}{rel_str}{hop_str}{curr_str}"
    output_dir = os.path.join("./results", args.commit_hash, experiment_name)
    

    # Load checkpoint
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if checkpoints:
        if args.checkpoint_number:
            checkpoint_path = [c for c in checkpoints if f"checkpoint-{args.checkpoint_number}" in c][0]
        else:
            checkpoint_path = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = LlamaMuPForCausalLM.from_pretrained(checkpoint_path)
        set_model_base_shapes(model, args.num_layers, tokenizer)
    else:
        raise ValueError(f"No checkpoint found in {output_dir}")

        # Get optimizer to load heldout sets
    _, _, _, heldout_sets = create_or_load_optimizer(dummy_model, args, checkpoint_path)


    # Load dataset
    train_dataset = load_and_prepare_datasets(
        tokenizer,
        args.N,
        orders=args.orders,
        relations=args.relations,
        hop_ratio=args.hop_ratio
    )
    
    if heldout_sets:
        train_dataset.heldout_sets = heldout_sets
    
    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Run evaluation
    results_df = evaluate_detailed_generalization(model, tokenizer, train_dataset, output_dir)
    print(f"Results saved to {output_dir}/detailed_generalization_data.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_params", type=int, default=500_000)
    parser.add_argument("--N", type=int, default=25000)
    parser.add_argument("--orders", type=int, nargs="+", default=[1,2])
    parser.add_argument("--hop_ratio", type=float, default=0.1)
    parser.add_argument("--relations", type=str, default=None)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.99)
    parser.add_argument("--optimizer_type", type=str, default="schedulefree")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--commit_hash", type=str, required=True,
                        help="Git commit hash for the experiment")
    parser.add_argument("--checkpoint_number", type=int, default=None,
                        help="Specific checkpoint to load (default: latest)")
    
    args = parser.parse_args()
    main(args) 