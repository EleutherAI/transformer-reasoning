import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_reasoning.train.train_utils import InfiniteQADataset, get_qa_token_ranges
from transformer_reasoning.utils import Classifier
from transformer_reasoning.evaluation.eval_utils import get_checkpoints
import argparse
from datasets import load_dataset
from tqdm import tqdm
import csv
import os

def extract_activations(model, batch, layer_idx):
    """Extract activations from a specific layer's residual stream."""
    outputs = model(
        input_ids=batch['input_ids'].to(model.device),
        output_hidden_states=True,
        return_dict=True
    )
    # Get hidden states for the specified layer
    hidden_states = outputs.hidden_states[layer_idx]
    return hidden_states, outputs.logits

def collect_qa_activations(model, dataloader, layer_idx, tokenizer, num_samples=10000, lookback=13):
    """Collect activations and labels for QA pairs."""
    device = next(model.parameters()).device
    # List of lists: each inner list contains activations for one position
    all_activations = [[] for _ in range(lookback)]
    all_labels = [[] for _ in range(lookback)]
    all_model_logits = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting activations"):
            # Get activations and logits
            hidden_states, logits = extract_activations(model, batch, layer_idx)
            
            # Find positions where labels != -100 (these are the answer tokens)
            label_positions = batch['labels'] != -100
            
            # Find start of answer sequences (where label_positions changes from False to True)
            answer_starts = label_positions & ~torch.roll(label_positions, 1, dims=1)
            # Set first position to False since roll gives us last position
            answer_starts[:, 0] = False
            
            # For each sequence in the batch
            for seq_idx, (seq_labels, seq_activations, seq_logits) in enumerate(zip(batch['labels'], hidden_states, logits)):
                # Get all answer start positions
                answer_positions = torch.where(answer_starts[seq_idx])[0]
                
                if len(answer_positions) == 0:
                    continue

                # Create window of input positions (ans_pos-10 to ans_pos-1)
                for ans_pos in answer_positions:
                    input_positions = torch.arange(ans_pos-lookback, ans_pos)
                    
                    # Filter out any positions that would be negative
                    valid_mask = input_positions >= 0
                    input_positions = input_positions[valid_mask]
                    
                    # Extract activations and corresponding labels
                    for pos, in_pos in enumerate(input_positions):
                        all_activations[pos].append(seq_activations[in_pos].cpu())
                        all_labels[pos].append(seq_labels[ans_pos].cpu())
                    
                    # Store model's logits for the first answer token
                    all_model_logits.append(seq_logits[ans_pos - 1].cpu())
                
                if len(all_activations[-1]) >= num_samples:
                    break
            
            if len(all_activations[-1]) >= num_samples:
                break

    # Convert to tensors and one-hot encode labels
    num_classes = len(tokenizer)
    for pos in range(lookback):
        if all_activations[pos]:  # Check if list is not empty
            all_activations[pos] = torch.stack(all_activations[pos])
            labels = torch.stack(all_labels[pos])
            one_hot = torch.zeros((len(labels), num_classes))
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            all_labels[pos] = one_hot

    # Convert model logits to tensor
    all_model_logits = torch.stack(all_model_logits) if all_model_logits else None

    return all_activations, all_labels, all_model_logits

def train_and_evaluate_probes(
        train_activations, 
        train_labels, 
        eval_activations, 
        eval_labels, 
        eval_model_logits, 
        input_dim, 
        device, 
        max_iter=1000, 
        lookback=13
    ):
    """Train separate probes for each position and evaluate."""
    # Train 10 separate probes

    probes = []
    train_losses = []
    for pos in tqdm(range(lookback), desc="Training position-specific probes"):
        probe = Classifier(input_dim=input_dim, num_classes=train_labels[pos].shape[-1], device=device)
        loss = probe.fit_cv(
            train_activations[pos].to(device),
            train_labels[pos].to(device),
            max_iter=max_iter,
            k=3,
            num_penalties=4,
        )
        probes.append(probe)
        train_losses.append(loss)
    
    # Evaluate each probe on the eval set
    eval_losses = []
    best_loss_positions = []
    last_token_losses = []
    model_losses = []  # Store model's losses
    uniform_losses = []  # Store uniform distribution losses

    # Calculate uniform distribution over first answer tokens
    all_first_answers = torch.sum(eval_labels[0], dim=0) + torch.sum(train_labels[0], dim=0)
    uniform_dist = all_first_answers / torch.sum(all_first_answers)

    for sample_idx in range(len(eval_activations[0])):
        sample_losses = []
        for pos in range(lookback):
            probe = probes[pos]
            with torch.no_grad():
                logits = probe(eval_activations[pos][sample_idx].unsqueeze(0).to(device))
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    eval_labels[pos][sample_idx].unsqueeze(0).to(device)
                )
                sample_losses.append(loss.item())
        
        # Calculate model's loss
        model_loss = torch.nn.functional.cross_entropy(
            eval_model_logits[sample_idx].unsqueeze(0).to(device),
            eval_labels[0][sample_idx].unsqueeze(0).to(device)
        )
        model_losses.append(model_loss.item())
        
        # Calculate uniform distribution loss
        uniform_loss = torch.nn.functional.cross_entropy(
            uniform_dist.unsqueeze(0).to(device),
            eval_labels[0][sample_idx].unsqueeze(0).to(device)
        )
        uniform_losses.append(uniform_loss.item())
        
        eval_losses.append(min(sample_losses))
        best_loss_positions.append(len(sample_losses) - sample_losses.index(min(sample_losses)))
        last_token_losses.append(sample_losses[-1])

    avg_best_loss = sum(eval_losses) / len(eval_losses)
    avg_best_loss_positions = sum(best_loss_positions) / len(best_loss_positions)
    avg_last_token_loss = sum(last_token_losses) / len(last_token_losses)
    avg_model_loss = sum(model_losses) / len(model_losses)
    avg_uniform_loss = sum(uniform_losses) / len(uniform_losses)
    
    return probes, train_losses, eval_losses, avg_best_loss, avg_best_loss_positions, avg_last_token_loss, avg_model_loss, avg_uniform_loss

def main(args):
    
    checkpoints = get_checkpoints(1, args.max_train_hops, args.N_profiles, args.n_params, 0.1, args.relations, layers=args.layers)
    checkpoints = [f for f in checkpoints if 'checkpoint-' in f]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    latest_checkpoint = checkpoints[-1]

    # Get parent directory and checkpoint name
    checkpoint_parent = os.path.dirname(os.path.dirname(latest_checkpoint))
    checkpoint_name = os.path.basename(latest_checkpoint)
    
    model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/llama_multihop_tokenizer')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    rel_str = f'_r{args.relations}' if args.relations is not None else ''
    # Load dataset
    profiles = load_dataset(f"EleutherAI/profiles_dataset_{args.N_profiles}_uniform{rel_str}", keep_in_memory=True)['train']
    
    # Create dataset for one-hop questions
    dataset = InfiniteQADataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=[args.hops],  # Only one-hop questions
        qa_indices=list(range(len(profiles)))
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=20
    )

    lookback = 13
    
    results = []
    for layer_idx in range(args.min_layer_idx, args.max_layer_idx + 1):
        print(f"\nProcessing layer {layer_idx}")
        # Replace args.layer_idx with layer_idx in existing code
        activations, labels, model_logits = collect_qa_activations(
            model, 
            dataloader, 
            layer_idx,  # Changed from args.layer_idx
            tokenizer,
            args.num_samples,
            lookback
        )
        
        # Split into train and eval sets
        train_size = int(0.95 * len(activations[0]))
        train_activations = [activations[i][:train_size] for i in range(lookback)]
        eval_activations = [activations[i][train_size:] for i in range(lookback)]
        train_labels = [labels[i][:train_size] for i in range(lookback)]
        eval_labels = [labels[i][train_size:] for i in range(lookback)]
        eval_model_logits = model_logits[train_size:]

        # Initialize and train probes
        input_dim = activations[0].shape[-1]
        probes, train_losses, eval_losses, avg_best_eval_loss, avg_best_loss_positions, avg_last_token_loss, avg_model_loss, avg_uniform_loss = train_and_evaluate_probes(
            train_activations,
            train_labels,
            eval_activations,
            eval_labels,
            eval_model_logits,
            input_dim,
            device,
            args.max_iter,
            lookback
        )
        
        print(f"Average best evaluation loss: {avg_best_eval_loss}")
        print(f"Average best loss positions: {avg_best_loss_positions}")
        print(f"Average last token loss: {avg_last_token_loss}")
        print(f"Average model loss: {avg_model_loss}")
        print(f"Average uniform distribution loss: {avg_uniform_loss}")
        
        result = {
            'layer': layer_idx,
            'avg_best_eval_loss': avg_best_eval_loss,
            'avg_last_token_loss': avg_last_token_loss,
            'avg_model_loss': avg_model_loss,
            'avg_uniform_loss': avg_uniform_loss
        }
        results.append(result)
        
        # Modify save paths to use parent directory and checkpoint name
        results_dir = os.path.join(checkpoint_parent, f'probe_results_{checkpoint_name}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save probes for this layer with checkpoint name
        torch.save({
            'probes': [probe.state_dict() for probe in probes],
            'train_losses': train_losses,
            'eval_losses': eval_losses,
            **result
        }, os.path.join(results_dir, f'probes_layer_{layer_idx}.pt'))

    # Save CSV in parent directory with checkpoint name
    csv_path = os.path.join(checkpoint_parent, f'probe_results_{checkpoint_name}.csv')
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_profiles", type=int, required=True,
                      help="Number of profiles to use")
    parser.add_argument("--min_layer_idx", type=int, required=True,
                      help="Minimum layer index to probe")
    parser.add_argument("--max_layer_idx", type=int, required=True,
                      help="Maximum layer index to probe")
    parser.add_argument("--n_params", type=int, default=25000,
                      help="Number of parameters in the model")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for collecting activations")
    parser.add_argument("--num_samples", type=int, default=100000,
                      help="Number of samples to collect")
    parser.add_argument("--max_iter", type=int, default=1000,
                      help="Maximum iterations for probe training")
    parser.add_argument("--hops", type=int, default=1,
                      help="Number of hops in the question")
    parser.add_argument("--max_train_hops", type=int, default=2,
                      help="Maximum number of hops to train on")
    parser.add_argument("--relations", type=int, default=None,
                      help="Num relations in dataset")
    parser.add_argument("--layers", type=int, default=4,
                      help="Num layers in model")
    
    args = parser.parse_args()
    main(args)
