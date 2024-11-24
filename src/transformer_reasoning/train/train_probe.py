import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_reasoning.train.train_utils import InfiniteQADataset, get_qa_token_ranges
from transformer_reasoning.utils import Classifier
from transformer_reasoning.evaluation.eval_utils import get_checkpoints
import argparse
from datasets import load_dataset
from tqdm import tqdm

def extract_activations(model, batch, layer_idx):
    """Extract activations from a specific layer's residual stream."""
    outputs = model(
        input_ids=batch['input_ids'].to(model.device),
        output_hidden_states=True,
        return_dict=True
    )
    # Get hidden states for the specified layer
    hidden_states = outputs.hidden_states[layer_idx]
    return hidden_states

def collect_qa_activations(model, dataloader, layer_idx, tokenizer, num_samples=10000):
    """Collect activations and labels for QA pairs."""
    device = next(model.parameters()).device
    # List of lists: each inner list contains activations for one position
    all_activations = [[] for _ in range(10)]
    all_labels = [[] for _ in range(10)]
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting activations"):
            # Get activations
            hidden_states = extract_activations(model, batch, layer_idx)
            
            # Find positions where labels != -100 (these are the answer tokens)
            label_positions = batch['labels'] != -100
            
            # Find start of answer sequences (where label_positions changes from False to True)
            answer_starts = label_positions & ~torch.roll(label_positions, 1, dims=1)
            # Set first position to False since roll gives us last position
            answer_starts[:, 0] = False
            
            # For each sequence in the batch
            for seq_idx, (seq_labels, seq_activations) in enumerate(zip(batch['labels'], hidden_states)):
                # Get all answer start positions
                answer_positions = torch.where(answer_starts[seq_idx])[0]
                
                if len(answer_positions) == 0:
                    continue

                # Create window of input positions (ans_pos-10 to ans_pos-1)
                for ans_pos in answer_positions:
                    input_positions = torch.arange(ans_pos-10, ans_pos)
                    
                    # Filter out any positions that would be negative
                    valid_mask = input_positions >= 0
                    input_positions = input_positions[valid_mask]
                    
                    # Extract activations and corresponding labels
                    for pos, in_pos in enumerate(input_positions):
                        all_activations[pos].append(seq_activations[in_pos].cpu())
                        all_labels[pos].append(seq_labels[ans_pos].cpu())
                
                if len(all_activations[-1]) >= num_samples:
                    break
            
            if len(all_activations[-1]) >= num_samples:
                break

    # Convert to tensors and one-hot encode labels
    num_classes = len(tokenizer)
    for pos in range(10):
        if all_activations[pos]:  # Check if list is not empty
            all_activations[pos] = torch.stack(all_activations[pos])
            labels = torch.stack(all_labels[pos])
            one_hot = torch.zeros((len(labels), num_classes))
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            all_labels[pos] = one_hot

    return all_activations, all_labels

def train_and_evaluate_probes(train_activations, train_labels, eval_activations, eval_labels, input_dim, device, max_iter=1000):
    """Train separate probes for each position and evaluate."""
    # Train 10 separate probes
    probes = []
    train_losses = []
    for pos in tqdm(range(10), desc="Training position-specific probes"):
        probe = Classifier(input_dim=input_dim, num_classes=train_labels[pos].shape[-1], device=device)
        loss = probe.fit(
            train_activations[pos].to(device),
            train_labels[pos].to(device),
            max_iter=max_iter
        )
        probes.append(probe)
        train_losses.append(loss)
    
    # Evaluate each probe on the eval set
    eval_losses = []
    for sample_idx in range(len(eval_activations[0])):
        sample_losses = []
        for pos in range(10):
            probe = probes[pos]
            with torch.no_grad():
                logits = probe(eval_activations[pos][sample_idx].unsqueeze(0).to(device))
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    eval_labels[pos][sample_idx].unsqueeze(0).to(device)
                )
                sample_losses.append(loss.item())
        # Take best loss among all positions
        eval_losses.append(min(sample_losses))
    
    avg_best_loss = sum(eval_losses) / len(eval_losses)
    breakpoint()
    return probes, train_losses, eval_losses,avg_best_loss

def main(args):
    
    checkpoints = get_checkpoints(1, 2, args.N_profiles, args.n_params, 0.1)
    checkpoints = [f for f in checkpoints if 'checkpoint-' in f]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    latest_checkpoint = checkpoints[-1]

    model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/llama_multihop_tokenizer')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load dataset
    profiles = load_dataset(f"EleutherAI/profiles_dataset_{args.N_profiles}_uniform", keep_in_memory=True)['train']
    
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
    
    # Collect activations
    print(f"Collecting activations from layer {args.layer_idx}")
    activations, labels = collect_qa_activations(
        model, 
        dataloader, 
        args.layer_idx,
        tokenizer,
        args.num_samples
    )
    
    # Split into train and eval sets
    train_size = int(0.8 * len(activations[0]))
    train_activations, eval_activations = [activations[i][:train_size] for i in range(10)], [activations[i][train_size:] for i in range(10)]
    train_labels, eval_labels = [labels[i][:train_size] for i in range(10)], [labels[i][train_size:] for i in range(10)]
    
    # Initialize and train probes
    input_dim = activations[0].shape[-1]
    probes, train_losses, eval_losses, avg_best_eval_loss = train_and_evaluate_probes(
        train_activations,
        train_labels,
        eval_activations,
        eval_labels,
        input_dim,
        device,
        args.max_iter
    )
    
    print(f"Average best evaluation loss: {avg_best_eval_loss}")
    
    # Save probes and results
    torch.save({
        'probes': [probe.state_dict() for probe in probes],
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'avg_best_eval_loss': avg_best_eval_loss,
    }, f'probes_layer_{args.layer_idx}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_profiles", type=int, required=True,
                      help="Number of profiles to use")
    parser.add_argument("--layer_idx", type=int, required=True,
                      help="Index of the layer to probe")
    parser.add_argument("--n_params", type=int, default=25000,
                      help="Number of parameters in the model")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for collecting activations")
    parser.add_argument("--num_samples", type=int, default=10000,
                      help="Number of samples to collect")
    parser.add_argument("--max_iter", type=int, default=1000,
                      help="Maximum iterations for probe training")
    parser.add_argument("--hops", type=int, default=1,
                      help="Number of hops in the question")
    
    args = parser.parse_args()
    main(args)
