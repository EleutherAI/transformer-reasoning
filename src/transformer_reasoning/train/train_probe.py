import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_reasoning.train.train_utils import InfiniteQADataset
from transformer_reasoning.utils import Classifier
from transformer_reasoning.evaluation.eval_utils import get_checkpoints
from transformer_reasoning.generate_dataset.generate_qa_dataset import SECOND_ORDER_TEMPLATE
import argparse
from datasets import load_dataset
from tqdm import tqdm
import csv
import os

def generate_probe_question(
                profile, 
                profiles, 
                order,
                mode,
                heldout_sets,
                relation,
                subject
            ):
    answer = profile[relation]['name']
    return {
        "question": SECOND_ORDER_TEMPLATE.format(name=profile['name'], relation=relation.replace("_", " "), subject=subject.replace("_", " ")),
        "answer": answer,
        "order": order
    }
    

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

def get_qa_token_ranges(input_ids, pre_name_tokens, relation_tokens, pre_attribute_token):

    name_ranges = torch.where(input_ids == pre_name_tokens[0])[0] + 1
    name_ranges = [list(range(i, i+2)) for i in name_ranges]
    relation_ranges = torch.where(input_ids == pre_attribute_token[0])[0][::2] + 1
    attribute_ranges = torch.where(input_ids == pre_attribute_token[0])[0][1::2] + 1

    return name_ranges, relation_ranges, attribute_ranges

def collect_qa_activations(
        model, 
        dataloader, 
        layer_idx, 
        tokenizer, 
        num_samples=10000,
        pre_name_tokens=None,
        relation_tokens=None,
        pre_attribute_token=None
    ):
    """Collect activations and labels for QA pairs, targeting specific token types."""
    device = next(model.parameters()).device
    # Separate activations for each token type
    name_activations = []
    relation_activations = []
    attribute_activations = []
    all_labels = []
    all_model_logits = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting activations"):
            # Get activations and logits
            hidden_states, logits = extract_activations(model, batch, layer_idx)
            
            # Find answer positions (where labels != -100)
            label_positions = batch['labels'] != -100
            answer_starts = label_positions & ~torch.roll(label_positions, 1, dims=1)
            answer_starts[:, 0] = False
            
            # Process each sequence
            for seq_idx, (seq_labels, seq_activations, seq_logits, seq_input_ids) in enumerate(
                zip(batch['labels'], hidden_states, logits, batch['input_ids'])
            ):
                answer_positions = torch.where(answer_starts[seq_idx])[0]
                if len(answer_positions) == 0:
                    continue

                for i, ans_pos in enumerate(answer_positions):
                    # Get token ranges for this sequence
                    ranges = get_qa_token_ranges(seq_input_ids, pre_name_tokens, relation_tokens, pre_attribute_token)
                    
                    if ranges is None:
                        continue
                        
                    name_ranges, relation_ranges, attribute_ranges = ranges

                    # Collect activations for each token type
                    name_acts = seq_activations[name_ranges[i]]
                    relation_acts = seq_activations[relation_ranges[i]]
                    attribute_acts = seq_activations[attribute_ranges[i]]
                    
                    name_activations.append(name_acts.reshape(1, -1).cpu())
                    relation_activations.append(relation_acts.reshape(1, -1).cpu())
                    attribute_activations.append(attribute_acts.reshape(1, -1).cpu())
                    all_labels.append(seq_labels[ans_pos].cpu())
                    all_model_logits.append(seq_logits[ans_pos - 1].cpu())
                    
                if len(name_activations) >= num_samples:
                    break
            
            if len(name_activations) >= num_samples:
                break

    # Convert to tensors and one-hot encode labels
    num_classes = len(tokenizer)
    name_activations = torch.concat(name_activations, dim=0)
    relation_activations = torch.concat(relation_activations, dim=0)
    attribute_activations = torch.concat(attribute_activations, dim=0)
    
    labels = torch.stack(all_labels)
    one_hot = torch.zeros((len(labels), num_classes))
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    all_labels = one_hot

    all_model_logits = torch.stack(all_model_logits)

    return [name_activations, relation_activations, attribute_activations], all_labels, all_model_logits

def train_and_evaluate_probes(
        train_activations, 
        train_labels, 
        eval_activations, 
        eval_labels, 
        eval_model_logits, 
        device, 
        max_iter=1000
    ):
    """Train separate probes for each activation type (name, relation, attribute) and evaluate."""
    activation_types = ['name', 'relation', 'attribute', 'all']
    probes = {typ: None for typ in activation_types}
    train_losses = {typ: [] for typ in activation_types}
    eval_losses = {typ: [] for typ in activation_types}
    
    # Train probes for each activation type
    for idx, typ in enumerate(activation_types):
        if typ == 'all':
            train_features = torch.concat(train_activations, dim=1).to(device)
        else:
            train_features = train_activations[idx].to(device)
        input_dim = train_features.shape[-1]
        probe = Classifier(input_dim=input_dim, num_classes=train_labels.shape[-1], device=device)

        loss = probe.fit_cv(
            train_features,
            train_labels.to(device),
            max_iter=max_iter,
            k=3,
            num_penalties=4,
        )
        probes[typ] = probe
        train_losses[typ] = loss
    
    # Calculate uniform distribution over first answer tokens
    all_first_answers = torch.sum(eval_labels, dim=0) + torch.sum(train_labels, dim=0)
    uniform_dist = all_first_answers / torch.sum(all_first_answers)
    
    # Evaluate probes
    for idx, typ in enumerate(activation_types):
        probe = probes[typ]
        with torch.no_grad():
            if typ == 'all':
                logits = probe(torch.concat(eval_activations, dim=1).to(device))
            else:
                logits = probe(eval_activations[idx].to(device))
            loss = torch.nn.functional.cross_entropy(
                logits,
                eval_labels.to(device),
                reduction='none'
            )
            eval_losses[typ] = loss.tolist()
    
    # Calculate model and uniform losses in batch
    model_losses = torch.nn.functional.cross_entropy(
        eval_model_logits.to(device),
        eval_labels.to(device),
        reduction='none'
    ).tolist()
    
    uniform_losses = torch.nn.functional.cross_entropy(
        uniform_dist.unsqueeze(0).repeat(len(eval_labels), 1).to(device),
        eval_labels.to(device),
        reduction='none'
    ).tolist()

    # Calculate average losses
    avg_losses = {
        typ: sum(eval_losses[typ]) / len(eval_losses[typ]) 
        for typ in activation_types
    }
    avg_model_loss = sum(model_losses) / len(model_losses)
    avg_uniform_loss = sum(uniform_losses) / len(uniform_losses)
    
    return (
        probes, 
        train_losses, 
        eval_losses, 
        avg_losses,
        avg_model_loss, 
        avg_uniform_loss
    )

def prepare_token_ids(tokenizer, profiles, relations, attributes):
    """
    Prepare token IDs for names, relations, and attributes.
    Returns tensors of token IDs trimmed to minimum length for each type.
    """
    # Tokenize relations (with 's)
    relation_tokens = [tokenizer(f" {r}'s", add_special_tokens=False)['input_ids'] 
                      for r in relations]
    min_relation_len = min(len(t) for t in relation_tokens)
    relation_ids = torch.tensor([t[:min_relation_len] for t in relation_tokens])
    
    # Tokenize attributes
    attribute_tokens = [tokenizer(f" {a}", add_special_tokens=False)['input_ids'] 
                       for a in attributes]
    min_attribute_len = min(len(t) for t in attribute_tokens)
    attribute_ids = torch.tensor([t[:min_attribute_len] for t in attribute_tokens])
    
    # Tokenize names from profiles
    name_tokens = [tokenizer(f" {p['name']}", add_special_tokens=False)['input_ids'] 
                  for p in profiles]
    min_name_len = min(len(t) for t in name_tokens)
    name_ids = torch.tensor([t[:min_name_len] for t in name_tokens])
    
    return name_ids, relation_ids, attribute_ids

def main(args):
    
    checkpoints = get_checkpoints(
        1, 
        args.max_train_hops, 
        args.N_profiles, 
        args.n_params, 
        0.1,
        args.commit,
        args.relations, 
        layers=args.layers
        )
    checkpoints = [f for f in checkpoints if 'checkpoint-' in f]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    latest_checkpoint = checkpoints[-1]

    # Get parent directory and checkpoint name
    checkpoint_parent = os.path.dirname(latest_checkpoint)

    checkpoint_name = os.path.basename(latest_checkpoint)
    
    model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/llama_multihop_tokenizer')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # Import relations and attributes
    from transformer_reasoning.generate_dataset.generate_qa_dataset import RELATIONS, ATTRIBUTES
    
    rel_str = f'_r{args.relations}' if args.relations is not None else ''
    # Load dataset
    profiles = load_dataset(f"EleutherAI/profiles_dataset_{args.N_profiles}_uniform{rel_str}", keep_in_memory=True)['train']
    
    # Prepare token IDs
    pre_name_tokens = [tokenizer("What was", add_special_tokens=False)['input_ids'][-1]] # len 1
    relation_tokens = [i[0] for i in tokenizer(RELATIONS, add_special_tokens=False)['input_ids']] # len 1x4
    pre_attribute_token = [tokenizer(" parent's", add_special_tokens=False)['input_ids'][-1]] # len 1
    
    # Create dataset for one-hop questions
    dataset = InfiniteQADataset(
        profiles_dataset=profiles,
        tokenizer=tokenizer,
        max_seq_len=512,
        orders=[args.hops],
        question_generator=generate_probe_question
    )

    # First 4 relations are 1 token long, exclude others
    dataset.heldout_sets['relations'] = RELATIONS[4:]

    dataset._generate_explicit_heldout_sets(0., len(profiles), RELATIONS, len(ATTRIBUTES))
    
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
        activations, labels, model_logits = collect_qa_activations(
            model, 
            dataloader, 
            layer_idx,
            tokenizer,
            args.num_samples,
            pre_name_tokens=pre_name_tokens,
            relation_tokens=relation_tokens,
            pre_attribute_token=pre_attribute_token
        )
        
        # Split into train and eval sets
        train_size = int(0.95 * len(activations[0]))
        
        train_activations = [activations[i][:train_size] for i in range(len(activations))]
        eval_activations = [activations[i][train_size:] for i in range(len(activations))]
        train_labels = labels[:train_size]
        eval_labels = labels[train_size:]
        eval_model_logits = model_logits[train_size:]

        # Initialize and train probes
        probes, train_losses, eval_losses, avg_losses, avg_model_loss, avg_uniform_loss = train_and_evaluate_probes(
            train_activations,
            train_labels,
            eval_activations,
            eval_labels,
            eval_model_logits,
            device,
            args.max_iter,
        )
        
        print(f"Average evaluation loss: {avg_losses}")
        print(f"Average model loss: {avg_model_loss}")
        print(f"Average uniform distribution loss: {avg_uniform_loss}")
        
        result = {
            'layer': layer_idx,
            'avg_eval_loss': avg_losses,
            'avg_model_loss': avg_model_loss,
            'avg_uniform_loss': avg_uniform_loss
        }
        results.append(result)
        
        # Modify save paths to use parent directory and checkpoint name
        results_dir = os.path.join(checkpoint_parent, f'probe_results_{checkpoint_name}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save probes for this layer with checkpoint name
        torch.save({
            'probes': {name: probe.state_dict() for name, probe in probes.items()},
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
    parser.add_argument("--commit", type=str, default=None,
                      help="Commit hash to use")
    
    args = parser.parse_args()
    main(args)
