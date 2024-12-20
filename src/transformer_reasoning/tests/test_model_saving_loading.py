import torch
from transformers import LlamaForCausalLM
from transformer_reasoning.train.train_utils import create_model_and_tokenizer, create_or_load_optimizer, load_and_prepare_datasets, train_single_model, calculate_model_size
import os
import glob
import itertools
import torch.distributed

def compare_model_states(model1, model2, optimizer1, optimizer2):
    model1.train()
    model2.train()
    if hasattr(optimizer1, 'train'):
        optimizer1.train()
    if hasattr(optimizer2, 'train'):
        optimizer2.train()
    res = True
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.allclose(p1, p2, rtol=1e-11, atol=1e-14):
            print(f"Local rank {torch.distributed.get_rank()}: Eval parameter mismatch in {n1}: {torch.norm(p1 - p2)}/{torch.norm(p1)}, nonzero: {torch.count_nonzero(p1 - p2)}")
            res = False
    model1.eval()
    model2.eval()
    if hasattr(optimizer1, 'eval'):
        optimizer1.eval()
    if hasattr(optimizer2, 'eval'):
        optimizer2.eval()
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.allclose(p1, p2, rtol=1e-11, atol=1e-14):
            print(f"Local rank {torch.distributed.get_rank()}: Parameter mismatch in {n1}: {torch.norm(p1 - p2)}/{torch.norm(p1)}, nonzero: {torch.count_nonzero(p1 - p2)}")
            res = False
    
    if res == True:
        print(f'Local rank {torch.distributed.get_rank()}: Parameter match')
    
    # Compare model metadata and architecture
    if model1.config != model2.config:
        print(f"Model configs differ: {model1.config} vs {model2.config}")
        res = False
    
    # Compare model buffers (like batch norm running stats)
    for (n1, b1), (n2, b2) in zip(model1.named_buffers(), model2.named_buffers()):
        if not torch.allclose(b1, b2, rtol=1e-11, atol=1e-14):
            print(f"Buffer mismatch in {n1}: {torch.norm(b1 - b2)}/{torch.norm(b1)}")
            res = False
    
    return res

def compare_optimizer_states(opt1, opt2):
    # Compare full optimizer state dict structure
    state1 = opt1.state_dict()
    state2 = opt2.state_dict()
    
    # Compare param_groups (learning rates, weight decay, etc)
    if state1['param_groups'] != state2['param_groups']:
        print("Optimizer param_groups differ")
        return False
        
    # Compare optimizer internal state attributes
    if opt1.defaults != opt2.defaults:
        print(f"Optimizer defaults differ: {opt1.defaults} vs {opt2.defaults}")
        return False
    
    if len(opt1.state) != len(opt2.state):
        print(f"Optimizer state lengths differ: {len(opt1.state)} vs {len(opt2.state)}")
        return False
        
    # Compare state dicts recursively
    for key in state1['state'].keys():
        for param_key in state1['state'][key].keys():
            if not torch.allclose(state1['state'][key][param_key], state2['state'][key][param_key], rtol=1e-11, atol=1e-14):
                print(f"Optimizer state mismatch in {key}.{param_key}: {torch.norm(state1['state'][key][param_key] - state2['state'][key][param_key])} vs {torch.norm(state1['state'][key][param_key])}")
                return False
    return True


def main():
    torch.use_deterministic_algorithms(True)
    
    class Args:
        num_params = 1000000
        num_layers = 8
        orders = [1,2]
        N = 1000
        relations = 17
        optimizer_type = "adamw-linear"
        lr = 1e-3
        beta1 = 0.9
        wd = 0.1
        hop_ratio = 0.1
        curriculum = False
        resume_from_checkpoint = False
        checkpoint_number = None
        push_to_hub = False
        num_epochs = 1
        train_batch_size = 32
        eval_batch_size = 32
        num_workers = 15
        max_seq_length = 512
        num_training_steps = 1000000
    
    args = Args()
    rel_str = f'_r{args.relations}' if args.relations else ''
    # Create single model and tokenizer
    model, tokenizer, real_num_params = create_model_and_tokenizer(args.num_params, args.num_layers)
    
    curr_str = "_curr" if args.curriculum else ""
    sf_str = "sf" if args.optimizer_type == "schedulefree" else "adamw"
    hop_str = f"_hr{args.hop_ratio}" if args.hop_ratio != 0.1 else ""
    output_dir = f"./results/n{args.N}_p{real_num_params}_omin{min(args.orders)}_omax{max(args.orders)}_wd{args.wd}_l{args.num_layers}_lr{args.lr}_beta1{args.beta1}_{sf_str}{rel_str}{hop_str}{curr_str}"

    model_size_mb = calculate_model_size(real_num_params)
    print(f"Estimated model size: {model_size_mb} MB")
    print(f"Epochs: {args.num_epochs}")
    optimizer, _ = train_single_model(model, tokenizer, args, output_dir, args.curriculum, n_steps=100, debug=True)

    args.resume_from_checkpoint = True
    args.checkpoint_number = 100

    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not args.checkpoint_number:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
    else:
        latest_checkpoint = [c for c in checkpoints if f"checkpoint-{args.checkpoint_number}" in c][0]
    print(f"Loading model from checkpoint: {latest_checkpoint}")
    model_loaded = LlamaForCausalLM.from_pretrained(latest_checkpoint)
    args.num_epochs = 0
    optimizer_loaded, _ = train_single_model(model_loaded, tokenizer, args, output_dir, args.curriculum, n_steps=0, debug=True)

    print(compare_model_states(model, model_loaded, optimizer, optimizer_loaded))
    print(compare_optimizer_states(optimizer, optimizer_loaded))

    args.num_epochs = 2
    print(f"Training model for {args.num_training_steps} steps, rank {torch.distributed.get_rank()}, original model")
    optimizer, last_batch = train_single_model(model, tokenizer, args, output_dir, args.curriculum, n_steps=1000, debug=True)
    print(f"Training model for {args.num_training_steps} steps, rank {torch.distributed.get_rank()}, loaded model")
    optimizer_loaded, last_batch_loaded = train_single_model(model_loaded, tokenizer, args, output_dir, args.curriculum, n_steps=1000, debug=True)
    print(compare_model_states(model, model_loaded, optimizer, optimizer_loaded))
    print(compare_optimizer_states(optimizer, optimizer_loaded))
    print(f"Last batch text match: {last_batch['text'] == last_batch_loaded['text']}")

if __name__ == "__main__":
    main()