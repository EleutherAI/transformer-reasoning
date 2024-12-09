import jax
import jax.numpy as jnp
import torch
import numpy as np
from transformers import LlamaConfig, AutoTokenizer
from transformers import FlaxLlamaForCausalLM, LlamaForCausalLM
import optax
from flax.training import train_state

import glob
import os   

from transformer_reasoning.train.train_utils import (
    create_model_and_tokenizer as create_torch_model,
    create_or_load_optimizer,
    evaluate_single_model as torch_evaluate_single_model
)
from transformer_reasoning.train.train_utils_jax import (
    create_train_state,
    train_step as jax_train_step,
    evaluate_dataset as jax_evaluate_dataset,
    eval_step as jax_eval_step
)
from transformer_reasoning.train.dataset import load_and_prepare_datasets


@jax.pmap
def debug_eval_step(state, batch):
    """Single evaluation step with schedule-free parameter adjustment."""
    eval_params = state.params
    
    attention_mask = jnp.ones_like(batch['input_ids'])
    position_ids = batch['position_ids']
    
    logits = state.apply_fn(
        {'params': eval_params}, 
        batch['input_ids'], 
        attention_mask, 
        position_ids,
        return_dict=False
    )[0]
    
    # Move logits and labels computation before masking
    shifted_logits = logits[..., :-1, :]
    shifted_labels = batch['labels'][..., 1:]
    
    # Instead of boolean indexing, use where/select
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=shifted_logits,
        labels=shifted_labels,
    )
    
    # Apply masking using where
    mask = shifted_labels != -100
    loss = jnp.where(mask, loss, 0.0)
    
    jax.debug.print("{loss}", loss=loss.sum()/mask.sum())
    jax.debug.print("{loss_sum}", loss_sum=loss.sum())
    print(f"loss shape: {loss.shape}")

    return loss.sum()

class Args:
    """Mock args class for testing"""
    def __init__(self):
        self.lr = 1e-3
        self.beta1 = 0.99
        self.wd = 0.1
        self.optimizer_type = "schedulefree"
        self.train_batch_size = 2
        self.seed = 42
        self.num_params = 500_000
        self.num_layers = 2

def create_test_batch(tokenizer, batch_size=2):
    """Create identical test batch for both frameworks"""
    text = ["This is a test input"] * batch_size
    torch_inputs = tokenizer(text, return_tensors="pt", padding=True)
    torch_inputs['labels'] = torch_inputs['input_ids'].clone()
    
    # Create JAX batch
    jax_inputs = {
        k: jnp.array(v.numpy()) 
        for k, v in torch_inputs.items()
    }
    
    # Add position_ids for JAX
    jax_inputs['position_ids'] = jnp.broadcast_to(
        jnp.arange(jax_inputs['input_ids'].shape[1])[None, :],
        jax_inputs['input_ids'].shape
    )
    
    return torch_inputs, jax_inputs

def torch_train_step(model, batch, optimizer):
    """Single PyTorch training step"""
    model.train()
    optimizer.train()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def test_model_equivalence():
    args = Args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    
    # Create models
    torch_model, tokenizer, _ = create_torch_model(args.num_params, args.num_layers)
    
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,  # Small for testing
        intermediate_size=128,
        num_hidden_layers=args.num_layers,
        num_attention_heads=4,
        max_position_embeddings=128,
        initializer_range=0.02,
        seed=args.seed,
    )
    
    # Save PyTorch model to temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        torch_model.save_pretrained(tmpdirname)
        # Load as Flax model
        jax_model_wrapped = FlaxLlamaForCausalLM.from_pretrained(tmpdirname, from_pt=True)
        jax_model = jax_model_wrapped.module
        jax_params = jax_model_wrapped.params
    
    # Create optimizers with copied parameters
    torch_optimizer, _, _ = create_or_load_optimizer(torch_model, args)
    jax_state = create_train_state(jax_model, args, rng, initial_params=jax_params)
    
    # Create test batch
    torch_batch, jax_batch = create_test_batch(tokenizer)
    
    del torch_batch['token_type_ids']
    # Forward pass comparison
    with torch.no_grad():
        torch_output = torch_model(**torch_batch).logits
    jax_output = jax_model.apply({'params': jax_state.params}, jax_batch['input_ids'], 
                                jnp.ones_like(jax_batch['input_ids']), jax_batch['position_ids'], 
                                return_dict=False)[0]
    
    forward_diff = np.max(np.abs(
        torch_output.numpy() - 
        jax.device_get(jax_output)
    ))
    normalized_diff = forward_diff / (torch_output.numpy().std() + 1e-6)
    print(f"Maximum difference in forward pass: {normalized_diff}")
    
    # Training step comparison
    torch_loss = torch_train_step(torch_model, torch_batch, torch_optimizer)
    
    # Reshape batch for JAX pmap
    n_devices = jax.device_count()
    jax_batch = {
        k: jnp.reshape(v, (n_devices, -1, *v.shape[1:]))
        for k, v in jax_batch.items()
    }
    jax_state = jax.device_put_replicated(jax_state, jax.devices())
    jax_state, jax_loss = jax_train_step(jax_state, jax_batch)
    jax_loss = jax.device_get(jax_loss).mean()
    
    print(f"PyTorch loss: {torch_loss}")
    print(f"JAX loss: {jax_loss}")
    print(f"Loss difference: {abs(torch_loss - jax_loss)}")
    
    # Parameter comparison after training
    torch_params = torch.cat([p.flatten() for p in torch_model.parameters()])
    jax_params = jnp.concatenate([
        p.flatten() 
        for p in jax.tree_util.tree_leaves(jax_state.params)
    ])
    
    param_diff = np.max(np.abs(
        torch_params.detach().numpy() - 
        jax.device_get(jax_params)
    ))
    print(f"Maximum difference in parameters after training: {param_diff}")
    
    return {
        'forward_diff': forward_diff,
        'loss_diff': abs(torch_loss - jax_loss),
        'param_diff': param_diff,
    }



def test_equivalence_trained_models():
    output_dir = "./results/n10000_p1172032_omin1_omax2_wd0.1_l12_lr0.001_beta10.99_sf_r17"

    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
    print(f"Loading model from checkpoint: {latest_checkpoint}")
    torch_model = LlamaForCausalLM.from_pretrained(latest_checkpoint)
    jax_model_wrapped = FlaxLlamaForCausalLM.from_pretrained(latest_checkpoint, from_pt=True)
    jax_model = jax_model_wrapped.module
    jax_params = jax_model_wrapped.params
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llama_multihop_tokenizer")

    jax_dataset, jax_1hop_dataset, jax_2hop_dataset = load_and_prepare_datasets(tokenizer, 10000, orders=[1,2], relations=17)

    jax_1hop_dataloader = torch.utils.data.DataLoader(jax_1hop_dataset, batch_size=32, drop_last=True, num_workers=0)#15, multiprocessing_context='spawn')
    jax_2hop_dataloader = torch.utils.data.DataLoader(jax_2hop_dataset, batch_size=32, drop_last=True, num_workers=0)#15, multiprocessing_context='spawn')

    args = Args()

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    jax_train_state = create_train_state(jax_model, args, rng, initial_params=jax_params)
    jax_train_state = jax.device_put_replicated(jax_train_state, jax.devices())

    iterator = iter(jax_1hop_dataloader)

    torch_batch = next(iterator)
    torch_batch_2 = next(iterator)

    # Get number of devices and batch size per device
    n_devices = jax.device_count()
    per_device_batch_size = 32 // n_devices

    jax_batch = {
        k: jnp.array(v.numpy()).reshape(n_devices, per_device_batch_size, -1)
        for k, v in torch_batch.items()
        if isinstance(v, torch.Tensor)
    }

    jax_batch_2 = {
        k: jnp.array(v.numpy()).reshape(n_devices, per_device_batch_size, -1)
        for k, v in torch_batch_2.items()
        if isinstance(v, torch.Tensor)
    }
    
    # Add position_ids if not present
    if 'position_ids' not in jax_batch:
        seq_length = jax_batch['input_ids'].shape[-1]
        position_ids = jnp.broadcast_to(
            jnp.arange(seq_length),
            (n_devices, per_device_batch_size, seq_length)
        )
        jax_batch['position_ids'] = position_ids

    del torch_batch['text']
    with torch.no_grad():
        torch_output = torch_model(**torch_batch).logits
        
    # Distribute input across devices and run forward pass
    jax_output = jax.pmap(
        lambda params, *args: jax_model.apply(
            {'params': params}, *args, return_dict=False
        )[0]
    )(
        jax_train_state.params,
        jax_batch['input_ids'],
        jnp.ones_like(jax_batch['input_ids']),
        jax_batch['position_ids']
    )

    jax_output_2 = jax.pmap(
        lambda params, *args: jax_model.apply(
            {'params': params}, *args, return_dict=False
        )[0]
    )(
        jax_train_state.params,
        jax_batch_2['input_ids'],
        jnp.ones_like(jax_batch_2['input_ids']),
        jax_batch_2['position_ids']
    )
    
    forward_diff = np.max(np.abs(
        torch_output.numpy() - 
        jax.device_get(jax_output)
    ))
    forward_diff_2 = np.max(np.abs(
        torch_output.numpy() - 
        jax.device_get(jax_output_2)
    ))
    normalized_diff = forward_diff / (torch_output.numpy().std() + 1e-6)
    normalized_diff_2 = forward_diff_2 / (torch_output.numpy().std() + 1e-6)
    print(f"Maximum difference in forward pass: {normalized_diff}")
    print(f"Maximum difference in forward pass 2: {normalized_diff_2}")

    jax_1hop_loss = jax_evaluate_dataset(jax_train_state, jax_1hop_dataloader, debug_eval_step, jax.device_count(), max(1, 32//jax.device_count()))
    jax_2hop_loss = jax_evaluate_dataset(jax_train_state, jax_2hop_dataloader, debug_eval_step, jax.device_count(), max(1, 32//jax.device_count()))
    print(f"One hop loss: {jax_1hop_loss}")
    print(f"Two hop loss: {jax_2hop_loss}")

    torch_1hop_loss = torch_evaluate_single_model(torch_model, jax_1hop_dataloader, 0, 1)
    torch_2hop_loss = torch_evaluate_single_model(torch_model, jax_2hop_dataloader, 0, 2)
    print(f"One hop loss: {torch_1hop_loss}")
    print(f"Two hop loss: {torch_2hop_loss}")

if __name__ == "__main__":
    results = test_model_equivalence()
    print("\nTest Results:")
    print(f"Forward pass difference: {results['forward_diff']:.2e}")
    print(f"Loss difference: {results['loss_diff']:.2e}")
    print(f"Parameter difference: {results['param_diff']:.2e}")
    
    test_equivalence_trained_models()
    # Define success thresholds
    THRESHOLDS = {
        'forward_diff': 1e-5,
        'loss_diff': 1e-5,
        'param_diff': 1e-5,
    }
    
    all_passed = all(
        results[k] < v 
        for k, v in THRESHOLDS.items()
    )
    
    print(f"\nOverall test {'PASSED' if all_passed else 'FAILED'}")