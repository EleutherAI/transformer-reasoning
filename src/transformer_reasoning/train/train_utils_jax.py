from flax.training import train_state
import jax
import jax.numpy as jnp
import glob
import os
import math
from tqdm import tqdm
import optax
import orbax.checkpoint as ocp
from flax.training import orbax_utils
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.multiprocessing as mp

from transformer_reasoning.train.dataset import load_and_prepare_datasets



def create_train_state(model, args, key, max_seq_len=5, initial_params=None):
    """Creates initial `TrainState` for model."""
    input_ids = jnp.ones((1, max_seq_len), dtype="i4")
    attention_mask = jnp.ones_like(input_ids)
    position_ids = jnp.broadcast_to(jnp.arange(max_seq_len)[None, :], input_ids.shape)
    
    params = initial_params if initial_params is not None else model.init(
        key, 
        input_ids, 
        attention_mask, 
        position_ids,
        return_dict=False
    )['params']
    
    tx = optax.contrib.schedule_free_adamw(
        learning_rate=args.lr,
        b1=args.beta1,
        b2=0.999,
        weight_decay=args.wd,
        warmup_steps=1000
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

@jax.pmap
def train_step(state, batch):
    """Single training step."""
    def loss_fn(params):
        attention_mask = jnp.ones_like(batch['input_ids'])
        position_ids = batch['position_ids']

        logits = state.apply_fn(
            {'params': params}, 
            batch['input_ids'], 
            attention_mask, 
            position_ids,
            return_dict=False
        )[0]
        
        # Move logits and labels computation before masking
        shifted_logits = logits[..., :-1, :]
        shifted_labels = batch['labels'][..., 1:]
        
        # Calculate loss without masking first
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=shifted_logits,
            labels=shifted_labels,
        )
        
        # Apply masking using where
        mask = shifted_labels != -100
        loss = jnp.where(mask, loss, 0.0)
        
        return loss.sum() / mask.sum()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.pmap
def eval_step(state, batch):
    """Single evaluation step with schedule-free parameter adjustment."""
    # Convert parameters to their evaluation form
    eval_params = optax.contrib.schedule_free_eval_params(state.opt_state, state.params)
    
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
    
    # Calculate loss without masking first
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=shifted_logits,
        labels=shifted_labels,
    )
    
    # Apply masking using where
    mask = shifted_labels != -100
    loss = jnp.where(mask, loss, 0.0)
    
    return loss.sum()


def save_checkpoint(state, checkpoint_dir, step):
    """Save model checkpoint using Orbax."""
    checkpoint_path = f"{checkpoint_dir}/checkpoint-{step}"
    if os.path.exists(checkpoint_path):
        return  # Skip if checkpoint already exists
        
    # Get the first replica of the state
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(checkpoint_path, state, save_args=save_args)

def restore_checkpoint(state, checkpoint_path):
    """Restore model checkpoint using Orbax."""
    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = orbax_utils.restore_args_from_target(state)
    # Get the first replica's structure for restoration
    state_struct = jax.tree_map(lambda x: x[0], state)
    restored_state = checkpointer.restore(checkpoint_path, item=state_struct, restore_kwargs={'restore_args': restore_args})
    # Replicate the restored state across devices
    return jax.device_put_replicated(restored_state, jax.devices())

def train_single_model(
        model,
        tokenizer,
        train_dataset, 
        onehop_dataset, 
        twohop_dataset, 
        args,
        output_dir=None,
        curriculum=False
        ):
    # Set start method to 'spawn' at the beginning of your script
    mp.set_start_method('spawn', force=True)
    
    # Initialize random key
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    # Create training state
    state = create_train_state(model, args, init_rng)
    
    # Load checkpoint if exists
    if args.resume_from_checkpoint and output_dir:
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            state = restore_checkpoint(state, latest_checkpoint)
            global_step = int(latest_checkpoint.split("-")[-1])
        else:
            global_step = 0
    else:
        global_step = 0

    early_saves = set([int(math.exp(i)) for i in jnp.linspace(math.log(100), math.log(100000), 10).tolist()])
    
    train_dataset, onehop_dataset, twohop_dataset = load_and_prepare_datasets(
        tokenizer, 
        args.N, 
        orders=[1,2], 
        relations=args.relations, 
        hop_ratio=args.hop_ratio, 
        jax=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        multiprocessing_context='spawn',
        drop_last=True
    )

    onehop_loader = DataLoader(
        onehop_dataset, 
        batch_size=args.eval_batch_size, 
        num_workers=args.num_workers,
        multiprocessing_context='spawn',
        drop_last=True
    )

    twohop_loader = DataLoader(
        twohop_dataset, 
        batch_size=args.eval_batch_size, 
        num_workers=args.num_workers,
        multiprocessing_context='spawn',
        drop_last=True
    )
    
    results_dicts = []
    curriculum_switched = not curriculum
    
    # Get number of devices
    n_devices = jax.device_count()
    
    # Ensure batch size is divisible by number of devices
    assert args.train_batch_size % n_devices == 0, f"Batch size must be divisible by number of devices ({n_devices})"
    per_device_batch_size = args.train_batch_size // n_devices
    
    # Replicate state across devices
    state = jax.device_put_replicated(state, jax.devices())
    
    # Training loop
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Reshape batch to add device dimension [n_devices, per_device_batch_size, ...]
            jax_batch = {
                k: jnp.reshape(
                    jnp.from_dlpack(torch.utils.dlpack.to_dlpack(v)),
                    (n_devices, per_device_batch_size, *v.shape[1:])
                )
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            
            state, loss = train_step(state, jax_batch)
            
            global_step += 1
            
            should_save = (global_step < 100000 and global_step in early_saves) or \
                (global_step >= 100000 and global_step % 100000 == 0)
            
            if should_save:
                # Evaluation
                onehop_loss = evaluate_dataset(state, onehop_loader, eval_step, n_devices, per_device_batch_size)
                twohop_loss = evaluate_dataset(state, twohop_loader, eval_step, n_devices, per_device_batch_size)
                
                results_dicts.extend([
                    {'loss': onehop_loss, 'global_step': global_step},
                    {'loss': twohop_loss, 'global_step': global_step}
                ])

                # Save checkpoint
                if output_dir:
                    save_checkpoint(state, output_dir, global_step)
                
                # Check curriculum condition
                if curriculum and not curriculum_switched and onehop_loss < 1.0:
                    print("\nSwitching to full curriculum - enabling 2-hop questions")
                    train_dataset.order_weights = [0.1, 1.0]
                    curriculum_switched = True

                pbar.set_postfix({
                    'onehop_loss': onehop_loss,
                    'twohop_loss': twohop_loss,
                    'step': global_step
                })
                results_df = pd.DataFrame(results_dicts)
                results_df.to_csv(f"{output_dir}/results.csv", index=False, mode='a')

def evaluate_dataset(state, dataloader, eval_step, n_devices, per_device_batch_size):
    """Evaluate model on dataset."""
    total_loss = 0
    eval_questions = 0
    
    for batch in tqdm(dataloader, 'evaluating'):
        # Filter for samples ending with -100 (complete sequences)
        labels = batch['labels']
        last_neg = labels[:,-1] == -100
        if not last_neg.any():
            continue
            
        # Filter inputs and labels
        filtered_inputs = {
            k: v[last_neg] for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
        
        # Count number of answer blocks
        is_pos = filtered_inputs['labels'] >= 0
        block_starts = ((is_pos) & (~torch.roll(is_pos, 1))).sum()
        eval_questions += block_starts.item()

        # Convert to JAX format
        jax_batch = {
            k: jnp.reshape(
                jnp.from_dlpack(torch.utils.dlpack.to_dlpack(v)),
                (n_devices, per_device_batch_size, *v.shape[1:])
            )
            for k, v in filtered_inputs.items()
        }
        
        loss = eval_step(state, jax_batch)
        total_loss += jax.device_get(loss)
        
        if eval_questions >= 5000:
            break
    
    return total_loss / eval_questions if eval_questions > 0 else float('inf')