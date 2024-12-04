from flax.training import train_state
import jax
import jax.numpy as jnp
import glob
import os
import math
from tqdm import tqdm
import optax
from src.transformer_reasoning.train.dataset import JAXQADataset
import orbax.checkpoint as ocp
from flax.training import orbax_utils

def create_train_state(model, args, key):
    """Creates initial `TrainState` for model."""
    params = model.init(key, jnp.ones((1, args.max_seq_len)))['params']
    
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

@jax.jit
def train_step(state, batch):
    """Single training step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['input_ids'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits[..., :-1, :],
            labels=batch['labels'][..., 1:],
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch):
    """Single evaluation step with schedule-free parameter adjustment."""
    # Convert parameters to their evaluation form
    eval_params = optax.contrib.schedule_free_eval_params(state.opt_state, state.params)
    
    logits = state.apply_fn({'params': eval_params}, batch['input_ids'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits[..., :-1, :],
        labels=batch['labels'][..., 1:],
    ).mean()
    return loss

def save_checkpoint(state, checkpoint_dir, step):
    """Save model checkpoint using Orbax."""
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(f"{checkpoint_dir}/checkpoint-{step}", state, save_args=save_args)

def restore_checkpoint(state, checkpoint_path):
    """Restore model checkpoint using Orbax."""
    checkpointer = ocp.PyTreeCheckpointer()
    # Create restore args from the state structure
    restore_args = orbax_utils.restore_args_from_target(state)
    # Restore with the specified structure
    return checkpointer.restore(checkpoint_path, item=state, restore_kwargs={'restore_args': restore_args})

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
    
    # Create data loaders
    train_dataset = JAXQADataset(
        profiles_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        orders=[1,2],
        qa_indices=train_indices
    )

    train_loader = train_dataset.get_loader(args.train_batch_size, num_workers=args.num_workers)

    onehop_dataset = JAXQADataset(
        profiles_dataset=onehop_dataset,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        orders=[1,2],
        qa_indices=onehop_indices
    )

    onehop_loader = onehop_dataset.get_loader(args.eval_batch_size, num_workers=args.num_workers)

    twohop_dataset = JAXQADataset(
        profiles_dataset=twohop_dataset,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        orders=[1,2],
        qa_indices=twohop_indices
    )

    twohop_loader = twohop_dataset.get_loader(args.eval_batch_size, num_workers=args.num_workers)
    
    results_dicts = []
    curriculum_switched = not curriculum

    # Training loop
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move batch to device and preprocess
            batch = {k: jnp.array(v) for k, v in batch.items() if k != 'text'}
            
            # Update state and get loss
            state, loss = train_step(state, batch)
            global_step += 1
            
            should_save = (global_step < 100000 and global_step in early_saves) or \
                (global_step >= 100000 and global_step % 100000 == 0)
            
            if should_save:
                # Evaluation
                onehop_loss = evaluate_dataset(state, onehop_loader, eval_step)
                twohop_loss = evaluate_dataset(state, twohop_loader, eval_step)
                
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

def evaluate_dataset(state, dataloader, eval_step):
    """Evaluate model on dataset."""
    total_loss = 0
    count = 0
    
    for batch in dataloader:
        batch = {k: jnp.array(v) for k, v in batch.items() if k != 'text'}
        loss = eval_step(state, batch)
        total_loss += loss
        count += 1
        if count >= 1000:
            break
    
    return total_loss / count if count > 0 else float('inf')