#!/bin/bash

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=7,6,5,4 torchrun --nproc_per_node=4 --master_port=29505 src/transformer_reasoning/train/train_llama.py --num_params 6000000 --orders 1 2 --num_layers 12 --relations 17 --N 15000 &

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3,2,1,0 torchrun --nproc_per_node=4 --master_port=29501 src/transformer_reasoning/train/train_llama.py --num_params 7000000 --orders 1 2 --num_layers 4 --relations 17 --N 25000 &


TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=7,6 torchrun --nproc_per_node=2 --master_port=29500 src/transformer_reasoning/train/train_llama.py --num_params 4000000 --orders 1 2 --num_layers 4 --relations 17 --N 20000 --resume_from_checkpoint --commit_hash_override 344bde1

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=29504 src/transformer_reasoning/train/train_llama.py --num_params 1000000 --orders 1 2 --num_layers 12 --relations 17 --N 1000 --resume_from_checkpoint --commit_hash_override ecb2fdc

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3,2 torchrun --nproc_per_node=2 --master_port=29502 src/transformer_reasoning/train/train_llama.py --num_params 3000000 --orders 1 2 --num_layers 4 --relations 17 --N 10000 --resume_from_checkpoint --commit_hash_override ecb2fdc

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1,0 torchrun --nproc_per_node=2 --master_port=29503 src/transformer_reasoning/train/train_llama.py --num_params 4000000 --orders 1 2 --num_layers 4 --relations 17 --N 10000 --resume_from_checkpoint --commit_hash_override ecb2fdc


python -m pdb src/transformer_reasoning/evaluation/measure_capacity.py --scheme 2-hop-big-hash --selection_scheme enumerate --relations 17 --commit_hashes ecb2fdc no_git pre_commit_stamp