#!/bin/bash

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=7,6,5,4 torchrun --nproc_per_node=4 --master_port=29505 src/transformer_reasoning/train/train_llama.py --num_params 6000000 --orders 1 2 --num_layers 12 --relations 17 --N 15000 &

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3,2,1,0 torchrun --nproc_per_node=4 --master_port=29501 src/transformer_reasoning/train/train_llama.py --num_params 7000000 --orders 1 2 --num_layers 4 --relations 17 --N 25000 &

wait