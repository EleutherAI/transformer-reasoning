#!/bin/bash

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29500 src/transformer_reasoning/train/train_llama.py --num_params 2000000 --orders 1 --num_layers 12 --relations 17 --N 15500 &

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 src/transformer_reasoning/train/train_llama.py --num_params 7000000 --orders 1 --num_layers 4 --relations 17 --N 24000 &


