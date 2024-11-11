# First run

CUDA_VISIBLE_DEVICES=0 python src/transformer_reasoning/train/train_llama.py --num_params 1300000 --N 10000 --orders 1 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8
CUDA_VISIBLE_DEVICES=1 python src/transformer_reasoning/train/train_llama.py --num_params 1300000 --N 10000 --orders 1 2 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8
CUDA_VISIBLE_DEVICES=2 python src/transformer_reasoning/train/train_llama.py --num_params 850000 --N 25000 --orders 1 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8
CUDA_VISIBLE_DEVICES=3 python src/transformer_reasoning/train/train_llama.py --num_params 850000 --N 25000 --orders 1 2 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8
CUDA_VISIBLE_DEVICES=4 python src/transformer_reasoning/train/train_llama.py --num_params 1500000 --N 25000 --orders 1 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8
CUDA_VISIBLE_DEVICES=5 python src/transformer_reasoning/train/train_llama.py --num_params 1500000 --N 25000 --orders 1 2 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8
CUDA_VISIBLE_DEVICES=6 python src/transformer_reasoning/train/train_llama.py --num_params 3000000 --N 25000 --orders 1 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8
CUDA_VISIBLE_DEVICES=7 python src/transformer_reasoning/train/train_llama.py --num_params 3000000 --N 25000 --orders 1 2 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8

# Resuming

CUDA_VISIBLE_DEVICES=0 python src/transformer_reasoning/train/train_llama.py --num_params 1300000 --N 10000 --orders 1 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8 --resume_from ./logs/n10000_p1300000_omin1_omax1_wd0.01
CUDA_VISIBLE_DEVICES=1 python src/transformer_reasoning/train/train_llama.py --num_params 1300000 --N 10000 --orders 1 2 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8 --resume_from ./logs/n10000_p1300000_omin1_omax2_wd0.01
CUDA_VISIBLE_DEVICES=2 python src/transformer_reasoning/train/train_llama.py --num_params 850000 --N 25000 --orders 1 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8 --resume_from ./logs/n25000_p850000_omin1_omax1_wd0.01
CUDA_VISIBLE_DEVICES=3 python src/transformer_reasoning/train/train_llama.py --num_params 850000 --N 25000 --orders 1 2 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8 --resume_from ./logs/n25000_p850000_omin1_omax2_wd0.01
CUDA_VISIBLE_DEVICES=4 python src/transformer_reasoning/train/train_llama.py --num_params 1500000 --N 25000 --orders 1 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8 --resume_from ./logs/n25000_p1500000_omin1_omax1_wd0.01
CUDA_VISIBLE_DEVICES=5 python src/transformer_reasoning/train/train_llama.py --num_params 1500000 --N 25000 --orders 1 2 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8 --resume_from ./logs/n25000_p1500000_omin1_omax2_wd0.01
CUDA_VISIBLE_DEVICES=6 python src/transformer_reasoning/train/train_llama.py --num_params 3000000 --N 25000 --orders 1 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8 --resume_from ./logs/n25000_p3000000_omin1_omax1_wd0.01
CUDA_VISIBLE_DEVICES=7 python src/transformer_reasoning/train/train_llama.py --num_params 3000000 --N 25000 --orders 1 2 --wd 0.01 --train_batch_size 32 --qa_ratio 0.8 --resume_from ./logs/n25000_p3000000_omin1_omax2_wd0.01


TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 python src/transformer_reasoning/train/train_llama.py --num_params 700000 --N 50000 --orders 1 2 --wd 0.1 --train_batch_size 32 --qa_ratio 1