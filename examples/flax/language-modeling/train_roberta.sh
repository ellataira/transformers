#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --job-name=roberta_train
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --output=logs/train_roberta.%j.out
#SBATCH --error=logs/train_roberta.%j.err

module load anaconda3/2022.05 cuda/12.1
conda activate greenai
python -c'import torch; print(torch.cuda.is_available())'

cd /home/taira.e/transformers/examples/flax/language-modeling

python run_mlm_flax.py \
    --output_dir="/scratch/taira.e/english-roberta-base/output" \
    --model_type="roberta" \
    --config_name="/scratch/taira.e/english-roberta-base" \
    --tokenizer_name="/scratch/taira.e/english-roberta-base" \
    --dataset_name="test" \
    --dataset_config_name="/scratch/taira.e/english-roberta-base/processed_roberta_ds.txt" \
    --max_seq_length="128" \
    --weight_decay="0.01" \
    --per_device_train_batch_size="128" \
    --per_device_eval_batch_size="128" \
    --learning_rate="3e-4" \
    --warmup_steps="1000" \
    --overwrite_output_dir \
    --num_train_epochs="18" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --logging_steps="500" \
    --save_steps="2500" \
    --eval_steps="2500" \
    --push_to_hub
