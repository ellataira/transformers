#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=dist_prepr
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --output=logs/dist_prepr.%j.out
#SBATCH --error=logs/dist_prepr.%j.err

module load anaconda3/2022.05 cuda/12.1
conda create --name pytorch_env python=3.9 -y
conda activate pytorch_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
python -c'import torch; print(torch.cuda.is_available())'

#cd /home/taira.e/transformers/examples/research_projects/distillation

pip install -r requirements.txt

# Binarize the data
python scripts/binarized_data.py \
    --file_path /scratch/taira.e/c4_10_dataset_distill.txt \
    --tokenizer_type bert \
    --tokenizer_name bert-base-uncased \
    --dump_file /scratch/taira.e/binarized_text &

distill_pid=$!

# Start continuous monitoring while the bert-vocab command is running
while ps -p $distill_pid > /dev/null; do
   # Get timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Get GPU power draw and append to CSV file
    power_draw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
    echo "$timestamp,$power_draw" >> /home/taira.e/power_stats/distill_prep4_6.csv

    sleep 600  # 10 mins

done

wait $distill_pid

# Compute token counts
python scripts/token_counts.py \
    --data_file /scratch/taira.e/binarized_text.bert-base-uncased.pickle \
    --token_counts_dump /scratch/taira.e/token_counts.bert-base-uncased.pickle \
    --vocab_size 30522 &

distill_pid=$!

# Start continuous monitoring while the bert-vocab command is running
while ps -p $distill_pid > /dev/null; do
   # Get timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Get GPU power draw and append to CSV file
    power_draw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
    echo "$timestamp,$power_draw" >> /home/taira.e/power_stats/distill_prep4_6.csv

    sleep 600  # 10 mins

done

wait $distill_pid

deactivate

