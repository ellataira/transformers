#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --job-name=distilltrain4.8
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --output=logs/distilltrain48.%j.out
#SBATCH --error=logs/distilltrain48.%j.err

module load anaconda3/2022.05 cuda/12.1
#conda create --name pytorch_env python=3.9 -y
source activate greenai
#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
python -c'import torch; print(torch.cuda.is_available())'

#cd /home/taira.e/transformers/examples/research_projects/distillation

pip install -r requirements.txt

pkill -f 'python -u train.py'

# Start training
python train.py \
    --force \
    --n_gpu 1 \
    --student_type distilbert \
    --student_config training_configs/distilbert-base-uncased.json \
    --teacher_type bert \
    --teacher_name bert-base-uncased \
    --alpha_ce 0.33 --alpha_mlm 0.33 --alpha_cos 0.33 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path /scratch/taira.e/models \
    --data_file /scratch/taira.e/binarized_text.bert-base-uncased.pickle \
    --token_counts /scratch/taira.e/token_counts.bert-base-uncased.pickle &

distill_pid=$!



# Start continuous monitoring while the bert-vocab command is running
while ps -p $distill_pid > /dev/null; do
   # Get timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Get GPU power draw and append to CSV file
    power_draw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
    echo "$timestamp,$power_draw" >> /home/taira.e/power_stats/distill_train4_8.csv

    sleep 600  # 10 mins

done

wait $distill_pid

conda deactivate

