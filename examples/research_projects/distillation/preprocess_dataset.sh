#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=preproc_c4
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --output=logs/myjob.%j.out
#SBATCH --error=logs/myjob.%j.err

module load anaconda3/2022.05 cuda/12.1
conda create --name pytorch_env python=3.9 -y
conda activate pytorch_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
python -c'import torch; print(torch.cuda.is_available())'

# # Run your Python script
# python preprocess_dataset.py
# Run your Python script in the background
python preprocess_dataset.py

# Deactivate the virtual environment
deactivate

# TODO : output file?
# sbatch dataset --- add nvidia-smi query
# carbon tracking
# GPU: nvidia-smi --query-gpu=power.draw --format=csv --filename=/home/taira.e/power_stats/gpu_test.csv

