#!/bin/bash
#SBATCH --job-name=mkunet_train
#SBATCH --output=logs/output_mod_%j.log
#SBATCH --error=logs/error_mod_%j.err
#SBATCH --partition=btech
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00        

# 1. Reroute all temporary and cache files to your scratch space
export TMPDIR=/scratch/b23cs1001/tmp
export HF_HOME=/scratch/b23cs1001/cache/huggingface
export TORCH_HOME=/scratch/b23cs1001/cache/torch

# 2. Run your python script directly inside your custom environment
# REPLACE "/path/to/your/actual_script.py" with your real Python file!
conda run --prefix /scratch/b23cs1001/mkunetenv python /scratch/b23cs1001/MK-UNet-final/train_polyp.py