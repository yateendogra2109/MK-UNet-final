#!/bin/bash
#SBATCH --job-name=mkunet_train
#SBATCH --output=logs/output_mod_%j.log
#SBATCH --error=logs/error_mod_%j.err
#SBATCH --partition=btech
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00        

export TMPDIR=/scratch/b23cs1001/tmp
export HF_HOME=/scratch/b23cs1001/cache/huggingface
export TORCH_HOME=/scratch/b23cs1001/cache/torch

cd /scratch/b23cs1001/MK-UNet-final

export PYTHONPATH=/scratch/b23cs1001/MK-UNet-final:$PYTHONPATH

conda run --prefix /scratch/b23cs1001/mkunetenv python train_polyp_gik.py