#!/bin/bash
set -e  # Stop on any error

DATA_FOLDER=${1:-data/training/Synthetic}

echo "=== Stage 1: VAE Training ==="
accelerate launch --num_processes 2 train_vae.py --cfg ./configs/custom_train_vae.yaml --data_folder "$DATA_FOLDER"

echo "=== Stage 2: LDM Training ==="
accelerate launch --num_processes 2 train_cond_ldm.py --cfg ./configs/custom_train_ldm.yaml --data_folder "$DATA_FOLDER"

echo "=== Training Complete ==="
