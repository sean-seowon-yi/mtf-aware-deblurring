#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=large-score
#SBATCH --output=logs/%x-%j.out

if [ ! -d "/tmp/$USER/envs/CSC2529" ]; then
  mkdir -p /tmp/$USER/envs
  tar -xzf ~/CSC2529-env.tar.gz -C /tmp/$USER/envs
fi
source /tmp/$USER/envs/CSC2529/bin/activate

cd ~/mtf-aware-deblurring

python train_tiny_score_model.py \
  --dataset-root ~/datasets/DIV2K/DIV2K_train_HR \
  --patch-size 96 \
  --patches-per-image 60 \
  --max-images 800 \
  --epochs 80 \
  --batch-size 8 \
  --lr 1e-4 \
  --base-channels 128 \
  --depth 8 \
  --device cuda \
  --output src/mtf_aware_deblurring/assets/large_score_unet.pth
