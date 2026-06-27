#!/bin/bash
#SBATCH --job-name=skcdf_syn20p
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/work/aac/zimuzhang2302/SKCDF/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/work/aac/zimuzhang2302/SKCDF/slurm_logs/%x_%j.err

set -euo pipefail

echo "===== SKCDF Synapse 20% training: $(date) on $(hostname) ====="

export SKCDF_ROOT=/gpfs/work/aac/zimuzhang2302/SKCDF
cd "${SKCDF_ROOT}"
mkdir -p slurm_logs logs

source slurm/xec/hpc_env.sh
bash slurm/xec/setup_data.sh

cd "${SKCDF_ROOT}/SKCDF/code"
python train_skcdf.py \
  --task synapse \
  --exp synapse_20p/skcdf/fold1 \
  --seed 0 \
  -sl labeled_20p \
  -su unlabeled_20p \
  -se eval \
  --max_epoch 1500 \
  --batch_size 2 \
  --num_workers 0 \
  --base_lr 0.3 \
  --gpu 0

echo "===== SKCDF Synapse 20% DONE: $(date) ====="
