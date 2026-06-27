#!/bin/bash
#SBATCH --job-name=skcdf_smoke
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/work/aac/zimuzhang2302/SKCDF/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/work/aac/zimuzhang2302/SKCDF/slurm_logs/%x_%j.err

set -euo pipefail

echo "===== SKCDF smoke (Synapse+AMOS): $(date) on $(hostname) ====="

export SKCDF_ROOT=/gpfs/work/aac/zimuzhang2302/SKCDF
cd "${SKCDF_ROOT}"
mkdir -p slurm_logs logs

source slurm/xec/hpc_env.sh
bash slurm/xec/setup_data.sh
python scripts/verify_skcdf_data.py

cd "${SKCDF_ROOT}/SKCDF/code"

# Synapse smoke
TRAIN_RC=0
python train_skcdf.py \
  --task synapse \
  --exp synapse_20p/smoke_test \
  --seed 0 \
  -sl labeled_20p \
  -su unlabeled_20p \
  -se eval \
  --max_epoch 1 \
  --batch_size 2 \
  --num_workers 0 \
  --base_lr 0.3 \
  --gpu 0 || TRAIN_RC=$?

SYN_CKPT="${SKCDF_ROOT}/logs/synapse_20p/smoke_test/ckpts/best_model.pth"
if [ ! -f "${SYN_CKPT}" ]; then
  echo "ERROR: Synapse checkpoint not found at ${SYN_CKPT}"
  exit 1
fi
echo "Synapse smoke OK: ${SYN_CKPT}"

# AMOS smoke
TRAIN_RC=0
python train_skcdf.py \
  --task amos \
  --exp amos_5p/smoke_test \
  --seed 0 \
  -sl labeled_5p \
  -su unlabeled_5p \
  -se eval \
  --max_epoch 1 \
  --batch_size 2 \
  --num_workers 0 \
  --base_lr 0.1 \
  --gpu 0 || TRAIN_RC=$?

AMOS_CKPT="${SKCDF_ROOT}/logs/amos_5p/smoke_test/ckpts/best_model.pth"
if [ ! -f "${AMOS_CKPT}" ]; then
  echo "ERROR: AMOS checkpoint not found at ${AMOS_CKPT}"
  exit 1
fi
echo "AMOS smoke OK: ${AMOS_CKPT}"

echo "===== SKCDF smoke PASSED: $(date) ====="
