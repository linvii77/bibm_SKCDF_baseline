#!/bin/bash
#SBATCH --job-name=skcdf_amos5p
#SBATCH --partition=gpua800
#SBATCH --qos=4a800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/gpfs/work/aac/yuangluo22/SKCDF/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/work/aac/yuangluo22/SKCDF/slurm_logs/%x_%j.err

set -euo pipefail

echo "===== SKCDF AMOS 5% training: $(date) on $(hostname) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

export SKCDF_ROOT=/gpfs/work/aac/yuangluo22/SKCDF
cd "${SKCDF_ROOT}"
mkdir -p slurm_logs logs

source slurm/hpc_env.sh
bash slurm/setup_hpc_data.sh
python scripts/verify_amos_data.py

cd "${SKCDF_ROOT}/SKCDF/code"
python train_skcdf.py \
  --task amos \
  --exp amos_5p/skcdf/fold1 \
  --seed 0 \
  -sl labeled_5p \
  -su unlabeled_5p \
  -se eval \
  --max_epoch 1500 \
  --batch_size 2 \
  --num_workers 0 \
  --base_lr 0.1 \
  --gpu 0

echo "===== SKCDF AMOS 5% training DONE: $(date) ====="
