#!/bin/bash
#SBATCH --job-name=skcdf_amos_eval
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/zimuzhang2302/SKCDF/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/work/aac/zimuzhang2302/SKCDF/slurm_logs/%x_%j.err

set -euo pipefail

export SKCDF_ROOT=/gpfs/work/aac/zimuzhang2302/SKCDF
cd "${SKCDF_ROOT}"
mkdir -p slurm_logs

source slurm/xec/hpc_env.sh
bash slurm/xec/setup_data.sh

CKPT="${SKCDF_ROOT}/logs/amos_5p/skcdf/fold1/ckpts/best_model.pth"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi

python scripts/eval_amos_4metrics.py \
  --exp amos_5p/skcdf/fold1 \
  --split test \
  --gpu 0

cat "${SKCDF_ROOT}/logs/amos_5p/skcdf/fold1/evaluation_4metrics.txt"
