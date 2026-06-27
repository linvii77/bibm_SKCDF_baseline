#!/bin/bash
#SBATCH --job-name=skcdf_off_test
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/zimuzhang2302/SKCDF/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/work/aac/zimuzhang2302/SKCDF/slurm_logs/%x_%j.err

set -euo pipefail

export SKCDF_ROOT=/gpfs/work/aac/zimuzhang2302/SKCDF
cd "${SKCDF_ROOT}"
mkdir -p slurm_logs

source slurm/xec/hpc_env.sh
bash slurm/xec/setup_data.sh

python scripts/eval_official_paper_test.py \
  --exp synapse_20p/skcdf/fold1 \
  --gpu 0

cat "${SKCDF_ROOT}/logs/synapse_20p/skcdf/fold1/evaluation_official_test.txt"
