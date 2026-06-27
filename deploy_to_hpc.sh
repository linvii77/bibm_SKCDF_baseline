#!/usr/bin/env bash
# Deploy SKCDF to XJTLU HPC and submit smoke -> formal Synapse 20% job.
set -euo pipefail

HPC_USER="yuangluo22"
HPC_HOST="login.hpc.xjtlu.edu.cn"
HPC_DIR="/gpfs/work/aac/yuangluo22/SKCDF"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
ECCV_LOCAL="${ECCV_LOCAL:-$HOME/Desktop/ECCV}"

if [ -z "${HPC_PASS:-}" ]; then
    echo "Set HPC_PASS environment variable"
    exit 1
fi

SSH="sshpass -p ${HPC_PASS} ssh -o StrictHostKeyChecking=no ${HPC_USER}@${HPC_HOST}"
RSYNC_SSH="sshpass -p ${HPC_PASS} ssh -o StrictHostKeyChecking=no"

echo "Syncing SKCDF code to ${HPC_USER}@${HPC_HOST}:${HPC_DIR} ..."
rsync -avz -e "${RSYNC_SSH}" --delete \
  --exclude '.git' \
  --exclude 'logs/' \
  --exclude 'slurm_logs/' \
  --exclude 'ECCV/' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "${LOCAL_DIR}/" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/"

echo "Syncing split files from ${ECCV_LOCAL} ..."
${SSH} "mkdir -p ${HPC_DIR}/ECCV/synapse_splits ${HPC_DIR}/ECCV/amos_splits"
rsync -avz -e "${RSYNC_SSH}" \
  "${ECCV_LOCAL}/synapse_splits/" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/ECCV/synapse_splits/"
rsync -avz -e "${RSYNC_SSH}" \
  "${ECCV_LOCAL}/amos_splits/" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/ECCV/amos_splits/"

echo "Submitting smoke + formal jobs ..."
${SSH} bash -s <<'REMOTE'
set -euo pipefail
ROOT=/gpfs/work/aac/yuangluo22/SKCDF
cd "$ROOT"
chmod +x slurm/*.sh deploy_to_hpc.sh
mkdir -p slurm_logs logs
SMOKE_ID=$(sbatch --parsable slurm/skcdf_smoke.slurm)
TRAIN_ID=$(sbatch --parsable --dependency=afterok:${SMOKE_ID} slurm/skcdf_synapse_20p.slurm)
echo "Smoke job:  ${SMOKE_ID}"
echo "Train job:  ${TRAIN_ID} (after smoke passes)"
squeue -u yuangluo22
REMOTE

echo "Done. Monitor: ssh ${HPC_USER}@${HPC_HOST} 'squeue -u yuangluo22'"
