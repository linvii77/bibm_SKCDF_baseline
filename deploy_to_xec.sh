#!/usr/bin/env bash
# Deploy SKCDF to XEC HPC (zimuzhang2302) and submit smoke -> formal jobs.
set -euo pipefail

HPC_USER="zimuzhang2302"
HPC_HOST="xeclogin.hpc.xjtlu.edu.cn"
HPC_DIR="/gpfs/work/aac/zimuzhang2302/SKCDF"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${HPC_PASS:-}" ]; then
    echo "Set HPC_PASS environment variable"
    exit 1
fi

SSH="sshpass -p ${HPC_PASS} ssh -o StrictHostKeyChecking=no ${HPC_USER}@${HPC_HOST}"
RSYNC_SSH="sshpass -p ${HPC_PASS} ssh -o StrictHostKeyChecking=no"

echo "Syncing SKCDF to ${HPC_USER}@${HPC_HOST}:${HPC_DIR} ..."
rsync -avz -e "${RSYNC_SSH}" --delete \
  --exclude '.git' \
  --exclude 'logs/' \
  --exclude 'slurm_logs/' \
  --exclude 'ECCV/' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "${LOCAL_DIR}/" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/"

echo "Submitting smoke + formal jobs on XEC ..."
${SSH} bash -s <<'REMOTE'
set -euo pipefail
ROOT=/gpfs/work/aac/zimuzhang2302/SKCDF
cd "$ROOT"
chmod +x slurm/xec/*.sh
mkdir -p slurm_logs logs
SMOKE_ID=$(sbatch --parsable slurm/xec/skcdf_smoke.slurm)
SYN_ID=$(sbatch --parsable --dependency=afterok:${SMOKE_ID} slurm/xec/skcdf_synapse_20p.slurm)
AMOS_ID=$(sbatch --parsable --dependency=afterok:${SMOKE_ID} slurm/xec/skcdf_amos_5p.slurm)
echo "Smoke job:   ${SMOKE_ID}"
echo "Synapse job: ${SYN_ID} (after smoke)"
echo "AMOS job:    ${AMOS_ID} (after smoke)"
squeue -u zimuzhang2302
REMOTE

echo "Done."
