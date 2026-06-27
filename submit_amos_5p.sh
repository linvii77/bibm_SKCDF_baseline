#!/usr/bin/env bash
# Submit AMOS 5% smoke -> formal training on HPC.
set -euo pipefail

HPC_USER="yuangluo22"
HPC_HOST="login.hpc.xjtlu.edu.cn"
HPC_DIR="/gpfs/work/aac/yuangluo22/SKCDF"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${HPC_PASS:-}" ]; then
    echo "Set HPC_PASS environment variable"
    exit 1
fi

SSH="sshpass -p ${HPC_PASS} ssh -o StrictHostKeyChecking=no ${HPC_USER}@${HPC_HOST}"
RSYNC_SSH="sshpass -p ${HPC_PASS} ssh -o StrictHostKeyChecking=no"

echo "Syncing AMOS scripts to HPC ..."
rsync -avz -e "${RSYNC_SSH}" \
  "${LOCAL_DIR}/slurm/skcdf_smoke_amos.slurm" \
  "${LOCAL_DIR}/slurm/skcdf_amos_5p.slurm" \
  "${LOCAL_DIR}/scripts/verify_amos_data.py" \
  "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/"

${SSH} bash -s <<'REMOTE'
set -euo pipefail
ROOT=/gpfs/work/aac/yuangluo22/SKCDF
cd "$ROOT"
mv -f skcdf_smoke_amos.slurm skcdf_amos_5p.slurm slurm/ 2>/dev/null || true
mv -f verify_amos_data.py scripts/ 2>/dev/null || true
chmod +x slurm/*.sh scripts/verify_amos_data.py
mkdir -p slurm_logs logs
SMOKE_ID=$(sbatch --parsable slurm/skcdf_smoke_amos.slurm)
TRAIN_ID=$(sbatch --parsable --dependency=afterok:${SMOKE_ID} slurm/skcdf_amos_5p.slurm)
echo "AMOS smoke job:  ${SMOKE_ID}"
echo "AMOS train job:  ${TRAIN_ID} (after smoke passes)"
squeue -u yuangluo22
REMOTE

echo "Done."
