#!/bin/bash
# Link ECCV-style paths to AD_project data on XEC HPC.
set -euo pipefail

SKCDF_ROOT="${SKCDF_ROOT:-/gpfs/work/aac/zimuzhang2302/SKCDF}"
AD_PROJECT="${AD_PROJECT:-/gpfs/work/aac/zimuzhang2302/AD_project}"
ECCV_ROOT="${SKCDF_ROOT}/ECCV"

mkdir -p "${ECCV_ROOT}/synapse_data"
ln -sfn "${AD_PROJECT}/Synapse_data/npy" "${ECCV_ROOT}/synapse_data/npy"
ln -sfn "${AD_PROJECT}/Synapse_data/split_txts" "${ECCV_ROOT}/synapse_splits"
ln -sfn "${AD_PROJECT}/AMOS_data/npy" "${ECCV_ROOT}/AMOS"
ln -sfn "${AD_PROJECT}/AMOS_data/splits" "${ECCV_ROOT}/amos_splits"

echo "ECCV_ROOT=${ECCV_ROOT}"
echo "Synapse npy pairs: $(ls "${ECCV_ROOT}/synapse_data/npy"/*_image.npy 2>/dev/null | wc -l)"
echo "AMOS npy pairs: $(ls "${ECCV_ROOT}/AMOS"/*_image.npy 2>/dev/null | wc -l)"
echo "synapse splits: $(ls "${ECCV_ROOT}/synapse_splits"/*.txt | wc -l)"
echo "amos splits: $(ls "${ECCV_ROOT}/amos_splits"/*.txt | wc -l)"
