#!/bin/bash
# Link ECCV data on HPC for SKCDF (volume-level npy).
set -euo pipefail

# Do not derive from $0: under SLURM, $0 points to spool dir and breaks paths.
SKCDF_ROOT="${SKCDF_ROOT:-/gpfs/work/aac/yuangluo22/SKCDF}"
ECCV_ROOT="${SKCDF_ROOT}/ECCV"
DHC_ROOT="/gpfs/work/aac/yuangluo22/DHC-main"

mkdir -p "${ECCV_ROOT}"

mkdir -p "${ECCV_ROOT}/synapse_data"
ln -sfn "${DHC_ROOT}/synapse_data/npy" "${ECCV_ROOT}/synapse_data/npy"
ln -sfn "${DHC_ROOT}/amos_data/npy" "${ECCV_ROOT}/AMOS"

if [ ! -d "${ECCV_ROOT}/synapse_splits" ] || [ -z "$(ls -A "${ECCV_ROOT}/synapse_splits"/*.txt 2>/dev/null)" ]; then
    if [ -d "/gpfs/work/aac/yuangluo22/SS-Net-eccv/ECCV/synapse_splits" ]; then
        ln -sfn "/gpfs/work/aac/yuangluo22/SS-Net-eccv/ECCV/synapse_splits" "${ECCV_ROOT}/synapse_splits"
    else
        echo "ERROR: missing synapse_splits"
        exit 1
    fi
fi

if [ ! -d "${ECCV_ROOT}/amos_splits" ] || [ -z "$(ls -A "${ECCV_ROOT}/amos_splits"/*.txt 2>/dev/null)" ]; then
    if [ -d "/gpfs/work/aac/yuangluo22/SS-Net-eccv/ECCV/amos_splits" ]; then
        ln -sfn "/gpfs/work/aac/yuangluo22/SS-Net-eccv/ECCV/amos_splits" "${ECCV_ROOT}/amos_splits"
    else
        echo "ERROR: missing amos_splits"
        exit 1
    fi
fi

echo "ECCV_ROOT=${ECCV_ROOT}"
echo "Synapse npy pairs: $(ls "${ECCV_ROOT}/synapse_data/npy"/*_image.npy 2>/dev/null | wc -l)"
echo "AMOS npy pairs: $(ls "${ECCV_ROOT}/AMOS"/*_image.npy 2>/dev/null | wc -l)"
echo "synapse splits: $(ls "${ECCV_ROOT}/synapse_splits"/*.txt | wc -l)"
echo "amos splits: $(ls "${ECCV_ROOT}/amos_splits"/*.txt | wc -l)"
