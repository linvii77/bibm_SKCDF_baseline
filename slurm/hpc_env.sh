#!/bin/bash
# SKCDF conda env on XJTLU HPC

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/miniconda3-22.11.1-l4fo6takdpx5xewhp463xsqr4jcd73dx/etc/profile.d/conda.sh

ENV_NAME=skcdf

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Creating ${ENV_NAME} from genericssl..."
    conda create -n "${ENV_NAME}" --clone genericssl -y
fi
conda activate "${ENV_NAME}"

if [ ! -f "${CONDA_PREFIX}/.skcdf_ready" ]; then
    echo "Installing SKCDF dependencies..."
    pip install tensorboardX einops
    touch "${CONDA_PREFIX}/.skcdf_ready"
fi

# Do not derive from $0: under SLURM, $0 points to spool dir and breaks paths.
export SKCDF_ROOT="${SKCDF_ROOT:-/gpfs/work/aac/yuangluo22/SKCDF}"
export ECCV_ROOT="${ECCV_ROOT:-${SKCDF_ROOT}/ECCV}"
export PYTHONPATH="${SKCDF_ROOT}/SKCDF/code:${PYTHONPATH:-}"

python -c "
import os, torch
import einops
from tensorboardX import SummaryWriter
from medpy import metric
assert torch.cuda.is_available(), 'CUDA not available'
print('skcdf env OK | torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0))
print('SKCDF_ROOT =', os.environ.get('SKCDF_ROOT'))
print('ECCV_ROOT =', os.environ.get('ECCV_ROOT'))
"
