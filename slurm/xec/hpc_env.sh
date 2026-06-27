#!/bin/bash
# SKCDF env on XEC HPC (zimuzhang2302)

module load anaconda3/2023.09-0-none-none-3te2njg
source activate dhc

if ! python -c "import tensorboardX" 2>/dev/null; then
    echo "Installing tensorboardX into dhc env..."
    pip install tensorboardX -q
fi

export SKCDF_ROOT="${SKCDF_ROOT:-/gpfs/work/aac/zimuzhang2302/SKCDF}"
export ECCV_ROOT="${ECCV_ROOT:-${SKCDF_ROOT}/ECCV}"
export PYTHONPATH="${SKCDF_ROOT}/SKCDF/code:${PYTHONPATH:-}"

python -c "
import os, torch
import einops
from tensorboardX import SummaryWriter
from medpy import metric
assert torch.cuda.is_available(), 'CUDA not available'
print('skcdf/xec env OK | torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0))
print('SKCDF_ROOT =', os.environ.get('SKCDF_ROOT'))
print('ECCV_ROOT  =', os.environ.get('ECCV_ROOT'))
"
