#!/usr/bin/env bash
# Copy repo checkpoints into logs/ for evaluation or inference.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "${ROOT}/logs/synapse_20p/skcdf/fold1/ckpts"
mkdir -p "${ROOT}/logs/amos_5p/skcdf/fold1/ckpts"

cp -f "${ROOT}/checkpoints/synapse_20p/fold1/best_model.pth" \
      "${ROOT}/logs/synapse_20p/skcdf/fold1/ckpts/best_model.pth"
cp -f "${ROOT}/checkpoints/amos_5p/fold1/best_model.pth" \
      "${ROOT}/logs/amos_5p/skcdf/fold1/ckpts/best_model.pth"

echo "Checkpoints ready under ${ROOT}/logs/"
