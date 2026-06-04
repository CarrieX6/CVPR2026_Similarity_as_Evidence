#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/sae_busi_seed1}"
SEED="${SEED:-1}"
GPU="${GPU:-0}"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" train.py \
  --root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-config-file configs/datasets/BUSI.yaml \
  --config-file configs/trainers/ALVLM/vit_b16.yaml \
  --method-config-file configs/methods/sae.yaml \
  --trainer BiomedCLIP \
  --seed "${SEED}"
