#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
SEED="${SEED:-1}"
GPU="${GPU:-0}"

DATASETS=(
  BTMRI BUSI CHMNIST COVID_19 DermaMNIST
  KneeXray Kvasir LungColon OCTMNIST RETINA
)

cd "${REPO_ROOT}"

for ds in "${DATASETS[@]}"; do
  OUT="${OUTPUT_BASE:-${REPO_ROOT}/output}/sae_${ds,,}_seed${SEED}"
  echo "==== [${ds}] -> ${OUT} ===="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" train.py \
    --root "${DATA_ROOT}" \
    --output-dir "${OUT}" \
    --dataset-config-file "configs/datasets/${ds}.yaml" \
    --config-file configs/trainers/ALVLM/vit_b16.yaml \
    --method-config-file configs/methods/sae.yaml \
    --trainer BiomedCLIP \
    --seed "${SEED}"
done

echo "All ten datasets finished."
