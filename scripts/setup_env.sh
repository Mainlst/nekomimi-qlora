#!/usr/bin/env bash
set -euo pipefail
# Create or update a local environment named 'lora-local'.
# If conda/mamba exists, prefer it. Otherwise fall back to venv.

ENV_NAME=${1:-lora-local}
CUDA_INDEX_URL=${CUDA_INDEX_URL:-"https://download.pytorch.org/whl/cu121"}

have_conda() { command -v conda >/dev/null 2>&1; }
have_mamba() { command -v mamba >/dev/null 2>&1; }

if have_mamba; then
  echo "[setup] Using mamba to create env '${ENV_NAME}'"
  mamba env update -f environment.yml -n "${ENV_NAME}" || mamba env create -f environment.yml -n "${ENV_NAME}" || true
  echo "[setup] To activate: conda activate ${ENV_NAME}"
  echo "[setup] Installing CUDA-specific torch wheels via pip (optional)"
  conda run -n "${ENV_NAME}" python -m pip install --index-url "${CUDA_INDEX_URL}" torch torchvision --upgrade
elif have_conda; then
  echo "[setup] Using conda to create env '${ENV_NAME}'"
  conda env update -f environment.yml -n "${ENV_NAME}" || conda env create -f environment.yml -n "${ENV_NAME}" || true
  echo "[setup] To activate: conda activate ${ENV_NAME}"
  echo "[setup] Installing CUDA-specific torch wheels via pip (optional)"
  conda run -n "${ENV_NAME}" python -m pip install --index-url "${CUDA_INDEX_URL}" torch torchvision --upgrade
else
  echo "[setup] No conda/mamba found. Using venv at .venv-${ENV_NAME}"
  python -m venv ".venv-${ENV_NAME}"
  source ".venv-${ENV_NAME}/bin/activate"
  python -m pip install --upgrade pip
  # Use CUDA extra-index if available
  python -m pip install --index-url "${CUDA_INDEX_URL}" torch torchvision --upgrade || true
  python -m pip install -r requirements.txt
  echo "[setup] To activate: source .venv-${ENV_NAME}/bin/activate"
fi
