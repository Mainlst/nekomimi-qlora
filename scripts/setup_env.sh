#!/usr/bin/env bash
set -euo pipefail
# Create or update a local environment named 'lora-local' using micromamba/mamba.
# Priority: micromamba > mamba > conda > venv

ENV_NAME=${1:-lora-local}
CUDA_INDEX_URL=${CUDA_INDEX_URL:-"https://download.pytorch.org/whl/cu121"}

have_conda() { command -v conda >/dev/null 2>&1; }
have_mamba() { command -v mamba >/dev/null 2>&1; }
have_micromamba() { command -v micromamba >/dev/null 2>&1; }

if have_micromamba; then
  echo "[setup] Using micromamba to create env '${ENV_NAME}'"
  # Try update first; if it fails create the env
  micromamba update -y -n "${ENV_NAME}" -f environment.yml || micromamba create -y -n "${ENV_NAME}" -f environment.yml || true
  echo "[setup] To activate: micromamba activate ${ENV_NAME}"
  echo "[setup] Installing CUDA-specific torch wheels via pip (optional)"
  micromamba run -n "${ENV_NAME}" python -m pip install --index-url "${CUDA_INDEX_URL}" torch torchvision --upgrade || true
elif have_mamba; then
  echo "[setup] Using mamba to create env '${ENV_NAME}'"
  mamba env update -f environment.yml -n "${ENV_NAME}" || mamba env create -f environment.yml -n "${ENV_NAME}" || true
  echo "[setup] To activate: conda activate ${ENV_NAME}"
  echo "[setup] Installing CUDA-specific torch wheels via pip (optional)"
  mamba run -n "${ENV_NAME}" python -m pip install --index-url "${CUDA_INDEX_URL}" torch torchvision --upgrade || true
elif have_conda; then
  echo "[setup] mamba install failed; using conda as fallback to create env '${ENV_NAME}'"
  conda env update -f environment.yml -n "${ENV_NAME}" || conda env create -f environment.yml -n "${ENV_NAME}" || true
  echo "[setup] To activate: conda activate ${ENV_NAME}"
  echo "[setup] Installing CUDA-specific torch wheels via pip (optional)"
  conda run -n "${ENV_NAME}" python -m pip install --index-url "${CUDA_INDEX_URL}" torch torchvision --upgrade || true
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
