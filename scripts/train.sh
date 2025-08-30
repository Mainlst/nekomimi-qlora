#!/usr/bin/env bash
set -euo pipefail
# Usage: scripts/train.sh configs/maid_1p5b_stable.yaml [--dry-run]
CONFIG=${1:-configs/maid_1p5b_stable.yaml}
shift || true

ENV_NAME=${ENV_NAME:-lora-local}

if command -v micromamba >/dev/null 2>&1; then
	PY=("micromamba" "run" "-n" "$ENV_NAME" "python")
elif command -v mamba >/dev/null 2>&1; then
	PY=("mamba" "run" "-n" "$ENV_NAME" "python")
elif command -v conda >/dev/null 2>&1; then
	PY=("conda" "run" "-n" "$ENV_NAME" "python")
else
	PY=(python)
fi

"${PY[@]}" -u train_maid.py --config "$CONFIG" "$@"
