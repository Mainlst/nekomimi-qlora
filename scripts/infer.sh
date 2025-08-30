#!/usr/bin/env bash
set -euo pipefail
# Usage: scripts/infer.sh "prompt text" [preset]
PROMPT=${1:-"明日の朝やるべきことを3つだけ教えて"}
PRESET=${2:-sweet}

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

"${PY[@]}" -u chat_maid.py --prompt "$PROMPT" --preset "$PRESET"
