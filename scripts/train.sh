#!/usr/bin/env bash
set -euo pipefail
# Usage: scripts/train.sh configs/maid_1p5b_stable.yaml [--dry-run]
CONFIG=${1:-configs/maid_1p5b_stable.yaml}
shift || true
python -u train_maid.py --config "$CONFIG" "$@"
