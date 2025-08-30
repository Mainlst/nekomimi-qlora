#!/usr/bin/env bash
set -euo pipefail

cfg="${1:-}"
if [[ -z "$cfg" ]]; then
  echo "Usage: scripts/run_exp.sh experiments/<date>-<name>/config.yaml" >&2
  exit 1
fi

python train_maid.py --config "$cfg"

# Optional: place for evaluation/summary script. No-op if missing.
if [[ -f scripts/summarize_metrics.py ]]; then
  python scripts/summarize_metrics.py "$(dirname "$cfg")/metrics.json"
fi

echo "Done: $(dirname "$cfg")"
