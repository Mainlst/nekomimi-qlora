#!/usr/bin/env bash
set -euo pipefail

cfg="${1:-}"
shift || true
extra_args=("$@")

if [[ -z "$cfg" ]]; then
  echo "Usage: scripts/run_exp.sh experiments/<date>-<name>/config.yaml [--dry-run] [extra args...]" >&2
  exit 1
fi

python train_maid.py --config "$cfg" "${extra_args[@]}"

# Optional: place for evaluation/summary script. No-op if missing.
if [[ -f scripts/summarize_metrics.py ]]; then
  python scripts/summarize_metrics.py "$(dirname "$cfg")/metrics.json"
fi

echo "Done: $(dirname "$cfg")"
