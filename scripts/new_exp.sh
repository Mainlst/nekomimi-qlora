#!/usr/bin/env bash
set -euo pipefail

name="${1:-}"
if [[ -z "$name" ]]; then
  echo "Usage: scripts/new_exp.sh <short-name>" >&2
  echo "  e.g. scripts/new_exp.sh lora-qlora" >&2
  exit 1
fi

today=$(date +%F)
exp_dir="experiments/${today}-${name}"

if [[ -d "$exp_dir" ]]; then
  echo "Already exists: $exp_dir" >&2
  exit 1
fi

mkdir -p "$exp_dir"
cp -r experiments/_template/* "$exp_dir"/

echo "Created: $exp_dir"
echo "Next: bash scripts/run_exp.sh $exp_dir/config.yaml"
