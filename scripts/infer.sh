#!/usr/bin/env bash
set -euo pipefail
# Usage: scripts/infer.sh "prompt text" [preset]
PROMPT=${1:-"明日の朝やるべきことを3つだけ教えて"}
PRESET=${2:-sweet}
python -u chat_maid.py --prompt "$PROMPT" --preset "$PRESET"
