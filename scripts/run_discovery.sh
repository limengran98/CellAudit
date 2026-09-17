#!/usr/bin/env bash
set -euo pipefail

task="${1:-bbbc036}"
device="${2:-cuda:0}"
output="runs/${task}_falsification_guided_discovery"

python -m cellaudit campaign \
  --stage falsification-guided \
  --task "$task" \
  --trajectories 10 \
  --seed-base 2026080701 \
  --device "$device" \
  --output-root "$output"
