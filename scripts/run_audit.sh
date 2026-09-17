#!/usr/bin/env bash
set -euo pipefail

task="${1:-bbbc036}"
device="${2:-cuda:0}"
discovery="runs/${task}_falsification_guided_discovery"
audit="runs/${task}_endpoint_audit"

python -m cellaudit audit freeze \
  --method falsification-guided \
  --task "$task" \
  --discovery-root "$discovery" \
  --output-root "$audit"
python -m cellaudit audit run --output-root "$audit" --fold 4 --device "$device"
python -m cellaudit audit summarize --output-root "$audit" --fold 4
python -m cellaudit audit authorize-fold5 --output-root "$audit"
python -m cellaudit audit run --output-root "$audit" --fold 5 --device "$device"
python -m cellaudit audit summarize --output-root "$audit" --fold 5
