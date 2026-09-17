#!/usr/bin/env bash
set -euo pipefail

task="${1:-bbbc047}"
device="${2:-cuda:0}"
discovery="${3:-runs/${task}_open_discovery}"
source_trajectory="${4:-}"
refits="runs/${task}_open_refits"
audit="runs/${task}_open_endpoint_audit"

prepare_args=(
  python -m cellaudit audit prepare-open-refits
  --task "$task"
  --discovery-root "$discovery"
  --device "$device"
  --output-root "$refits"
)
if [[ -n "$source_trajectory" ]]; then
  prepare_args+=(--source-trajectory "$source_trajectory")
fi
"${prepare_args[@]}"
python -m cellaudit audit freeze \
  --method open \
  --task "$task" \
  --discovery-root "$refits" \
  --output-root "$audit"
python -m cellaudit audit run --output-root "$audit" --fold 4 --device "$device"
python -m cellaudit audit summarize --output-root "$audit" --fold 4
python -m cellaudit audit authorize-fold5 --output-root "$audit"
python -m cellaudit audit run --output-root "$audit" --fold 5 --device "$device"
python -m cellaudit audit summarize --output-root "$audit" --fold 5
