#!/usr/bin/env bash
# Fit the PACE controller on the three benchmarks and report test metrics.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-data}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
DATASETS=${DATASETS:-"DB15K MKG-W MKG-Y"}
SEEDS=${SEEDS:-"20260904 20260905 20260906"}
DEVICE=${DEVICE:-cuda:0}
FEATURES=${FEATURES:-5}
# structural_historical reproduces the paper's tables. anchor_hard is the
# alternative rule (hardest legal non-target under the anchor fusion).
NEGATIVE_RULE=${NEGATIVE_RULE:-structural_historical}

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    python train.py \
      --data-root "$DATA_ROOT" \
      --dataset "$dataset" \
      --features "$FEATURES" \
      --negative-rule "$NEGATIVE_RULE" \
      --output-dir "$OUTPUT_ROOT/${dataset}_s${seed}" \
      --device "$DEVICE" \
      --seed "$seed"
  done
done
