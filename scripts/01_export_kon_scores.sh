#!/usr/bin/env bash
# Export frozen K-ON (semantic expert) scores for the train and test splits.
#
# Required environment:
#   KON_ROOT   checkout of the K-ON code base (k_on.py, dataset.py, checkpoint_utils.py)
#   CKPT_ROOT  folder holding one checkpoint directory per dataset, e.g.
#                $CKPT_ROOT/DB15K/checkpoint-4140/
#
# Optional:
#   DATASETS   default "DB15K MKG-W MKG-Y"
#   OUTPUT_ROOT  default "data"
set -euo pipefail
cd "$(dirname "$0")/.."

: "${KON_ROOT:?set KON_ROOT to the K-ON checkout}"
: "${CKPT_ROOT:?set CKPT_ROOT to the folder holding the K-ON checkpoints}"

OUTPUT_ROOT=${OUTPUT_ROOT:-data}
DATASETS=${DATASETS:-"DB15K MKG-W MKG-Y"}
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-8}

# Checkpoint sub-directory reported for each benchmark.
checkpoint_for() {
  case "$1" in
    DB15K) echo "$CKPT_ROOT/DB15K/checkpoint-4140" ;;
    MKG-W) echo "$CKPT_ROOT/MKG-W/checkpoint-12220" ;;
    MKG-Y) echo "$CKPT_ROOT/MKG-Y-b2-lr57e5/checkpoint-12654" ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

for dataset in $DATASETS; do
  config="configs/${dataset}.yaml"
  checkpoint=$(checkpoint_for "$dataset")
  for split in train test; do
    python experts/kon/export_kon_scores.py \
      --kon-root "$KON_ROOT" \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --dataset "$dataset" \
      --split "$split" \
      --output-root "$OUTPUT_ROOT" \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE"
  done
done
