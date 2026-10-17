#!/usr/bin/env bash
# Export frozen LBMKGC (structural expert) scores for the train and test splits.
#
# Required environment:
#   LBM_ROOT    checkout of LBMKGC (needs mmkgc/ and embeddings/*.pth)
#   LBM_CKPT_ROOT  folder holding one checkpoint zip per dataset
#
# Optional:
#   REFERENCE_VOCAB_ROOT  named entity vocabularies used when the checkpoints
#                         were prepared; required unless NO_ALIGNMENT=1
#   GRAPH_ROOT   default "data/graphs"
#   OUTPUT_ROOT  default "data"
set -euo pipefail
cd "$(dirname "$0")/.."

: "${LBM_ROOT:?set LBM_ROOT to the LBMKGC checkout}"
: "${LBM_CKPT_ROOT:?set LBM_CKPT_ROOT to the folder holding the LBMKGC checkpoints}"

GRAPH_ROOT=${GRAPH_ROOT:-data/graphs}
OUTPUT_ROOT=${OUTPUT_ROOT:-data}
DATASETS=${DATASETS:-"DB15K MKG-W MKG-Y"}
DEVICE=${DEVICE:-cuda:0}
NO_ALIGNMENT=${NO_ALIGNMENT:-0}

checkpoint_for() {
  case "$1" in
    DB15K) echo "$LBM_CKPT_ROOT/DB15K-checkpoint.zip" ;;
    MKG-W) echo "$LBM_CKPT_ROOT/MKG-W-checkpoint.zip" ;;
    MKG-Y) echo "$LBM_CKPT_ROOT/MKG-Y-checkpoint.zip" ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

for dataset in $DATASETS; do
  checkpoint=$(checkpoint_for "$dataset")
  graph="$GRAPH_ROOT/$dataset"
  extra=()
  if [ "$NO_ALIGNMENT" = "1" ]; then
    extra+=(--no-alignment)
  else
    : "${REFERENCE_VOCAB_ROOT:?set REFERENCE_VOCAB_ROOT or NO_ALIGNMENT=1}"
    extra+=(--reference-data "$REFERENCE_VOCAB_ROOT/$dataset")
  fi
  for split in train test; do
    python experts/lbmkgc/export_lbmkgc_scores.py \
      --lbm-root "$LBM_ROOT" \
      --data-dir "$graph" \
      "${extra[@]}" \
      --checkpoint "$checkpoint" \
      --dataset "$dataset" \
      --split "$split" \
      --output-root "$OUTPUT_ROOT" \
      --device "$DEVICE"
  done
done
