#!/usr/bin/env bash
# Build the PACE query contracts for the three benchmarks.
#
# Expects one OpenKE-style folder per dataset (see experts/README.md):
#   $GRAPH_ROOT/DB15K/{train,valid,test}2id.txt, entity2id.txt, relation2id.txt
set -euo pipefail
cd "$(dirname "$0")/.."

GRAPH_ROOT=${GRAPH_ROOT:-data/graphs}
OUT=${OUTPUT_ROOT:-data}
DATASETS=${DATASETS:-"DB15K MKG-W MKG-Y"}

for dataset in $DATASETS; do
  for split in train test; do
    if [ "$split" = "train" ]; then
      filter_source=train     # in-sample: no validation or test fact may leak
    else
      filter_source=all       # standard filtered-ranking protocol
    fi
    python experts/build_contract.py \
      --data-dir "$GRAPH_ROOT/$dataset" \
      --dataset "$dataset" \
      --split "$split" \
      --filter-source "$filter_source" \
      --output-root "$OUT" \
      --overwrite
  done
done
