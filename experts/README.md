# Expert adapters

PACE never trains an expert. It consumes two frozen score matrices that share
one candidate indexing, so the only work on the expert side is *exporting*
scores into the layout `data.py` expects.

```
data/{dataset}/{split}_{direction}/
    semantic.npy     K-ON scores      [queries, entities]
    structural.npy   LBMKGC scores    [queries, entities]
    contract.npz     target index + shared other-answer filter
```

`direction` is `head` or `tail`. Query order inside a file is the row order of
the source `{split}2id.txt`; head rows and tail rows live in separate folders.
Both arrays are float32 with higher meaning better.

## 1. Contracts

`build_contract.py` needs nothing but the graph in OpenKE text format and
writes `contract.npz` for any split:

```bash
python experts/build_contract.py --data-dir data/graphs/DB15K --dataset DB15K \
    --split test  --filter-source all   --output-root data
python experts/build_contract.py --data-dir data/graphs/DB15K --dataset DB15K \
    --split train --filter-source train --output-root data
```

`--filter-source all` (evaluation) filters every other known answer in
train/valid/test; `--filter-source train` (in-sample fitting) uses training
facts only, so no validation or test fact can leak into a feature or a training
negative.

## 2. K-ON — semantic expert

`kon/export_kon_scores.py` runs the frozen K-ON checkpoint over the query
prompts and writes `semantic.npy`.

Needs the K-ON code base on the path (`k_on.py`, `dataset.py`,
`checkpoint_utils.py`) plus its `data/{dataset}/preprocessed.pkl`. The
checkpoint directories reported for the paper are:

| Dataset | Config | Checkpoint |
| --- | --- | --- |
| DB15K | `configs/DB15K.yaml` | `outputs/DB15K/checkpoint-4140` |
| MKG-W | `configs/MKG-W.yaml` | `outputs/MKG-W/checkpoint-12220` |
| MKG-Y | `configs/MKG-Y.yaml` | `outputs/MKG-Y-b2-lr57e5/checkpoint-12654` |

```bash
KON_ROOT=/path/to/K-ON CKPT_ROOT=/path/to/K-ON/outputs \
    bash scripts/01_export_kon_scores.sh
```

## 3. LBMKGC — structural expert

`lbmkgc/export_lbmkgc_scores.py` loads the official AdvRelRotatE checkpoint and
scores every entity for each query, writing `structural.npy` with columns
reordered into the K-ON entity-id order.

`lbmkgc/structure_expert.py` is the frozen scoring wrapper. It needs the
LBMKGC checkout for `mmkgc.module.model.AdvRelRotatE` and its multimodal
tables `embeddings/{dataset}-visual.pth` / `-textual.pth`.

The LBMKGC benchmark vocabularies contain only counts, so the entity
correspondence is recovered from a *named* vocabulary snapshot — the same URI
list that was used when the checkpoints were prepared. Pass it with
`REFERENCE_VOCAB_ROOT`, or use `NO_ALIGNMENT=1` when both experts already share
an identical entity-id order.

```bash
LBM_ROOT=/path/to/LBMKGC LBM_CKPT_ROOT=/path/to/LBMKGC/Checkpoints/LBMKGC \
REFERENCE_VOCAB_ROOT=/path/to/named_vocab \
    bash scripts/02_export_lbmkgc_scores.sh
```

## Adding a different expert pair

Any frozen method with a candidate-level score works, as long as both experts
agree on entity ids, candidate column order and score orientation. Write the
two matrices into the layout above; nothing in `pace.py` or `train.py` needs to
change.

Expert implementations, checkpoints, pretrained weights and benchmark data are
**not** redistributed with this repository.
