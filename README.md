# PACE

Core code for bounded adaptive collaboration of frozen entity predictors.
PACE operates on two score matrices with a shared candidate entity order;
it does not update either expert.

## Included

- `pace.py`: query-wise alignment, prediction profiling, the 16-unit bounded
  controller, bounded fusion, and negative selection.
- `selection.py`: training-query anchor selection and ranking metrics.
- `data.py`: portable input validation and loading.
- `train.py`: controller fitting and filtered head/tail evaluation.
- `experts/`: adapters that export frozen expert scores into the score layout
  (K-ON semantic, LBMKGC structural) and the contract builder.
- `configs/`: expert settings used in the paper, one file per benchmark.
- `scripts/`: end-to-end launchers for the main experiment.
- `docs/RESULTS.md`: which command reproduces which paper artifact.
- `examples/make_toy_data.py`: a synthetic input example.

The controller, alignment, feature construction, fusion and rank operations
retain the original numerical implementation. Machine monitoring, historical
sweeps and reporting scratch scripts are omitted. The command line and input
paths have been made portable.

The release covers the main comparison. Ablations, sensitivity sweeps and the
inference-overhead benchmark are not part of this archive.

## Repository layout

```text
pace.py  selection.py  data.py  train.py     core method and score interface
experts/build_contract.py                    query contracts from a graph
experts/kon/export_kon_scores.py             semantic expert scores
experts/lbmkgc/export_lbmkgc_scores.py       structural expert scores
scripts/00..03_*.sh                          one launcher per stage
configs/{DB15K,MKG-W,MKG-Y}.yaml             expert settings
docs/RESULTS.md                              reproduction map
```

The pipeline is `contracts → expert scores → controller`. See
`docs/RESULTS.md` for the exact command per stage and `experts/README.md` for
the expert-side requirements.

## Environment

Python 3.10 or newer. The research environment used Python 3.11.15,
PyTorch 2.7.0 and CUDA 12.6. Install PyTorch appropriate for your CUDA
runtime (or CPU), then install the dependencies:

```bash
pip install -r requirements.txt
```

`numpy` and `torch` are all the controller needs. The remaining entries
(`pyyaml`, `easydict`, `transformers`, `peft`) are used only by the two expert
score exporters; skip them if you already have exported scores.

## Input scores

Export scores from any frozen semantic and structural expert pair. Use
identical entity IDs, candidate-column order, query-row order and score
orientation (higher is better). For every dataset, provide:

```text
data/DB15K/
  train_head/{semantic.npy,structural.npy,contract.npz}
  train_tail/{semantic.npy,structural.npy,contract.npz}
  test_head/{semantic.npy,structural.npy,contract.npz}
  test_tail/{semantic.npy,structural.npy,contract.npz}
```

Both score arrays are finite float matrices of shape `[queries, entities]`.
Their scores are loaded with memory mapping and processed in row batches.
Each `contract.npz` has three integer arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `target` | `[queries]` | Current correct entity ID for each query |
| `filter_offsets` | `[queries+1]` | CSR offsets into `filter_ids`, starting at zero |
| `filter_ids` | `[excluded answers]` | Other known correct entity IDs to exclude per query |

The current target must remain outside its filter list. Both experts use
exactly the same filter. The filter list must be prepared for the intended
training or evaluation protocol by the data exporter; this code does not
infer which graph splits should contribute known answers. At least two
legal candidates must remain. Query order is head queries followed by tail
queries. Raw datasets, expert implementations, score caches and pretrained
weights are not bundled; obtain the experts and benchmark data separately
and export their scores to this interface.

## Train and evaluate

```bash
python train.py --data-root data --dataset DB15K --output-dir outputs/DB15K --device cuda:0
```

By default the fixed anchor is selected on training queries over
`0.0, 0.1, ..., 1.0`. The original selection utility averages MRR and
Hits@1/3/10, with MRR, Hits@1 and smaller alpha used to break ties.
To use an explicit anchor and radius, pass `--anchor-alpha` and
`--gate-radius`. Without an explicit radius it is
`min(0.2, alpha0, 1-alpha0)`. Explicit radii are not silently clipped:
convex fusion requires `0 <= alpha0-radius` and `alpha0+radius <= 1`.

Features preserve the original code order:
`[semantic margin, structural margin, absolute margin difference,
top-1 agreement, absolute top-1 score difference]`.
Dispersion features are not controller inputs in this five-statistic variant.
Feature means and standard deviations are fitted on training queries and reused
at evaluation. The structural weight is `alpha0 + radius * tanh(MLP(features))`.

`--features 7` switches the profile to the seven-statistic vector written in the
manuscript, `[m_K, m_L, d_K, d_L, |m_K-m_L|, a, |t_K-t_L|]`, where `d_K` and
`d_L` are zero-filled sample deviations of the aligned score vectors. The
controller then has 7 inputs and 145 parameters instead of 5 and 113.

The shared other-answer filter is applied before score alignment and
feature construction, preserving the original experiment protocol.

Two training-negative rules are available:

- `--negative-rule structural_historical` (default): the structural expert's
  top-1 candidate, falling back to its runner-up when both experts agree on
  top-1. This reproduces the reported tables.
- `--negative-rule anchor_hard`: the highest-scoring legal non-target entity
  under fixed anchor fusion, with the lowest entity ID breaking score ties.

Both rules exclude the target and keep the negative fixed during a fit, so the
two recipes differ only in which legal non-target supplies the margin.

Defaults: Adam, 80 epochs, learning rate 0.001, weight decay 0.0001,
batch size 4096, ranking-loss temperature 0.2, and anchor penalty 0.05.
Evaluation uses pessimistic ties. MRR and Hits in `metrics.json` are
fractions; multiply by 100 for percentage-style tables.

Each run writes its controller, standardization statistics, configuration,
test ranks and fusion weights under the requested output directory.
These generated files are excluded from version control.

## Small CPU example

```bash
python examples/make_toy_data.py --output data/Toy
python train.py --data-root data --dataset Toy --output-dir outputs/toy --device cpu --epochs 2 --anchor-alpha 0.6 --gate-radius 0.2
```

This checks the input interface and training/evaluation path using synthetic
scores; it is not a benchmark result.

## Inference API

`pace.predict_scores` accepts a controller, two score tensors, training-fitted
feature statistics and the fixed anchor/radius. It returns fused scores
and query-wise weights without target labels. For filtered evaluation,
use `train.py`'s evaluation path with the appropriate query contracts.

## Release scope

This archive contains the PACE method, the score-based training interface and
the expert score exporters for the two experts used in the paper. It contains
no benchmark outputs, logs, checkpoints, expert weights, datasets, cluster
configuration or repository history.

Ablation, sensitivity and inference-overhead studies are not included. The
released scope is the main comparison: the learned controller against fixed
fusion under a shared frozen score cache.

Expert training and checkpoint selection are outside this release: the two
exporters consume frozen checkpoints and only write candidate score matrices.
See `experts/README.md`.
