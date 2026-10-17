# PACE

Core code for bounded adaptive collaboration of frozen entity predictors.
PACE operates on two score matrices with a shared candidate entity order;
it does not update either expert.

## Included

- `pace.py`: query-wise alignment, five-feature prediction profiling,
  5-16-1 controller (113 parameters), bounded fusion, and negative selection.
- `selection.py`: training-query anchor selection and ranking metrics.
- `data.py`: portable input validation and loading.
- `train.py`: controller fitting and filtered head/tail evaluation.
- `examples/make_toy_data.py`: a synthetic input example.

The controller, alignment, feature construction, fusion and rank operations
retain the original numerical implementation. Experiment scheduling,
machine monitoring, historical sweeps and reporting scripts are omitted.
The command-line and input paths have been made portable.

## Environment

Python 3.10 or newer. The research environment used Python 3.11.15,
PyTorch 2.7.0 and CUDA 12.6. Install PyTorch appropriate for your CUDA
runtime (or CPU), then install the dependencies:

```bash
pip install -r requirements.txt
```

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
Dispersion features are not controller inputs. Feature means and standard
deviations are fitted on training queries and reused at evaluation.
The structural weight is `alpha0 + radius * tanh(MLP(features))`.

The shared other-answer filter is applied before score alignment and
feature construction, preserving the original experiment protocol.
The training negative is the highest-scoring legal non-target entity
under fixed anchor fusion, with the lowest entity ID breaking score ties.
The anchor and negatives remain fixed during a controller fit. Historical
structural-negative selection remains explicitly labelled in the helper
code and is not used by the training entry point.

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

This archive contains the core PACE method and a portable score-based
training interface. It contains no benchmark outputs, logs, checkpoints,
expert weights, datasets, cluster configuration or repository history.
Full expert training/export pipelines and paper-specific experiment suites
are outside this compact release.
