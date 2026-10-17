"""Training-only fixed-anchor selection from the original implementation."""
from __future__ import annotations
import numpy as np

ALPHAS = np.arange(0.0, 1.0001, 0.1, dtype=np.float64)

def metric_dict(ranks: np.ndarray) -> dict[str, float | int]:
    x = ranks.astype(np.float64)
    return {
        "num_queries": int(x.size),
        "mr": float(x.mean()),
        "mrr": float((1.0 / x).mean()),
        "h1": float((x <= 1).mean()),
        "h3": float((x <= 3).mean()),
        "h10": float((x <= 10).mean()),
    }

def utility(metrics: dict[str, float | int]) -> float:
    # Equal-weight multi-metric train objective; all four reported metrics
    # contribute, while MR is intentionally excluded because it is on another
    # scale and is not a higher-is-better measure.
    return float(
        (float(metrics["mrr"])
         + float(metrics["h1"])
         + float(metrics["h3"])
         + float(metrics["h10"]))
        / 4.0
    )

def choose_alpha(cube: np.ndarray, mask: np.ndarray) -> tuple[float, dict[str, float]]:
    if not mask.any():
        raise ValueError("empty selection group")
    candidates = []
    for index, alpha in enumerate(ALPHAS):
        item = metric_dict(cube[index, mask])
        candidates.append((utility(item), float(item["mrr"]), float(item["h1"]), -float(alpha), float(alpha), item))
    _, _, _, _, alpha, item = max(candidates)
    return alpha, {"utility": utility(item), **{k: float(v) for k, v in item.items() if k != "num_queries"}}

