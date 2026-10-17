"""Generate small synthetic frozen scores to check the public data interface."""
import argparse
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, default=Path("data/Toy"))
args = p.parse_args()
rng = np.random.default_rng(42)
candidates = 16
relations = 4
for split, n in (("train", 32), ("test", 12)):
    for direction in ("head", "tail"):
        folder = args.output / f"{split}_{direction}"
        folder.mkdir(parents=True, exist_ok=False)
        target = rng.integers(0, candidates, n, dtype=np.int64)
        semantic = rng.normal(size=(n, candidates)).astype(np.float32)
        structural = rng.normal(size=(n, candidates)).astype(np.float32)
        semantic[np.arange(n), target] += 0.5
        structural[np.arange(n), target] += 0.8
        np.save(folder / "semantic.npy", semantic)
        np.save(folder / "structural.npy", structural)
        np.savez_compressed(
            folder / "contract.npz",
            target=target,
            filter_offsets=np.arange(n + 1, dtype=np.int64),
            filter_ids=(target + 1) % candidates,
            # Optional graph columns, present in contracts built by
            # experts/build_contract.py; they drive stratified subsampling.
            h=rng.integers(0, candidates, n, dtype=np.int64),
            r=rng.integers(0, relations, n, dtype=np.int64),
            t=target.copy(),
            filter_source=np.asarray(["toy"]),
        )
print(args.output)
