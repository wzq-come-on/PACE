"""Portable frozen-score input interface; no server-specific paths."""
from pathlib import Path
import numpy as np


def load_part(data_root, dataset, split, direction):
    folder = Path(data_root) / dataset / f"{split}_{direction}"
    semantic = np.load(folder / "semantic.npy", mmap_mode="r", allow_pickle=False)
    structural = np.load(folder / "structural.npy", mmap_mode="r", allow_pickle=False)
    with np.load(folder / "contract.npz", allow_pickle=False) as f:
        contract = {key: f[key] for key in f.files}
    if semantic.ndim != 2 or semantic.shape != structural.shape:
        raise ValueError(f"Score shapes differ at {folder}")
    n, candidates = semantic.shape
    if n == 0 or candidates < 2:
        raise ValueError("Need at least one query and two candidates")
    target = contract["target"]
    offsets, ids = contract["filter_offsets"], contract["filter_ids"]
    if target.shape != (n,) or offsets.shape != (n + 1,):
        raise ValueError("Contract row count does not match scores")
    if any(not np.issubdtype(v.dtype, np.integer) for v in (target, offsets, ids)):
        raise ValueError("Targets, filter offsets and filter IDs must be integers")
    if offsets[0] != 0 or offsets[-1] != len(ids) or np.any(np.diff(offsets) < 0):
        raise ValueError("Invalid filter offsets")
    if np.any(target < 0) or np.any(target >= candidates) or np.any(ids < 0) or np.any(ids >= candidates):
        raise ValueError("Candidate ID out of range")
    for row in range(n):
        if target[row] in ids[offsets[row]:offsets[row + 1]]:
            raise ValueError("The current target must not appear in the other-answer filter")
    return semantic, structural, contract
