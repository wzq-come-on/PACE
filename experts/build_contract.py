#!/usr/bin/env python3
"""Build the PACE query contracts from a knowledge graph in OpenKE text format.

PACE does not read the graph.  It only needs, per query, the correct entity and
the list of *other* known correct entities that must be filtered out before
ranking.  This script derives exactly those two things and writes the
``contract.npz`` files that ``data.py`` consumes:

    data/{dataset}/{split}_{direction}/contract.npz

Query order is all head queries followed by all tail queries, matching the
row order of the score matrices produced by the exporters.

The filter policy follows the standard filtered-ranking protocol:

``--filter-source all``   (default, evaluation)
    A candidate is filtered if it appears as the same (h, r, ?) or (?, r, t) in
    train, valid or test.
``--filter-source train``  (in-sample fitting)
    Only training facts are used, so no validation or test fact leaks into the
    features or the negative selection.

Usage
-----
    python experts/build_contract.py --data-dir data/DB15K --dataset DB15K --output-root data
    python experts/build_contract.py --data-dir data/DB15K --dataset DB15K \
        --output-root data --split train --filter-source train
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

SPLITS = ("train", "valid", "test")


def read_triples(folder: Path, split: str) -> np.ndarray:
    """Read ``{split}2id.txt`` and return rows as (h, r, t).

    OpenKE files are ``head tail relation``; PACE works with (h, r, t).
    """
    path = folder / f"{split}2id.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if raw.shape[1] != 3:
        raise ValueError(f"Expected three columns in {path}, got {raw.shape}")
    return raw[:, [0, 2, 1]].copy()


def read_count(folder: Path, name: str) -> int:
    """Vocabulary size, for both OpenKE count files and named ``id name`` files."""
    path = folder / name
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"empty vocabulary file {path}")
    if len(rows) == 1 and len(rows[0]) == 1:
        return int(rows[0][0])
    return len(rows)


def build_lookup(folder: Path, splits) -> dict[tuple[int, int, int], list[int]]:
    values: dict[tuple[int, int, int], set] = defaultdict(set)
    for split in splits:
        for h, r, t in read_triples(folder, split):
            values[(int(h), int(r), -1)].add(int(t))
            values[(-1, int(r), int(t))].add(int(h))
    return {key: sorted(items) for key, items in values.items()}


def directional_filters(triples: np.ndarray, lookup, direction: str):
    offsets = [0]
    flat: list[int] = []
    targets: list[int] = []
    for h, r, t in triples:
        if direction == "tail":
            target = int(t)
            filtered = lookup.get((int(h), int(r), -1), [])
        elif direction == "head":
            target = int(h)
            filtered = lookup.get((-1, int(r), int(t)), [])
        else:
            raise ValueError(direction)
        flat.extend(idx for idx in filtered if idx != target)
        offsets.append(len(flat))
        targets.append(target)
    return (np.asarray(offsets, dtype=np.int64),
            np.asarray(flat, dtype=np.int64),
            np.asarray(targets, dtype=np.int64))


def write_contract(path: Path, dataset: str, split: str, direction: str,
                   triples: np.ndarray, offsets, flat, targets,
                   n_entity: int, n_relation: int, filter_source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a failed run never leaves a half-written contract.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    query_ids = np.asarray(
        [f"{dataset}:{split}:{direction}:{i}" for i in range(len(triples))],
        dtype=f"<U{max(32, len(dataset) + len(direction) + 24)}",
    )
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            query_id=query_ids,
            h=triples[:, 0], r=triples[:, 1], t=triples[:, 2],
            target=targets,
            filter_offsets=offsets,
            filter_ids=flat,
            num_entities=np.asarray([n_entity], dtype=np.int64),
            num_relations=np.asarray([n_relation], dtype=np.int64),
            filter_source=np.asarray([filter_source]),
        )
    os.replace(temporary, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="folder holding {split}2id.txt, entity2id.txt, relation2id.txt")
    p.add_argument("--dataset", required=True, help="dataset name, used for the output folder")
    p.add_argument("--split", choices=SPLITS, default="test",
                   help="which query split to build (default: test)")
    p.add_argument("--directions", nargs="+", choices=("head", "tail"),
                   default=("head", "tail"))
    p.add_argument("--filter-source", choices=("all", "train"), default="all")
    p.add_argument("--output-root", type=Path, default=Path("data"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    sources = SPLITS if args.filter_source == "all" else ("train",)
    lookup = build_lookup(args.data_dir, sources)
    triples = read_triples(args.data_dir, args.split)
    n_entity = read_count(args.data_dir, "entity2id.txt")
    n_relation = read_count(args.data_dir, "relation2id.txt")

    for direction in args.directions:
        target_folder = args.output_root / args.dataset / f"{args.split}_{direction}"
        path = target_folder / "contract.npz"
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite to rebuild it")
        offsets, flat, targets = directional_filters(triples, lookup, direction)
        if np.any(targets < 0) or np.any(targets >= n_entity):
            raise ValueError(f"{args.split}/{direction}: target id out of range")
        write_contract(path, args.dataset, args.split, direction, triples,
                       offsets, flat, targets, n_entity, n_relation, args.filter_source)
        print(f"{path}: {len(triples)} queries, {len(flat)} filtered entries "
              f"(filter source: {args.filter_source})", flush=True)


if __name__ == "__main__":
    main()
