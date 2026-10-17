#!/usr/bin/env python3
"""Export frozen LBMKGC (structural expert) scores into the PACE score layout.

LBMKGC scores every candidate entity from its multimodal embedding bank.  This
exporter runs the frozen checkpoint over the queries of one split and writes

    {output_root}/{dataset}/{split}_head/structural.npy
    {output_root}/{dataset}/{split}_tail/structural.npy

with shape ``[queries, entities]``, float32, higher is better.  Columns are
reordered into the semantic expert's entity-id order so that both experts share
one candidate indexing.

Requirements
------------
``structure_expert.py`` (shipped next to this file) loads the official
AdvRelRotatE wrapper.  The LBMKGC checkpoints are not redistributed.

Entity alignment
----------------
The LBMKGC benchmark vocabularies contain only counts, so the correspondence
between LBMKGC entity ids and semantic-expert entity ids is recovered from a
named vocabulary snapshot (``--reference-data``): the same word/URI list that
was used when the checkpoints were prepared.  Use ``--no-alignment`` only when
both experts already share the identical id order.

Usage
-----
    python experts/lbmkgc/export_lbmkgc_scores.py \
        --lbm-root /path/to/LBMKGC \
        --reference-data /path/to/vocab/DB15K \
        --data-dir /path/to/DB15K \
        --checkpoint /path/to/LBMKGC/Checkpoints/LBMKGC/DB15K-checkpoint.zip \
        --dataset DB15K --split test --output-root data
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structure_expert import build_lbm_teacher  # noqa: E402

SPLITS = ("train", "valid", "test")


def read_rows(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_triples(folder: Path, split: str) -> np.ndarray:
    raw = np.loadtxt(folder / f"{split}2id.txt", dtype=np.int64, ndmin=2)
    # OpenKE order is (head, tail, relation).
    return raw[:, [0, 2, 1]].copy()


def read_count(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").split()[0])


def normalize_name(name: str) -> str:
    return " ".join(name.replace("_", " ").split()).lower()


def read_vocab(path: Path, dataset: str, entity: bool) -> dict[str, int]:
    """Read a named vocabulary, tolerating the dataset-specific formats."""
    result: dict[str, int] = {}
    for fields in read_rows(path):
        if len(fields) == 1:
            continue  # count-only OpenKE file
        idx = int(fields[-1])
        name_fields = fields[:-1]
        if dataset == "MKG-W" and entity and len(name_fields) >= 2 and name_fields[0].isdigit():
            name_fields = name_fields[1:]
        name = normalize_name(" ".join(name_fields))
        if name in result:
            raise ValueError(f"duplicate normalized name {name!r} in {path}")
        result[name] = idx
    return result


def load_entity_alignment(paper_data: Path, reference_data: Path, lbm_data: Path,
                          dataset: str):
    """Return (semantic_id -> lbmkgc_id, lbmkgc_id -> semantic_id, metadata)."""
    paper_entities = read_vocab(paper_data / "entity2id.txt", dataset, entity=True)
    reference_entities = read_vocab(reference_data / "entity2id.txt", dataset, entity=True)
    if set(paper_entities) != set(reference_entities):
        raise ValueError(f"{dataset}: entity vocabularies differ")
    if read_count(lbm_data / "entity2id.txt") != len(reference_entities):
        raise ValueError(f"{dataset}: LBMKGC entity count does not match the reference vocabulary")

    paper_to_lbm = np.empty(len(paper_entities), dtype=np.int64)
    for name, paper_id in paper_entities.items():
        paper_to_lbm[paper_id] = reference_entities[name]
    if sorted(paper_to_lbm.tolist()) != list(range(len(reference_entities))):
        raise ValueError(f"{dataset}: entity mapping is not bijective")
    metadata = {"semantic_entities": len(paper_entities),
                "lbmkgc_entities": len(reference_entities),
                "entity_permutation": bool(np.any(paper_to_lbm != np.arange(len(paper_to_lbm))))}
    return paper_to_lbm, np.argsort(paper_to_lbm).astype(np.int64), metadata


def score_split(*, triples, n_entity, n_relation, lbm_root, checkpoint, dataset,
                column_order, device, query_batch_size, candidate_chunk_size,
                head_path, tail_path) -> None:
    teacher = build_lbm_teacher(lbm_root=lbm_root, ckpt=checkpoint, n_ent=n_entity,
                                n_rel=n_relation, dataset_name=dataset, device=device)
    teacher.eval()
    order = torch.as_tensor(column_order, dtype=torch.long, device=device)

    def open_memmap(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npy")
        if temporary.exists():
            temporary.unlink()
        return np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                         shape=(len(triples), n_entity)), temporary

    head_mmap, head_tmp = open_memmap(head_path)
    tail_mmap, tail_tmp = open_memmap(tail_path)
    begin = time.time()
    try:
        for start in range(0, len(triples), query_batch_size):
            stop = min(start + query_batch_size, len(triples))
            block = triples[start:stop]
            h = torch.as_tensor(column_order[block[:, 0]], dtype=torch.long, device=device)
            r = torch.as_tensor(block[:, 1], dtype=torch.long, device=device)
            t = torch.as_tensor(column_order[block[:, 2]], dtype=torch.long, device=device)
            with torch.inference_mode():
                both = teacher.score_full_bank_doubled(h, r, t,
                                                       chunk_size=candidate_chunk_size)
                # The wrapper returns tail rows first, then head rows, with
                # columns in LBMKGC id order; reorder columns to the shared ids.
                both = both[:, order].float().cpu().numpy()
            tail_mmap[start:stop] = both[: stop - start]
            head_mmap[start:stop] = both[stop - start:]
            if stop == len(triples) or (start // query_batch_size + 1) % 10 == 0:
                print(f"scored {stop}/{len(triples)} queries "
                      f"({(time.time() - begin) / 60:.1f} min)", flush=True)
    except Exception:
        del head_mmap, tail_mmap
        for temporary in (head_tmp, tail_tmp):
            if temporary.exists():
                temporary.unlink()
        raise
    for mmap, temporary, path in ((head_mmap, head_tmp, head_path),
                                  (tail_mmap, tail_tmp, tail_path)):
        mmap.flush()
        del mmap
        os.replace(temporary, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lbm-root", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="queried split in OpenKE text format")
    p.add_argument("--reference-data", type=Path,
                   help="named vocabulary used when the checkpoints were prepared")
    p.add_argument("--no-alignment", action="store_true",
                   help="experts already share the identical entity-id order")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", choices=SPLITS, default="test")
    p.add_argument("--output-root", type=Path, default=Path("data"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--query-batch-size", type=int, default=32)
    p.add_argument("--candidate-chunk-size", type=int, default=4096)
    args = p.parse_args()

    device = torch.device(args.device)
    folder = args.output_root / args.dataset
    triples = read_triples(args.data_dir, args.split)
    lbm_data = args.lbm_root / "benchmarks" / args.dataset

    if args.no_alignment:
        n_entity = read_count(lbm_data / "entity2id.txt")
        column_order = np.arange(n_entity, dtype=np.int64)
        print(f"no alignment: {n_entity} entities in native LBMKGC order", flush=True)
    else:
        if args.reference_data is None:
            raise SystemExit("--reference-data is required unless --no-alignment is given")
        paper_to_lbm, _, metadata = load_entity_alignment(
            args.data_dir, args.reference_data, lbm_data, args.dataset)
        n_entity = len(paper_to_lbm)
        column_order = paper_to_lbm
        print(f"entity alignment: {metadata}", flush=True)

    score_split(triples=triples, n_entity=n_entity,
                n_relation=read_count(args.data_dir / "relation2id.txt"),
                lbm_root=args.lbm_root, checkpoint=args.checkpoint, dataset=args.dataset,
                column_order=column_order, device=device,
                query_batch_size=args.query_batch_size,
                candidate_chunk_size=args.candidate_chunk_size,
                head_path=folder / f"{args.split}_head" / "structural.npy",
                tail_path=folder / f"{args.split}_tail" / "structural.npy")
    print(f"wrote {folder}/{args.split}_{{head,tail}}/structural.npy", flush=True)


if __name__ == "__main__":
    main()
