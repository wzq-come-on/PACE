"""Train/evaluate the PACE controller on frozen expert scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from data import load_part
from pace import (Gate, FEATURE_NAMES_BY_COUNT, align_block, filter_block,
                  profile, fused_scores, ranks, negatives)
from selection import ALPHAS, choose_alpha, metric_dict


def blocks(args, split):
    for direction in ("head", "tail"):
        semantic, structural, contract = load_part(args.data_root, args.dataset, split, direction)
        for start in range(0, len(semantic), args.row_batch):
            stop = min(start + args.row_batch, len(semantic))
            k, l = [torch.from_numpy(np.array(x[start:stop], dtype=np.float32, copy=True)).to(args.device)
                    for x in (semantic, structural)]
            if not torch.isfinite(k).all() or not torch.isfinite(l).all():
                raise ValueError("Raw scores must be finite; supply exclusions through contract.npz")
            # Preserve the original filtered-feature protocol and operation order.
            filter_block(k, contract, start)
            filter_block(l, contract, start)
            k, l = align_block(k, l)
            if (torch.isfinite(k).sum(1) < 2).any():
                raise ValueError("Each query needs at least two legal candidates after filtering")
            target = torch.as_tensor(contract["target"][start:stop], dtype=torch.long, device=args.device)
            yield k, l, target


@torch.inference_mode()
def prepare_training(args):
    features, cubes = [], []
    for k, l, target in blocks(args, "train"):
        features.append(profile(k, l, args.features).cpu().numpy())
        cubes.append(np.stack([ranks(fused_scores(k, l, float(a)), target).cpu().numpy()
                               for a in ALPHAS]))
    raw, cube = np.concatenate(features), np.concatenate(cubes, axis=1)
    selected, selection = choose_alpha(cube, np.ones(cube.shape[1], dtype=bool))
    anchor = float(selected if args.anchor_alpha is None else args.anchor_alpha)
    radius = args.gate_radius
    if radius is None:
        radius = min(0.2, anchor, 1.0 - anchor)
    if not 0 <= anchor <= 1 or radius < 0:
        raise ValueError("Anchor must be in [0, 1] and radius must be nonnegative")
    # Do not silently clip user-specified sensitivity settings.
    # The original bounded parameterization is preserved even outside [0,1].
    # The released main table uses the structural-led negative rule: the
    # structural expert's top candidate, falling back to its runner-up when both
    # experts agree on top-1.  ``anchor_hard`` is the alternative rule that picks
    # the strongest legal non-target under the fixed anchor fusion.
    gaps = []
    for k, l, target in blocks(args, "train"):
        _, gap = negatives(k, l, target, anchor, args.negative_rule)
        gaps.append(gap.cpu().numpy())
    mean, std = raw.mean(0), raw.std(0)
    std[std < 1e-6] = 1.0
    return (raw - mean) / std, np.concatenate(gaps), mean, std, anchor, radius, selection


@torch.inference_mode()
def evaluate(args, model, mean, std, anchor, radius):
    all_ranks, all_alpha = [], []
    for k, l, target in blocks(args, "test"):
        raw = profile(k, l, args.features).cpu().numpy()
        x = torch.from_numpy((raw - mean) / std).to(args.device)
        alpha = model(x, anchor, radius, True)
        all_ranks.append(ranks(fused_scores(k, l, alpha[:, None]), target).cpu().numpy())
        all_alpha.append(alpha.cpu().numpy())
    return np.concatenate(all_ranks), np.concatenate(all_alpha)


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.row_batch < 1 or args.loss_temperature <= 0:
        raise ValueError("Epochs, batch sizes and loss temperature must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    x, d, mean, std, anchor, radius, selected = prepare_training(args)
    model = Gate(args.features, "bounded-mlp").to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xx, dd = torch.from_numpy(x), torch.from_numpy(d)
    generator = torch.Generator().manual_seed(args.seed)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for bi in torch.randperm(len(xx), generator=generator).split(args.batch_size):
            xb, db = xx[bi].to(args.device), dd[bi].to(args.device)
            alpha = model(xb, anchor, radius, True)
            gap = (1 - alpha) * db[:, 0] + alpha * db[:, 1]
            loss = nn.functional.softplus(-gap / args.loss_temperature).mean()
            loss = loss + args.anchor_penalty * ((alpha - anchor) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(bi)
        print(f"epoch={epoch + 1} loss={total / len(xx):.6f}", flush=True)
    model.eval()
    rank, alpha = evaluate(args, model, mean, std, anchor, radius)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {"dataset": args.dataset, "seed": args.seed,
              "features": list(FEATURE_NAMES_BY_COUNT[args.features]),
              "controller": [args.features, 16, 1],
              "parameters": sum(p.numel() for p in model.parameters()),
              "anchor_alpha": anchor,
              "gate_radius": radius, "negative_rule": args.negative_rule,
              "epochs": args.epochs, "batch_size": args.batch_size,
              "lr": args.lr, "weight_decay": args.weight_decay,
              "anchor_penalty": args.anchor_penalty, "loss_temperature": args.loss_temperature,
              "anchor_selection": "train" if args.anchor_alpha is None else "explicit",
              "training_selected_anchor_metrics": selected,
              "feature_protocol": "shared other-answer filter before alignment/profiling",
              "tie_policy": "pessimistic", "metrics_scale": "fraction"}
    torch.save({"state_dict": model.state_dict(), "feature_mean": torch.from_numpy(mean.copy()),
                "feature_std": torch.from_numpy(std.copy()), "config": config}, out / "controller.pt")
    np.save(out / "test_ranks.npy", rank)
    np.save(out / "test_alpha.npy", alpha)
    result = {"config": config, "metrics": metric_dict(rank)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--features", type=int, choices=(5, 7), default=5,
                   help="5 keeps the dispersion-free profile, 7 is the manuscript phi(q)")
    p.add_argument("--negative-rule", choices=("structural_historical", "anchor_hard"),
                   default="structural_historical",
                   help="structural_historical reproduces the paper's tables; "
                        "anchor_hard selects the hardest legal non-target under the anchor fusion")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/pace"))
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--row-batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--anchor-penalty", type=float, default=0.05)
    p.add_argument("--loss-temperature", type=float, default=0.2)
    p.add_argument("--anchor-alpha", type=float, help="If omitted, select on training queries")
    p.add_argument("--gate-radius", type=float, help="If omitted, min(0.2, anchor, 1-anchor)")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
