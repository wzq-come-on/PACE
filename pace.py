"""PACE controller and score operations.

Expert order is semantic, structural. Alpha is the structural weight.
The five input features retain the training code's original ordering.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

FEATURE_NAMES = (
    "semantic_margin", "structural_margin", "margin_difference",
    "top1_agreement", "top1_score_difference",
)

class Gate(nn.Module):
    def __init__(self, d, architecture="bounded-mlp"):
        super().__init__()
        self.architecture = architecture
        if architecture in ("linear", "bounded-linear"):
            self.net=nn.Sequential(nn.Linear(d,1))
        elif architecture == "bounded-mlp":
            self.net=nn.Sequential(nn.Linear(d,16),nn.ReLU(),nn.Linear(16,1))
        else:
            raise ValueError(f"unknown architecture: {architecture}")
    def forward(self,x,anchor,radius,bounded=True):
        raw=self.net(x).squeeze(-1)
        return anchor+radius*(torch.tanh(raw) if bounded else raw)

def align_block(k, l):
    """Query-wise z-score alignment; filtered entries are excluded."""
    aligned=[]
    for x in (k,l):
        finite=torch.isfinite(x)
        count=finite.sum(dim=1).clamp_min(1)
        safe=torch.where(finite,x,torch.zeros_like(x))
        mean=safe.sum(dim=1)/count
        var=torch.where(finite,(x-mean[:,None])**2,torch.zeros_like(x)).sum(dim=1)/count
        z=(x-mean[:,None])/torch.sqrt(var.clamp_min(1e-12))[:,None]
        aligned.append(torch.where(finite,z,torch.full_like(z,-torch.inf)))
    return aligned[0],aligned[1]

def profile(k, l):
    kv, ki = k.topk(2, 1)
    lv, li = l.topk(2, 1)
    km, lm = kv[:, 0] - kv[:, 1], lv[:, 0] - lv[:, 1]
    return torch.stack((km, lm, (km - lm).abs(), (ki[:, 0] == li[:, 0]).float(),
                        (kv[:, 0] - lv[:, 0]).abs()), 1)

def fused_scores(k, l, alpha):
    # Avoid 0 * -inf at alpha endpoints and exclude masks for extrapolation too.
    valid = torch.isfinite(k) & torch.isfinite(l)
    s = (1 - alpha) * k + alpha * l
    return torch.where(valid, s, torch.full_like(s, -torch.inf))

def ranks(scores, targets):
    target = scores.gather(1, targets[:, None])
    if not torch.isfinite(target).all():
        raise ValueError("Target score is not finite")
    ids = torch.arange(scores.shape[1], device=scores.device)[None, :]
    return (1 + (scores > target).sum(1)
            + ((scores == target) & (ids != targets[:, None])).sum(1))

def filter_block(x, contract, start):
    """Same filter as release, applied by one indexed assignment per block."""
    offsets = contract["filter_offsets"]
    stop = start + len(x)
    lengths = np.diff(offsets[start : stop + 1]).astype(np.int64)
    ids = contract["filter_ids"][int(offsets[start]) : int(offsets[stop])]
    if len(ids):
        rows = np.repeat(np.arange(len(x), dtype=np.int64), lengths)
        x[torch.as_tensor(rows, device=x.device),
          torch.as_tensor(ids, dtype=torch.long, device=x.device)] = -torch.inf

def negatives(k, l, target, anchor, rule):
    kn, ln = k.clone(), l.clone()
    kn.scatter_(1, target[:, None], -torch.inf)
    ln.scatter_(1, target[:, None], -torch.inf)
    if rule == 'anchor_hard':
        best, neg = fused_scores(kn, ln, anchor).max(1, keepdim=True)
        assert torch.isfinite(best).all()
    elif rule == 'structural_historical':
        ki, li = kn.topk(2, 1).indices, ln.topk(2, 1).indices
        neg = torch.where(ki[:, :1] == li[:, :1], li[:, 1:2], li[:, :1])
    else:
        raise ValueError(rule)
    assert (neg[:, 0] != target).all()
    assert torch.isfinite(k.gather(1, neg)).all() and torch.isfinite(l.gather(1, neg)).all()
    gaps = torch.cat((k.gather(1, target[:, None]) - k.gather(1, neg),
                      l.gather(1, target[:, None]) - l.gather(1, neg)), 1)
    return neg[:, 0], gaps


@torch.inference_mode()
def predict_scores(model, semantic, structural, mean, std, anchor, radius):
    """Fuse a shared candidate set without targets or ground-truth labels.

    Inputs are [queries, candidates] tensors, with identical candidate indexing.
    Any pre-existing candidate mask must be shared by both tensors.
    """
    model.eval()
    k, l = align_block(semantic, structural)
    features = profile(k, l)
    mean = torch.as_tensor(mean, dtype=features.dtype, device=features.device)
    std = torch.as_tensor(std, dtype=features.dtype, device=features.device)
    alpha = model((features - mean) / std, anchor, radius, True)
    return fused_scores(k, l, alpha[:, None]), alpha
