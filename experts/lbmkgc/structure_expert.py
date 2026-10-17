"""Frozen LBMKGC scorer adapter.

Loads the official AdvRelRotatE checkpoint and exposes full-entity scoring for
a batch of queries.  Both the checkpoint and the multimodal embedding tables
are frozen; nothing in this module trains.
"""
from __future__ import annotations

import math
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _as_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def load_torch_state(path) -> dict:
    path = Path(path)
    if path.is_dir():
        for name in (
            "checkpoint.pt",
            "pytorch_model.bin",
            "model.pt",
        ):
            cand = path / name
            if cand.is_file():
                path = cand
                break
        else:
            zips = sorted(path.glob("*.zip")) + sorted(path.glob("*.pt")) + sorted(path.glob("*.pth"))
            if not zips:
                raise FileNotFoundError(f"no checkpoint under {path}")
            path = zips[0]

    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        obj = None
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                names = [n for n in zf.namelist() if n.endswith((".pt", ".pth", ".bin", ".pkl"))]
                if not names:
                    # some OpenKE dumps are the zip itself
                    raise
                with zf.open(names[0]) as f:
                    obj = torch.load(f, map_location="cpu")
        if obj is None:
            raise
    obj = _as_state_dict(obj)
    if not isinstance(obj, dict):
        raise RuntimeError(f"unexpected checkpoint type {type(obj)} from {path}")
    # strip common prefixes
    cleaned = {}
    for k, v in obj.items():
        nk = k
        for pref in ("module.", "model.", "lbm.", "teacher."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v
    return cleaned


class StructureExpert(nn.Module):
    """Geometry-only RotatE. Used as the tiny residual expert, not the teacher."""

    def __init__(
        self,
        n_ent: int,
        n_rel: int,
        dim: int = 250,
        margin: float = 6.0,
        epsilon: float = 2.0,
        use_multimodal: bool = False,
        img_dim: Optional[int] = None,
        text_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.dim = dim
        self.dim_e = dim * 2
        self.dim_r = dim
        self.use_multimodal = use_multimodal
        self.epsilon = epsilon

        self.ent_embeddings = nn.Embedding(n_ent, self.dim_e)
        self.rel_embeddings = nn.Embedding(n_rel, self.dim_r)
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(margin + epsilon) / self.dim_e]), requires_grad=False
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(margin + epsilon) / self.dim_r]), requires_grad=False
        )
        self.margin = nn.Parameter(torch.Tensor([margin]), requires_grad=False)
        self.pi_const = nn.Parameter(torch.Tensor([math.pi]), requires_grad=False)

        nn.init.uniform_(
            self.ent_embeddings.weight,
            -self.ent_embedding_range.item(),
            self.ent_embedding_range.item(),
        )
        nn.init.uniform_(
            self.rel_embeddings.weight,
            -self.rel_embedding_range.item(),
            self.rel_embedding_range.item(),
        )

        self.img_embeddings = None
        self.text_embeddings = None
        self.img_proj = None
        self.text_proj = None
        self.ent_attn = None
        self.rel_gate = None
        if use_multimodal:
            if img_dim is None or text_dim is None:
                raise ValueError("img_dim and text_dim required when use_multimodal=True")
            self.img_embeddings = nn.Embedding(n_ent, img_dim)
            self.text_embeddings = nn.Embedding(n_ent, text_dim)
            self.img_embeddings.weight.requires_grad_(False)
            self.text_embeddings.weight.requires_grad_(False)
            self.img_proj = nn.Sequential(
                nn.Linear(img_dim, self.dim_e),
                nn.ReLU(),
                nn.Linear(self.dim_e, self.dim_e),
            )
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, self.dim_e),
                nn.ReLU(),
                nn.Linear(self.dim_e, self.dim_e),
            )
            self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
            self.rel_gate = nn.Embedding(n_rel, 1)

    def get_joint_entity(self, idx: torch.Tensor, rel_idx: torch.Tensor) -> torch.Tensor:
        es = self.ent_embeddings(idx)
        if not self.use_multimodal:
            return es
        ev = self.img_proj(self.img_embeddings(idx))
        et = self.text_proj(self.text_embeddings(idx))
        stacked = torch.stack((ev, et), dim=-2)
        scores = self.ent_attn(torch.tanh(stacked)).squeeze(-1)
        attn = torch.softmax(scores, dim=-1)
        mm = (attn.unsqueeze(-1) * stacked).sum(dim=-2)
        rg = torch.sigmoid(self.rel_gate(rel_idx))
        while rg.dim() < es.dim():
            rg = rg.unsqueeze(-1)
        return rg * es + (1.0 - rg) * mm

    def rotate_score(self, h, r, t, mode: str = "tail_batch") -> torch.Tensor:
        re_h, im_h = torch.chunk(h, 2, dim=-1)
        re_t, im_t = torch.chunk(t, 2, dim=-1)
        phase = r / (self.rel_embedding_range.to(r.device) / self.pi_const.to(r.device))
        re_r, im_r = torch.cos(phase), torch.sin(phase)
        target_dim = max(re_h.dim(), re_r.dim(), re_t.dim())
        while re_r.dim() < target_dim:
            re_r = re_r.unsqueeze(-2)
            im_r = im_r.unsqueeze(-2)
        while re_h.dim() < target_dim:
            re_h = re_h.unsqueeze(-2)
            im_h = im_h.unsqueeze(-2)
        while re_t.dim() < target_dim:
            re_t = re_t.unsqueeze(-2)
            im_t = im_t.unsqueeze(-2)
        if mode == "head_batch":
            re_score = re_r * re_t + im_r * im_t
            im_score = re_r * im_t - im_r * re_t
            re_score = re_score - re_h
            im_score = im_score - im_h
        else:
            re_score = re_h * re_r - im_h * im_r
            im_score = re_h * im_r + im_h * re_r
            re_score = re_score - re_t
            im_score = im_score - im_t
        dist = torch.stack([re_score, im_score], dim=0).norm(dim=0).sum(dim=-1)
        return self.margin.to(dist.device) - dist

    def score_pair_batch(self, h, r, ents, mode: str = "tail_batch") -> torch.Tensor:
        B, K = ents.shape
        if mode == "tail_batch":
            he = self.get_joint_entity(h, r)
            r_exp = r.unsqueeze(-1).expand(B, K)
            te = self.get_joint_entity(ents, r_exp)
            re = self.rel_embeddings(r)
            return self.rotate_score(he, re, te, mode="tail_batch")
        te = self.get_joint_entity(h, r)
        r_exp = r.unsqueeze(-1).expand(B, K)
        he = self.get_joint_entity(ents, r_exp)
        re = self.rel_embeddings(r)
        return self.rotate_score(he, re, te, mode="head_batch")

    def score_full_bank_doubled(self, h, r, t, chunk_size: int = 4096) -> torch.Tensor:
        device = self.ent_embeddings.weight.device
        h = h.to(device)
        r = r.to(device)
        t = t.to(device)
        B = h.shape[0]
        n = self.n_ent
        out_tail = h.new_empty((B, n), dtype=self.ent_embeddings.weight.dtype)
        out_head = h.new_empty((B, n), dtype=self.ent_embeddings.weight.dtype)
        he_all = self.get_joint_entity(h, r)
        te_all = self.get_joint_entity(t, r)
        re_all = self.rel_embeddings(r)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            cands = torch.arange(start, end, device=device)
            C = cands.shape[0]
            r_exp = r.unsqueeze(1).expand(B, C)
            c_exp = cands.unsqueeze(0).expand(B, C)
            if self.use_multimodal:
                te = self.get_joint_entity(c_exp, r_exp)
                he_c = self.get_joint_entity(c_exp, r_exp)
            else:
                ce = self.ent_embeddings(cands)
                te = ce.unsqueeze(0).expand(B, C, -1)
                he_c = te
            he = he_all.unsqueeze(1).expand(B, C, -1)
            re = re_all.unsqueeze(1).expand(B, C, -1)
            te_f = te_all.unsqueeze(1).expand(B, C, -1)
            out_tail[:, start:end] = self.rotate_score(he, re, te, mode="tail_batch")
            out_head[:, start:end] = self.rotate_score(he_c, re, te_f, mode="head_batch")
        return torch.cat([out_tail, out_head], dim=0)

    def score_all_entities(self, h, r, t, chunk_size: int = 4096, include_head: bool = True):
        full = self.score_full_bank_doubled(h, r, t, chunk_size=chunk_size)
        if include_head:
            return full
        return full[: h.shape[0]]


class ResidualExpert(nn.Module):
    def __init__(self, n_ent: int, n_rel: int, dim: int = 100, margin: float = 6.0):
        super().__init__()
        self.inner = StructureExpert(
            n_ent=n_ent, n_rel=n_rel, dim=dim, margin=margin, use_multimodal=False
        )

    @property
    def ent_embeddings(self):
        return self.inner.ent_embeddings

    def score_pair_batch(self, h, r, ents, mode="tail_batch"):
        return self.inner.score_pair_batch(h, r, ents, mode=mode)

    def score_full_bank_doubled(self, h, r, t, chunk_size=4096):
        return self.inner.score_full_bank_doubled(h, r, t, chunk_size=chunk_size)

    def score_all_entities(self, h, r, t, chunk_size=4096, include_head=True):
        return self.inner.score_all_entities(h, r, t, chunk_size=chunk_size, include_head=include_head)


class LBMTeacher(nn.Module):
    """Thin wrapper around the official AdvRelRotatE. Scores are frozen."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.n_ent = int(model.ent_tot)
        self.n_rel = int(model.rel_tot)
        self.dim = int(model.dim_r)
        self.ent_embeddings = model.ent_embeddings

    def train(self, mode: bool = True):
        # teacher stays in eval; dropout / running stats must not change
        return super().train(False)

    def _score_flat(self, batch_h, batch_t, batch_r, mode: str) -> torch.Tensor:
        data = {
            "batch_h": batch_h,
            "batch_t": batch_t,
            "batch_r": batch_r,
            "mode": mode,
        }
        return self.model(data).reshape(-1)

    @torch.no_grad()
    def score_pair_batch(self, h, r, ents, mode: str = "tail_batch") -> torch.Tensor:
        B, K = ents.shape
        rr = r.unsqueeze(1).expand(B, K).reshape(-1)
        ee = ents.reshape(-1)
        if mode == "tail_batch":
            hh = h.unsqueeze(1).expand(B, K).reshape(-1)
            scores = self._score_flat(hh, ee, rr, "tail_batch")
        else:
            tt = h.unsqueeze(1).expand(B, K).reshape(-1)
            scores = self._score_flat(ee, tt, rr, "head_batch")
        return scores.view(B, K)

    @torch.no_grad()
    def score_full_bank_doubled(self, h, r, t, chunk_size: int = 2048) -> torch.Tensor:
        device = h.device
        B = h.shape[0]
        n = self.n_ent
        out_tail = h.new_empty((B, n), dtype=torch.float32)
        out_head = h.new_empty((B, n), dtype=torch.float32)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            cands = torch.arange(start, end, device=device)
            C = int(cands.numel())
            hh = h.unsqueeze(1).expand(B, C).reshape(-1)
            tt = t.unsqueeze(1).expand(B, C).reshape(-1)
            rr = r.unsqueeze(1).expand(B, C).reshape(-1)
            cc = cands.unsqueeze(0).expand(B, C).reshape(-1)
            out_tail[:, start:end] = self._score_flat(hh, cc, rr, "tail_batch").view(B, C)
            out_head[:, start:end] = self._score_flat(cc, tt, rr, "head_batch").view(B, C)
        return torch.cat([out_tail, out_head], dim=0)

    def score_all_entities(self, h, r, t, chunk_size: int = 2048, include_head: bool = True):
        full = self.score_full_bank_doubled(h, r, t, chunk_size=chunk_size)
        if include_head:
            return full
        return full[: h.shape[0]]


def build_lbm_teacher(
    lbm_root,
    ckpt,
    n_ent: int,
    n_rel: int,
    dataset_name: str,
    device,
) -> LBMTeacher:
    lbm_root = Path(lbm_root)
    if str(lbm_root) not in sys.path:
        sys.path.insert(0, str(lbm_root))
    from mmkgc.module.model import AdvRelRotatE

    state = load_torch_state(ckpt)
    if "ent_embeddings.weight" not in state:
        raise RuntimeError(f"LBM checkpoint missing ent_embeddings.weight; keys={list(state)[:20]}")
    dim = int(state["ent_embeddings.weight"].shape[1] // 2)
    margin = float(state["margin"].reshape(-1)[0]) if "margin" in state else 6.0

    vis = lbm_root / "embeddings" / f"{dataset_name}-visual.pth"
    txt = lbm_root / "embeddings" / f"{dataset_name}-textual.pth"
    image = torch.load(vis, map_location="cpu")
    text = torch.load(txt, map_location="cpu")
    if hasattr(image, "shape") and int(image.shape[0]) != n_ent:
        raise RuntimeError(f"visual emb {tuple(image.shape)} vs n_ent={n_ent}")
    if hasattr(text, "shape") and int(text.shape[0]) != n_ent:
        raise RuntimeError(f"textual emb {tuple(text.shape)} vs n_ent={n_ent}")

    model = AdvRelRotatE(
        ent_tot=n_ent,
        rel_tot=n_rel,
        dim=dim,
        margin=margin,
        epsilon=2.0,
        img_emb=image.float() if torch.is_tensor(image) else image,
        text_emb=text.float() if torch.is_tensor(text) else text,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    # adv_scores may be missing in some dumps; embeddings + gate must be present
    must = [
        "ent_embeddings.weight",
        "rel_embeddings.weight",
        "img_proj.0.weight",
        "text_proj.0.weight",
        "ent_attn.weight",
        "rel_gate.weight",
    ]
    absent = [k for k in must if k not in state]
    if absent:
        raise RuntimeError(f"LBM checkpoint missing teacher keys: {absent}")
    if missing:
        print(f"[lbmkgc] load missing={list(missing)} unexpected={list(unexpected)}", flush=True)
    else:
        print(f"[lbmkgc] teacher loaded dim={dim} margin={margin} unexpected={list(unexpected)}", flush=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    teacher = LBMTeacher(model)
    teacher.to(device)
    teacher.eval()
    return teacher
