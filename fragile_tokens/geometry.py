"""Static predictors from the embedding tables: where a token's rows sit relative to the glitch class.

Fragile tokens sit next to glitch tokens in embedding space. The proximity score (mean cosine to the
nearest glitch-class rows) separates fragile from stable tokens on every model with untied embeddings
in the paper; the projection onto a single glitch direction, the score earlier detectors use, does not
on every model. Both need no forward pass, so they rank a whole vocabulary in seconds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .types import FloatArray, IntArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Embeddings:
    """Input and output embedding tables as float32 CPU tensors; ``e_out`` has the final-norm gain
    folded in so that its rows are the ones the logits are read from."""

    e_in: torch.Tensor
    e_out: torch.Tensor
    tied: bool

    @property
    def vocab_size(self) -> int:
        return int(self.e_in.shape[0])


def unit(x: np.ndarray) -> np.ndarray:
    norms: np.ndarray = np.linalg.norm(x, axis=1, keepdims=True)
    out: np.ndarray = x / (norms + 1e-9)
    return out


def auc(score: Sequence[float] | np.ndarray, label: Sequence[bool] | np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney with average ranks for ties). NaN if one class is empty."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(label, dtype=np.bool_)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def glitch_projection(rows: np.ndarray, mean: np.ndarray, ref_rows: np.ndarray) -> FloatArray:
    """Projection of centred rows onto the glitch direction (mean glitch row minus the global mean)."""
    d = ref_rows.mean(axis=0) - mean
    d = d / (np.linalg.norm(d) + 1e-9)
    return np.asarray((rows - mean) @ d, dtype=np.float32)


def glitch_proximity(rows: np.ndarray, mean: np.ndarray, ref_rows: np.ndarray, k: int = 5,
                     self_ref_index: np.ndarray | None = None) -> FloatArray:
    """Mean cosine of each centred row to its ``k`` nearest glitch-class rows. ``self_ref_index[i]``
    gives, for rows that are themselves in the reference, their column in ``ref_rows`` (or -1), so a
    glitch row is not its own neighbour."""
    x = unit(rows - mean)
    r = unit(ref_rows - mean)
    k = min(k, r.shape[0])
    out = np.empty(x.shape[0], dtype=np.float32)
    step = 4096
    for s in range(0, x.shape[0], step):
        sim = x[s: s + step] @ r.T
        if self_ref_index is not None:
            idx = self_ref_index[s: s + step]
            own = np.where(idx >= 0)[0]
            sim[own, idx[own]] = -2.0
        out[s: s + step] = np.sort(sim, axis=1)[:, -k:].mean(axis=1)
    return out


@dataclass(frozen=True)
class GeometryScores:
    """Per-token static scores for the scanned tokens, and how well each separates fragile from stable
    tokens among those that pass the gate."""

    token_ids: IntArray
    projection: FloatArray
    proximity_out: FloatArray
    proximity_in: FloatArray
    reference_ids: IntArray
    auc_projection: float
    auc_proximity_out: float
    auc_proximity_in: float
    auc_token_id: float
    tied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_ids": self.reference_ids.tolist(),
            "tied_embeddings": self.tied,
            "auc_fragile_vs_stable": {"projection": self.auc_projection, "proximity_out": self.auc_proximity_out,
                                      "proximity_in": self.auc_proximity_in, "token_id": self.auc_token_id},
            "tokens": [{"token_id": int(t), "projection": float(p), "proximity_out": float(po), "proximity_in": float(pi)}
                       for t, p, po, pi in zip(self.token_ids, self.projection, self.proximity_out, self.proximity_in)],
        }


def geometry_scores(emb: Embeddings, token_ids: IntArray, reference_ids: IntArray, fragile: np.ndarray,
                    stable: np.ndarray, k: int = 5) -> GeometryScores:
    """Score the scanned tokens against a glitch reference class (vocabulary ids)."""
    e_in = emb.e_in.numpy()
    e_out = emb.e_out.numpy()
    mu_in = e_in.mean(axis=0)
    mu_out = e_out.mean(axis=0)
    ref_pos = {int(t): j for j, t in enumerate(reference_ids)}
    self_idx = np.asarray([ref_pos.get(int(t), -1) for t in token_ids], dtype=np.int64)
    rows_in, rows_out = e_in[token_ids], e_out[token_ids]
    proj = glitch_projection(rows_in, mu_in, e_in[reference_ids])
    prox_out = glitch_proximity(rows_out, mu_out, e_out[reference_ids], k, self_idx)
    prox_in = glitch_proximity(rows_in, mu_in, e_in[reference_ids], k, self_idx)
    mask = fragile | stable
    y = fragile[mask]
    scores = GeometryScores(token_ids=token_ids, projection=proj, proximity_out=prox_out, proximity_in=prox_in,
                            reference_ids=reference_ids, auc_projection=auc(proj[mask], y),
                            auc_proximity_out=auc(prox_out[mask], y), auc_proximity_in=auc(prox_in[mask], y),
                            auc_token_id=auc(token_ids[mask].astype(np.float64), y), tied=emb.tied)
    logger.info("geometry AUC fragile vs stable: projection %.3f, proximity(out) %.3f, proximity(in) %.3f, token id %.3f",
                scores.auc_projection, scores.auc_proximity_out, scores.auc_proximity_in, scores.auc_token_id)
    return scores


def rank_vocabulary(emb: Embeddings, reference_ids: IntArray, k: int = 5, space: str = "out") -> FloatArray:
    """Proximity to the glitch class for every row of the vocabulary: the static screen that needs no
    forward pass. Higher means closer to the glitch class."""
    table = emb.e_out if space == "out" else emb.e_in
    x = table.numpy()
    mu = x.mean(axis=0)
    self_idx = np.full(x.shape[0], -1, dtype=np.int64)
    self_idx[reference_ids] = np.arange(len(reference_ids))
    return glitch_proximity(x, mu, x[reference_ids], k, self_idx)


__all__ = ["Embeddings", "GeometryScores", "auc", "geometry_scores", "glitch_projection", "glitch_proximity",
           "rank_vocabulary", "unit"]
