"""Token classes, failure modes and the label-free glitch reference class."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

from .config import ScanConfig
from .contexts import Context
from .scoring import FragilityMatrix, is_scannable
from .types import BoolArray, FloatArray, IntArray, TokenizerLike


class TokenClass(str, Enum):
    """Where a token sits once the probe and the matrix are both known.

    ``glitch``: fails the probe and fails in at least ``glitch_context_share`` of contexts.
    ``fragile``: passes the probe, fails in at least ``fragile_threshold`` of contexts. These are the
        tokens isolation-based detectors miss.
    ``intermittent``: passes the probe, fails somewhere but below the fragile threshold.
    ``stable``: passes the probe and never fails.
    ``probe_only``: fails the probe but copies in context (at most ``fragile_threshold`` of contexts fail);
        a probe-format failure, not a defective token.
    ``ungated``: fails the probe with mixed behaviour in context.
    """

    GLITCH = "glitch"
    FRAGILE = "fragile"
    INTERMITTENT = "intermittent"
    STABLE = "stable"
    PROBE_ONLY = "probe_only"
    UNGATED = "ungated"


class FailureMode(str, Enum):
    """What the model emitted at the copy slot instead of the token.

    ``deletion``: the next context word, as if the slot were empty. ``truncation``: a proper prefix of
    the token. ``substitution``: anything else (the paper's "translation" is a human reading of a
    substitution and is not decided here). ``correct``: the token itself.
    """

    CORRECT = "correct"
    DELETION = "deletion"
    TRUNCATION = "truncation"
    SUBSTITUTION = "substitution"


@dataclass(frozen=True)
class WorstCell:
    """The context where a token's copy log-probability is lowest, and what happened there."""

    context_index: int
    logprob: float
    emitted_id: int
    emitted_text: str
    entropy_bits: float
    mode: FailureMode
    confident: bool

    def to_dict(self) -> dict[str, Any]:
        return {"context_index": self.context_index, "logprob": self.logprob, "emitted_id": self.emitted_id,
                "emitted_text": self.emitted_text, "entropy_bits": self.entropy_bits, "mode": self.mode.value,
                "confident": self.confident}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WorstCell":
        return cls(int(d["context_index"]), float(d["logprob"]), int(d["emitted_id"]), str(d["emitted_text"]),
                   float(d["entropy_bits"]), FailureMode(d["mode"]), bool(d["confident"]))


def failure_mode(emitted_id: int, token_id: int, context: Context, tok: TokenizerLike) -> FailureMode:
    """Classify the emitted id at a context slot, following the specimen-store heuristic of the paper."""
    if emitted_id == token_id:
        return FailureMode.CORRECT
    if context.slot + 1 < len(context.ids) and emitted_id == context.ids[context.slot + 1]:
        return FailureMode.DELETION
    emitted, target = tok.decode([emitted_id]).strip(), tok.decode([token_id]).strip()
    if emitted and target.startswith(emitted) and len(emitted) < len(target):
        return FailureMode.TRUNCATION
    return FailureMode.SUBSTITUTION


def classify_token(passes_gate: bool, fragility: float, cfg: ScanConfig) -> TokenClass:
    if passes_gate:
        if fragility >= cfg.fragile_threshold:
            return TokenClass.FRAGILE
        return TokenClass.INTERMITTENT if fragility > 0 else TokenClass.STABLE
    if fragility >= cfg.glitch_context_share:
        return TokenClass.GLITCH
    if fragility <= cfg.fragile_threshold:
        return TokenClass.PROBE_ONLY
    return TokenClass.UNGATED


def gate_mask(fm: FragilityMatrix, cfg: ScanConfig, gate: str = "greedy") -> BoolArray:
    """Which tokens pass the single-token probe under the greedy (default) or absolute gate."""
    thr = cfg.greedy_gate_logprob if gate == "greedy" else cfg.absolute_gate_logprob
    return np.asarray(fm.single_logprob > thr, dtype=np.bool_)


def worst_cells(fm: FragilityMatrix, tok: TokenizerLike, cfg: ScanConfig) -> list[WorstCell]:
    """For every token, its worst context and the failure mode there."""
    worst_idx = fm.logprob.argmin(axis=1)
    out: list[WorstCell] = []
    for i, c in enumerate(worst_idx):
        c_i = int(c)
        ctx = fm.bank.contexts[c_i]
        emitted = int(fm.top1[i, c_i])
        ent = float(fm.entropy_bits[i, c_i])
        out.append(WorstCell(context_index=c_i, logprob=float(fm.logprob[i, c_i]), emitted_id=emitted,
                             emitted_text=tok.decode([emitted]), entropy_bits=ent,
                             mode=failure_mode(emitted, int(fm.token_ids[i]), ctx, tok),
                             confident=ent < cfg.confident_entropy_bits))
    return out


def confident_share(fm: FragilityMatrix, cfg: ScanConfig) -> FloatArray:
    """Per token: the share of its failing cells whose entropy is below the confidence threshold
    (NaN for tokens that never fail)."""
    fails = fm.fails()
    conf = fails & (fm.entropy_bits < cfg.confident_entropy_bits)
    n_fail = fails.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = conf.sum(axis=1) / n_fail
    return np.where(n_fail > 0, share, np.nan).astype(np.float32)


def classify_matrix(fm: FragilityMatrix, cfg: ScanConfig, gate: str = "greedy") -> list[TokenClass]:
    gated = gate_mask(fm, cfg, gate)
    frag = fm.fragility()
    return [classify_token(bool(g), float(f), cfg) for g, f in zip(gated, frag)]


def glitch_reference(fm: FragilityMatrix, tok: TokenizerLike, cfg: ScanConfig, n: int | None = None,
                     special_ids: frozenset[int] = frozenset()) -> IntArray:
    """Indices (into the scan) of the label-free glitch reference class: the printable tokens with the
    lowest probe log-probability that also fail in at least ``glitch_context_share`` of contexts."""
    n = cfg.n_reference if n is None else n
    frag = fm.fragility()
    order = np.argsort(fm.single_logprob)
    picked = [int(i) for i in order if frag[i] >= cfg.glitch_context_share and is_scannable(tok, int(fm.token_ids[i]), special_ids)]
    return np.asarray(picked[:n], dtype=np.int64)


__all__ = ["FailureMode", "TokenClass", "WorstCell", "classify_matrix", "classify_token", "confident_share",
           "failure_mode", "gate_mask", "glitch_reference", "worst_cells"]
