"""Teacher-forced copy scores: the single-token probe and the fragility matrix."""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .contexts import Context, ContextBank
from .prompt import CopyPrompt
from .types import CausalLMLike, FloatArray, IntArray, TokenizerLike

logger = logging.getLogger(__name__)
LN2 = math.log(2.0)


@dataclass(frozen=True)
class CellScores:
    """Scores of every token at one context: copy log-probability of the token, the id the model
    would emit instead, and the next-token entropy in bits."""

    logprob: FloatArray
    top1: IntArray
    entropy_bits: FloatArray


@dataclass(frozen=True)
class FragilityMatrix:
    """Tokens x contexts. ``logprob[i, c]`` is the copy log-probability of token ``i`` at the slot of
    context ``c``; ``top1`` the id with the highest probability there; ``entropy_bits`` the entropy of
    that distribution. ``single_logprob`` is the single-token probe."""

    token_ids: IntArray
    single_logprob: FloatArray
    logprob: FloatArray
    top1: IntArray
    entropy_bits: FloatArray
    bank: ContextBank
    fail_logprob: float

    def __post_init__(self) -> None:
        n, c = self.logprob.shape
        if not (len(self.token_ids) == len(self.single_logprob) == n and self.top1.shape == (n, c)
                and self.entropy_bits.shape == (n, c) and len(self.bank) == c):
            raise ValueError("inconsistent fragility matrix shapes")

    @property
    def n_tokens(self) -> int:
        return int(self.logprob.shape[0])

    @property
    def n_contexts(self) -> int:
        return int(self.logprob.shape[1])

    def fails(self) -> np.ndarray:
        return self.logprob < self.fail_logprob

    def fragility(self) -> FloatArray:
        """Share of contexts in which each token fails."""
        return self.fails().mean(axis=1).astype(np.float32)

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(path, token_ids=self.token_ids, single_logprob=self.single_logprob, logprob=self.logprob,
                            top1=self.top1, entropy_bits=self.entropy_bits, fail_logprob=np.float64(self.fail_logprob),
                            bank_ids=np.array([list(c.ids) + [-1] * (max(len(x.ids) for x in self.bank.contexts) - len(c.ids))
                                               for c in self.bank.contexts], dtype=np.int64),
                            bank_slots=np.array([c.slot for c in self.bank.contexts], dtype=np.int64))

    @classmethod
    def load_npz(cls, path: str | Path) -> "FragilityMatrix":
        z = np.load(path)
        contexts = tuple(Context(tuple(int(i) for i in row if i >= 0), int(slot)) for row, slot in zip(z["bank_ids"], z["bank_slots"]))
        return cls(token_ids=z["token_ids"].astype(np.int64), single_logprob=z["single_logprob"].astype(np.float32),
                   logprob=z["logprob"].astype(np.float32), top1=z["top1"].astype(np.int64),
                   entropy_bits=z["entropy_bits"].astype(np.float32), bank=ContextBank(contexts),
                   fail_logprob=float(z["fail_logprob"]))


def _batches(items: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for s in range(0, len(items), size):
        yield items[s: s + size]


@torch.no_grad()
def single_token_logprobs(model: CausalLMLike, prompt: CopyPrompt, token_ids: Sequence[int], device: torch.device,
                          batch_size: int = 48) -> FloatArray:
    """The single-token probe: log-probability of copying each token back when it is the whole text."""
    out = np.empty(len(token_ids), dtype=np.float32)
    pos = 0
    for chunk in _batches(list(token_ids), batch_size):
        x = torch.tensor([prompt.single(int(t)) for t in chunk], device=device)
        logits = model(input_ids=x).logits[:, -2, :].float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, x[:, -1:]).squeeze(-1)
        out[pos: pos + len(chunk)] = lp.cpu().numpy()
        pos += len(chunk)
    return out


@torch.no_grad()
def score_context(model: CausalLMLike, prompt: CopyPrompt, context: Context, token_ids: Sequence[int],
                  device: torch.device, batch_size: int = 24) -> CellScores:
    """Every token at one context. All sequences share a length, so there is no padding."""
    base = prompt.sequence(list(context.ids))
    tpos, cpos = prompt.positions(len(context.ids), context.slot)
    n = len(token_ids)
    lp_out = np.empty(n, dtype=np.float32)
    top_out = np.empty(n, dtype=np.int64)
    ent_out = np.empty(n, dtype=np.float32)
    pos = 0
    for chunk in _batches(list(token_ids), batch_size):
        seqs = []
        for t in chunk:
            q = list(base)
            q[tpos] = int(t)
            q[cpos] = int(t)
            seqs.append(q)
        x = torch.tensor(seqs, device=device)
        logits = model(input_ids=x).logits[:, cpos - 1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        p = lp.exp()
        lp_out[pos: pos + len(chunk)] = lp.gather(-1, x[:, cpos: cpos + 1]).squeeze(-1).cpu().numpy()
        top_out[pos: pos + len(chunk)] = lp.argmax(dim=-1).cpu().numpy()
        ent_out[pos: pos + len(chunk)] = (-(p * lp).sum(dim=-1) / LN2).cpu().numpy()
        pos += len(chunk)
    return CellScores(lp_out, top_out, ent_out)


def fragility_matrix(model: CausalLMLike, prompt: CopyPrompt, bank: ContextBank, token_ids: Sequence[int],
                     device: torch.device, *, fail_logprob: float, batch_size: int = 24,
                     single_logprob: FloatArray | None = None) -> FragilityMatrix:
    """Score every token at every context of the bank (and the single-token probe if not supplied)."""
    ids = np.asarray([int(t) for t in token_ids], dtype=np.int64)
    if single_logprob is None:
        logger.info("single-token probe over %d tokens", len(ids))
        single_logprob = single_token_logprobs(model, prompt, ids.tolist(), device, batch_size=max(batch_size, 48))
    n, c = len(ids), len(bank)
    lp = np.empty((n, c), dtype=np.float32)
    top1 = np.empty((n, c), dtype=np.int64)
    ent = np.empty((n, c), dtype=np.float32)
    for k, ctx in enumerate(bank.contexts):
        cell = score_context(model, prompt, ctx, ids.tolist(), device, batch_size=batch_size)
        lp[:, k], top1[:, k], ent[:, k] = cell.logprob, cell.top1, cell.entropy_bits
        logger.info("context %d/%d (len %d): %.1f%% of tokens fail", k + 1, c, len(ctx.ids),
                    100.0 * float((cell.logprob < fail_logprob).mean()))
    return FragilityMatrix(token_ids=ids, single_logprob=np.asarray(single_logprob, dtype=np.float32), logprob=lp,
                           top1=top1, entropy_bits=ent, bank=bank, fail_logprob=fail_logprob)


def is_scannable(tok: TokenizerLike, token_id: int, special_ids: frozenset[int]) -> bool:
    """Printable, non-blank, non-special: the tokens worth scoring."""
    if token_id in special_ids:
        return False
    s = tok.decode([token_id])
    return bool(s.strip()) and s.isprintable()


def sample_token_ids(tok: TokenizerLike, vocab_size: int, n: int, seed: int, *, exclude: Iterable[int] = (),
                     special_ids: Iterable[int] = ()) -> list[int]:
    """A stratified sample of scannable ids: equal thirds of the id range, so rare (high-id) pieces are
    represented as well as common ones. Deterministic for a seed."""
    rng = random.Random(seed)
    ex = set(int(i) for i in exclude)
    sp = frozenset(int(i) for i in special_ids)
    out: list[int] = []
    bounds = [0, vocab_size // 3, 2 * vocab_size // 3, vocab_size]
    per = -(-n // 3)
    for lo, hi in zip(bounds, bounds[1:]):
        pool = list(range(lo, hi))
        rng.shuffle(pool)
        picked = 0
        for t in pool:
            if picked >= per:
                break
            if t in ex or not is_scannable(tok, t, sp):
                continue
            out.append(t)
            picked += 1
    return sorted(out[:n])


def tokens_of_text(tok: TokenizerLike, text: str) -> list[int]:
    return [int(i) for i in tok(text, add_special_tokens=False)["input_ids"]]


def summarize_cells(cells: Any) -> dict[str, float]:  # pragma: no cover - convenience for notebooks
    lp = np.asarray(cells.logprob)
    return {"mean_logprob": float(lp.mean()), "min_logprob": float(lp.min())}


__all__ = ["CellScores", "FragilityMatrix", "fragility_matrix", "is_scannable", "sample_token_ids", "score_context",
           "single_token_logprobs", "tokens_of_text"]
