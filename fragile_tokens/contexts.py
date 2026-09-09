"""Context banks: random sequences of common words with one interior slot for the token under test."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .types import TokenizerLike

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Context:
    """A filler sequence and the slot index at which the token under test is placed."""

    ids: tuple[int, ...]
    slot: int

    def __post_init__(self) -> None:
        if not 0 <= self.slot < len(self.ids):
            raise ValueError(f"slot {self.slot} outside a context of {len(self.ids)} tokens")

    def with_token(self, token_id: int) -> list[int]:
        out = list(self.ids)
        out[self.slot] = int(token_id)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"ids": list(self.ids), "slot": self.slot}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Context":
        return cls(tuple(int(x) for x in d["ids"]), int(d["slot"]))


@dataclass(frozen=True)
class ContextBank:
    contexts: tuple[Context, ...]

    def __len__(self) -> int:
        return len(self.contexts)

    def __iter__(self) -> Iterable[Context]:  # pragma: no cover - trivial
        return iter(self.contexts)

    def to_list(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.contexts]

    @classmethod
    def from_list(cls, items: Sequence[Mapping[str, Any]]) -> "ContextBank":
        return cls(tuple(Context.from_dict(d) for d in items))

    def describe(self, tok: TokenizerLike) -> list[str]:
        """Each context rendered with a placeholder at the slot."""
        out = []
        for c in self.contexts:
            words = [tok.decode([i]) for i in c.ids]
            words[c.slot] = " <token>"
            out.append("".join(words))
        return out


def common_word_ids(tok: TokenizerLike, vocab_size: int, limit: int = 2000) -> list[int]:
    """Low-id tokens of the form ``' lowercaseword'``. In a byte-pair vocabulary the earliest merges
    are the most frequent pieces, so these are the common words of the dominant language. Returns at
    most ``limit`` ids; a tokenizer that marks word starts differently (SentencePiece ``▁``) is handled
    by also accepting that marker."""
    out: list[int] = []
    for i in range(vocab_size):
        s = tok.decode([i])
        if len(s) >= 3 and s[0] in (" ", "▁") and s[1:].isalpha() and s[1:].islower():
            out.append(i)
            if len(out) >= limit:
                break
    if not out:
        logger.warning("no ' lowercaseword' tokens found among the first %d ids", vocab_size)
    return out


def build_bank(filler_ids: Sequence[int], lengths: Sequence[int], per_length: int, seed: int) -> ContextBank:
    """Draw ``per_length`` contexts of each length from ``filler_ids`` without replacement within a
    context, each with a random interior slot (never the first or last position)."""
    if len(filler_ids) < max(lengths):
        raise ValueError(f"need at least {max(lengths)} filler ids, have {len(filler_ids)}")
    rng = random.Random(seed)
    contexts: list[Context] = []
    for k in lengths:
        for _ in range(per_length):
            ids = rng.sample(list(filler_ids), k)
            slot = rng.randint(1, k - 2)
            contexts.append(Context(tuple(ids), slot))
    logger.info("context bank: %d contexts, lengths %s", len(contexts), list(lengths))
    return ContextBank(tuple(contexts))


__all__ = ["Context", "ContextBank", "build_bank", "common_word_ids"]
