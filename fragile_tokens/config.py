"""Scan configuration: the thresholds and sizes of the measurement, with the paper's values as defaults."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class Framing(str, Enum):
    """How the copy prompt is presented to the model.

    ``raw``: plain text, ``Repeat the text exactly.`` followed by the demonstrations and ``Text:`` /
    ``Copy:`` lines. Works for base-style copying on OLMo-2, Qwen2.5 and Qwen3.

    ``chat``: the same instruction inside the user turn of the model's chat template and ``Copy:`` at
    the start of the assistant turn, with thinking disabled where the template knows the switch.
    Needed for models that otherwise open the copy with a newline (Qwen3.5 and Qwen3.8).

    ``auto``: ``chat`` when the tokenizer carries a chat template, ``raw`` otherwise.
    """

    RAW = "raw"
    CHAT = "chat"
    AUTO = "auto"


DEFAULT_DEMOS: tuple[str, ...] = (" apple pie is good", " the quick brown fox")


@dataclass(frozen=True)
class ScanConfig:
    """Everything that defines a scan besides the model and the tokens.

    Attributes:
        context_lengths: lengths (in tokens) of the random common-word contexts.
        contexts_per_length: contexts drawn per length; 4 lengths x 6 gives the paper's 24.
        fail_logprob: a cell fails when the copy log-probability is below this (-0.5, p < 0.61).
        greedy_gate_p: a token passes the single-token probe when its probability alone exceeds this
            (0.5, so it is the greedy output of its own probe). This is the paper's cross-model gate.
        absolute_gate_logprob: the stricter gate used on the reference model (log-probability > -0.1).
        fragile_threshold: fragility (share of failing contexts) at or above which a gated token is fragile.
        glitch_context_share: a token that fails the probe and fails in at least this share of contexts
            is in the glitch class.
        confident_entropy_bits: a failing cell is "confident" when the next-token entropy is below this.
        n_reference: size of the label-free glitch reference class used by the geometry scores.
        common_word_limit: how many low-id ' lowercaseword' tokens to consider as context filler.
        demos: the few-shot demonstrations of the copy prompt.
        framing: raw, chat or auto (see :class:`Framing`).
        seed: seed for token sampling and context drawing.
        batch_size: sequences per forward pass.
    """

    context_lengths: tuple[int, ...] = (8, 16, 32, 64)
    contexts_per_length: int = 6
    fail_logprob: float = -0.5
    greedy_gate_p: float = 0.5
    absolute_gate_logprob: float = -0.1
    fragile_threshold: float = 0.10
    glitch_context_share: float = 0.9
    confident_entropy_bits: float = 2.0
    n_reference: int = 100
    common_word_limit: int = 2000
    demos: tuple[str, ...] = DEFAULT_DEMOS
    framing: Framing = Framing.AUTO
    seed: int = 0
    batch_size: int = 24

    def __post_init__(self) -> None:
        if not self.context_lengths or min(self.context_lengths) < 3:
            raise ValueError("context_lengths must be non-empty and at least 3 (the slot is interior)")
        if self.contexts_per_length < 1:
            raise ValueError("contexts_per_length must be positive")
        if not 0.0 < self.greedy_gate_p < 1.0:
            raise ValueError("greedy_gate_p must be in (0, 1)")
        if not 0.0 <= self.fragile_threshold <= 1.0 or not 0.0 <= self.glitch_context_share <= 1.0:
            raise ValueError("thresholds on shares must lie in [0, 1]")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    @property
    def n_contexts(self) -> int:
        return len(self.context_lengths) * self.contexts_per_length

    @property
    def greedy_gate_logprob(self) -> float:
        return math.log(self.greedy_gate_p)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["framing"] = self.framing.value
        d["context_lengths"] = list(self.context_lengths)
        d["demos"] = list(self.demos)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ScanConfig":
        kw: dict[str, Any] = dict(d)
        kw["framing"] = Framing(kw.get("framing", Framing.AUTO.value))
        kw["context_lengths"] = tuple(int(x) for x in kw.get("context_lengths", (8, 16, 32, 64)))
        kw["demos"] = tuple(str(x) for x in kw.get("demos", DEFAULT_DEMOS))
        return cls(**kw)


def default_config() -> ScanConfig:
    return ScanConfig()


__all__ = ["DEFAULT_DEMOS", "Framing", "ScanConfig", "default_config", "field"]
