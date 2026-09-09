"""The detector: one object that owns a model, builds the copy prompt and the context bank, and turns
token ids into fragility reports."""
from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np
import torch

from .classify import TokenClass, glitch_reference
from .config import ScanConfig
from .contexts import ContextBank, build_bank, common_word_ids
from .geometry import Embeddings, GeometryScores, geometry_scores
from .models import LoadedModel, load_model
from .prompt import CopyPrompt
from .report import ScanResult, TokenReport, build_reports
from .scoring import FragilityMatrix, fragility_matrix, is_scannable, sample_token_ids, single_token_logprobs, tokens_of_text
from .types import CausalLMLike, IntArray, TokenizerLike

logger = logging.getLogger(__name__)


class FragileTokenDetector:
    """Measure fragility for a model.

    Typical use::

        det = FragileTokenDetector.from_pretrained("Qwen/Qwen3-1.7B")
        result = det.scan(n_tokens=4200)          # or det.scan(token_ids=[...])
        result.save("scan.json")
        for r in result.by_class(TokenClass.FRAGILE)[:20]:
            ...

    ``check_text`` scores the tokens of a prompt you are about to send, against the bank of a previous
    scan or a fresh one.
    """

    def __init__(self, loaded: LoadedModel, config: ScanConfig | None = None) -> None:
        self.loaded = loaded
        self.config = config or ScanConfig()
        self.prompt = CopyPrompt.build(loaded.tokenizer, self.config.framing, self.config.demos)
        self._bank: ContextBank | None = None
        self._special: frozenset[int] = frozenset(int(i) for i in getattr(loaded.tokenizer, "all_special_ids", []) or [])
        logger.info("copy prompt framing: %s (%d head tokens)", self.prompt.framing.value, len(self.prompt.head))

    @classmethod
    def from_pretrained(cls, name: str, config: ScanConfig | None = None, **load_kwargs: Any) -> "FragileTokenDetector":
        return cls(load_model(name, **load_kwargs), config)

    # ---- pieces --------------------------------------------------------------------------------
    @property
    def model(self) -> CausalLMLike:
        return self.loaded.model

    @property
    def tokenizer(self) -> TokenizerLike:
        return self.loaded.tokenizer

    @property
    def device(self) -> torch.device:
        return self.loaded.device

    @property
    def vocab_size(self) -> int:
        v = self.loaded.info.get("vocab_size")
        return int(v) if v else len(self.tokenizer)

    def filler_ids(self, exclude: Sequence[int] = ()) -> list[int]:
        """Common words the model copies exactly on its own: the material contexts are built from."""
        cw = [t for t in common_word_ids(self.tokenizer, self.vocab_size, self.config.common_word_limit) if t not in set(exclude)]
        if not cw:
            raise RuntimeError("no common-word tokens found to build contexts from")
        lp = single_token_logprobs(self.model, self.prompt, cw, self.device, batch_size=max(self.config.batch_size, 48))
        keep = [t for t, v in zip(cw, lp) if v > self.config.greedy_gate_logprob]
        logger.info("filler pool: %d of %d common words copy exactly alone", len(keep), len(cw))
        if len(keep) < max(self.config.context_lengths):
            raise RuntimeError(f"only {len(keep)} common words copy exactly; the framing is probably wrong for this model "
                               f"(try framing=chat)")
        return keep

    def bank(self, exclude: Sequence[int] = ()) -> ContextBank:
        if self._bank is None:
            self._bank = build_bank(self.filler_ids(exclude), self.config.context_lengths, self.config.contexts_per_length,
                                    self.config.seed)
        return self._bank

    def use_bank(self, bank: ContextBank) -> None:
        """Reuse the bank of an earlier scan so new tokens are measured on the same contexts."""
        self._bank = bank

    # ---- scans ---------------------------------------------------------------------------------
    def matrix(self, token_ids: Sequence[int], bank: ContextBank | None = None) -> FragilityMatrix:
        bank = bank or self.bank(exclude=token_ids)
        return fragility_matrix(self.model, self.prompt, bank, token_ids, self.device, fail_logprob=self.config.fail_logprob,
                                batch_size=self.config.batch_size)

    def scan(self, token_ids: Sequence[int] | None = None, n_tokens: int = 4200, gate: str = "greedy",
             all_tokens: bool = False) -> ScanResult:
        """Score a set of tokens on the bank and classify them. Without ``token_ids`` a stratified sample
        of ``n_tokens`` scannable ids is drawn (``all_tokens`` scans the whole vocabulary)."""
        if token_ids is None:
            if all_tokens:
                ids = [t for t in range(self.vocab_size) if is_scannable(self.tokenizer, t, self._special)]
            else:
                ids = sample_token_ids(self.tokenizer, self.vocab_size, n_tokens, self.config.seed, special_ids=self._special)
        else:
            ids = [int(t) for t in token_ids]
        logger.info("scanning %d tokens on %s", len(ids), self.loaded.name)
        bank = self.bank(exclude=ids)
        fm = self.matrix(ids, bank)
        reports = build_reports(fm, self.tokenizer, self.config, gate)
        result = ScanResult(model_name=self.loaded.name, model_info=dict(self.loaded.info),
                            prompt_info=self.prompt.describe(self.tokenizer), config=self.config, gate=gate, matrix=fm,
                            reports=reports, bank_text=bank.describe(self.tokenizer))
        s = result.summary()
        logger.info("%d of %d gated tokens are fragile (%.1f%%); split-half %.2f; context share %.3f", s["fragile"],
                    s["passes_gate"], 100 * s["fragile_share_of_gated"], s["split_half_correlation"], s["context_variance_share"])
        return result

    def check_text(self, text: str, scan: ScanResult | None = None) -> list[TokenReport]:
        """Fragility of every token in ``text``. Tokens present in ``scan`` are read from it; the rest
        are scored on the same bank."""
        ids = tokens_of_text(self.tokenizer, text)
        known = scan.lookup() if scan is not None else {}
        if scan is not None:
            self.use_bank(scan.matrix.bank)
        missing = sorted({t for t in ids if t not in known and is_scannable(self.tokenizer, t, self._special)})
        if missing:
            logger.info("scoring %d tokens of the text not in the scan", len(missing))
            fm = self.matrix(missing, self.bank())
            for r in build_reports(fm, self.tokenizer, self.config, scan.gate if scan is not None else "greedy"):
                known[r.token_id] = r
        return [known[t] for t in ids if t in known]

    # ---- geometry ------------------------------------------------------------------------------
    def geometry(self, scan: ScanResult, emb: Embeddings, k: int = 5) -> GeometryScores:
        """Static scores for the scanned tokens, using the scan's own label-free glitch class."""
        ref_idx = glitch_reference(scan.matrix, self.tokenizer, self.config, special_ids=self._special)
        if len(ref_idx) == 0:
            raise RuntimeError("no glitch-class tokens in the scan; the geometry needs a reference class")
        ref_ids: IntArray = scan.matrix.token_ids[ref_idx]
        classes = np.asarray([r.token_class.value for r in scan.reports])
        fragile = classes == TokenClass.FRAGILE.value
        stable = classes == TokenClass.STABLE.value
        return geometry_scores(emb, scan.matrix.token_ids, ref_ids, fragile, stable, k)


def fragility_of(result: ScanResult, token_id: int) -> float:
    """Convenience: a token's fragility from a scan, NaN if it was not scanned."""
    r = result.lookup().get(token_id)
    return r.fragility if r is not None else math.nan


__all__ = ["FragileTokenDetector", "fragility_of"]
