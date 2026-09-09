"""Per-token reports and the scan result, with JSON, CSV and Markdown output."""
from __future__ import annotations

import csv
import io
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ._version import __version__ as _version
from .classify import TokenClass, WorstCell, classify_token, confident_share, gate_mask, worst_cells
from .config import ScanConfig
from .contexts import ContextBank
from .scoring import FragilityMatrix
from .types import TokenizerLike

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenReport:
    token_id: int
    text: str
    single_logprob: float
    passes_gate: bool
    fragility: float
    n_fail: int
    token_class: TokenClass
    worst: WorstCell
    confident_share: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"token_id": self.token_id, "text": self.text, "single_logprob": self.single_logprob,
                "passes_gate": self.passes_gate, "fragility": self.fragility, "n_fail": self.n_fail,
                "class": self.token_class.value, "worst": self.worst.to_dict(), "confident_share": self.confident_share}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TokenReport":
        cs = d.get("confident_share")
        return cls(int(d["token_id"]), str(d["text"]), float(d["single_logprob"]), bool(d["passes_gate"]),
                   float(d["fragility"]), int(d["n_fail"]), TokenClass(d["class"]), WorstCell.from_dict(d["worst"]),
                   None if cs is None else float(cs))


def build_reports(fm: FragilityMatrix, tok: TokenizerLike, cfg: ScanConfig, gate: str = "greedy") -> list[TokenReport]:
    gated = gate_mask(fm, cfg, gate)
    frag = fm.fragility()
    n_fail = fm.fails().sum(axis=1)
    conf = confident_share(fm, cfg)
    worst = worst_cells(fm, tok, cfg)
    out: list[TokenReport] = []
    for i, t in enumerate(fm.token_ids):
        cs = float(conf[i])
        out.append(TokenReport(token_id=int(t), text=tok.decode([int(t)]), single_logprob=float(fm.single_logprob[i]),
                               passes_gate=bool(gated[i]), fragility=float(frag[i]), n_fail=int(n_fail[i]),
                               token_class=classify_token(bool(gated[i]), float(frag[i]), cfg), worst=worst[i],
                               confident_share=None if math.isnan(cs) else cs))
    return out


def split_half_correlation(fm: FragilityMatrix, mask: np.ndarray) -> float:
    """Correlation of fragility on even- and odd-numbered contexts among ``mask`` tokens."""
    fails = fm.fails()[mask]
    if fails.shape[0] < 3 or fails.shape[1] < 2:
        return float("nan")
    a, b = fails[:, 0::2].mean(axis=1), fails[:, 1::2].mean(axis=1)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def context_variance_share(fm: FragilityMatrix, mask: np.ndarray) -> float:
    """Share of the log-probability variance carried by the context (two-way additive decomposition)."""
    m = fm.logprob[mask].astype(np.float64)
    if m.size == 0:
        return float("nan")
    mu = m.mean()
    tot = float(((m - mu) ** 2).sum())
    if tot == 0:
        return float("nan")
    ctx = m.mean(axis=0, keepdims=True) - mu
    return float((np.broadcast_to(ctx, m.shape) ** 2).sum() / tot)


@dataclass
class ScanResult:
    """Everything a scan produced: provenance, configuration, the context bank, the matrix and the
    per-token reports."""

    model_name: str
    model_info: dict[str, Any]
    prompt_info: dict[str, Any]
    config: ScanConfig
    gate: str
    matrix: FragilityMatrix
    reports: list[TokenReport]
    bank_text: list[str] = field(default_factory=list)
    tool_version: str = _version

    # ---- summaries ---------------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        c = {k.value: 0 for k in TokenClass}
        for r in self.reports:
            c[r.token_class.value] += 1
        return c

    def summary(self) -> dict[str, Any]:
        gated = np.asarray([r.passes_gate for r in self.reports], dtype=np.bool_)
        frag = np.asarray([r.fragility for r in self.reports], dtype=np.float64)
        n_gated = int(gated.sum())
        fragile = gated & (frag >= self.config.fragile_threshold)
        conf = [r.confident_share for r in self.reports if r.token_class is TokenClass.FRAGILE and r.confident_share is not None]
        return {
            "model": self.model_name,
            "tokens_scanned": len(self.reports),
            "contexts": self.matrix.n_contexts,
            "gate": self.gate,
            "passes_gate": n_gated,
            "fragile": int(fragile.sum()),
            "fragile_share_of_gated": float(fragile.sum() / n_gated) if n_gated else float("nan"),
            "never_fail_share_of_gated": float(((frag == 0) & gated).sum() / n_gated) if n_gated else float("nan"),
            "split_half_correlation": split_half_correlation(self.matrix, gated),
            "context_variance_share": context_variance_share(self.matrix, gated),
            "confident_share_among_fragile": float(np.mean(conf)) if conf else float("nan"),
            "classes": self.counts(),
        }

    def by_class(self, cls: TokenClass) -> list[TokenReport]:
        return sorted((r for r in self.reports if r.token_class is cls), key=lambda r: (-r.fragility, r.single_logprob))

    def lookup(self) -> dict[int, TokenReport]:
        return {r.token_id: r for r in self.reports}

    # ---- serialization -----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_version": self.tool_version,
            "model_name": self.model_name,
            "model_info": self.model_info,
            "prompt": self.prompt_info,
            "config": self.config.to_dict(),
            "gate": self.gate,
            "summary": self.summary(),
            "bank": self.matrix.bank.to_list(),
            "bank_text": self.bank_text,
            "tokens": [r.to_dict() for r in self.reports],
            "matrix": {
                "token_ids": self.matrix.token_ids.tolist(),
                "single_logprob": [round(float(x), 5) for x in self.matrix.single_logprob],
                "logprob": np.round(self.matrix.logprob, 5).tolist(),
                "top1": self.matrix.top1.tolist(),
                "entropy_bits": np.round(self.matrix.entropy_bits, 4).tolist(),
                "fail_logprob": self.matrix.fail_logprob,
            },
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("scan written to %s (%d tokens, %d contexts)", p, len(self.reports), self.matrix.n_contexts)

    @classmethod
    def load(cls, path: str | Path) -> "ScanResult":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = d["matrix"]
        matrix = FragilityMatrix(token_ids=np.asarray(m["token_ids"], dtype=np.int64),
                                 single_logprob=np.asarray(m["single_logprob"], dtype=np.float32),
                                 logprob=np.asarray(m["logprob"], dtype=np.float32), top1=np.asarray(m["top1"], dtype=np.int64),
                                 entropy_bits=np.asarray(m["entropy_bits"], dtype=np.float32),
                                 bank=ContextBank.from_list(d["bank"]), fail_logprob=float(m["fail_logprob"]))
        return cls(model_name=str(d["model_name"]), model_info=dict(d.get("model_info", {})), prompt_info=dict(d.get("prompt", {})),
                   config=ScanConfig.from_dict(d["config"]), gate=str(d.get("gate", "greedy")), matrix=matrix,
                   reports=[TokenReport.from_dict(t) for t in d["tokens"]], bank_text=list(d.get("bank_text", [])),
                   tool_version=str(d.get("tool_version", "unknown")))

    def to_csv(self, path: str | Path, classes: Iterable[TokenClass] | None = None) -> int:
        keep = set(classes) if classes is not None else set(TokenClass)
        rows = [r for r in self.reports if r.token_class in keep]
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["token_id", "text", "class", "single_logprob", "passes_gate", "fragility", "n_fail",
                        "worst_context", "worst_logprob", "emitted_id", "emitted_text", "mode", "entropy_bits", "confident",
                        "confident_share"])
            for r in sorted(rows, key=lambda r: (-r.fragility, r.single_logprob)):
                w.writerow([r.token_id, r.text, r.token_class.value, f"{r.single_logprob:.4f}", int(r.passes_gate),
                            f"{r.fragility:.4f}", r.n_fail, r.worst.context_index, f"{r.worst.logprob:.4f}", r.worst.emitted_id,
                            r.worst.emitted_text, r.worst.mode.value, f"{r.worst.entropy_bits:.3f}", int(r.worst.confident),
                            "" if r.confident_share is None else f"{r.confident_share:.3f}"])
        logger.info("%d rows written to %s", len(rows), path)
        return len(rows)

    def to_markdown(self, top: int = 40, classes: Sequence[TokenClass] = (TokenClass.FRAGILE,)) -> str:
        s = self.summary()
        out = io.StringIO()
        out.write(f"# Fragile-token scan: {self.model_name}\n\n")
        out.write(f"{s['tokens_scanned']} tokens x {s['contexts']} contexts, gate: {s['gate']}. ")
        out.write(f"{s['passes_gate']} pass the probe; {s['fragile']} of those are fragile "
                  f"({100 * s['fragile_share_of_gated']:.1f}%). Split-half correlation {s['split_half_correlation']:.2f}, "
                  f"context share of variance {s['context_variance_share']:.3f}.\n\n")
        out.write("| class | count |\n|:--|--:|\n")
        for k, v in s["classes"].items():
            out.write(f"| {k} | {v} |\n")
        for cls in classes:
            rows = self.by_class(cls)[:top]
            if not rows:
                continue
            out.write(f"\n## {cls.value} tokens (top {len(rows)} by fragility)\n\n")
            out.write("| token | id | alone | fragility | worst context | emitted | mode | entropy (bits) |\n|:--|--:|--:|--:|--:|:--|:--|--:|\n")
            for r in rows:
                out.write(f"| `{_cell(r.text)}` | {r.token_id} | {math.exp(r.single_logprob):.3f} | {r.fragility:.2f} | "
                          f"{r.worst.context_index} | `{_cell(r.worst.emitted_text)}` | {r.worst.mode.value} | {r.worst.entropy_bits:.2f} |\n")
        return out.getvalue()


def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("`", "'").replace("\n", "\\n")


__all__ = ["ScanResult", "TokenReport", "build_reports", "context_variance_share", "split_half_correlation"]
