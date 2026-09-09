"""Command line: ``fragile-tokens scan | report | check | geometry``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np

from . import __version__
from .classify import TokenClass, glitch_reference
from .config import Framing, ScanConfig
from .detector import FragileTokenDetector
from .geometry import rank_vocabulary
from .models import load_embeddings, load_model
from .report import ScanResult

logger = logging.getLogger("fragile_tokens")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="Hugging Face id or local path")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--load-4bit", action="store_true", help="NF4 quantization (needs accelerate and bitsandbytes)")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--cpu", action="store_true", help="keep the model off any GPU")
    p.add_argument("--max-gpu-memory", default=None, help="per-card cap such as 56GiB")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--framing", choices=[f.value for f in Framing], default=Framing.AUTO.value,
                   help="raw prompt, chat template, or auto (chat when the tokenizer has a template)")
    p.add_argument("--context-lengths", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--contexts-per-length", type=int, default=6, help="4 lengths x 6 gives the paper's 24 contexts")
    p.add_argument("--fail-logprob", type=float, default=-0.5)
    p.add_argument("--gate", choices=["greedy", "absolute"], default="greedy")
    p.add_argument("--gate-p", type=float, default=0.5, help="greedy gate: the token's probability alone must exceed this")
    p.add_argument("--fragile-threshold", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=24)


def _config_from(a: argparse.Namespace) -> ScanConfig:
    return ScanConfig(context_lengths=tuple(a.context_lengths), contexts_per_length=a.contexts_per_length,
                      fail_logprob=a.fail_logprob, greedy_gate_p=a.gate_p, fragile_threshold=a.fragile_threshold,
                      framing=Framing(a.framing), seed=a.seed, batch_size=a.batch)


def _detector(a: argparse.Namespace, cfg: ScanConfig) -> FragileTokenDetector:
    loaded = load_model(a.model, dtype=a.dtype, load_4bit=a.load_4bit, trust_remote_code=a.trust_remote_code,
                        force_cpu=a.cpu, max_gpu_memory=a.max_gpu_memory)
    return FragileTokenDetector(loaded, cfg)


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", "\\n")


def _read_ids(path: str) -> list[int]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = d["token_ids"] if isinstance(d, dict) else d
    return [int(t) for t in ids]


# ---- subcommands ------------------------------------------------------------------------------
def cmd_scan(a: argparse.Namespace) -> int:
    cfg = _config_from(a)
    det = _detector(a, cfg)
    ids: list[int] | None = None
    if a.token_ids:
        ids = _read_ids(a.token_ids)
    elif a.text:
        from .scoring import tokens_of_text

        ids = sorted(set(tokens_of_text(det.tokenizer, a.text)))
    result = det.scan(token_ids=ids, n_tokens=a.n_tokens, gate=a.gate, all_tokens=a.all_tokens)
    result.save(a.out)
    if a.csv:
        result.to_csv(a.csv)
    if a.markdown:
        Path(a.markdown).write_text(result.to_markdown(top=a.top), encoding="utf-8")
        logger.info("markdown report written to %s", a.markdown)
    return 0


def cmd_report(a: argparse.Namespace, out: TextIO = sys.stdout) -> int:
    result = ScanResult.load(a.scan)
    classes = [TokenClass(c) for c in a.classes] if a.classes else [TokenClass.FRAGILE]
    if a.csv:
        result.to_csv(a.csv, classes if a.classes else None)
    if a.json:
        Path(a.json).write_text(json.dumps(result.summary(), indent=1), encoding="utf-8")
        logger.info("summary written to %s", a.json)
    out.write(result.to_markdown(top=a.top, classes=classes))
    return 0


def cmd_check(a: argparse.Namespace, out: TextIO = sys.stdout) -> int:
    scan = ScanResult.load(a.scan) if a.scan else None
    if scan is None and not a.model:
        raise SystemExit("check needs --scan, --model, or both")
    cfg = scan.config if scan is not None else _config_from(a)
    if a.model:
        det = _detector(a, cfg)
        reports = det.check_text(a.text, scan)
    else:
        assert scan is not None
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(scan.model_name)
        known = scan.lookup()
        ids = [int(i) for i in tok(a.text, add_special_tokens=False)["input_ids"]]
        reports = [known[t] for t in ids if t in known]
        unknown = [t for t in ids if t not in known]
        if unknown:
            logger.warning("%d tokens of the text were not in the scan; pass --model to score them", len(unknown))
    out.write("| token | id | class | fragility | alone | worst emitted | mode |\n|:--|--:|:--|--:|--:|:--|:--|\n")
    for r in reports:
        out.write(f"| `{_esc(r.text)}` | {r.token_id} | {r.token_class.value} | {r.fragility:.2f} | "
                  f"{np.exp(r.single_logprob):.3f} | `{_esc(r.worst.emitted_text)}` | {r.worst.mode.value} |\n")
    flagged = [r for r in reports if r.token_class in (TokenClass.FRAGILE, TokenClass.GLITCH)]
    logger.info("%d of %d tokens flagged (fragile or glitch)", len(flagged), len(reports))
    return 0


def cmd_geometry(a: argparse.Namespace) -> int:
    scan = ScanResult.load(a.scan)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(scan.model_name, trust_remote_code=a.trust_remote_code)
    emb = load_embeddings(scan.model_name, trust_remote_code=a.trust_remote_code)
    special = frozenset(int(i) for i in getattr(tok, "all_special_ids", []) or [])
    ref_idx = glitch_reference(scan.matrix, tok, scan.config, special_ids=special)
    if len(ref_idx) == 0:
        raise SystemExit("the scan has no glitch-class tokens to serve as a reference")
    ref_ids = scan.matrix.token_ids[ref_idx]
    from .geometry import geometry_scores

    classes = np.asarray([r.token_class.value for r in scan.reports])
    scores = geometry_scores(emb, scan.matrix.token_ids, ref_ids, classes == TokenClass.FRAGILE.value,
                             classes == TokenClass.STABLE.value, k=a.k)
    payload = scores.to_dict()
    if a.rank_vocabulary:
        prox = rank_vocabulary(emb, ref_ids, k=a.k)
        scanned = set(int(t) for t in scan.matrix.token_ids)
        order = np.argsort(-prox)
        top = [{"token_id": int(t), "text": tok.decode([int(t)]), "proximity_out": float(prox[t])}
               for t in order if int(t) not in scanned][: a.top]
        payload["vocabulary_candidates"] = top
        logger.info("ranked %d vocabulary rows by proximity to the glitch class; top %d unscanned candidates kept",
                    len(prox), len(top))
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("geometry written to %s", a.out)
    return 0


# ---- parser ------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fragile-tokens",
                                description="Find tokens that pass the single-token glitch probe and fail to copy in context.")
    p.add_argument("--version", action="version", version=f"fragile-tokens {__version__}")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v for INFO (default), -vv for DEBUG")
    p.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="measure fragility for a sample of tokens, a list of ids, or the tokens of a text")
    _add_model_args(s)
    _add_config_args(s)
    s.add_argument("--n-tokens", type=int, default=4200, help="stratified sample size when no ids are given")
    s.add_argument("--all-tokens", action="store_true", help="scan the whole vocabulary")
    s.add_argument("--token-ids", help="JSON file: a list of ids or {\"token_ids\": [...]}")
    s.add_argument("--text", help="scan the distinct tokens of this text")
    s.add_argument("--out", default="fragile_tokens_scan.json")
    s.add_argument("--csv", help="also write a CSV of every token")
    s.add_argument("--markdown", help="also write a Markdown report")
    s.add_argument("--top", type=int, default=40)
    s.set_defaults(func=cmd_scan)

    r = sub.add_parser("report", help="summarize a scan and list its fragile tokens")
    r.add_argument("scan")
    r.add_argument("--classes", nargs="*", choices=[c.value for c in TokenClass], help="classes to list (default: fragile)")
    r.add_argument("--top", type=int, default=40)
    r.add_argument("--csv")
    r.add_argument("--json", help="write the summary as JSON")
    r.set_defaults(func=cmd_report)

    c = sub.add_parser("check", help="fragility of every token in a text, from a scan and/or a fresh measurement")
    c.add_argument("--scan")
    c.add_argument("--text", required=True)
    c.add_argument("--model", help="score tokens missing from the scan (loads the model)")
    c.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    c.add_argument("--load-4bit", action="store_true")
    c.add_argument("--trust-remote-code", action="store_true")
    c.add_argument("--cpu", action="store_true")
    c.add_argument("--max-gpu-memory", default=None)
    _add_config_args(c)
    c.set_defaults(func=cmd_check)

    g = sub.add_parser("geometry", help="static embedding scores against the scan's glitch class; needs no forward pass")
    g.add_argument("scan")
    g.add_argument("--out", default="fragile_tokens_geometry.json")
    g.add_argument("--k", type=int, default=5, help="nearest glitch-class rows averaged")
    g.add_argument("--rank-vocabulary", action="store_true", help="also rank every vocabulary row by proximity")
    g.add_argument("--top", type=int, default=200)
    g.add_argument("--trust-remote-code", action="store_true")
    g.set_defaults(func=cmd_geometry)
    return p


def configure_logging(verbose: int, quiet: bool) -> None:
    level = logging.WARNING if quiet else (logging.DEBUG if verbose >= 2 else logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    configure_logging(a.verbose, a.quiet)
    return int(a.func(a))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
