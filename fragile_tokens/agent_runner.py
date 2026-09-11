"""Run the agent tasks on one model, with every task word's fragility measured on that same model.

Two phases share one model load.

1. Fragility. Each task word is scored by the paper's copy protocol on THIS model: the filler pool is
   the model's low-id common words that copy exactly alone, the bank is 24 random contexts (lengths 8,
   16, 32 and 64, six each) with an interior slot, and a cell fails when the copy log-probability at the
   slot is below -0.5. Fragility is the share of failing contexts. A word that is not a single token on
   this model has no fragility, since the protocol places one token in the slot; it is flagged.
2. Agent tasks. Every word runs through every task; all live episodes advance one round per generate
   call, so a batch engine serves them together. Each record carries its word's fragility here.

Engines: vLLM (batched serving; scores are prompt log-probabilities) or HuggingFace transformers
(left-padded batched generation; scores from a teacher-forced forward pass). The two should agree on
the scores to rounding and on greedy text almost everywhere; the dry run checks both.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .agent import (TASK_BY_NAME, TASKS, AgentTask, CallFormat, ChatTokenizer, Episode, detect_format,
                    episode_summary, render)
from .config import Framing, ScanConfig
from .contexts import build_bank, common_word_ids
from .prompt import CopyPrompt

logger = logging.getLogger(__name__)
ROLES = ("fragile", "case_twin", "freq_control")


# ---- engines --------------------------------------------------------------------------------------
class Engine(Protocol):
    name: str
    max_len: int | None

    def generate(self, prompts: Sequence[Sequence[int]], max_new: int) -> list[list[int]]: ...

    def score(self, sequences: Sequence[Sequence[int]], positions: Sequence[int]) -> list[float]:
        """Log-probability of ``sequences[i][positions[i]]`` given everything before it."""
        ...


class VLLMEngine:
    """vLLM offline engine: greedy generation, and scores read from prompt log-probabilities."""

    name = "vllm"

    def __init__(self, model: str, tensor_parallel_size: int = 1, max_model_len: int = 32768,
                 gpu_memory_utilization: float = 0.90, trust_remote_code: bool = False) -> None:
        self._vllm: Any = importlib.import_module("vllm")
        self.llm: Any = self._vllm.LLM(model=model, tensor_parallel_size=tensor_parallel_size,
                                       max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                                       trust_remote_code=trust_remote_code, enable_prefix_caching=True, seed=0)
        self.max_len: int | None = max_model_len

    def generate(self, prompts: Sequence[Sequence[int]], max_new: int) -> list[list[int]]:
        sp = self._vllm.SamplingParams(temperature=0.0, max_tokens=max_new, skip_special_tokens=False)
        outs = self.llm.generate([{"prompt_token_ids": [int(t) for t in p]} for p in prompts], sp, use_tqdm=False)
        return [[int(t) for t in o.outputs[0].token_ids] for o in outs]

    def score(self, sequences: Sequence[Sequence[int]], positions: Sequence[int]) -> list[float]:
        sp = self._vllm.SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        outs = self.llm.generate([{"prompt_token_ids": [int(t) for t in s]} for s in sequences], sp, use_tqdm=False)
        res: list[float] = []
        for o, s, p in zip(outs, sequences, positions):
            res.append(float(o.prompt_logprobs[p][int(s[p])].logprob))
        return res


class HFEngine:
    """HuggingFace transformers: left-padded batched greedy generation and teacher-forced scoring."""

    name = "hf"

    def __init__(self, model: Any, tok: Any, device: Any, batch_size: int = 8, max_len: int | None = None) -> None:
        self.model = model
        self.tok = tok
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        eos_list = eos if isinstance(eos, list) else [eos]
        self.eos = {int(e) for e in eos_list + [tok.eos_token_id] if e is not None}
        self.pad = int(tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0))

    def generate(self, prompts: Sequence[Sequence[int]], max_new: int) -> list[list[int]]:
        import torch

        out: list[list[int]] = [[] for _ in prompts]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for b in range(0, len(order), self.batch_size):
            idx = order[b: b + self.batch_size]
            width = max(len(prompts[i]) for i in idx)
            ids = torch.full((len(idx), width), self.pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, i in enumerate(idx):
                p = [int(t) for t in prompts[i]]
                ids[r, width - len(p):] = torch.tensor(p)
                mask[r, width - len(p):] = 1
            with torch.no_grad():
                g = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                        max_new_tokens=max_new, do_sample=False, pad_token_id=self.pad)
            for r, i in enumerate(idx):
                new = [int(t) for t in g[r, width:].tolist()]
                cut = next((k for k, t in enumerate(new) if t in self.eos), None)
                out[i] = new[: cut + 1] if cut is not None else new
        return out

    def score(self, sequences: Sequence[Sequence[int]], positions: Sequence[int]) -> list[float]:
        import torch

        res = [0.0] * len(sequences)
        by_len: dict[int, list[int]] = {}
        for i, s in enumerate(sequences):
            by_len.setdefault(len(s), []).append(i)
        step = max(1, self.batch_size * 3)
        for idx in by_len.values():
            for b in range(0, len(idx), step):
                chunk = idx[b: b + step]
                x = torch.tensor([[int(t) for t in sequences[i]] for i in chunk], device=self.device)
                with torch.no_grad():
                    logits = self.model(input_ids=x).logits
                for r, i in enumerate(chunk):
                    p = positions[i]
                    lp = torch.log_softmax(logits[r, p - 1].float(), dim=-1)
                    res[i] = float(lp[int(sequences[i][p])])
        return res


# ---- words ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WordEntry:
    word: str
    role: str
    pair: str
    token_ids: tuple[int, ...]


def word_token_ids(tok: ChatTokenizer, word: str) -> tuple[int, ...]:
    """The word as it appears after a space: the pieces that follow ``a`` in ``a <word>``."""
    base = [int(i) for i in tok("a", add_special_tokens=False)["input_ids"]]
    full = [int(i) for i in tok("a " + word, add_special_tokens=False)["input_ids"]]
    if len(full) > len(base) and full[: len(base)] == base:
        ids = full[len(base):]
    else:
        ids = [int(i) for i in tok(" " + word, add_special_tokens=False)["input_ids"]]
    if tok.decode(ids).strip() != word:
        logger.warning("word %r tokenizes to %r, which decodes as %r", word, ids, tok.decode(ids))
    return tuple(ids)


def load_words(path: str | Path, roles: Sequence[str], limit_pairs: int = 0) -> list[tuple[str, str, str]]:
    """(word, role, pair) rows from a word list whose entries carry ``fragile``, ``case_twin`` and
    ``freq_control`` fields."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))["entries"]
    if limit_pairs:
        entries = entries[:limit_pairs]
    rows: list[tuple[str, str, str]] = []
    for e in entries:
        pair = str(e.get("pair") or e["fragile"])
        for role in roles:
            w = e.get(role)
            if w:
                rows.append((str(w), role, pair))
    return rows


# ---- phase 1: fragility --------------------------------------------------------------------------
@dataclass
class WordFragility:
    word: str
    token_ids: list[int]
    p_alone: float | None
    passes_gate: bool | None
    fragility: float | None
    cells: list[float] = field(default_factory=list)

    @property
    def band(self) -> str:
        if self.fragility is None:
            return "multi_token"
        if not self.passes_gate:
            return "fails_alone"
        if self.fragility >= 0.10:
            return "fragile_here"
        return "stable_here" if self.fragility == 0.0 else "between"


def paper_framing(model: str) -> Framing:
    """The paper's copy-prompt framing: chat for the Qwen3.8 models, raw for the rest."""
    return Framing.CHAT if "qwen3.8" in model.lower() else Framing.RAW


def measure_fragility(engine: Engine, tok: Any, words: Sequence[WordEntry],
                      cfg: ScanConfig) -> tuple[dict[str, WordFragility], dict[str, Any]]:
    """The copy protocol for the task words on this model, as ``FragileTokenDetector`` runs it."""
    prompt = CopyPrompt.build(tok, cfg.framing, cfg.demos)
    test_ids = sorted({e.token_ids[0] for e in words if len(e.token_ids) == 1})
    exclude = set(test_ids)
    cw = [t for t in common_word_ids(tok, len(tok), cfg.common_word_limit) if t not in exclude]
    pos_alone = prompt.positions(1, 0)[1]
    probe = cw + test_ids
    alone = dict(zip(probe, engine.score([prompt.single(t) for t in probe], [pos_alone] * len(probe))))
    filler = [t for t in cw if alone[t] > cfg.greedy_gate_logprob]
    logger.info("fragility: framing %s, filler pool %d of %d common words copy exactly alone",
                prompt.framing.value, len(filler), len(cw))
    if len(filler) < max(cfg.context_lengths):
        raise RuntimeError(f"only {len(filler)} common words copy exactly alone; the framing is probably wrong")
    bank = build_bank(filler, cfg.context_lengths, cfg.contexts_per_length, cfg.seed)
    seqs: list[list[int]] = []
    poss: list[int] = []
    for c in bank.contexts:
        base = prompt.sequence(list(c.ids))
        tpos, cpos = prompt.positions(len(c.ids), c.slot)
        for t in test_ids:
            q = list(base)
            q[tpos] = t
            q[cpos] = t
            seqs.append(q)
            poss.append(cpos)
    t0 = time.time()
    flat = engine.score(seqs, poss) if seqs else []
    cells = np.asarray(flat, dtype=np.float64).reshape(len(bank.contexts), len(test_ids)).T if seqs else np.zeros((0, 0))
    logger.info("fragility: %d cells scored in %.1fs", len(flat), time.time() - t0)
    col = {t: k for k, t in enumerate(test_ids)}
    out: dict[str, WordFragility] = {}
    for e in words:
        if len(e.token_ids) != 1:
            out[e.word] = WordFragility(e.word, list(e.token_ids), None, None, None)
            continue
        t = e.token_ids[0]
        row = cells[col[t]]
        out[e.word] = WordFragility(e.word, [t], math.exp(alone[t]), bool(alone[t] > cfg.greedy_gate_logprob),
                                    float((row < cfg.fail_logprob).mean()), [round(float(v), 4) for v in row])
    info = {"framing": prompt.framing.value, "prompt": prompt.describe(tok), "common_words": len(cw),
            "filler_pool": len(filler), "bank": bank.to_list(), "bank_text": bank.describe(tok),
            "config": cfg.to_dict()}
    return out, info


# ---- phase 2: agent tasks -------------------------------------------------------------------------
def run_episodes(engine: Engine, tok: Any, fmt: CallFormat, specials: Sequence[str], words: Sequence[WordEntry],
                 tasks: Sequence[AgentTask], thinking: bool, max_new: int, max_rounds: int,
                 case_sensitive: bool = False) -> list[Episode]:
    """Every word through every task. All live episodes advance one round per engine call."""
    episodes = [Episode.start(task, w.word, w.role, w.pair, w.token_ids, fmt, thinking, case_sensitive)
                for task in tasks for w in words]
    for rnd in range(max_rounds):
        live = [e for e in episodes if not e.done]
        if not live:
            break
        prompts: list[list[int]] = []
        runnable: list[Episode] = []
        for e in live:
            p = render(tok, e.messages, e.fmt, e.thinking, e.word_ids)
            if engine.max_len is not None and len(p) + max_new > engine.max_len:
                e.close_at_cap()
                e.transcript.append({"role": "note", "text": f"context full at {len(p)} tokens"})
                continue
            prompts.append(p)
            runnable.append(e)
        t0 = time.time()
        outs = engine.generate(prompts, max_new) if prompts else []
        for e, ids in zip(runnable, outs):
            e.advance(tok.decode(ids, skip_special_tokens=False), specials)
        logger.info("round %d: %d episodes advanced in %.1fs; %d finished of %d", rnd + 1, len(runnable),
                    time.time() - t0, sum(e.done for e in episodes), len(episodes))
    for e in episodes:
        e.close_at_cap()
    return episodes


def _group(records: Sequence[dict[str, Any]], *keys: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(" | ".join(str(r.get(k)) for k in keys), []).append(r)
    return {k: episode_summary(v) for k, v in sorted(groups.items())}


def run(model: str, words_path: str, out: str, *, engine_name: str = "vllm", thinking: str = "off",
        task_names: Sequence[str] | None = None, roles: Sequence[str] = ROLES, max_new: int = 1024,
        max_new_thinking: int = 4096, max_rounds: int = 6, case_sensitive: bool = False, framing: str = "paper",
        tensor_parallel_size: int = 1, max_model_len: int = 32768, gpu_memory_utilization: float = 0.90,
        batch_size: int = 8, trust_remote_code: bool = False, limit_pairs: int = 0) -> dict[str, Any]:
    """Both phases on one model; writes the JSON report to ``out`` and returns it."""
    from transformers import AutoTokenizer

    t_start = time.time()
    tok: Any = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    fmt = detect_format(tok)
    specials = sorted({s for s in getattr(tok, "all_special_tokens", []) if s}, key=len, reverse=True)
    engine: Engine
    if engine_name == "vllm":
        engine = VLLMEngine(model, tensor_parallel_size, max_model_len, gpu_memory_utilization, trust_remote_code)
    else:
        from .models import load_model

        loaded = load_model(model, trust_remote_code=trust_remote_code)
        engine = HFEngine(loaded.model, loaded.tokenizer, loaded.device, batch_size, max_model_len)
    words = [WordEntry(w, role, pair, word_token_ids(tok, w)) for w, role, pair in load_words(words_path, roles, limit_pairs)]
    fr = paper_framing(model) if framing == "paper" else Framing(framing)
    logger.info("%s: engine %s, tool format %s, copy framing %s, %d words, thinking %s", model, engine.name,
                fmt.value, fr.value, len(words), thinking)

    frag, frag_info = measure_fragility(engine, tok, words, ScanConfig(framing=fr))
    for w in words:
        f = frag[w.word]
        logger.info("  %-14s %-12s fragility %s  alone %s", w.word, w.role,
                    "n/a" if f.fragility is None else f"{f.fragility:.2f}", "n/a" if f.p_alone is None else f"{f.p_alone:.3f}")

    tasks = [TASK_BY_NAME[n] for n in task_names] if task_names else list(TASKS)
    modes = {"off": [False], "on": [True], "both": [False, True]}[thinking]
    records: list[dict[str, Any]] = []
    for think in modes:
        eps = run_episodes(engine, tok, fmt, specials, words, tasks, think,
                           max_new_thinking if think else max_new, max_rounds, case_sensitive)
        for e in eps:
            f = frag[e.word]
            r = e.record()
            r.update({"fragility_here": f.fragility, "p_alone_here": f.p_alone, "passes_gate_here": f.passes_gate,
                      "band": f.band})
            records.append(r)

    payload: dict[str, Any] = {
        "model": model, "engine": engine.name, "tool_format": fmt.value, "native_tools": fmt.native,
        "thinking": thinking, "tasks": [{"name": t.name, "text": t.text.replace("\x00", " <word>")} for t in tasks],
        "case_sensitive_world": case_sensitive, "max_rounds": max_rounds,
        "fragility": {k: asdict(v) | {"band": v.band} for k, v in frag.items()}, "fragility_protocol": frag_info,
        "summary": {"by_role": _group(records, "thinking", "role"), "by_band": _group(records, "thinking", "band"),
                    "by_pair_role": _group(records, "thinking", "pair", "role")},
        "records": records, "seconds": round(time.time() - t_start, 1),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    for key, s in payload["summary"]["by_role"].items():
        logger.info("%-22s n=%3d  query had word %.2f  final about word %.2f  form ok %.2f  finished %.2f  fabricated %.2f",
                    key, int(s["n"]), s.get("query_had_word", float("nan")), s.get("final_about_word", float("nan")),
                    s.get("format_ok", float("nan")), s.get("finished", float("nan")), s.get("fabricated", float("nan")))
    logger.info("wrote %s (%.0fs)", out, payload["seconds"])
    return payload


# ---- command line ---------------------------------------------------------------------------------
def cmd_agent(a: argparse.Namespace) -> int:
    run(a.model, a.words, a.out, engine_name=a.engine, thinking=a.thinking, task_names=a.tasks, roles=a.roles,
        max_new=a.max_new, max_new_thinking=a.max_new_thinking, max_rounds=a.max_rounds,
        case_sensitive=a.case_sensitive, framing=a.framing, tensor_parallel_size=a.tp,
        max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization, batch_size=a.batch,
        trust_remote_code=a.trust_remote_code, limit_pairs=a.limit_pairs)
    return 0


def add_agent_subcommand(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    p = sub.add_parser("agent", help="agent tasks with tool calling on a word list, with each word's fragility on the same model")
    p.add_argument("--model", required=True)
    p.add_argument("--words", required=True, help="JSON word list: entries with fragile, case_twin and freq_control")
    p.add_argument("--out", required=True)
    p.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--thinking", choices=["off", "on", "both"], default="off")
    p.add_argument("--tasks", nargs="*", choices=list(TASK_BY_NAME), help="default: all six")
    p.add_argument("--roles", nargs="*", choices=list(ROLES), default=list(ROLES))
    p.add_argument("--max-new", type=int, default=1024, help="new tokens per turn without thinking")
    p.add_argument("--max-new-thinking", type=int, default=4096, help="new tokens per turn with thinking")
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--case-sensitive", action="store_true", help="the mocked search ignores case-changed queries")
    p.add_argument("--framing", choices=["paper", "raw", "chat", "auto"], default="paper",
                   help="copy-prompt framing for the fragility phase (paper: chat for Qwen3.8, raw otherwise)")
    p.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel GPUs")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--batch", type=int, default=8, help="HF engine batch size")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--limit-pairs", type=int, default=0, help="dry runs: only the first N word pairs")
    p.set_defaults(func=cmd_agent)


__all__ = ["Engine", "HFEngine", "VLLMEngine", "WordEntry", "WordFragility", "add_agent_subcommand", "cmd_agent",
           "load_words", "measure_fragility", "paper_framing", "run", "run_episodes", "word_token_ids"]
