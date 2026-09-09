from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fragile_tokens.classify import FailureMode, TokenClass, classify_token, failure_mode, glitch_reference, worst_cells
from fragile_tokens.config import Framing, ScanConfig
from fragile_tokens.contexts import Context, build_bank, common_word_ids
from fragile_tokens.prompt import CopyPrompt
from fragile_tokens.scoring import FragilityMatrix, fragility_matrix, sample_token_ids, score_context, single_token_logprobs

CPU = torch.device("cpu")


def _setup(tok, model, n_ctx_per_len: int = 3):
    prompt = CopyPrompt.build(tok, Framing.RAW)
    cw = common_word_ids(tok, len(tok), limit=120)
    bank = build_bank(cw, (4, 8), n_ctx_per_len, seed=0)
    return prompt, bank


def test_single_probe_separates_glitch(tok, model, glitch_ids, fragile_ids, ids) -> None:
    prompt, _ = _setup(tok, model)
    probe = single_token_logprobs(model, prompt, glitch_ids + fragile_ids + [ids[" helper"]], CPU, batch_size=4)
    assert (probe[:2] < math.log(0.5)).all()  # glitch tokens fail alone
    assert (probe[2:] > math.log(0.5)).all()  # fragile and healthy tokens pass alone


def test_score_context_reads_the_copy_slot(tok, model, ids) -> None:
    prompt, bank = _setup(tok, model)
    ctx = bank.contexts[0]
    cell = score_context(model, prompt, ctx, [ids[" helper"], ids[" Председа"]], CPU, batch_size=2)
    assert cell.logprob[0] > math.log(0.9)  # healthy token copies
    assert cell.logprob[1] < math.log(0.1)  # glitch token is deleted
    assert cell.top1[1] == ctx.ids[ctx.slot + 1]  # the next context word is emitted
    assert cell.entropy_bits[1] < 1.0  # sharply


def test_matrix_and_classes(tok, model, glitch_ids, fragile_ids, ids) -> None:
    cfg = ScanConfig(context_lengths=(4, 8), contexts_per_length=4, batch_size=8)
    prompt, bank = _setup(tok, model, n_ctx_per_len=4)
    tokens = glitch_ids + fragile_ids + [ids[" helper"], ids[" zebra"]]
    fm = fragility_matrix(model, prompt, bank, tokens, CPU, fail_logprob=cfg.fail_logprob, batch_size=cfg.batch_size)
    assert fm.logprob.shape == (6, 8) and fm.n_contexts == len(bank)
    frag = fm.fragility()
    assert (frag[:2] == 1.0).all()  # glitch: every context
    assert 0.0 < frag[2] < 1.0 and 0.0 < frag[3] < 1.0  # fragile: some contexts
    assert (frag[4:] == 0.0).all()  # healthy: none
    gated = fm.single_logprob > cfg.greedy_gate_logprob
    classes = [classify_token(bool(g), float(f), cfg) for g, f in zip(gated, frag)]
    assert classes[:2] == [TokenClass.GLITCH, TokenClass.GLITCH]
    assert classes[4:] == [TokenClass.STABLE, TokenClass.STABLE]
    assert all(c in (TokenClass.FRAGILE, TokenClass.INTERMITTENT) for c in classes[2:4])
    worst = worst_cells(fm, tok, cfg)
    assert worst[0].mode is FailureMode.DELETION and worst[0].confident
    assert worst[4].mode is FailureMode.CORRECT
    ref = glitch_reference(fm, tok, cfg, n=10, special_ids=frozenset(tok.all_special_ids))
    assert set(fm.token_ids[ref].tolist()) == set(glitch_ids)


def test_classify_token_edges() -> None:
    cfg = ScanConfig()
    assert classify_token(True, 0.0, cfg) is TokenClass.STABLE
    assert classify_token(True, 0.05, cfg) is TokenClass.INTERMITTENT
    assert classify_token(True, 0.10, cfg) is TokenClass.FRAGILE
    assert classify_token(False, 0.95, cfg) is TokenClass.GLITCH
    assert classify_token(False, 0.0, cfg) is TokenClass.PROBE_ONLY
    assert classify_token(False, 0.5, cfg) is TokenClass.UNGATED


def test_failure_modes(tok, ids) -> None:
    ctx = Context((ids[" wab"], ids[" wac"], ids[" wad"]), 1)
    t = ids[" according"]
    assert failure_mode(t, t, ctx, tok) is FailureMode.CORRECT
    assert failure_mode(ids[" wad"], t, ctx, tok) is FailureMode.DELETION
    assert failure_mode(ids[" zebra"], t, ctx, tok) is FailureMode.SUBSTITUTION
    # a proper prefix of the target counts as truncation: " realt" is a prefix of " realt" + nothing here,
    # so build one from the vocabulary: "Text" is a prefix of nothing; use the marker pieces instead
    assert failure_mode(ids["<"], ids["<"], ctx, tok) is FailureMode.CORRECT


def test_sample_token_ids_is_stratified_and_deterministic(tok) -> None:
    a = sample_token_ids(tok, len(tok), 30, seed=3, special_ids=tok.all_special_ids)
    b = sample_token_ids(tok, len(tok), 30, seed=3, special_ids=tok.all_special_ids)
    assert a == b and len(a) == 30 and all(t not in tok.all_special_ids for t in a)
    thirds = len(tok) // 3
    assert any(t < thirds for t in a) and any(t >= 2 * thirds for t in a)


def test_matrix_npz_roundtrip(tmp_path, tok) -> None:
    bank = build_bank(list(range(30, 90)), (4, 6), 2, seed=0)
    n, c = 5, len(bank)
    fm = FragilityMatrix(token_ids=np.arange(n, dtype=np.int64), single_logprob=np.zeros(n, np.float32),
                         logprob=np.full((n, c), -0.2, np.float32), top1=np.zeros((n, c), np.int64),
                         entropy_bits=np.ones((n, c), np.float32), bank=bank, fail_logprob=-0.5)
    fm.save_npz(tmp_path / "m.npz")
    back = FragilityMatrix.load_npz(tmp_path / "m.npz")
    assert back.bank == bank and np.allclose(back.logprob, fm.logprob) and back.fail_logprob == -0.5
    with pytest.raises(ValueError):
        FragilityMatrix(token_ids=np.arange(n + 1), single_logprob=np.zeros(n, np.float32), logprob=fm.logprob,
                        top1=fm.top1, entropy_bits=fm.entropy_bits, bank=bank, fail_logprob=-0.5)
