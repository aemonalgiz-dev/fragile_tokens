from __future__ import annotations

import pytest

from fragile_tokens.config import Framing, ScanConfig
from fragile_tokens.contexts import Context, ContextBank, build_bank, common_word_ids
from fragile_tokens.prompt import INSTRUCTION, CopyPrompt


def test_raw_prompt_positions(tok) -> None:
    p = CopyPrompt.build(tok, Framing.RAW)
    assert p.framing is Framing.RAW
    assert tok.decode(list(p.head)).startswith(INSTRUCTION)
    assert tok.decode(list(p.pre_text)) == "Text:" and tok.decode(list(p.pre_copy)) == "\nCopy:"
    ctx = [10, 11, 12, 13]
    seq = p.sequence(ctx)
    tpos, cpos = p.positions(len(ctx), 2)
    assert seq[tpos] == 12 and seq[cpos] == 12
    assert seq[cpos - len(ctx) - len(p.pre_copy)] == 12  # the copy mirrors the text
    assert seq[-len(ctx):] == ctx  # the copy span is last
    with pytest.raises(ValueError):
        p.positions(4, 4)


def test_single_probe_sequence(tok) -> None:
    p = CopyPrompt.build(tok, Framing.RAW)
    s = p.single(42)
    assert s[-1] == 42 and s[p.text_start] == 42 and len(s) == len(p.head) + len(p.pre_text) + 1 + len(p.pre_copy) + 1


def test_auto_framing_falls_back_to_raw_without_template(tok) -> None:
    p = CopyPrompt.build(tok, Framing.AUTO)
    assert p.framing is Framing.RAW
    q = CopyPrompt.build(tok, Framing.CHAT)  # requested but impossible: raw with a warning
    assert q.framing is Framing.RAW


def test_chat_framing_splits_template(chat_tok) -> None:
    p = CopyPrompt.build(chat_tok, Framing.AUTO)
    assert p.framing is Framing.CHAT
    assert p.pre_text == ()
    head = chat_tok.decode(list(p.head))
    assert head.startswith("<|im_start|>user\n" + INSTRUCTION) and head.endswith("Text:")
    assert chat_tok.decode(list(p.pre_copy)).endswith("assistant\nCopy:")
    seq = p.sequence([10, 11])
    assert seq[-2:] == [10, 11]


def test_common_words_and_bank(tok, ids) -> None:
    cw = common_word_ids(tok, len(tok), limit=50)
    assert len(cw) == 50
    assert all(tok.decode([i])[0] == " " and tok.decode([i])[1:].isalpha() and tok.decode([i])[1:].islower() for i in cw)
    bank = build_bank(cw, (4, 8), 3, seed=1)
    assert len(bank) == 6
    for c in bank.contexts:
        assert len(set(c.ids)) == len(c.ids) and 1 <= c.slot <= len(c.ids) - 2
    assert build_bank(cw, (4, 8), 3, seed=1) == bank  # deterministic
    with pytest.raises(ValueError):
        build_bank(cw[:5], (8,), 1, seed=0)
    d = bank.to_list()
    assert ContextBank.from_list(d) == bank
    assert bank.describe(tok)[0].count("<token>") == 1


def test_context_validation() -> None:
    with pytest.raises(ValueError):
        Context((1, 2, 3), 3)
    assert Context((1, 2, 3), 1).with_token(9) == [1, 9, 3]


def test_config_roundtrip_and_validation() -> None:
    cfg = ScanConfig(context_lengths=(8, 16), contexts_per_length=2, framing=Framing.CHAT)
    assert cfg.n_contexts == 4
    assert ScanConfig.from_dict(cfg.to_dict()) == cfg
    assert abs(cfg.greedy_gate_logprob + 0.6931) < 1e-3
    with pytest.raises(ValueError):
        ScanConfig(context_lengths=())
    with pytest.raises(ValueError):
        ScanConfig(greedy_gate_p=1.5)
