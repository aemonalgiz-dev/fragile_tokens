from __future__ import annotations

import io
import json
import logging

import numpy as np
import torch

from fragile_tokens import cli
from fragile_tokens.classify import TokenClass
from fragile_tokens.config import Framing, ScanConfig
from fragile_tokens.detector import FragileTokenDetector, fragility_of
from fragile_tokens.geometry import Embeddings, auc, glitch_projection, glitch_proximity, rank_vocabulary
from fragile_tokens.models import LoadedModel
from fragile_tokens.report import ScanResult

from conftest import VOCAB, FakeCopyModel, common


def _detector(tok, model, **cfg_kw) -> FragileTokenDetector:
    cfg = ScanConfig(context_lengths=(4, 8), contexts_per_length=3, batch_size=8, framing=Framing.RAW, common_word_limit=120, **cfg_kw)
    loaded = LoadedModel(model=model, tokenizer=tok, device=torch.device("cpu"), name="fake/copier",
                         info={"vocab_size": len(VOCAB)})
    return FragileTokenDetector(loaded, cfg)


def test_scan_classifies_and_roundtrips(tmp_path, tok, model, glitch_ids, fragile_ids, ids) -> None:
    det = _detector(tok, model)
    tokens = glitch_ids + fragile_ids + [ids[" helper"], ids[" zebra"]]
    result = det.scan(token_ids=tokens)
    by_id = result.lookup()
    assert all(by_id[t].token_class is TokenClass.GLITCH for t in glitch_ids)
    assert all(by_id[t].token_class in (TokenClass.FRAGILE, TokenClass.INTERMITTENT) for t in fragile_ids)
    assert by_id[ids[" helper"]].token_class is TokenClass.STABLE
    s = result.summary()
    assert s["tokens_scanned"] == 6 and s["contexts"] == 6 and s["passes_gate"] == 4
    assert fragility_of(result, ids[" helper"]) == 0.0 and np.isnan(fragility_of(result, 999_999))
    path = tmp_path / "scan.json"
    result.save(path)
    back = ScanResult.load(path)
    assert back.config == result.config and back.matrix.bank == result.matrix.bank
    assert [r.to_dict() for r in back.reports] == [r.to_dict() for r in result.reports]
    assert np.allclose(back.matrix.logprob, result.matrix.logprob, atol=1e-4)
    md = result.to_markdown(top=5, classes=(TokenClass.GLITCH, TokenClass.STABLE))
    assert "glitch tokens" in md and "stable tokens" in md and "Председа" in md
    n = result.to_csv(tmp_path / "scan.csv", classes=[TokenClass.GLITCH])
    assert n == 2


def test_check_text_scores_unknown_tokens(tok, model, ids, caplog) -> None:
    det = _detector(tok, model)
    scan = det.scan(token_ids=[ids[" helper"]])
    calls_before = model.calls
    with caplog.at_level(logging.INFO, logger="fragile_tokens"):
        reports = det.check_text(" helper according zebra", scan)
    assert [r.token_id for r in reports] == [ids[" helper"], ids[" according"], ids[" zebra"]]
    assert reports[1].fragility > 0 and reports[2].fragility == 0
    assert model.calls > calls_before  # the two unknown tokens were measured on the scan's bank
    assert any("not in the scan" in m for m in caplog.messages)


def test_geometry_scores_prefer_the_planted_structure(tok, model, glitch_ids, fragile_ids, ids) -> None:
    det = _detector(tok, model)
    tokens = glitch_ids + fragile_ids + [ids[common(i)] for i in range(150, 190)]
    result = det.scan(token_ids=tokens)
    rng = np.random.default_rng(0)
    e = rng.standard_normal((len(VOCAB), 16)).astype(np.float32)
    anchor = rng.standard_normal(16).astype(np.float32) * 4
    for t in glitch_ids + fragile_ids:  # glitch and fragile rows share a neighbourhood
        e[t] = anchor + 0.3 * rng.standard_normal(16)
    emb = Embeddings(e_in=torch.tensor(e), e_out=torch.tensor(e), tied=True)
    g = det.geometry(result, emb, k=1)
    assert set(g.reference_ids.tolist()) == set(glitch_ids)
    assert g.auc_proximity_out > 0.9 and g.auc_projection > 0.9
    ranked = rank_vocabulary(emb, g.reference_ids, k=1)
    assert ranked.shape == (len(VOCAB),)
    assert all(ranked[t] > np.median(ranked) for t in fragile_ids)


def test_auc_and_scores() -> None:
    assert auc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == 0.75
    assert auc([1, 1, 1, 1], [True, False, True, False]) == 0.5
    assert np.isnan(auc([1, 2], [True, True]))
    rows = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    ref = np.array([[1.0, 0.0]])
    proj = glitch_projection(rows, rows.mean(0), ref)
    assert proj[0] > proj[1] > proj[2]
    prox = glitch_proximity(rows, np.zeros(2), ref, k=1, self_ref_index=np.array([0, -1, -1]))
    assert prox[0] == -2.0 and prox[1] < 1.0  # a glitch row is not its own neighbour


def test_cli_report_and_check(tmp_path, tok, model, glitch_ids, ids, monkeypatch) -> None:
    det = _detector(tok, model)
    scan = det.scan(token_ids=glitch_ids + [ids[" helper"], ids[" according"]])
    p = tmp_path / "scan.json"
    scan.save(p)
    # report
    out = io.StringIO()
    args = cli.build_parser().parse_args(["report", str(p), "--classes", "glitch", "--top", "5", "--json", str(tmp_path / "s.json")])
    assert cli.cmd_report(args, out) == 0
    assert "glitch tokens" in out.getvalue() and json.loads((tmp_path / "s.json").read_text())["tokens_scanned"] == 4
    # check without a model: tokens come from the scan through a tokenizer the CLI loads
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", staticmethod(lambda *a, **k: tok), raising=False)
    out = io.StringIO()
    args = cli.build_parser().parse_args(["check", "--scan", str(p), "--text", " helper according"])
    assert cli.cmd_check(args, out) == 0
    lines = out.getvalue().strip().split("\n")
    assert len(lines) == 4 and "stable" in lines[2] and ("fragile" in lines[3] or "intermittent" in lines[3])


def test_cli_parser_and_logging() -> None:
    p = cli.build_parser()
    a = p.parse_args(["scan", "--model", "x/y", "--n-tokens", "10", "--framing", "chat", "--context-lengths", "8", "16"])
    cfg = cli._config_from(a)
    assert cfg.framing is Framing.CHAT and cfg.context_lengths == (8, 16) and cfg.n_contexts == 12
    cli.configure_logging(verbose=0, quiet=True)
    assert logging.getLogger().level == logging.WARNING


def test_fake_model_is_a_causal_lm(model) -> None:
    x = torch.tensor([[1, 2, 3]])
    assert model(input_ids=x).logits.shape == (1, 3, len(VOCAB))
    assert isinstance(model, FakeCopyModel)
