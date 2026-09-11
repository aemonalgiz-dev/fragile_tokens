"""Does token-level fragility predict the plausible-near-miss hallucination on
real entities -- beyond how rare the entity is and how well the model knows it?

The fragility result (fragility.py) is that some ordinary tokens are copied
perfectly in isolation and confidently deleted, substituted or truncated in a
minority of contexts. Structurally that is a hallucination in miniature: fluent,
confident, wrong, a plausible neighbour of the truth, with the correct
information still present internally. The class of real hallucination it could
explain is the NEAR-MISS on a rare entity: the fabricated API that almost
exists, the name with one syllable swapped, the method with its prefix dropped.

This is the bridge test. Real entities (live Python/torch/transformers
identifiers; real proper nouns) are used in a NATURAL task, not a copy prompt:
"write one line that imports or calls `X`", "write one sentence that mentions
X". The output is scored as EXACT (the entity appears verbatim), NEAR-MISS (no
exact match, but a span with fuzzy similarity >= 0.75 -- the hallucination), or
MISS (nothing close).

Predictor under test: fragility of the entity's constituent tokens, aggregated
(max, mean, fraction fragile), from the N x C matrix in which those tokens were
deliberately included.

Controls it must beat, all fitted jointly in a cross-validated logistic model:
  RARITY        log token ids (BPE merge order ~ frequency), token count,
                character length, entity type
  SINGLE PROBE  the standard one-context copy score on the constituents
                (what every existing detector would measure)
  FAMILIARITY   the model's own teacher-forced log-probability of the entity
                after a neutral prefix -- how well it knows the whole string.
                The fairest control available: if fragility adds nothing over
                this, the two phenomena are lookalikes.

Verdict: fragility earns a place in the hallucination story only if adding it
to (rarity + single probe + familiarity) raises out-of-fold AUC for near-miss
with a bootstrap CI that excludes zero improvement.
"""
from __future__ import annotations
import argparse, difflib, json, re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .stats import auc
from .fragility_predict import logistic_cv, boot_auc, spearman

CODE_TMPL = ("Write exactly one line of Python that imports or calls `{e}`. "
             "Output only the code, nothing else.")
NOUN_TMPL = ("Write one short sentence that mentions{e}. Output only the sentence.")


def build(tok, text):
    try:
        s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                    add_generation_prompt=True)
        return tok(s, add_special_tokens=False)["input_ids"]
    except Exception:
        return tok(text + "\nAnswer:", add_special_tokens=False)["input_ids"]


@torch.no_grad()
def generate(model, tok, dev, prompts, max_new=40, batch=12):
    pad = tok.pad_token_id or tok.eos_token_id or 0
    order = sorted(range(len(prompts)), key=lambda k: len(prompts[k]))
    outs = [None] * len(prompts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        ch = [prompts[k] for k in idx]
        mx = max(len(p) for p in ch)
        ids = torch.tensor([[pad] * (mx - len(p)) + p for p in ch], device=dev)
        att = torch.tensor([[0] * (mx - len(p)) + [1] * len(p) for p in ch], device=dev)
        g = model.generate(ids, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=pad)
        for k, row in zip(idx, g):
            outs[k] = tok.decode(row[mx:].tolist(), skip_special_tokens=True)
    return outs


@torch.no_grad()
def familiarity(model, tok, dev, items, batch=16):
    """Mean per-token logprob of the entity after a neutral prefix."""
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    pad = tok.pad_token_id or tok.eos_token_id or 0
    out = np.zeros(len(items))
    for s in range(0, len(items), batch):
        ch = items[s:s + batch]
        seqs, spans = [], []
        for it in ch:
            pre = e("The name is:") if it["type"] == "noun" else e("The identifier is `")
            seqs.append(pre + it["ids"]); spans.append((len(pre), len(pre) + len(it["ids"])))
        mx = max(len(q) for q in seqs)
        ids = torch.tensor([q + [pad] * (mx - len(q)) for q in seqs], device=dev)
        lp = torch.log_softmax(model(input_ids=ids).logits[:, :-1].float(), -1)
        for k, (a_, b_) in enumerate(spans):
            tgt = ids[k, a_:b_]
            out[s + k] = lp[k, a_ - 1:b_ - 1].gather(-1, tgt.unsqueeze(-1)).mean().item()
    return out


def score(entity, etype, out):
    """0 exact, 1 near-miss (fuzzy >= 0.75), 2 miss. Also returns the best span."""
    if etype == "code":
        spans = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", out)
        target = entity
    else:
        words = re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", out)
        n = len(entity.split())
        spans = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        target = entity
    if target in out or (etype == "noun" and target.lower() in out.lower()):
        return 0, target
    best, best_r = "", 0.0
    for sp in spans:
        r = difflib.SequenceMatcher(None, sp.lower(), target.lower()).ratio()
        if r > best_r:
            best, best_r = sp, r
    return (1 if best_r >= 0.75 else 2), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--entities", default="results/entities.json")
    ap.add_argument("--pt", default="results/fragility_L.pt")
    ap.add_argument("--out", default="results/hallucination_bridge.json")
    a = ap.parse_args()

    items = json.load(open(a.entities))["items"]
    d = torch.load(a.pt, weights_only=False)
    M = d["M"]; thr = d["fail_thr"]; single = d["single"]
    frag_of = {int(t): float((M[k] < thr).mean()) for k, t in enumerate(d["tokens"])}
    single_of = {int(t): float(single[k]) for k, t in enumerate(d["tokens"])}
    items = [it for it in items if all(int(i) in frag_of for i in it["ids"])]
    print(f"{len(items)} entities with fragility for every constituent token")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

    prompts = [build(tok, CODE_TMPL.format(e=it["entity"]) if it["type"] == "code"
                     else NOUN_TMPL.format(e=it["prompt_form"])) for it in items]
    print("generating...", flush=True)
    outs = generate(model, tok, dev, prompts)
    fam = familiarity(model, tok, dev, items)

    rows = []
    for it, o, f in zip(items, outs, fam):
        cls, span = score(it["entity"], it["type"], o)
        ids = [int(i) for i in it["ids"]]
        fr = np.array([frag_of[i] for i in ids]); sg = np.array([single_of[i] for i in ids])
        rows.append({"entity": it["entity"], "type": it["type"], "source": it["source"],
                     "cls": cls, "out": o[:160], "span": span,
                     "n_tok": len(ids), "n_chars": len(it["entity"]),
                     "log_id_mean": float(np.log1p(ids).mean()),
                     "log_id_max": float(np.log1p(ids).max()),
                     "single_min": float(sg.min()), "single_mean": float(sg.mean()),
                     "familiarity": float(f),
                     "frag_max": float(fr.max()), "frag_mean": float(fr.mean()),
                     "frag_any": float((fr > 0).mean())})
    cls = np.array([r["cls"] for r in rows])
    typ = np.array([r["type"] for r in rows])

    print()
    print("=" * 78)
    print("OUTCOMES  (0 exact, 1 near-miss = the hallucination, 2 miss)")
    print("=" * 78)
    for t in ("code", "noun"):
        m = typ == t
        print(f"  {t:>5} (n={m.sum():4d}): exact {(cls[m]==0).mean():.3f}   "
              f"near-miss {(cls[m]==1).mean():.3f}   miss {(cls[m]==2).mean():.3f}")
    print("\n  near-miss specimens:")
    shown = 0
    for r in sorted(rows, key=lambda r: -r["frag_max"]):
        if r["cls"] == 1 and shown < 10:
            print(f"    {r['entity']!r:>40} -> {r['span']!r:<40} frag_max {r['frag_max']:.2f} "
                  f"fam {r['familiarity']:+.2f}")
            shown += 1

    # ---------- prediction ----------
    feats = {k: np.array([r[k] for r in rows], float) for k in
             ("n_tok", "n_chars", "log_id_mean", "log_id_max", "single_min", "single_mean",
              "familiarity", "frag_max", "frag_mean", "frag_any")}
    feats["is_code"] = (typ == "code").astype(float)
    y_fail = (cls != 0).astype(int)
    y_near = (cls == 1).astype(int)

    print()
    print("=" * 78)
    print("PER-FEATURE AUC   target A: any failure   target B: near-miss vs everything else")
    print("=" * 78)
    print(f"{'feature':>14} | {'AUC fail':>8} | {'AUC near':>8} | {'spearman(fail)':>14}")
    print("-" * 56)
    per = {}
    for nm, v in feats.items():
        rho = spearman(v, y_fail.astype(float)); s = v if rho >= 0 else -v
        A1 = auc(s, y_fail) if 0 < y_fail.sum() < len(y_fail) else float("nan")
        A2 = auc(s, y_near) if 0 < y_near.sum() < len(y_near) else float("nan")
        per[nm] = {"auc_fail": A1, "auc_near": A2, "spearman": rho}
        print(f"{nm:>14} | {A1:8.3f} | {A2:8.3f} | {rho:+14.3f}")

    print()
    print("=" * 78)
    print("CROSS-VALIDATED LOGISTIC: does fragility add to the controls?")
    print("=" * 78)
    R = lambda *ks: np.column_stack([feats[k] for k in ks])
    rarity = ("n_tok", "n_chars", "log_id_mean", "log_id_max", "is_code")
    ctrl = rarity + ("single_min", "single_mean", "familiarity")
    sets = {"rarity only": R(*rarity),
            "rarity + single probe": R(*(rarity + ("single_min", "single_mean"))),
            "rarity + single + familiarity (controls)": R(*ctrl),
            "controls + fragility": R(*(ctrl + ("frag_max", "frag_mean", "frag_any"))),
            "fragility only": R("frag_max", "frag_mean", "frag_any")}
    cv = {}
    for tgt_name, y in (("any failure", y_fail), ("near-miss", y_near)):
        if y.sum() < 10 or y.sum() == len(y):
            print(f"  {tgt_name}: degenerate ({y.sum()} positives)"); continue
        print(f"\n  target = {tgt_name}  ({y.sum()} positives / {len(y)})")
        preds = {}
        for nm, X in sets.items():
            p = logistic_cv(X, y); preds[nm] = p
            A, lo, hi = boot_auc(p, y)
            cv[f"{tgt_name}|{nm}"] = {"auc": A, "ci": [lo, hi]}
            print(f"    {nm:>42}: AUC {A:.3f} [{lo:.3f}, {hi:.3f}]")
        # paired bootstrap of the IMPROVEMENT from adding fragility to the controls
        rng = np.random.default_rng(5); diffs = []
        pc, pf = preds["rarity + single + familiarity (controls)"], preds["controls + fragility"]
        for _ in range(1000):
            ii = rng.integers(0, len(y), len(y))
            if 0 < y[ii].sum() < len(ii):
                diffs.append(auc(pf[ii], y[ii]) - auc(pc[ii], y[ii]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        cv[f"{tgt_name}|improvement"] = [float(np.mean(diffs)), float(lo), float(hi)]
        print(f"    improvement from adding fragility: {np.mean(diffs):+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]{'   *' if lo > 0 else ''}")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    imp = cv.get("near-miss|improvement")
    if imp and imp[1] > 0:
        print(f"  fragility ADDS to rarity + single probe + familiarity for near-miss "
              f"hallucination: +{imp[0]:.3f} AUC, CI excludes 0.")
    elif imp:
        print(f"  fragility does NOT reliably add over the controls for near-miss "
              f"(+{imp[0]:.3f}, CI [{imp[1]:+.3f}, {imp[2]:+.3f}]). The two phenomena "
              f"share a cause (rarity) but fragility is not an independent predictor here.")
    else:
        print("  near-miss target degenerate; no verdict")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "n": len(rows), "per_feature": per, "cv": cv,
               "rates": {t: {"exact": float((cls[typ == t] == 0).mean()),
                             "near": float((cls[typ == t] == 1).mean()),
                             "miss": float((cls[typ == t] == 2).mean())} for t in ("code", "noun")},
               "rows": rows}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
