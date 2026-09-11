"""The substitutions are lexical completions, not embedding neighbours.

embed_predict.py tested whether the direct path E_in[t].E_out[j] predicts which
token a model substitutes when it fails to reproduce t. It does not: top-1
agreement with the actual output was 0.003, and the margin's AUC for predicting
failure at all (0.546) was worse than the trivial row-norm baseline (0.581).

But the failures themselves are strikingly regular:

    'ction'    -> 'tion'          'ience'    -> 'science'
    'osition'  -> 'position'      'plement'  -> 'supplement'
    'ension'   -> 'tension'       'istance'  -> 'distance'
    'ensions'  -> 'extensions'    'structor' -> 'instructor'

The model is not retrieving a nearby embedding. It is COMPLETING A FRAGMENT INTO
THE NEAREST REAL WORD -- behaving like a spelling corrector. That predicts the
same thing the geometry failed to predict, from string properties alone, and it
retro-explains the n-gram results: 'ance'+'led' came back as "ancelled" read as
"cancelled".

So the hypothesis here is orthographic, not geometric:

  H1  A token fails to reproduce when it is a PROPER FRAGMENT of a real word --
      an affix that the lexicon can complete -- and succeeds when it is a
      standalone word or an unambiguous symbol.
  H2  The substitute is predictable: it is the completion, not a neighbour.

The lexicon is built from the model's own vocabulary: in BPE a token decoding to
a leading space plus letters is a whole word, which gives a word list that is
automatically matched to the model's own distribution and needs no external
resource.

If H1 holds it is a static, model-free predictor of exactly the tokens that
break, which is what the geometric route was supposed to deliver and did not.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .stats import auc
from .ngram_identify import reproduce


def build_lexicon(tok, V):
    """Whole words, from tokens that decode to ' xxxx' with xxxx alphabetic."""
    lex = set()
    for i in range(V):
        s = tok.decode([i])
        if s.startswith(" ") and s[1:].isalpha() and len(s) > 2:
            lex.add(s[1:].lower())
    return lex


def features(s, lex, by_suffix, by_prefix):
    """String-only properties of one token's surface form."""
    w = s.strip().lower()
    out = {"is_word": float(w in lex),
           "alpha": float(w.isalpha() and len(w) > 0),
           "len": float(len(w))}
    out["suffix_of"] = float(len(by_suffix.get(w, ())) > 0) if w.isalpha() else 0.0
    out["prefix_of"] = float(len(by_prefix.get(w, ())) > 0) if w.isalpha() else 0.0
    out["n_completions"] = float(len(by_suffix.get(w, ())) + len(by_prefix.get(w, ()))) \
        if w.isalpha() else 0.0
    # the headline feature: an alphabetic fragment that is NOT itself a word but
    # that the lexicon can complete
    out["completable_fragment"] = float(
        out["alpha"] and not out["is_word"] and out["n_completions"] > 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--n-probe", type=int, default=4000)
    ap.add_argument("--out", default="results/orthographic.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    V = model.get_input_embeddings().weight.shape[0]

    lex = build_lexicon(tok, V)
    print(f"lexicon from the model's own vocabulary: {len(lex)} words")
    by_suffix, by_prefix = {}, {}
    for w in lex:
        for k in range(2, len(w)):
            by_suffix.setdefault(w[-k:], []).append(w)
            by_prefix.setdefault(w[:k], []).append(w)

    rng = random.Random(0)
    ids = sorted(rng.sample(range(V), min(a.n_probe, V)))   # same seed as embed_predict
    print(f"copy probe on {len(ids)} tokens...", flush=True)
    ex, H, outs = reproduce(model, tok, dev, [[t] for t in ids])
    y = (~ex).astype(int)
    print(f"failures: {y.sum()}/{len(y)} ({y.mean():.3f})")

    F = [features(tok.decode([t]), lex, by_suffix, by_prefix) for t in ids]
    keys = list(F[0])

    print()
    print("=" * 74)
    print("H1  DOES 'COMPLETABLE FRAGMENT' PREDICT REPRODUCTION FAILURE?")
    print("=" * 74)
    print(f"{'feature':>26} | {'AUC':>7} | {'fail|feat=1':>11} | {'fail|feat=0':>11}")
    print("-" * 64)
    res = {}
    for k in keys:
        v = np.array([f[k] for f in F])
        A = auc(v, y)
        if set(np.unique(v)) <= {0.0, 1.0} and v.sum() > 5 and (1 - v).sum() > 5:
            r1, r0 = y[v == 1].mean(), y[v == 0].mean()
            print(f"{k:>26} | {A:7.3f} | {r1:11.3f} | {r0:11.3f}")
        else:
            print(f"{k:>26} | {A:7.3f} | {'-':>11} | {'-':>11}")
        res[k] = float(A)

    print()
    print("=" * 74)
    print("H2  IS THE SUBSTITUTE THE COMPLETION?")
    print("=" * 74)
    fails = [i for i in range(len(ids)) if y[i] == 1]
    is_comp = exact_comp = in_lex = scored = 0
    shown = []
    for i in fails:
        s = tok.decode([ids[i]]).strip().lower()
        got = outs[i].strip().lower()
        if not got or not s:
            continue
        scored += 1
        # the emitted string contains the token as a substring, i.e. the model
        # padded the fragment rather than replacing it
        if s in got and got != s:
            is_comp += 1
        cands = set(by_suffix.get(s, ())) | set(by_prefix.get(s, ()))
        if got in cands:
            exact_comp += 1
        if got in lex:
            in_lex += 1
        if len(shown) < 12:
            shown.append((tok.decode([ids[i]]), outs[i][:24],
                          sorted(cands, key=len)[:3]))
    print(f"scored failures: {scored}")
    if scored:
        print(f"  output CONTAINS the token (padded fragment): {is_comp/scored:.3f}")
        print(f"  output is an exact lexicon completion       : {exact_comp/scored:.3f}")
        print(f"  output is a real word at all                : {in_lex/scored:.3f}")
    print()
    print(f"{'token':>16} | {'emitted':>26} | shortest completions")
    print("-" * 74)
    for w, got, c in shown:
        print(f"{w[:16]!r:>16} | {got!r:>26} | {c}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "n": len(ids), "fail_rate": float(y.mean()),
               "auc": res, "n_scored": scored,
               "contains_token": is_comp / max(scored, 1),
               "exact_completion": exact_comp / max(scored, 1),
               "is_real_word": in_lex / max(scored, 1)},
              open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
