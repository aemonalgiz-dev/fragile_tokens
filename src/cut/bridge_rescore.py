"""Rescore the hallucination bridge on the entity's FINAL COMPONENT.

The first scorer compared the whole dotted entity to identifier spans in the
output, so `numpy.ravel_multi_index` -> `np.ravel_multi_index` and
`transformers.DepthEstimationPipeline` -> `from transformers import
DepthEstimationPipeline` were counted as near-misses. They are correct code.
The hallucination of interest is the NAME being mangled -- `curses.BUTTON5_PRESSED`
-> `curses.C_BUTTON5_PRESSED` -- so:

  exact      the final component (after the last '.') appears verbatim as an
             identifier in the output, under any module prefix or none
  near-miss  it does not, but some identifier span has fuzzy similarity >= 0.75
             to it (or, for dotted output, its own final component does)
  miss       nothing close

Nouns are unchanged except that apostrophe variants (ʻ vs ') are normalised.
Runs locally on the saved outputs and per-row features; no model needed.
"""
from __future__ import annotations
import argparse, difflib, json, re
import numpy as np

from .stats import auc

ALIAS_OK = {"numpy": {"np", "numpy"}, "torch": {"torch"}, "transformers": {"transformers"}}


def logistic_cv(X, y, folds=5, l2=1.0, seed=0, iters=25):
    rng = np.random.default_rng(seed)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.hstack([X, np.ones((len(X), 1))])
    idx = rng.permutation(len(y)); pred = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X[tr] @ w))
            g = X[tr].T @ (p - y[tr]) + l2 * w
            H = (X[tr] * (p * (1 - p))[:, None]).T @ X[tr] + l2 * np.eye(len(w))
            w -= np.linalg.solve(H, g)
        pred[te] = 1 / (1 + np.exp(-X[te] @ w))
    return pred


def boot_auc(s, y, n=1000, seed=1):
    rng = np.random.default_rng(seed); A = auc(s, y); bs = []
    for _ in range(n):
        ii = rng.integers(0, len(y), len(y))
        if 0 < y[ii].sum() < len(ii):
            bs.append(auc(s[ii], y[ii]))
    return A, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def norm_apos(s):
    return s.replace("ʻ", "'").replace("’", "'").replace("`", "'")


def rescore(r):
    out = r["out"]
    if r["type"] == "noun":
        e = norm_apos(r["entity"]).lower(); o = norm_apos(out).lower()
        if e in o:
            return 0, r["entity"]
        words = re.findall(r"[^\W\d_]+(?:['\-][^\W\d_]+)*", o); n = len(e.split())
        spans = [" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]
        best = max(spans, key=lambda sp: difflib.SequenceMatcher(None, sp, e).ratio(), default="")
        ratio = difflib.SequenceMatcher(None, best, e).ratio() if best else 0.0
        return (1 if ratio >= 0.75 else 2), best
    final = r["entity"].split(".")[-1]
    idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", out)
    if final in idents:
        return 0, final
    best, best_r = "", 0.0
    for sp in idents:
        ratio = difflib.SequenceMatcher(None, sp, final).ratio()
        if ratio > best_r:
            best, best_r = sp, ratio
    return (1 if best_r >= 0.75 else 2), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/hallucination_bridge.json")
    ap.add_argument("--out", default="results/hallucination_bridge_rescored.json")
    a = ap.parse_args()
    d = json.load(open(a.inp, encoding="utf-8"))
    rows = d["rows"]
    for r in rows:
        r["cls2"], r["span2"] = rescore(r)
    cls = np.array([r["cls2"] for r in rows]); typ = np.array([r["type"] for r in rows])
    old = np.array([r["cls"] for r in rows])

    print("=" * 78)
    print("RESCORED OUTCOMES on the entity's final component  (0 exact, 1 near-miss, 2 miss)")
    print("=" * 78)
    for t in ("code", "noun"):
        m = typ == t
        print(f"  {t:>5} (n={m.sum():4d}): exact {(cls[m]==0).mean():.3f}   near-miss "
              f"{(cls[m]==1).mean():.3f}   miss {(cls[m]==2).mean():.3f}     "
              f"(old scorer: exact {(old[m]==0).mean():.3f}, near {(old[m]==1).mean():.3f})")
    print("\n  genuine name-mangling near-misses (sorted by constituent fragility):")
    shown = 0
    for r in sorted(rows, key=lambda r: -r["frag_max"]):
        if r["cls2"] == 1 and shown < 14:
            print(f"    {r['entity']!r:>44} -> {r['span2']!r:<36} frag_max {r['frag_max']:.2f}  "
                  f"fam {r['familiarity']:+.2f}")
            shown += 1
    print("\n  misses that are NOT near anything (sample):")
    shown = 0
    for r in rows:
        if r["cls2"] == 2 and r["type"] == "code" and shown < 5:
            print(f"    {r['entity']!r:>44} -> {r['out'][:70]!r}"); shown += 1

    feats = {k: np.array([r[k] for r in rows], float) for k in
             ("n_tok", "n_chars", "log_id_mean", "log_id_max", "single_min", "single_mean",
              "familiarity", "frag_max", "frag_mean", "frag_any")}
    feats["is_code"] = (typ == "code").astype(float)
    y_fail = (cls != 0).astype(int); y_near = (cls == 1).astype(int)
    print()
    print("=" * 78)
    print("PER-FEATURE AUC (rescored)")
    print("=" * 78)
    print(f"{'feature':>14} | {'AUC fail':>8} | {'AUC near':>8}")
    print("-" * 38)
    for nm, v in feats.items():
        A1 = max(auc(v, y_fail), 1 - auc(v, y_fail)); A2 = max(auc(v, y_near), 1 - auc(v, y_near))
        print(f"{nm:>14} | {A1:8.3f} | {A2:8.3f}")

    R = lambda *ks: np.column_stack([feats[k] for k in ks])
    rarity = ("n_tok", "n_chars", "log_id_mean", "log_id_max", "is_code")
    ctrl = rarity + ("single_min", "single_mean", "familiarity")
    sets = {"rarity only": R(*rarity), "controls (rarity + single + familiarity)": R(*ctrl),
            "controls + fragility": R(*(ctrl + ("frag_max", "frag_mean", "frag_any"))),
            "fragility only": R("frag_max", "frag_mean", "frag_any")}
    res = {}
    for tgt, y in (("any failure", y_fail), ("near-miss (name mangled)", y_near)):
        print(f"\n  target = {tgt}  ({y.sum()} positives / {len(y)})")
        preds = {}
        for nm, X in sets.items():
            p = logistic_cv(X, y); preds[nm] = p
            A, lo, hi = boot_auc(p, y); res[f"{tgt}|{nm}"] = [A, lo, hi]
            print(f"    {nm:>42}: AUC {A:.3f} [{lo:.3f}, {hi:.3f}]")
        rng = np.random.default_rng(5); diffs = []
        pc, pf = preds["controls (rarity + single + familiarity)"], preds["controls + fragility"]
        for _ in range(1000):
            ii = rng.integers(0, len(y), len(y))
            if 0 < y[ii].sum() < len(ii):
                diffs.append(auc(pf[ii], y[ii]) - auc(pc[ii], y[ii]))
        lo, hi = np.percentile(diffs, [2.5, 97.5]); res[f"{tgt}|improvement"] = [float(np.mean(diffs)), float(lo), float(hi)]
        print(f"    improvement from adding fragility: {np.mean(diffs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"{'   *' if lo > 0 else ''}")
    json.dump({"rates": {t: {"exact": float((cls[typ==t]==0).mean()), "near": float((cls[typ==t]==1).mean()),
                             "miss": float((cls[typ==t]==2).mean())} for t in ("code", "noun")},
               "cv": res, "rows": rows}, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
