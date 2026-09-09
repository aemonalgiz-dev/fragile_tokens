"""Does the step-256 detector work when scored against TRUE corpus counts?

On Pythia we could only score the detector against a behavioural proxy. Here the
labels are exact: we know how many times each token appears in the model's data.
This is the clean test of the Idea-1 claim.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from ..cut.analyze_trajectory import auc

IND = {"unemb_cnorm": -1, "in_cnorm": -1, "unemb_norm": -1,
       "unemb_cos": +1, "d_unemb": -1, "resid_o": -1, "resid_i": -1,
       "shared_o": +1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="baseline")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tok", default="data/tok")
    ap.add_argument("--thresh", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tr = torch.load(Path(a.runs) / a.arm / "traj.pt", weights_only=False)
    counts = np.load(Path(a.tok) / "counts.npy")
    n = len(counts)
    labels = (counts < a.thresh).astype(int)
    steps = tr["steps"]
    total = steps[-1]
    print(f"arm={a.arm}   doomed = true count < {a.thresh}  ->  {labels.sum()} of {n} tokens")

    rows = []
    print()
    print(f"{'step':>7} | " + " | ".join(f"{k:>11}" for k in IND) + " | " + f"{'combined':>9}")
    print("-" * (10 + 14 * len(IND) + 12))

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    tr_i, te_i = perm[: n // 2], perm[n // 2:]
    pos_w = torch.tensor(float((labels == 0).sum() / max(labels.sum(), 1)))

    for i, st in enumerate(steps):
        vals = {}
        for k, sgn in IND.items():
            vals[k] = auc(sgn * tr[k][i].numpy()[:n], labels)
        X = np.stack([tr[k][i].numpy()[:n] for k in IND], 1)
        X = (X - X[tr_i].mean(0)) / (X[tr_i].std(0) + 1e-8)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(labels, dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.1)
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                Xt[tr_i] @ w + b, yt[tr_i], pos_weight=pos_w).backward()
            opt.step()
        comb = auc((Xt[te_i] @ w + b).detach().numpy(), labels[te_i])
        vals["combined"] = comb
        rows.append({"step": st, **vals})
        print(f"{st:7d} | " + " | ".join(f"{vals[k]:11.3f}" for k in IND) + f" | {comb:9.3f}")

    print()
    print(f"{'indicator':>12} | {'AUC>=.90':>28} | {'AUC>=.95':>28}")
    print("-" * 76)
    tbl = {}
    for k in list(IND) + ["combined"]:
        curve = [(r["step"], r[k]) for r in rows]
        e90 = next((s for s, v in curve if v >= 0.90), None)
        e95 = next((s for s, v in curve if v >= 0.95), None)
        f = lambda e: (f"step {e:<6} ({100*e/total:6.3f}% of training)" if e is not None else "never")
        tbl[k] = {"auc90": e90, "auc95": e95, "peak": max(v for _, v in curve)}
        print(f"{k:>12} | {f(e90):>28} | {f(e95):>28}")

    print()
    print("update decomposition, doomed / healthy ratio:")
    print(f"{'step':>7} | {'total |dW|':>11} | {'shared':>9} | {'residual':>9}")
    print("-" * 46)
    dec = []
    for i, st in enumerate(steps):
        if st < 2:
            continue
        g = lambda k: tr[k][i].numpy()[:n]
        r = {"step": st}
        for nm, k in [("total", "d_unemb"), ("shared", "shared_o"), ("resid", "resid_o")]:
            v = g(k)
            r[nm] = float(v[labels == 1].mean() / v[labels == 0].mean())
        dec.append(r)
        if st in (2, 8, 32, 64, 128, 256, 512, 1000, 4000, total):
            print(f"{st:7d} | {r['total']:11.3f} | {r['shared']:9.3f} | {r['resid']:9.3f}")

    out = a.out or f"{a.runs}/{a.arm}/detector.json"
    json.dump({"arm": a.arm, "thresh": a.thresh, "n_doomed": int(labels.sum()),
               "auc": rows, "earliest": tbl, "decomposition": dec},
              open(out, "w"), indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
