"""Idea 1: when during pretraining do glitch tokens become identifiable?

Ground truth is BEHAVIOURAL and measured at the final checkpoint (copy-probe over
the full vocabulary). We then ask, for each earlier checkpoint t, how well the
embedding geometry AT t predicts that final behaviour.

Two questions:
  Q1  earliest step at which final glitch status is predictable (AUC threshold)
  Q2  are glitch rows never updated, or updated then abandoned?
      ("Hub of Short Rows", 2608.29702, claims they ARE updated -- testable here.)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U / rank-based AUC. labels: 1 = glitch."""
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n1 = labels.sum()
    n0 = len(labels) - n1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="results/traj_1.4b.pt")
    ap.add_argument("--gt", default="results/behav_gt_1.4b.pt")
    ap.add_argument("--glitch-pct", type=float, default=1.0)
    ap.add_argument("--out", default="results/trajectory_analysis.json")
    a = ap.parse_args()

    tr = torch.load(a.traj, weights_only=False)
    gt = torch.load(a.gt, weights_only=False)
    lp = gt["copy_logprob"].numpy()
    n = len(lp)

    thr = np.percentile(lp, a.glitch_pct)
    labels = (lp <= thr).astype(int)
    print(f"model {tr['model']}   glitch = worst {a.glitch_pct}% by copy-logprob "
          f"(<= {thr:+.3f})  ->  {labels.sum()} tokens vs {n - labels.sum()} healthy")

    steps = tr["steps"]
    # indicators available at every checkpoint, no access to the final model
    IND = {"unemb_cnorm": -1, "in_cnorm": -1, "unemb_norm": -1,
           "unemb_cos": +1, "d_unemb": -1, "resid_o": -1, "resid_i": -1,
           "shared_o": +1}
    # sign: +1 means "higher = more glitchy"; -1 means "lower = more glitchy"

    print(f"\n{'step':>8} | " + " | ".join(f"{k:>11}" for k in IND))
    print("-" * (10 + 14 * len(IND)))
    rows = []
    for i, st in enumerate(steps):
        vals = {}
        for k, sgn in IND.items():
            v = tr[k][i].numpy()[:n]
            vals[k] = auc(sgn * v, labels)
        rows.append({"step": st, **vals})
        print(f"{st:8d} | " + " | ".join(f"{vals[k]:11.3f}" for k in IND))

    # combined predictor: logistic regression over all indicators at each
    # checkpoint, fit on half the vocabulary and scored on the held-out half.
    print()
    print("combined (logistic regression on all indicators, 50/50 split):")
    rng = np.random.default_rng(0)
    perm = rng.permutation(n); tr_i, te_i = perm[: n // 2], perm[n // 2:]
    pos_w = torch.tensor(float((labels == 0).sum() / max(labels.sum(), 1)))
    for i, st in enumerate(steps):
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
        sc = (Xt[te_i] @ w + b).detach().numpy()
        rows[i]["combined"] = auc(sc, labels[te_i])
        print(f"  step {st:8d}  AUC {rows[i]['combined']:.3f}")

    total = steps[-1]
    print()
    print("EARLIEST DETECTION (first checkpoint reaching each AUC threshold):")
    print(f"{'indicator':>14} | {'AUC>=.90':>26} | {'AUC>=.95':>26}")
    print("-" * 72)
    early_tbl = {}
    def fmt(e):
        return f"step {e:<7} ({100*e/total:5.2f}% of training)" if e is not None else "never"
    for k in list(IND) + ["combined"]:
        curve = [(r["step"], r[k]) for r in rows]
        e90 = next((s_ for s_, v in curve if v >= 0.90), None)
        e95 = next((s_ for s_, v in curve if v >= 0.95), None)
        early_tbl[k] = {"auc90": e90, "auc95": e95,
                        "peak": max(v for _, v in curve)}
        print(f"{k:>14} | {fmt(e90):>26} | {fmt(e95):>26}")
    early, early95 = early_tbl["combined"]["auc90"], early_tbl["combined"]["auc95"]
    # Q2: update magnitude, glitch vs healthy
    print(f"\nQ2 -- were glitch rows ever updated?  (mean per-row |dW| between saved ckpts)")
    print(f"{'step':>8} | {'glitch':>10} | {'healthy':>10} | {'ratio':>7}")
    print("-" * 44)
    q2 = []
    for i, st in enumerate(steps):
        d = tr["d_unemb"][i].numpy()[:n]
        g, h = d[labels == 1].mean(), d[labels == 0].mean()
        q2.append({"step": st, "glitch": float(g), "healthy": float(h),
                   "ratio": float(g / h) if h else None})
        if st in (0, 128, 512, 1000, 4000, 16000, 64000, 143000):
            print(f"{st:8d} | {g:10.5f} | {h:10.5f} | {g/h if h else 0:7.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": tr["model"], "glitch_pct": a.glitch_pct,
               "n_glitch": int(labels.sum()), "auc": rows,
               "earliest_auc90": early, "earliest_auc95": early95,
               "earliest_table": early_tbl,
               "earliest_table": early_tbl,
               "total_steps": total, "update_magnitude": q2},
              open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
