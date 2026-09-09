"""Score the trajectory detector against EXTERNAL, published ground truth.

Everything so far defined "glitch" with my own copy probe, which invites the
objection that the detector and the labels share a source. Magikarp
(Fishing for Magikarp, EMNLP 2024) published per-token verification for
EleutherAI/pythia-6.9b: 1,124 candidates behaviourally tested, 36 verified
under-trained.

All Pythia sizes share one tokenizer AND the same Pile data in the same order,
so a token starved in 6.9b is starved in 1.4b. That makes this list valid ground
truth for the 1.4b trajectories, and it is fully independent of my probe.

It also ships a category label, including UNREACHABLE_MULTI_TOKEN: tokens the
tokenizer can only ever emit inside a longer sequence. That is the closest
labelled analogue in the literature to the n-gram question.
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
import torch

from .analyze_trajectory import auc
from .stats import paired_boot

IND = {"unemb_cnorm": -1, "in_cnorm": -1, "unemb_norm": -1,
       "unemb_cos": +1, "d_unemb": -1, "resid_o": -1, "resid_i": -1,
       "shared_o": +1}


def load_labels(path, n):
    rows = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]
    verified = np.zeros(n, dtype=int)
    tested = np.zeros(n, dtype=bool)
    cat = {}
    dec = {}
    for r in rows:
        i = int(r["i"])
        if i >= n:
            continue
        cat[i] = r.get("category")
        dec[i] = r.get("decoded")
        if "magikarp" in r:
            tested[i] = True
            verified[i] = 1 if "verified" in r["magikarp"] else 0
    return verified, tested, cat, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="results/traj_1.4b.pt")
    ap.add_argument("--ext", default="external/pythia_6_9b.jsonl.gz")
    ap.add_argument("--gt", default="results/behav_gt_1.4b.pt")
    ap.add_argument("--out", default="results/external_validation.json")
    a = ap.parse_args()

    tr = torch.load(a.traj, weights_only=False)
    n = tr["unemb_cnorm"].shape[1]
    ver, tested, cat, dec = load_labels(a.ext, n)
    steps = tr["steps"]; total = steps[-1]

    print(f"external labels: {tested.sum()} tokens behaviourally tested by Magikarp, "
          f"{ver.sum()} verified under-trained")
    print(f"categories: " + ", ".join(
        f"{k}={sum(1 for v in cat.values() if v == k)}"
        for k in ["OK", "UNREACHABLE_MULTI_TOKEN", "UNREACHABLE_SINGLE_TOKEN", "UNDECODEABLE"]))

    # ---- task 1: verified under-trained vs the rest of the TESTED pool ----
    # Scoring only inside Magikarp's tested pool is the hard version: those 1,124
    # tokens were all flagged as suspicious by their indicators, so this asks
    # whether we separate true positives from their false positives.
    pool = np.where(tested)[0]
    y = ver[pool]
    print(f"\nTASK 1 -- verified vs rejected, inside Magikarp's own candidate pool "
          f"(n={len(pool)}, {y.sum()} positive)")
    print(f"{'step':>7} | " + " | ".join(f"{k:>10}" for k in IND) + f" | {'combined':>9}")
    print("-" * (10 + 13 * len(IND) + 12))
    rows = []
    rng = np.random.default_rng(0)
    for i, st in enumerate(steps):
        vals = {k: auc(s * tr[k][i].numpy()[pool], y) for k, s in IND.items()}
        X = np.stack([tr[k][i].numpy()[pool] for k in IND], 1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        # leave-half-out to keep it honest
        perm = rng.permutation(len(pool)); tr_i, te_i = perm[:len(pool)//2], perm[len(pool)//2:]
        w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.1)
        pw = torch.tensor(float((y == 0).sum() / max(y.sum(), 1)))
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                Xt[tr_i] @ w + b, yt[tr_i], pos_weight=pw).backward()
            opt.step()
        vals["combined"] = auc((Xt[te_i] @ w + b).detach().numpy(), y[te_i])
        rows.append({"step": st, **vals})
        print(f"{st:7d} | " + " | ".join(f"{vals[k]:10.3f}" for k in IND)
              + f" | {vals['combined']:9.3f}")

    print(f"\n{'indicator':>12} | {'AUC>=.80':>26} | {'AUC>=.90':>26}")
    print("-" * 70)
    tbl = {}
    for k in list(IND) + ["combined"]:
        c = [(r["step"], r[k]) for r in rows]
        e8 = next((s for s, v in c if v >= 0.80), None)
        e9 = next((s for s, v in c if v >= 0.90), None)
        f = lambda e: (f"step {e:<6} ({100*e/total:6.3f}% of train)" if e is not None else "never")
        tbl[k] = {"auc80": e8, "auc90": e9, "peak": max(v for _, v in c)}
        print(f"{k:>12} | {f(e8):>26} | {f(e9):>26}")

    # ---- task 2: the UNREACHABLE_MULTI_TOKEN class ----
    multi = np.array([cat.get(i) == "UNREACHABLE_MULTI_TOKEN" for i in range(n)])
    ok = np.array([cat.get(i) == "OK" for i in range(n)])
    print(f"\nTASK 2 -- UNREACHABLE_MULTI_TOKEN ({multi.sum()}) vs OK ({ok.sum()})")
    sub = np.where(multi | ok)[0]
    y2 = multi[sub].astype(int)
    print(f"{'step':>7} | {'combined AUC':>12}")
    print("-" * 24)
    rows2 = []
    for i, st in enumerate(steps):
        X = np.stack([tr[k][i].numpy()[sub] for k in IND], 1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y2, dtype=torch.float32)
        perm = rng.permutation(len(sub)); tr_i, te_i = perm[:len(sub)//2], perm[len(sub)//2:]
        w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.1)
        pw = torch.tensor(float((y2 == 0).sum() / max(y2.sum(), 1)))
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                Xt[tr_i] @ w + b, yt[tr_i], pos_weight=pw).backward()
            opt.step()
        v = auc((Xt[te_i] @ w + b).detach().numpy(), y2[te_i])
        rows2.append({"step": st, "combined": v})
        if st in (0, 2, 32, 256, 1000, 16000, 143000):
            print(f"{st:7d} | {v:12.3f}")

    # ---- task 3: what do we flag early that Magikarp never tested? ----
    i256 = steps.index(256) if 256 in steps else len(steps) // 2
    X = np.stack([tr[k][i256].numpy() for k in IND], 1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(ver, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.1)
    pw = torch.tensor(float((ver == 0).sum() / max(ver.sum(), 1)))
    for _ in range(400):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(
            Xt[tested] @ w + b, yt[tested], pos_weight=pw).backward()
        opt.step()
    score = (Xt @ w + b).detach().numpy()
    untested = ~tested
    top = np.argsort(-score)
    novel = [int(i) for i in top if untested[i]][:20]
    print(f"\nTASK 3 -- highest step-256 scores among tokens Magikarp never tested:")
    for i in novel[:15]:
        print(f"  score={score[i]:+6.2f}  cat={cat.get(i,'?'):24s} {dec.get(i, '?')!r}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"task1": rows, "earliest": tbl, "task2": rows2,
               "n_verified": int(ver.sum()), "n_tested": int(tested.sum()),
               "novel_candidates": novel}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
