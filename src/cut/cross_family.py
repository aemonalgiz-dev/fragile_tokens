"""Does the abandonment signature survive outside Pythia?

Six models, three corpora (Pile / Dolma v1.5 / OLMo-2 mix), two tokenizer
families, 410M to 7B, dense and MoE. Labels are Magikarp's published behavioural
verifications for each family.

Three questions:
  1. MECHANISM -- do doomed rows show total-magnitude parity or excess while their
     token-specific residual is suppressed? This is the central claim, and if it
     is a Pile artifact the paper is wrong.
  2. DETECTION -- does the early signal rank verified tokens, scored against each
     family's own labels?
  3. GEOMETRY -- do glitch tokens become hubs (appearing in many other tokens'
     k-NN lists), and does hubness onset track abandonment onset?
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
import torch

from .analyze_trajectory import auc

MODELS = [
    ("pythia-410m",   "results/traj_pythia410m.pt", "external/pythia_6_9b.jsonl.gz", "Pile"),
    ("pythia-1.4b",   "results/traj_1.4b.pt",        "external/pythia_6_9b.jsonl.gz", "Pile"),
    ("pythia-6.9b",   "results/traj_pythia6.9b.pt",  "external/pythia_6_9b.jsonl.gz", "Pile*"),
    ("OLMo-1B",       "results/traj_olmo1b.pt",      "external/olmo_7b.jsonl.gz",     "Dolma"),
    ("OLMo-2-1B",     "results/traj_olmo2_1b.pt",    "external/allenai_OLMo_2_1124_7B.jsonl.gz", "OLMo2"),
    ("OLMo-2-7B",     "results/traj_olmo2_7b.pt",    "external/allenai_OLMo_2_1124_7B.jsonl.gz", "OLMo2"),
    ("OLMoE-1B-7B",   "results/traj_olmoe.pt",       "external/allenai_OLMoE_1B_7B_0924.jsonl.gz", "OLMo/MoE"),
]
IND = ["unemb_cnorm", "in_cnorm", "unemb_norm", "unemb_cos",
       "d_unemb", "resid_o", "resid_i", "shared_o"]
SIGN = {"unemb_cnorm": -1, "in_cnorm": -1, "unemb_norm": -1, "unemb_cos": +1,
        "d_unemb": -1, "resid_o": -1, "resid_i": -1, "shared_o": +1}


def labels(path, n):
    ver = np.zeros(n, int)
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line)
        i = int(r["i"])
        if i < n and "verified" in r.get("magikarp", ""):
            ver[i] = 1
    return ver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cross_family.json")
    a = ap.parse_args()
    report = {}

    print("=" * 78)
    print("Q1  MECHANISM: glitch/healthy update ratios (total | shared | residual)")
    print("=" * 78)
    for name, tp, lp, corpus in MODELS:
        if not Path(tp).exists():
            print(f"{name}: missing {tp}"); continue
        tr = torch.load(tp, weights_only=False)
        n = tr["unemb_cnorm"].shape[1]
        ver = labels(lp, n)
        if ver.sum() < 5:
            print(f"{name}: only {ver.sum()} labels, skipping"); continue
        steps = tr["steps"]
        print(f"\n{name}  ({corpus}, {ver.sum()} verified, {len(steps)} ckpts)")
        print(f"   {'step':>9} | {'total':>7} | {'shared':>7} | {'residual':>9}")
        rows = []
        for i, st in enumerate(steps):
            if i == 0:
                continue
            g = lambda k: tr[k][i].numpy()[:n]
            r = {"step": st}
            for nm, k in [("total", "d_unemb"), ("shared", "shared_o"), ("resid", "resid_o")]:
                v = g(k)
                h = v[ver == 0].mean()
                r[nm] = float(v[ver == 1].mean() / h) if h else float("nan")
            rows.append(r)
            if i <= 6 or i == len(steps) - 1 or st in (256, 1000, 16000):
                print(f"   {st:9d} | {r['total']:7.3f} | {r['shared']:7.3f} | {r['resid']:9.3f}")
        report.setdefault(name, {})["mechanism"] = rows
        early = [r for r in rows if r["resid"] == r["resid"]][:6]
        if early:
            print(f"   -> early residual ratio mean {np.mean([r['resid'] for r in early]):.3f}"
                  f"   total {np.mean([r['total'] for r in early]):.3f}")

    print()
    print("=" * 78)
    print("Q2  DETECTION: single-indicator AUC vs each family's own labels")
    print("=" * 78)
    print(f"{'model':>14} | {'first ckpt':>10} | {'resid_i':>8} | {'unemb_cos':>10} | {'best late':>10}")
    print("-" * 66)
    for name, tp, lp, corpus in MODELS:
        if not Path(tp).exists():
            continue
        tr = torch.load(tp, weights_only=False)
        n = tr["unemb_cnorm"].shape[1]
        ver = labels(lp, n)
        if ver.sum() < 5:
            continue
        steps = tr["steps"]
        j = 1 if len(steps) > 1 else 0
        ri = auc(-tr["resid_i"][j].numpy()[:n], ver)
        uc = auc(tr["unemb_cos"][j].numpy()[:n], ver)
        late = max(auc(SIGN[k] * tr[k][-1].numpy()[:n], ver) for k in IND)
        report.setdefault(name, {})["detect"] = {"step": steps[j], "resid_i": ri,
                                                 "unemb_cos": uc, "best_late": late}
        print(f"{name:>14} | {steps[j]:10d} | {ri:8.3f} | {uc:10.3f} | {late:10.3f}")

    print()
    print("=" * 78)
    print("Q3  GEOMETRY: are glitch tokens hubs?  (k-occurrence, k=10)")
    print("=" * 78)
    print(f"{'model':>14} | {'hub glitch':>11} | {'hub healthy':>12} | {'ratio':>7} | {'AUC':>7}")
    print("-" * 62)
    for name, tp, lp, corpus in MODELS:
        if not Path(tp).exists():
            continue
        tr = torch.load(tp, weights_only=False)
        if "hub_o" not in tr:
            print(f"{name:>14} | (no geometry recorded)")
            continue
        n = tr["unemb_cnorm"].shape[1]
        ver = labels(lp, n)
        if ver.sum() < 5:
            continue
        h = tr["hub_o"][-1].numpy()[:n]
        hg, hh = h[ver == 1].mean(), h[ver == 0].mean()
        report.setdefault(name, {})["hub"] = {"glitch": float(hg), "healthy": float(hh),
                                              "auc": auc(h, ver)}
        print(f"{name:>14} | {hg:11.2f} | {hh:12.2f} | {hg/max(hh,1e-9):7.2f} | {auc(h, ver):7.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
