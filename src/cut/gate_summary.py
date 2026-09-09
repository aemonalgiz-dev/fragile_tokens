"""Fragility under a calibration-free gate, for cross-model comparison.

The paper's "clean-looking" gate is an absolute single-probe log-probability
above -0.1 (copied alone at > 0.9). On the scale ladder that gate turned out
to be model-specific: Qwen3-32B copies tokens in context at 0.997 but, under the
bare few-shot probe, at 0.72-0.89 for ordinary tokens, so the -0.1 cut kept 353
of 4,200 tokens and called none of them fragile -- a statement about the probe's
calibration on that model, not about fragility.

The gate here is "greedy-copyable alone": p_alone > 0.5, which guarantees the
token is the argmax of its own copy probe on any model. Fragility is reported
at both failure thresholds (in-context p < 0.61, the paper's -0.5; and p < 0.5,
greedy failure), with the two-way variance decomposition on the gated set.
Reads a fragility .pt and writes gate_<tag>.json for scale_summary.py.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch


def decompose(M):
    mu = M.mean(); te = M.mean(1, keepdims=True) - mu; ce = M.mean(0, keepdims=True) - mu
    r = M - mu - te - ce
    ss = lambda x: float((x ** 2).sum()); tot = ss(M - mu)
    return {"token": ss(np.broadcast_to(te, M.shape)) / tot,
            "context": ss(np.broadcast_to(ce, M.shape)) / tot, "interaction": ss(r) / tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-p", type=float, default=0.5)
    a = ap.parse_args()
    d = torch.load(a.pt, weights_only=False, map_location="cpu")
    s = np.asarray(d["single"]); M = np.asarray(d["M"], np.float64); is_g = np.asarray(d["is_glitch"])
    tokens = list(d["tokens"])
    gate = np.log(a.gate_p)
    clean = (s > gate) & (is_g == 0)
    out = {"pt": a.pt, "N": int(len(s)), "C": int(M.shape[1]), "gate_p": a.gate_p,
           "n_clean": int(clean.sum()),
           "single_pct": {str(p): float(np.percentile(s, p)) for p in (10, 25, 50, 75, 90)},
           "ctx_median_lp": float(np.median(M)),
           "shares_clean": decompose(M[clean]) if clean.sum() > 2 else None}
    for fail_p, key in ((0.61, "fail_p61"), (0.5, "fail_p50")):
        frag = (M[clean] < np.log(fail_p)).mean(1)
        halves = (M[clean][:, ::2] < np.log(fail_p)).mean(1), (M[clean][:, 1::2] < np.log(fail_p)).mean(1)
        out[key] = {"ge10": float((frag >= 0.10).mean()), "ge25": float((frag >= 0.25).mean()),
                    "never": float((frag == 0).mean()), "mean": float(frag.mean()),
                    "n_ge10": int((frag >= 0.10).sum()),
                    "half_corr": float(np.corrcoef(halves[0], halves[1])[0, 1]) if clean.sum() > 2 else None}
        top = np.argsort(-frag)[:8]
        idx = np.where(clean)[0]
        out[key]["top"] = [{"id": int(tokens[idx[i]]), "frag": float(frag[i]),
                            "single": float(s[idx[i]]), "mean_lp": float(M[idx[i]].mean())} for i in top]
    print(f"{a.pt}: N={out['N']} gate p>{a.gate_p} -> n_clean {out['n_clean']}; "
          f"single pct50 {out['single_pct']['50']:.3f} pct90 {out['single_pct']['90']:.3f}; "
          f"ctx median {out['ctx_median_lp']:.4f}")
    for key in ("fail_p61", "fail_p50"):
        o = out[key]
        print(f"  {key}: frag>=0.10 {o['ge10']:.3f} (n={o['n_ge10']})  never {o['never']:.3f}  "
              f"mean {o['mean']:.4f}  half-corr {o['half_corr']:.3f}")
    if out["shares_clean"]:
        sh = out["shares_clean"]
        print(f"  shares (clean, lp): token {sh['token']:.3f} context {sh['context']:.3f} "
              f"interaction {sh['interaction']:.3f}")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
