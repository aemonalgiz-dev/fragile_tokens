"""Re-score a slice of a stored fragility matrix with batch size 1, as a check on the run.

Loads a (compact or full) fragility .pt, picks tokens (the most fragile gated ones plus a
random draw of never-failing ones and of the reference class), rebuilds every stored
context exactly, and scores each (token, context) cell one sequence at a time: no padding,
no batching, and on a multi-card node the same device_map pipeline the original run used.
Reports agreement with the stored cells and the per-token fragility both ways.

  python -m src.cut.rescore_slice --model Qwen/Qwen2.5-72B-Instruct \
      --pt results/fragility_qwen25_72b.pt --n-fragile 120 --n-clean 120 --n-ref 30 \
      --out results/rescore_qwen25_72b.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from .fragility import score_context, FAIL
from .loadmodel import add_model_args, load_from_args, describe

GREEDY = float(np.log(0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-fragile", type=int, default=120)
    ap.add_argument("--n-clean", type=int, default=120)
    ap.add_argument("--n-ref", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    add_model_args(ap)
    a = ap.parse_args()
    d = torch.load(a.pt, weights_only=False, map_location="cpu")
    tokens = np.asarray(d["tokens"]); s = np.asarray(d["single"], np.float64)
    M = np.asarray(d["M"], np.float64); bank = d["bank"]; thr = float(d.get("fail_thr", FAIL))
    g = np.asarray(d["is_glitch"]); frag = (M < thr).mean(1)
    clean = (s > GREEDY) & (g == 0)
    rng = np.random.default_rng(a.seed)
    fr = [i for i in np.argsort(-frag) if clean[i] and frag[i] >= 0.10][:a.n_fragile]
    ok_pool = np.where(clean & (frag == 0))[0]
    ok = list(rng.choice(ok_pool, min(a.n_clean, len(ok_pool)), replace=False))
    ref = [i for i in np.argsort(s) if frag[i] >= 0.9][:a.n_ref]
    idx = np.array(fr + ok + ref)
    kind = ["fragile"] * len(fr) + ["clean"] * len(ok) + ["reference"] * len(ref)
    sel = [int(tokens[i]) for i in idx]
    print(f"{a.pt}: re-scoring {len(sel)} tokens x {len(bank)} contexts at batch size 1 "
          f"({len(fr)} fragile, {len(ok)} clean, {len(ref)} reference)", flush=True)

    model, tok, dev = load_from_args(a)
    L = model.config.num_hidden_layers
    layers = {"mid": L // 2, "last": L}
    M2 = np.zeros((len(sel), len(bank)), np.float32)
    for c, (ctx, slot) in enumerate(bank):
        lp, _ = score_context(model, tok, dev, list(ctx), int(slot), sel, layers, batch=1)
        M2[:, c] = lp
        if c % 4 == 0:
            print(f"   context {c + 1}/{len(bank)}  K={len(ctx)}  mean lp {lp.mean():.3f}", flush=True)
    M1 = M[idx]
    diff = M2 - M1
    frag1 = (M1 < thr).mean(1); frag2 = (M2 < thr).mean(1)
    agree = ((M1 < thr) == (M2 < thr)).mean()
    out = {"model": a.model, "pt": a.pt, "model_info": describe(model, tok, a.model), "n_tokens": len(sel),
           "n_contexts": len(bank), "fail_thr": thr,
           "cell_abs_diff": {"median": float(np.median(np.abs(diff))), "p90": float(np.percentile(np.abs(diff), 90)),
                             "max": float(np.abs(diff).max())},
           "cell_fail_agreement": float(agree),
           "fragility_corr": float(np.corrcoef(frag1, frag2)[0, 1]) if len(sel) > 2 else None,
           "fragility_mean_stored_vs_rescored": {k: [float(frag1[np.array(kind) == k].mean()), float(frag2[np.array(kind) == k].mean())]
                                                 for k in ("fragile", "clean", "reference") if k in kind},
           "per_token": [{"id": int(t), "string": tok.decode([int(t)]), "kind": k,
                          "frag_stored": float(f1), "frag_rescored": float(f2),
                          "mean_lp_stored": float(m1.mean()), "mean_lp_rescored": float(m2.mean())}
                         for t, k, f1, f2, m1, m2 in zip(sel, kind, frag1, frag2, M1, M2)]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\ncell |diff| median {out['cell_abs_diff']['median']:.4f}  p90 {out['cell_abs_diff']['p90']:.4f}  "
          f"max {out['cell_abs_diff']['max']:.3f}")
    print(f"fail/pass agreement per cell {agree:.4f};  fragility corr stored vs rescored {out['fragility_corr']:.4f}")
    for k, (f1, f2) in out["fragility_mean_stored_vs_rescored"].items():
        print(f"  {k:>9}: mean fragility stored {f1:.3f}  rescored {f2:.3f}")
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
