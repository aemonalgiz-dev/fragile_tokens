"""Is COMMITMENT STRENGTH the causal variable?

Day 1b showed an interaction but a small one. If compositional under-training is
real, the damage from violating a prediction should scale with how committed the
model was -- a dose-response on p(top1) at the seam, not just on eps.

This is the decisive cut: a flat line across commitment quartiles kills the
hypothesis even though the day-1b interaction was nominally significant.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from .modelio import load, next_logits
from .chains import extract_chain
from .repetition import score_batch, exact_match
from .stats import paired_boot
from .run_day1b import pick_seeds, sub_at_band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--sweep", default="results/sweep_1.4b.pt")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--chain-len", type=int, default=5)
    ap.add_argument("--out", default="results/strength.json")
    a = ap.parse_args()

    model, tok, device = load(a.model)
    res = torch.load(a.sweep, weights_only=False)

    # Pool welded AND diffuse seeds together, then stratify the COMBINED pool by
    # measured commitment. This gives a continuous commitment axis rather than a
    # binary contrast, so the dose-response can be read directly.
    seeds = pick_seeds(res, a.n, True) + pick_seeds(res, a.n, False)
    rows = []
    for s in seeds:
        ch, ents, _ = extract_chain(model, tok, device, s, max_len=a.chain_len)
        if len(ch) != a.chain_len or not ents:
            continue
        j = min(range(len(ents)), key=lambda i: ents[i])
        p = torch.softmax(next_logits(
            model, torch.tensor(ch[:j + 1], device=device).unsqueeze(0)), -1)[0]
        t, p_sub = sub_at_band(model, device, ch[:j + 1], a.eps)
        q = list(ch); q[j + 1] = t
        rows.append({"chain": ch, "sub": q, "seam": j,
                     "p_top1": float(p.max()), "ent": ents[j], "p_sub": p_sub})
    print(f"chains: {len(rows)}   p_top1 range {min(r['p_top1'] for r in rows):.3f}"
          f" - {max(r['p_top1'] for r in rows):.3f}")

    intact = score_batch(model, tok, device, [r["chain"] for r in rows]).numpy()
    subbed = score_batch(model, tok, device, [r["sub"] for r in rows]).numpy()
    drop = intact - subbed
    p1 = np.array([r["p_top1"] for r in rows])

    em_i, _ = exact_match(model, tok, device, [r["chain"] for r in rows])
    em_s, _ = exact_match(model, tok, device, [r["sub"] for r in rows])
    em_i, em_s = em_i.numpy(), em_s.numpy()

    edges = np.quantile(p1, [0, .25, .5, .75, 1.0])
    print(f"\n{'commitment p(top1)':>24} | {'n':>4} | {'copy-logprob drop':>26} | {'exact-match drop':>18}")
    print("-" * 84)
    out = []
    for i in range(4):
        m = (p1 >= edges[i]) & (p1 <= edges[i + 1] if i == 3 else p1 < edges[i + 1])
        d, lo, hi = paired_boot(drop[m], np.zeros(m.sum()))
        emd = float(em_i[m].mean() - em_s[m].mean())
        print(f"  Q{i+1} [{edges[i]:.3f},{edges[i+1]:.3f}] | {m.sum():4d} | "
              f"{d:+7.3f}  [{lo:+.3f},{hi:+.3f}] | {100*emd:+9.1f} pp")
        out.append({"q": i + 1, "lo": float(edges[i]), "hi": float(edges[i + 1]),
                    "n": int(m.sum()), "drop": d, "ci": [lo, hi], "em_drop": emd})

    r = np.corrcoef(p1, drop)[0, 1]
    top, bot = drop[p1 >= edges[3]], drop[p1 < edges[1]]
    diff, dlo, dhi = paired_boot(top[:min(len(top), len(bot))], bot[:min(len(top), len(bot))])
    print(f"\n  corr(commitment, drop) = {r:+.3f}")
    print(f"  Q4 - Q1 drop           = {diff:+.3f}  95% CI [{dlo:+.3f}, {dhi:+.3f}]")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "eps": a.eps, "n": len(rows), "quartiles": out,
               "corr": float(r), "q4_minus_q1": [diff, dlo, dhi],
               "p_top1": p1.tolist(), "drop": drop.tolist()}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
