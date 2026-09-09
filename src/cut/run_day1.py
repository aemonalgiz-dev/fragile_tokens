"""Day-one probe: does seam violation glitch where random zero-count does not?"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch

from .modelio import load
from .entropy_sweep import sweep, constituent_health, CONTEXT_BANK
from .chains import extract_chain, build_arms, norm_pool
from .repetition import score_batch, exact_match
from .stats import paired_boot, welch, cohen_d

ARMS = ["A_weld_intact", "B_seam_violation", "C_perm_control", "D_random_control"]


def select_seeds(res, n, norm_floor_pct=2.0):
    """Most context-invariantly committed tokens, EXCLUDING anything a
    Magikarp-style single-token detector would already flag."""
    me, ag = res["max_ent"], res["agree"]
    norm = res["unemb_norm"][: res["n_tok"]]
    floor = np.percentile(norm.numpy(), norm_floor_pct)
    healthy = torch.tensor(norm.numpy() > floor)
    ok = ag & healthy
    cand = torch.nonzero(ok).squeeze(-1)
    order = me[cand].argsort()
    return cand[order][:n].tolist(), float(floor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--sweep", default="results/sweep.pt")
    ap.add_argument("--n-seeds", type=int, default=300)
    ap.add_argument("--chain-len", type=int, default=5)
    ap.add_argument("--exact-match-n", type=int, default=80)
    ap.add_argument("--out", default="results/day1.json")
    a = ap.parse_args()

    model, tok, device = load(a.model)
    res = torch.load(a.sweep, weights_only=False)
    rng = random.Random(0)

    seeds, floor = select_seeds(res, a.n_seeds)
    print(f"seed pool: {len(seeds)} welded tokens (unemb-norm floor {floor:.3f})")
    print("examples:", [repr(tok.decode([s])) for s in seeds[:8]])

    pool = norm_pool(res["unemb_norm"][: res["n_tok"]])
    items, meta = {k: [] for k in ARMS}, []
    for s in seeds:
        chain, ents, seconds = extract_chain(model, tok, device, s,
                                             max_len=a.chain_len)
        arms = build_arms(chain, ents, seconds, res["unemb_norm"], rng, pool)
        if arms is None:
            continue
        for k in ARMS:
            items[k].append(arms[k])
        meta.append({"seed": s, "seed_str": tok.decode([s]),
                     "chain": chain, "chain_str": tok.decode(chain),
                     "ents": ents, "seam_step": int(min(range(len(seconds)), key=lambda i: ents[i])) if seconds else -1,
                     "violated_str": tok.decode(arms["B_seam_violation"])})
    n = len(meta)
    print(f"chains kept: {n}\n")

    scores = {}
    for k in ARMS:
        scores[k] = score_batch(model, tok, device, items[k]).numpy()
        print(f"{k:18s} copy-logprob  mean {scores[k].mean():7.3f}  "
              f"median {np.median(scores[k]):7.3f}")

    em, gens = {}, {}
    for k in ARMS:
        em[k], gens[k] = exact_match(model, tok, device, items[k][: a.exact_match_n])
        em[k] = em[k].numpy()
        print(f"{k:18s} exact-match   {100*em[k].mean():5.1f}%")

    print("\n--- paired contrasts on copy-logprob (negative = arm glitchier) ---")
    contrasts = {}
    for other in ["A_weld_intact", "C_perm_control", "D_random_control"]:
        m, lo, hi = paired_boot(scores["B_seam_violation"], scores[other])
        mm, mlo, mhi = paired_boot(scores["B_seam_violation"], scores[other], stat="median")
        t, df = welch(scores["B_seam_violation"], scores[other])
        d = cohen_d(scores["B_seam_violation"], scores[other])
        contrasts[f"B_vs_{other}"] = {"mean_diff": m, "ci": [lo, hi],
                                      "welch_t": t, "df": df, "cohen_d": d,
                                      "median_diff": mm, "median_ci": [mlo, mhi]}
        print(f"B - {other:18s} mean {m:+8.3f} [{lo:+.2f},{hi:+.2f}]   "
              f"median {mm:+7.3f} [{mlo:+.2f},{mhi:+.2f}]  d={d:+.2f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "n": n, "contexts": CONTEXT_BANK,
               "scores": {k: v.tolist() for k, v in scores.items()},
               "exact_match": {k: float(v.mean()) for k, v in em.items()},
               "generations": {k: v for k, v in gens.items()},
               "contrasts": contrasts, "meta": meta},
              open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
