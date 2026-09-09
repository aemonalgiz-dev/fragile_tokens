"""Day 1b: the matched-improbability 2x2.

Day 1a substituted the 2nd-argmax at a weld, which turned out not to be a
violation at all (median p = 0.037; 44% of "violations" were plausible
alternatives). Result: no effect. This design fixes that and isolates the
actual variable of interest.

                     intact            substituted at probability band eps
  WELDED chain   |  cell A          |  cell B      (violating a sharp prediction)
  DIFFUSE chain  |  cell C          |  cell D      (violating a flat prediction)

Both substitutions insert a token of the SAME conditional probability eps, at the
SAME position index. So "how surprising is this token here" is held constant and
the only thing that varies is whether the model was COMMITTED at that position.

H1 (compositional under-training): the intact->substituted drop is much larger
    for welded chains than diffuse ones. That is an INTERACTION, and it is the
    entire claim. A main effect of eps alone is just ordinary OOD degradation
    and would support the Broken Tokens null instead.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch

from .modelio import load, next_logits
from .chains import extract_chain
from .repetition import score_batch, exact_match
from .stats import paired_boot

EPS_BANDS = [1e-2, 1e-3, 1e-4, 1e-5]


@torch.no_grad()
def sub_at_band(model, device, prefix: list[int], eps: float):
    """Token whose conditional probability after `prefix` is closest to eps."""
    p = torch.softmax(next_logits(model, torch.tensor(prefix, device=device).unsqueeze(0)), -1)[0]
    return int((p.log() - np.log(eps)).abs().argmin()), float(p[(p.log() - np.log(eps)).abs().argmin()])


def pick_seeds(res, n, welded: bool):
    """Welded = lowest context-invariant entropy + cross-context agreement.
    Diffuse = mid-band entropy, no agreement. Both screened for constituent
    health so nothing here is a classic single-token glitch."""
    me, ag = res["max_ent"], res["agree"]
    norm = res["unemb_norm"][: res["n_tok"]]
    healthy = torch.tensor(norm.numpy() > np.percentile(norm.numpy(), 2.0))
    if welded:
        ok = ag & healthy
        return torch.nonzero(ok).squeeze(-1)[me[torch.nonzero(ok).squeeze(-1)].argsort()][:n].tolist()
    lo, hi = np.percentile(me.numpy(), 45), np.percentile(me.numpy(), 55)
    ok = (~ag) & healthy & (me > lo) & (me < hi)
    cand = torch.nonzero(ok).squeeze(-1).tolist()
    random.Random(1).shuffle(cand)
    return cand[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--sweep", default="results/sweep_1.4b.pt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--chain-len", type=int, default=5)
    ap.add_argument("--out", default="results/day1b.json")
    a = ap.parse_args()

    model, tok, device = load(a.model)
    res = torch.load(a.sweep, weights_only=False)

    chains = {}
    for kind in ["welded", "diffuse"]:
        seeds = pick_seeds(res, a.n, kind == "welded")
        out = []
        for s in seeds:
            ch, ents, _ = extract_chain(model, tok, device, s, max_len=a.chain_len)
            if len(ch) == a.chain_len and ents:
                # welded: violate the most-committed step. diffuse: the least.
                j = (min if kind == "welded" else max)(range(len(ents)), key=lambda i: ents[i])
                out.append({"chain": ch, "ents": ents, "seam": j})
        chains[kind] = out
        print(f"{kind:8s}: {len(out)} chains  median seam-entropy "
              f"{np.median([c['ents'][c['seam']] for c in out]):.3f} nats")

    n = min(len(chains["welded"]), len(chains["diffuse"]))
    report = {"model": a.model, "n_per_cell": n, "bands": {}}

    intact = {}
    for kind in ["welded", "diffuse"]:
        seqs = [c["chain"] for c in chains[kind][:n]]
        intact[kind] = score_batch(model, tok, device, seqs).numpy()
    print(f"\nintact  welded {intact['welded'].mean():+7.3f}   "
          f"diffuse {intact['diffuse'].mean():+7.3f}   (n={n} per cell)\n")

    print(f"{'eps':>8} | {'welded drop':>22} | {'diffuse drop':>22} | {'INTERACTION':>24}")
    print("-" * 88)
    for eps in EPS_BANDS:
        drops, subs = {}, {}
        for kind in ["welded", "diffuse"]:
            seqs = []
            for c in chains[kind][:n]:
                j = c["seam"]
                t, _ = sub_at_band(model, device, c["chain"][:j + 1], eps)
                q = list(c["chain"]); q[j + 1] = t
                seqs.append(q)
            subs[kind] = seqs
            sc = score_batch(model, tok, device, seqs).numpy()
            drops[kind] = intact[kind] - sc          # positive = substitution hurt
        m_w, lo_w, hi_w = paired_boot(drops["welded"], np.zeros(n))
        m_d, lo_d, hi_d = paired_boot(drops["diffuse"], np.zeros(n))
        inter, ilo, ihi = paired_boot(drops["welded"], drops["diffuse"])
        star = "  <== interaction" if ilo > 0 else ""
        print(f"{eps:8.0e} | {m_w:7.3f} [{lo_w:+.2f},{hi_w:+.2f}] | "
              f"{m_d:7.3f} [{lo_d:+.2f},{hi_d:+.2f}] | {inter:+7.3f} [{ilo:+.2f},{ihi:+.2f}]{star}")
        report["bands"][f"{eps:.0e}"] = {
            "welded_drop": m_w, "welded_ci": [lo_w, hi_w],
            "diffuse_drop": m_d, "diffuse_ci": [lo_d, hi_d],
            "interaction": inter, "interaction_ci": [ilo, ihi]}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
