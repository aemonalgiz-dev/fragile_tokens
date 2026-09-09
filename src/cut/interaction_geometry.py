"""Is the non-contiguous interaction I(a,b) visible in the geometry -- static
embeddings, or the representation as it evolves through the forward pass -- and
does that depend on distance?

Predictors, per ordered pair (a at slot i, b at slot j):

  STATIC (two matrices, no forward pass)
    cos_in        cos(E_in[a], E_in[b])
    cross_ab      E_in[a] . E_out[b]   direct-path excitation of b's readout by a
    cross_ba      E_in[b] . E_out[a]   (final-LN gain folded into E_out, centring
                                        into E_in, as in embed_predict.py)
    norm_min/max  |E_in| over the pair     } RARITY BASELINE. Every other
    id_min/max    token id over the pair   } predictor must beat these.

  DYNAMIC (activation-space interference, recorded during the screen)
    R_pos_layer   1 - cos( h(pos | a,b),  mean_c h(pos | a,c) )
                  how much b's presence perturbs the representation at a's
                  slot, relative to the same slot with a control partner.
                  pos in {copy_i, copy_j, text_j}: with causal attention the
                  Text-span slot i cannot see slot j, so it is not measured.
                  Computed pooled over replicates and within distance bands.

Targets: the FDR-significance mask (off-diagonal) for AUC, and the continuous
I for Spearman. Bootstrap CIs on every AUC.

Pre-registered: a predictor counts only if AUC >= 0.60 with CI excluding 0.5
AND it beats the best rarity-baseline AUC. Otherwise: "interaction exists but is
not geometrically predictable by these features".
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .analyze_trajectory import auc
from .embed_predict import final_norm_weight

BANDS = {"short": (2, 4, 8), "mid": (16, 32), "long": (64, 128, 256, 512)}


def boot_auc(s, y, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    A = auc(s, y); bs = []
    for _ in range(n):
        ii = rng.integers(0, len(y), len(y))
        if 0 < y[ii].sum() < len(ii):
            bs.append(auc(s[ii], y[ii]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return A, float(lo), float(hi)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def interference(states, meta, N, reps_sel, pos, ref_kind):
    """R[a,b] = mean over selected replicates of 1 - cos(h(pos|a,b), mean_c h(pos|ref))."""
    H = states.float()
    idx11 = {}; ref = {}
    for k, m in enumerate(meta):
        if m["r"] not in reps_sel:
            continue
        if m["kind"] == "11":
            idx11[(m["a"], m["b"], m["r"])] = k
        elif m["kind"] == ref_kind:
            key = (m["a"] if ref_kind == "10" else m["b"], m["r"])
            ref.setdefault(key, []).append(k)
    refmean = {key: H[v].mean(0) for key, v in ref.items()}
    Rm = np.zeros((N, N))
    for (ai, bi, r), k in idx11.items():
        key = (ai, r) if ref_kind == "10" else (bi, r)
        Rm[ai, bi] += 1.0 - float(torch.nn.functional.cosine_similarity(
            H[k], refmean[key], dim=0))
    return Rm / len(reps_sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--pt", default="results/interaction_screen.pt")
    ap.add_argument("--out", default="results/interaction_geometry.json")
    a = ap.parse_args()

    d = torch.load(a.pt, weights_only=False)
    pool = d["pool"]; N = len(pool); meta = d["meta"]; states = d["states"]
    layers = d["layers"]; dists = d["dists"]; R = len(dists)
    I = d["I"]; I_r = d["I_r"]; rej = d["rej"]
    off = ~np.eye(N, dtype=bool)
    y = rej[off].astype(int); Ioff = I[off]
    print(f"{N}x{N} pairs, R={R}, {int(y.sum())} FDR-significant off-diagonal")
    have_sig = y.sum() >= 5
    if not have_sig:
        print("fewer than 5 significant pairs: AUCs vs the significance mask are not "
              "meaningful; Spearman with continuous I is the informative column")

    # ---------- static ----------
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    E_in = model.get_input_embeddings().weight.detach().float()
    E_out = model.get_output_embeddings().weight.detach().float()
    g = final_norm_weight(model, E_in.shape[1]).cpu()
    del model
    Eo = (E_out * g.unsqueeze(0))[pool]
    Ei = (E_in - E_in.mean(dim=-1, keepdim=True))[pool]
    Eraw = E_in[pool]
    nrm = Eraw.norm(dim=-1).numpy()
    En = torch.nn.functional.normalize(Eraw, dim=-1)
    cross = (Ei @ Eo.T).numpy()
    ids = np.array(pool, float)
    feats = {
        "cos_in": (En @ En.T).numpy(),
        "cross_ab": cross, "cross_ba": cross.T, "cross_max": np.maximum(cross, cross.T),
        "norm_min (baseline)": np.minimum.outer(nrm, nrm),
        "norm_max (baseline)": np.maximum.outer(nrm, nrm),
        "id_min (baseline)": np.minimum.outer(ids, ids),
        "id_max (baseline)": np.maximum.outer(ids, ids),
    }

    # ---------- dynamic, pooled ----------
    all_r = set(range(R))
    for nm in layers:
        for pos, refk in (("copy_i", "10"), ("copy_j", "01"), ("text_j", "01")):
            feats[f"R_{pos}_{nm}"] = interference(states[f"{nm}_{pos}"], meta, N, all_r, pos, refk)

    def evaluate(F, target_y, target_I, label):
        v = F[off]
        rho = spearman(v, target_I)
        s = -v if rho > 0 else v            # orient so AUC>0.5 = predicts damage
        A, lo, hi = boot_auc(s, target_y) if have_sig else (float("nan"),) * 3
        return {"spearman": rho, "auc": A, "ci": [lo, hi]}

    print()
    print("=" * 84)
    print("GEOMETRIC PREDICTORS OF THE INTERACTION  (off-diagonal, pooled over distance)")
    print("=" * 84)
    print(f"{'predictor':>22} | {'spearman(I)':>11} | {'AUC(sig)':>8} | {'95% CI':>16}")
    print("-" * 70)
    res = {}; base_auc = 0.5
    for nm, F in feats.items():
        r = evaluate(F, y, Ioff, nm); res[nm] = r
        A, (lo, hi) = r["auc"], r["ci"]
        if "baseline" in nm and A == A:
            base_auc = max(base_auc, A)
        star = " *" if (A == A and lo > 0.5 and A >= 0.60) else ""
        print(f"{nm:>22} | {r['spearman']:+11.3f} | {A:8.3f} | [{lo:.3f}, {hi:.3f}]{star}")

    # ---------- dynamic, by distance band ----------
    print()
    print("=" * 84)
    print("ACTIVATION-SPACE INTERFERENCE BY DISTANCE BAND  (last layer, copy span)")
    print("=" * 84)
    print(f"{'band':>6} | {'R':>2} | {'mean R copy_i':>13} | {'spearman(I_band)':>16} | "
          f"{'AUC(sig)':>8}")
    print("-" * 62)
    band_res = {}
    for bname, bds in BANDS.items():
        rr = {r for r, dd in enumerate(dists) if dd in bds}
        if len(rr) < 2:
            continue
        Ib = I_r[:, :, sorted(rr)].mean(2)[off]
        Rb = interference(states["last_copy_i"], meta, N, rr, "copy_i", "10")
        v = Rb[off]; rho = spearman(v, Ib)
        A = auc(-v if rho > 0 else v, y) if have_sig else float("nan")
        band_res[bname] = {"R": len(rr), "mean_R": float(v.mean()), "spearman": rho, "auc": A}
        print(f"{bname:>6} | {len(rr):2d} | {v.mean():13.4f} | {rho:+16.3f} | {A:8.3f}")

    print()
    print("=" * 84)
    print("PRE-REGISTERED VERDICT")
    print("=" * 84)
    print(f"  best rarity-baseline AUC: {base_auc:.3f}")
    winners = [nm for nm, r in res.items() if "baseline" not in nm and r["auc"] == r["auc"]
               and r["auc"] >= 0.60 and r["ci"][0] > 0.5 and r["auc"] > base_auc]
    if winners:
        print("  predictors that pass (AUC>=0.60, CI excludes 0.5, beats rarity):")
        for nm in winners:
            print(f"    {nm}: AUC {res[nm]['auc']:.3f} {res[nm]['ci']}")
    else:
        print("  NO geometric predictor passes. Interaction, if present, is not")
        print("  predictable by these static or activation-space features.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "N": N, "n_sig": int(y.sum()), "features": res,
               "by_band": band_res, "baseline_auc": base_auc, "winners": winners},
              open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
