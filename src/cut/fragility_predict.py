"""Is fragility -- failing to copy in some contexts while passing the standard
single probe -- predictable from geometry?

Three tiers, kept strictly separate because they cost different things:

  STATIC       two matrices, no forward pass. Row norms, centroid distance,
               nearest-neighbour cosine, local density, input/output row
               alignment cos(E_in[t], E_out[t]), the direct-path self score /
               competitor / margin, and projection onto the "glitch direction"
               (mean verified-glitch embedding minus mean embedding). Plus
               SURFACE baselines every geometric feature must beat: token id
               (rarity), character length, leading space, alphabetic, ascii.

  DYNAMIC      the token's evolving representation on HALF the contexts (train):
               spread of the Text-span slot state across contexts, retention of
               its own identity (cos to its unembedding row), spread of the
               Copy-span state. Predicts failures on the OTHER half (test), so
               nothing is circular.

  BEHAVIOURAL  just measure copy on the train half. This is the trivial
               predictor; if you can afford half the contexts you can afford to
               measure. Geometry earns its keep only by adding information over
               this, or by working with zero forward passes.

Target population: tokens that PASS the single probe (lp_single > -0.1) and are
not Magikarp-verified -- the ones every existing detector calls fine.
Target: frag_test >= 0.10 (fails in at least 10% of held-out contexts).

Evaluation: per-feature AUC with bootstrap CI and Spearman with continuous
frag_test; 5-fold cross-validated L2-logistic AUC per tier (numpy, no sklearn).

CELL LEVEL: among clean-looking tokens on test contexts, does token-context
similarity predict WHICH context breaks a token, beyond the two main effects
(token fragility from train, context hostility from other tokens)?

Pre-registered: a geometric tier passes if CV AUC >= 0.65, CI excludes 0.5, and
it beats the best surface baseline. Otherwise: fragility is real but not
predictable by these features.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .stats import auc
from .embed_predict import direct_path
from .loadmodel import load_embeddings


def boot_auc(s, y, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    A = auc(s, y); bs = []
    for _ in range(n):
        ii = rng.integers(0, len(y), len(y))
        if 0 < y[ii].sum() < len(ii):
            bs.append(auc(s[ii], y[ii]))
    return A, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def logistic_cv(X, y, folds=5, l2=1.0, seed=0, iters=25):
    """Out-of-fold predictions from an L2 logistic regression fit by Newton."""
    rng = np.random.default_rng(seed)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.hstack([X, np.ones((len(X), 1))])
    idx = rng.permutation(len(y)); pred = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X[tr] @ w))
            g = X[tr].T @ (p - y[tr]) + l2 * w
            H = (X[tr] * (p * (1 - p))[:, None]).T @ X[tr] + l2 * np.eye(len(w))
            w -= np.linalg.solve(H, g)
        pred[te] = 1 / (1 + np.exp(-X[te] @ w))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--pt", default="results/fragility.pt")
    ap.add_argument("--frag-thr", type=float, default=0.10)
    ap.add_argument("--gate-lp", type=float, default=-0.1,
                    help="single-probe lp above which a token is 'clean-looking'; "
                         "use -0.693 (p>0.5) for cross-model comparison")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--out", default="results/fragility_predict.json")
    a = ap.parse_args()

    d = torch.load(a.pt, weights_only=False)
    tokens = np.array(d["tokens"]); is_glitch = d["is_glitch"]; single = d["single"]
    M = d["M"]; bank = d["bank"]; states = d["states"]; thr = d["fail_thr"]
    N, C = M.shape
    F = M < thr
    train = np.arange(C) % 2 == 0; test = ~train
    frag_test = F[:, test].mean(1); frag_train = F[:, train].mean(1)
    mu_train = M[:, train].mean(1)
    # --gate-lp: the paper's absolute -0.1 is model-specific (see gate_summary.py);
    # ln(0.5) = -0.693 is the calibration-free "greedy-copyable alone" gate.
    clean = (single > a.gate_lp) & (is_glitch == 0)
    y = (frag_test[clean] >= a.frag_thr).astype(int)
    print(f"{N} tokens x {C} contexts; clean-looking non-glitch: {clean.sum()}; "
          f"fragile (frag_test >= {a.frag_thr}): {y.sum()}  (base rate {y.mean():.3f})")
    if y.sum() < 10 or y.sum() == len(y):
        print("degenerate target; stopping"); return

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    E_in, E_out, g, tied = load_embeddings(a.model, trust_remote_code=a.trust_remote_code)
    if tied:
        print("NOTE: tied embeddings -- E_out is E_in, so the output-row and direct-path "
              "features are not independent of the input-row ones; reported, flagged.")
    V = E_in.shape[0]
    import os
    dev = "cuda" if (torch.cuda.is_available() and os.environ.get("GLITCH_FORCE_CPU") != "1") else "cpu"

    # ---------- static ----------
    Eo_eff = E_out * g.unsqueeze(0)
    Ei_c = E_in - E_in.mean(dim=-1, keepdim=True)
    self_s, best_s, _ = direct_path(Ei_c, Eo_eff, dev)
    mu_in = E_in.mean(0, keepdim=True); mu_out = E_out.mean(0, keepdim=True)
    Xn = torch.nn.functional.normalize(E_in, dim=-1)
    sel = torch.tensor(tokens)
    S = (Xn[sel].to(dev) @ Xn.to(dev).T)
    S[torch.arange(len(sel), device=dev), sel.to(dev)] = -2.0
    topk = S.topk(10, dim=-1).values.cpu()
    del S
    # The "glitch direction": mean embedding of the known-bad tokens minus the
    # global mean. Indexed by VOCABULARY ID -- an earlier version indexed E_in by
    # position in the token list, which silently used the wrong rows. Without
    # labels, the worst tokens by the single probe stand in, as in
    # reasoning_drift.py.
    gl = np.where(is_glitch == 1)[0]
    if len(gl) == 0:
        gl = np.argsort(single)[:100]
    gdir = E_in[torch.as_tensor(tokens[gl], dtype=torch.long)].mean(0) - mu_in[0]
    gdir = gdir / gdir.norm()
    dec = [tok.decode([int(t)]) for t in tokens]
    static = {
        "norm_in": E_in[sel].norm(dim=-1).numpy(),
        "norm_out": E_out[sel].norm(dim=-1).numpy(),
        "cdist_in": (E_in[sel] - mu_in).norm(dim=-1).numpy(),
        "cdist_out": (E_out[sel] - mu_out).norm(dim=-1).numpy(),
        "nn_cos": topk[:, 0].numpy(),
        "knn10_density": topk.mean(1).numpy(),
        "inout_cos": torch.nn.functional.cosine_similarity(E_in[sel], Eo_eff[sel]).numpy(),
        "dp_self": self_s[tokens], "dp_competitor": best_s[tokens],
        "dp_margin": (self_s - best_s)[tokens],
        "glitch_dir": (E_in[sel] @ gdir).numpy(),
    }
    surface = {
        "id (baseline)": tokens.astype(float),
        "n_chars (baseline)": np.array([len(s.strip()) for s in dec], float),
        "leading_space (baseline)": np.array([s.startswith(" ") for s in dec], float),
        "alpha (baseline)": np.array([s.strip().isalpha() for s in dec], float),
        "ascii (baseline)": np.array([s.isascii() for s in dec], float),
    }

    # ---------- dynamic (train contexts only) ----------
    def spread(H):                               # H: (N, C_train, d) -> mean pairwise 1-cos
        X = torch.nn.functional.normalize(H.float(), dim=-1)
        G = torch.einsum("ncd,nkd->nck", X, X)
        n = G.shape[1]
        return ((G.sum((1, 2)) - n) / (n * (n - 1))).numpy() * -1 + 1
    tr_idx = torch.tensor(np.where(train)[0])
    Ht_last = states["text_last"][:, tr_idx]; Ht_mid = states["text_mid"][:, tr_idx]
    Hc_last = states["copy_last"][:, tr_idx]
    ret = torch.nn.functional.cosine_similarity(
        Ht_last.float(), Eo_eff[sel].unsqueeze(1).expand_as(Ht_last).float(), dim=-1)
    retc = torch.nn.functional.cosine_similarity(
        Hc_last.float(), Eo_eff[sel].unsqueeze(1).expand_as(Hc_last).float(), dim=-1)
    dynamic = {
        "text_spread_mid": spread(Ht_mid), "text_spread_last": spread(Ht_last),
        "copy_spread_last": spread(Hc_last),
        "identity_retention_text": ret.mean(1).numpy(),
        "identity_retention_copy": retc.mean(1).numpy(),
        "identity_retention_min": retc.min(1).values.numpy(),
        "text_norm_mean": Ht_last.float().norm(dim=-1).mean(1).numpy(),
    }
    behav = {"mu_train": mu_train, "frag_train": frag_train}

    ft = frag_test[clean]
    print()
    print("=" * 84)
    print("PER-FEATURE:  predicting fragile (frag_test >= %.2f) among clean-looking tokens"
          % a.frag_thr)
    print("=" * 84)
    print(f"{'tier':>11} | {'feature':>26} | {'spearman':>8} | {'AUC':>6} | {'95% CI':>15}")
    print("-" * 84)
    res = {}; best_surface = 0.5
    for tier, feats in (("surface", surface), ("static", static), ("dynamic", dynamic),
                        ("behaviour", behav)):
        for nm, v in feats.items():
            v = np.asarray(v, float)[clean]
            rho = spearman(v, ft)
            s = v if rho >= 0 else -v
            A, lo, hi = boot_auc(s, y)
            res[nm] = {"tier": tier, "spearman": rho, "auc": A, "ci": [lo, hi]}
            if tier == "surface":
                best_surface = max(best_surface, A)
            star = " *" if (lo > 0.5 and A >= 0.65) else ""
            print(f"{tier:>11} | {nm:>26} | {rho:+8.3f} | {A:6.3f} | [{lo:.3f}, {hi:.3f}]{star}")

    print()
    print("=" * 84)
    print("CROSS-VALIDATED LOGISTIC (5-fold, out-of-fold AUC) by tier")
    print("=" * 84)
    cv = {}; oof = {}
    stack = lambda D: np.column_stack([np.asarray(v, float)[clean] for v in D.values()])
    for nm, X in (("surface only", stack(surface)),
                  ("static geometry", stack(static)),
                  ("static + surface", np.hstack([stack(static), stack(surface)])),
                  ("dynamic (train half)", stack(dynamic)),
                  ("static + dynamic", np.hstack([stack(static), stack(dynamic)])),
                  ("behavioural (train half)", stack(behav)),
                  ("everything", np.hstack([stack(static), stack(surface), stack(dynamic),
                                            stack(behav)]))):
        p = logistic_cv(X, y); oof[nm] = p
        A, lo, hi = boot_auc(p, y)
        cv[nm] = {"auc": A, "ci": [lo, hi]}
        print(f"  {nm:>26}: AUC {A:.3f}  [{lo:.3f}, {hi:.3f}]")

    # the decisive statistic: a PAIRED bootstrap of the improvement over surface.
    # Two overlapping marginal CIs say nothing about the difference; this does.
    print("\n  paired bootstrap, improvement over 'surface only':")
    brng = np.random.default_rng(11)
    for nm in ("static geometry", "dynamic (train half)", "static + dynamic"):
        diffs = []
        for _ in range(2000):
            ii = brng.integers(0, len(y), len(y))
            if 0 < y[ii].sum() < len(ii):
                diffs.append(auc(oof[nm][ii], y[ii]) - auc(oof["surface only"][ii], y[ii]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        cv[nm]["vs_surface"] = [float(np.mean(diffs)), float(lo), float(hi)]
        print(f"    {nm:>22}: {np.mean(diffs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"{'   *' if lo > 0 else ''}")

    # ---------- cell level: which context breaks which token? ----------
    print()
    print("=" * 84)
    print("CELL LEVEL: predicting fail(t,c) on test contexts among clean-looking tokens")
    print("=" * 84)
    ci = np.where(clean)[0]; te = np.where(test)[0]
    rng = np.random.default_rng(3)
    half = rng.permutation(len(ci)); hA, hB = ci[half[: len(ci) // 2]], ci[half[len(ci) // 2:]]
    host_from_A = F[hA][:, te].mean(0)                    # context hostility, other tokens
    ctx_mean_emb = torch.stack([E_in[torch.tensor(ctx)].mean(0) for ctx, _ in bank])[te]
    rows = []
    for t in hB:
        for k, c in enumerate(te):
            sim = float(torch.nn.functional.cosine_similarity(E_in[int(tokens[t])], ctx_mean_emb[k], dim=0))
            rows.append((F[t, c], frag_train[t], host_from_A[k], sim,
                         float(torch.nn.functional.cosine_similarity(
                             states["text_last"][t, c].float(), Eo_eff[int(tokens[t])], dim=0))))
    R = np.array(rows, float)
    yc = R[:, 0].astype(int)
    print(f"  cells: {len(yc)}  fail rate {yc.mean():.4f}")
    if 10 <= yc.sum() < len(yc):
        names = ["token frag (train)", "context hostility (other tokens)",
                 "cos(E_in[t], mean E_in[ctx])  static", "identity retention in THIS context  dynamic"]
        for j, nm in enumerate(names, 1):
            rho = spearman(R[:, j], yc.astype(float))
            s = R[:, j] if rho >= 0 else -R[:, j]
            print(f"    {nm:>46}: AUC {auc(s, yc):.3f}")
        base = logistic_cv(R[:, 1:3], yc); A0 = auc(base, yc)
        stat = logistic_cv(R[:, 1:4], yc); A1 = auc(stat, yc)
        dyn = logistic_cv(R[:, [1, 2, 4]], yc); A2 = auc(dyn, yc)
        print(f"  CV logistic: main effects only {A0:.3f} | + static similarity {A1:.3f} | "
              f"+ dynamic retention {A2:.3f}")
        cell = {"main": A0, "main+static": A1, "main+dynamic": A2}
    else:
        cell = {}

    print()
    print("=" * 84)
    print("PRE-REGISTERED VERDICT")
    print("=" * 84)
    print(f"  best surface-baseline single-feature AUC: {best_surface:.3f};  "
          f"surface-only CV AUC {cv['surface only']['auc']:.3f}")
    passed = []
    for nm in ("static geometry", "dynamic (train half)", "static + dynamic"):
        A, (lo, hi) = cv[nm]["auc"], cv[nm]["ci"]
        ok = A >= 0.65 and lo > 0.5 and A > max(best_surface, cv["surface only"]["auc"])
        print(f"  {nm:>22}: CV AUC {A:.3f} [{lo:.3f},{hi:.3f}]  -> {'PASS' if ok else 'fail'}")
        if ok:
            passed.append(nm)
    print(f"  behavioural (train half) for reference: {cv['behavioural (train half)']['auc']:.3f}")
    if not passed:
        print("  NO geometric tier passes: fragility is real but not predictable by these features.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "tied": bool(tied),
               "n_clean": int(clean.sum()), "n_fragile": int(y.sum()),
               "base_rate": float(y.mean()), "features": res, "cv": cv, "cell": cell,
               "best_surface": best_surface, "passed": passed}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
