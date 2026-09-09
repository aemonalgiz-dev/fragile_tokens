"""Does proximity to the glitch class in embedding space predict fragility where the one-axis
projection does not?  The UMAP panels of Figure 2 are built from cosine neighbourhoods; the AUC
printed beside them is a projection onto a single axis (mean glitch embedding minus global mean).
For every model with a local embedding sample this scores each gated token by its cosine proximity
to the glitch class (nearest member, mean of the five nearest, share of glitch members among its
twenty nearest neighbours) in the unembedding space the UMAP uses and in the input-embedding space,
and compares the AUC for fragility against the projection, against token id, and against a control
built the same way from random never-fragile tokens.  CPU only; reads the npz samples."""
import os, json, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
sys.path.insert(0, ".")
from src.cut.figures_paper import MODELS, _geometry_arrays, auc


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def boot_ci(score, y, n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() == 0 or y[b].sum() == len(b):
            continue
        vals.append(auc(score[b], y[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def proximity_features(X, clean_idx, ref_idx, k_nn=20):
    """X unit rows. Returns dict of per-clean-token scores."""
    C = X[clean_idx]; R = X[ref_idx]
    S = C @ R.T                                   # cosine to every glitch-class member
    top5 = np.sort(S, axis=1)[:, -5:]
    # share of glitch members among the 20 nearest neighbours in the pool clean ∪ ref
    pool = np.concatenate([clean_idx, ref_idx]); is_ref = np.r_[np.zeros(len(clean_idx), bool), np.ones(len(ref_idx), bool)]
    P = X[pool]
    SP = C @ P.T
    SP[np.arange(len(clean_idx)), np.arange(len(clean_idx))] = -2      # drop self
    nn = np.argpartition(-SP, k_nn, axis=1)[:, :k_nn]
    knn_share = is_ref[nn].mean(1)
    return {"max cos to glitch class": S.max(1), "mean cos to 5 nearest glitch": top5.mean(1),
            f"share of glitch among {k_nn} nearest": knn_share}


out = {}
rows = []
for tag, name, model, npz in MODELS:
    A = _geometry_arrays(tag, model, npz)
    clean = np.where(A["clean"])[0]
    ref = np.asarray(A["ref"]); ref = ref[~np.isin(ref, clean)]
    y = A["fragile"][clean].astype(int)
    if y.sum() < 5 or len(ref) < 5:
        print(f"skip {name}: {y.sum()} fragile, {len(ref)} ref"); continue
    res = {"n_gated": int(len(clean)), "n_fragile": int(y.sum()), "n_ref": int(len(ref))}
    res["projection on glitch direction"] = auc(A["proj"][clean], y)
    res["token id"] = auc(A["tok"][clean].astype(float), y)
    rng = np.random.default_rng(0)
    never = clean[y == 0]
    for space, E, mu in (("unembedding", A["E_out"], A["mu_out"]), ("input", A["E_in"], A["E_in"].mean(0))):
        X = unit(E - mu)
        feats = proximity_features(X, clean, ref)
        for fname, sc in feats.items():
            a = auc(sc, y); lo, hi = boot_ci(sc, y)
            res[f"{space}: {fname}"] = a; res[f"{space}: {fname} ci"] = [lo, hi]
        # control: the same features against random never-fragile reference sets of the same size
        ctrl = []
        for r in range(10):
            rnd = rng.choice(never, len(ref), replace=False)
            keep = ~np.isin(clean, rnd)
            f = proximity_features(X, clean[keep], rnd)
            ctrl.append(auc(f["max cos to glitch class"], y[keep]))
        res[f"{space}: max cos to random never-fragile set (control, mean of 10)"] = float(np.mean(ctrl))
    out[tag] = res
    rows.append((name, res))
    print(f"\n== {name}: gated {res['n_gated']}, fragile {res['n_fragile']}, glitch class {res['n_ref']}")
    for k, v in res.items():
        if isinstance(v, float):
            ci = res.get(k + " ci")
            print(f"   {k:64s} {v:.3f}" + (f"  [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))

json.dump(out, open("results/glitch_proximity.json", "w", encoding="utf-8"), indent=1)
print("\nwrote results/glitch_proximity.json")
