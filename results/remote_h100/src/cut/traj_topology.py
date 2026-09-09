"""Does the hidden-state trajectory of a COMPLETION develop topological
structure, and does it differ for glitch-seeded chains?

Every geometric predictor tried so far has been static -- a property of a row of
the embedding matrix, or of a set of rows. All of them failed on combinations:
row norm 0.58, direct-path margin 0.55, cross-excitation 0.503, nearest-neighbour
cosine 0.36 (backwards), completable-fragment 0.545.

A completion is a different object. The model conditions on its own output, so
the per-step residual states trace a path through activation space, and a path
has shape that no per-token statistic can express. The specific structure worth
looking for is a 1-cycle: the trajectory returning to a region it already
visited. Rumination, circling back, and the degenerate repetition seen earlier
(' week aDecoder lesbisk His ...' -> repeated CJK) are all literally loops.

Persistent homology of the trajectory point cloud measures exactly that. For
each chain, the states at each generated position form a point cloud in R^d;
H0 bars describe how it fragments into clusters, H1 bars describe cycles, and
the persistence of a bar is how robust that feature is to scale.

THE CONTROL THAT DECIDES THIS. A chain that repeats tokens revisits nearly
identical states, so it manufactures cycles by construction. If H1 merely tracks
repetition rate it is an expensive repetition detector and adds nothing. So
every topological feature is reported (a) as a raw discriminator, and (b) after
stratifying on repetition, and is compared head-to-head against repetition rate
itself. Beating repetition is the bar.

Distances are cosine: the residual stream grows in norm across positions, so a
Euclidean cloud is dominated by that drift rather than by direction. Each chain
is also centred before the distance matrix is built, for the same reason the
static analyses centred -- the shared component is large and carries no
chain-specific information.
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .analyze_trajectory import auc


def load_verified(path, n):
    ver = np.zeros(n, int); tested = np.zeros(n, bool)
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line); i = int(r["i"])
        if i < n and "magikarp" in r:
            tested[i] = True
            if "verified" in r["magikarp"]:
                ver[i] = 1
    return ver, tested


def build(tok, t):
    """Reasoning-style prompt with the seed id spliced in directly."""
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    try:
        s = tok.apply_chat_template([{"role": "user", "content": "\x00"}],
                                    tokenize=False, add_generation_prompt=True)
        a, b = s.split("\x00")
        return (e(a + "What is this token? Identify it exactly, then explain "
                  "step by step what it means: ") + [t]
                + e(b + "Let me think about this carefully."))
    except Exception:
        return e("Q: What is this token? ") + [t] + e("\nA: Let me think.")


@torch.no_grad()
def chain(model, tok, dev, ids, max_new, layers):
    p = torch.tensor(ids, device=dev).unsqueeze(0)
    eos = tok.eos_token_id
    out = model.generate(p, max_new_tokens=max_new, min_new_tokens=max_new,
                         do_sample=False, pad_token_id=eos or 0,
                         suppress_tokens=[eos] if eos is not None else None)
    full = out[0]
    o = model(input_ids=full.unsqueeze(0), output_hidden_states=True)
    # the last layer is already specialised for next-token readout; a mid layer
    # is where the residual stream still carries general structure, so both are
    # taken rather than assuming which one any topology would live in.
    return (full[p.shape[1]:].tolist(),
            {nm: o.hidden_states[l][0, p.shape[1]:].float().cpu().numpy()
             for nm, l in layers.items()})


def ph_features(H, maxdim=1):
    """Persistent homology of one chain's state cloud, on cosine distance."""
    from ripser import ripser
    X = H - H.mean(0, keepdims=True)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    D = 1.0 - X @ X.T
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0, None)
    dgms = ripser(D, distance_matrix=True, maxdim=maxdim)["dgms"]
    f = {}
    for k, d in enumerate(dgms):
        if len(d) == 0:
            f[f"H{k}_sum"] = 0.0; f[f"H{k}_max"] = 0.0; f[f"H{k}_n"] = 0.0
            continue
        life = d[:, 1] - d[:, 0]
        life = life[np.isfinite(life)]
        f[f"H{k}_sum"] = float(life.sum()) if len(life) else 0.0
        f[f"H{k}_max"] = float(life.max()) if len(life) else 0.0
        f[f"H{k}_n"] = float((life > 0.05).sum()) if len(life) else 0.0
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--ext", default="external/allenai_OLMo_2_1124_7B.jsonl.gz")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--out", default="results/traj_topology.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    V = model.get_input_embeddings().weight.shape[0]
    ver, tested = load_verified(a.ext, V)

    rng = np.random.default_rng(0)
    pos = [int(i) for i in np.where(ver == 1)[0]]
    rng.shuffle(pos); pos = pos[:a.n]
    negpool = np.where((ver == 0) & tested)[0]
    neg = [int(i) for i in rng.choice(negpool, size=len(pos), replace=False)]
    allids = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    print(f"{a.model}: {len(pos)} glitch vs {len(neg)} healthy seeds, "
          f"{a.max_new} steps, {dev}")

    L = model.config.num_hidden_layers
    layers = {"mid": L // 2, "last": L}
    print(f"  layers probed: {layers}")
    rows, gens = [], []
    for k, t in enumerate(allids):
        g, Hs = chain(model, tok, dev, build(tok, t), a.max_new, layers)
        f = {}
        for nm, H in Hs.items():
            for kk, vv in ph_features(H).items():
                f[f"{nm}:{kk}"] = vv
        f["rep"] = 1.0 - len(set(g)) / max(len(g), 1)
        f["distinct"] = float(len(set(g)))
        rows.append(f); gens.append(g)
        if k % 32 == 0:
            print(f"  {k}/{len(allids)}", flush=True)

    keys = [k for k in rows[0]]
    M = {k: np.array([r[k] for r in rows]) for k in keys}

    print()
    print("=" * 74)
    print("TOPOLOGY OF THE COMPLETION TRAJECTORY")
    print("=" * 74)
    print(f"{'feature':>16} | {'glitch':>10} | {'healthy':>10} | {'AUC':>7} | {'95% CI':>16}")
    print("-" * 70)
    res = {}
    # bootstrap CI, because an AUC near 0.6 on a few dozen chains is not
    # distinguishable from chance and should not be reported as if it were
    boot_rng = np.random.default_rng(1)
    for k in keys:
        v = M[k]
        A = auc(v, y)
        bs = []
        for _ in range(2000):
            ii = boot_rng.integers(0, len(y), len(y))
            if 0 < y[ii].sum() < len(ii):
                bs.append(auc(v[ii], y[ii]))
        lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
        res[k] = {"glitch": float(v[y == 1].mean()), "healthy": float(v[y == 0].mean()),
                  "auc": A, "ci": [float(lo), float(hi)]}
        sig = " *" if (lo > 0.5 or hi < 0.5) else ""
        print(f"{k:>16} | {v[y==1].mean():10.4f} | {v[y==0].mean():10.4f} | "
              f"{A:7.3f} | [{lo:.3f}, {hi:.3f}]{sig}")

    print()
    print("=" * 74)
    print("CONTROL: is H1 just measuring repetition?")
    print("=" * 74)
    rep = M["rep"]
    for k in [x for x in keys if "H1_" in x or "H0_sum" in x]:
        c = float(np.corrcoef(M[k], rep)[0, 1])
        print(f"  corr({k}, repetition rate) = {c:+.3f}")
    print(f"  AUC of repetition rate alone = {auc(rep, y):.3f}")

    # stratify: split at the median repetition rate and re-score inside each half
    med = np.median(rep)
    print(f"\n  within-stratum AUC (median repetition split at {med:.3f}):")
    print(f"{'feature':>12} | {'low-rep':>9} | {'high-rep':>9}")
    print("-" * 36)
    strat = {}
    for k in [x for x in keys if "H1_" in x]:
        out = []
        for lab, m in (("low", rep <= med), ("high", rep > med)):
            yy = y[m]
            out.append(auc(M[k][m], yy) if 0 < yy.sum() < len(yy) else float("nan"))
        strat[k] = out
        print(f"{k:>12} | {out[0]:9.3f} | {out[1]:9.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "n": len(pos), "steps": a.max_new,
               "features": res, "rep_auc": auc(rep, y),
               "corr_rep": {k: float(np.corrcoef(M[k], rep)[0, 1])
                            for k in keys if k not in ("rep", "distinct")},
               "stratified": {k: [None if np.isnan(x) else x for x in v]
                              for k, v in strat.items()}},
              open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
