"""Can embedding geometry alone predict WHICH tokens and combinations a model
will confabulate, and WHAT it will confabulate them into?

Established earlier in this project: the glitch failure mode is confident
SUBSTITUTION. Asked to identify or copy a token, the model emits a different
token at low entropy and reasons fluently about the wrong one. Not high entropy,
not derailment -- substitution.

Substitution has a direct geometric account. Copying a token back runs through
the induction path: the earlier occurrence's INPUT embedding is moved into the
residual stream and read out against the UNEMBEDDING. To first order, ignoring
what the layers add, the readout at the copy position ranks candidate j by

    E_in[t] . E_out[j]

so the token that actually comes back is the argmax over j, and the quantity
that decides whether it is t itself is

    margin(t) = E_in[t].E_out[t] - max_{j != t} E_in[t].E_out[j]

Negative margin means another row wins the readout. This gives two predictions
of very different strength:

  WEAK    margin ranks tokens by reproduction failure (an AUC).
  STRONG  when the model fails on t, it emits argmax_j E_in[t].E_out[j].

The strong form predicts the identity of the substitute from two matrices with
no forward pass. An AUC can be produced by many uninteresting correlates of
rarity; predicting the specific wrong token it comes back as cannot.

For COMBINATIONS the same algebra gives cross-excitation: within a sequence,
E_in[a].E_out[b] for a != b is how much a excites b's readout, so
max-cross-excitation is a combination-level quantity rather than an aggregate of
per-token ones.

Approximation, stated plainly: this is the direct path only. The final layer
norm's gain is folded into E_out and its centring into E_in, but everything the
attention and MLP blocks contribute is ignored. That is the point -- if a
zero-layer approximation predicts the substitute, the phenomenon lives in the
embedding geometry. Baselines it must beat are the standard per-token
indicators, which are all rarity proxies: row norm, distance to the embedding
centroid, and nearest-neighbour cosine.
"""
from __future__ import annotations
import argparse, gzip, json, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .analyze_trajectory import auc
from .ngram_identify import reproduce

NORM_PATHS = ("model.norm", "gpt_neox.final_layer_norm", "transformer.ln_f",
              "model.transformer.norm", "model.final_layernorm")


def final_norm_weight(model, d):
    for path in NORM_PATHS:
        o, ok = model, True
        for p in path.split("."):
            if not hasattr(o, p):
                ok = False; break
            o = getattr(o, p)
        if ok and hasattr(o, "weight") and o.weight is not None:
            return o.weight.detach().float().flatten()[:d]
    return torch.ones(d)


@torch.no_grad()
def direct_path(E_in, E_out, dev, chunk=256, topk=5):
    """Per token: self score, best competitor score, and the top-k competitors.

    E_in @ E_out.T is V x V and cannot be materialised for a 100k vocab, so this
    walks E_in in row chunks and keeps only the reductions.
    """
    V = E_in.shape[0]
    self_s = torch.zeros(V)
    best_s = torch.zeros(V)
    best_j = torch.zeros(V, topk, dtype=torch.long)
    Eo = E_out.to(dev)
    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        S = (E_in[s:e].to(dev) @ Eo.T).float()               # (chunk, V)
        idx = torch.arange(s, e, device=dev)
        self_s[s:e] = S[torch.arange(e - s, device=dev), idx].cpu()
        S[torch.arange(e - s, device=dev), idx] = -float("inf")   # exclude self
        v, j = S.topk(topk, dim=-1)
        best_s[s:e] = v[:, 0].cpu()
        best_j[s:e] = j.cpu()
        del S
    return self_s.numpy(), best_s.numpy(), best_j.numpy()


@torch.no_grad()
def geometry(E_in, dev, chunk=512, k=8):
    """Rarity-proxy baselines: norm, distance to centroid, nearest-neighbour cos."""
    V = E_in.shape[0]
    norm = E_in.norm(dim=-1).float().numpy()
    mu = E_in.float().mean(0, keepdim=True)
    cdist = (E_in.float() - mu).norm(dim=-1).numpy()
    X = (E_in.float() / (E_in.float().norm(dim=-1, keepdim=True) + 1e-9)).to(dev)
    nn = torch.zeros(V)
    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        S = X[s:e] @ X.T
        S[torch.arange(e - s, device=dev), torch.arange(s, e, device=dev)] = -2.0
        nn[s:e] = S.max(dim=-1).values.cpu()
        del S
    return norm, cdist, nn.numpy()


def load_verified(path, n):
    ver = np.zeros(n, int); tested = np.zeros(n, bool)
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line); i = int(r["i"])
        if i < n and "magikarp" in r:
            tested[i] = True
            if "verified" in r["magikarp"]:
                ver[i] = 1
    return ver, tested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--ext", default="external/allenai_OLMo_2_1124_7B.jsonl.gz")
    ap.add_argument("--n-probe", type=int, default=4000,
                    help="tokens to run the behavioural copy probe on")
    ap.add_argument("--n-seq", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=4)
    ap.add_argument("--out", default="results/embed_predict.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    if model.config.tie_word_embeddings:
        print("WARNING: tied embeddings -- E_in and E_out are the same matrix, so "
              "margin is degenerate. Results are not meaningful here.")
    E_in = model.get_input_embeddings().weight.detach().float().cpu()
    E_out = model.get_output_embeddings().weight.detach().float().cpu()
    V, d = E_in.shape
    # fold the final layer norm into the direct path: gain into E_out, centring
    # into E_in. Without this the dot products are in the wrong basis.
    g = final_norm_weight(model, d).cpu()
    E_out_eff = E_out * g.unsqueeze(0)
    E_in_c = E_in - E_in.mean(dim=-1, keepdim=True)
    print(f"{a.model}: V={V} d={d}, computing direct path...", flush=True)

    self_s, best_s, best_j = direct_path(E_in_c, E_out_eff, dev)
    margin = self_s - best_s
    norm, cdist, nncos = geometry(E_in, dev)
    print(f"margin: mean {margin.mean():.3f}, negative for "
          f"{(margin < 0).mean():.4f} of the vocabulary")

    # ---------- behavioural labels ----------
    rng = random.Random(0)
    ids = sorted(rng.sample(range(V), min(a.n_probe, V)))
    print(f"running the copy probe on {len(ids)} tokens...", flush=True)
    ex, H, outs = reproduce(model, tok, dev, [[t] for t in ids])
    y = (~ex).astype(int)
    print(f"reproduction failures: {y.sum()}/{len(y)}  ({y.mean():.3f})")

    print()
    print("=" * 74)
    print("WEAK FORM: does geometry rank single-token reproduction failure?")
    print("=" * 74)
    feats = {"copy margin (direct path)": -margin[ids],
             "best competitor score": best_s[ids],
             "self score": -self_s[ids],
             "row norm (baseline)": -norm[ids],
             "dist to centroid (baseline)": -cdist[ids],
             "nearest-nbr cos (baseline)": nncos[ids]}
    print(f"{'feature':>30} | {'AUC':>7}")
    print("-" * 42)
    aucs = {}
    for nm, s in feats.items():
        aucs[nm] = auc(s, y)
        print(f"{nm:>30} | {aucs[nm]:7.3f}")

    print()
    print("=" * 74)
    print("STRONG FORM: when the model fails, does it emit the PREDICTED token?")
    print("=" * 74)
    fails = [i for i in range(len(ids)) if y[i] == 1]
    hit1 = hit5 = scored = 0
    shown = []
    for i in fails:
        t = ids[i]
        got = outs[i].strip()
        if not got:
            continue
        scored += 1
        pred = [tok.decode([int(j)]).strip() for j in best_j[t]]
        if pred and pred[0] and got.startswith(pred[0]):
            hit1 += 1
        if any(p and got.startswith(p) for p in pred):
            hit5 += 1
        if len(shown) < 15:
            shown.append((tok.decode([t]), got, pred[:3]))
    # chance baseline: predicting a fixed arbitrary token
    print(f"scored failures: {scored}")
    if scored:
        print(f"  top-1 predicted substitute matches output: {hit1/scored:.3f}")
        print(f"  top-5 predicted substitute matches output: {hit5/scored:.3f}")
    print()
    print(f"{'token':>20} | {'model emitted':>22} | predicted substitutes")
    print("-" * 78)
    for w, got, pred in shown:
        print(f"{w[:20]!r:>20} | {got[:22]!r:>22} | {pred}")

    # ---------- combinations ----------
    print()
    print("=" * 74)
    print(f"COMBINATIONS: {a.seq_len}-token sequences of individually-clean tokens")
    print("=" * 74)
    clean = [ids[i] for i in range(len(ids)) if ex[i]]
    seqs = [[rng.choice(clean) for _ in range(a.seq_len)] for _ in range(a.n_seq)]
    sx, sH, souts = reproduce(model, tok, dev, seqs)
    sy = (~sx).astype(int)
    print(f"sequence failures: {sy.sum()}/{len(sy)}  ({sy.mean():.3f})")
    if sy.sum() >= 5 and sy.sum() < len(sy):
        Ein_d = E_in_c.to(dev); Eout_d = E_out_eff.to(dev)
        cross, minmarg, meanmarg = [], [], []
        for s in seqs:
            u = list(dict.fromkeys(s))
            M = (Ein_d[u] @ Eout_d[u].T).float().cpu().numpy()
            np.fill_diagonal(M, -np.inf)
            cross.append(float(M.max()) if len(u) > 1 else 0.0)
            minmarg.append(float(margin[s].min()))
            meanmarg.append(float(margin[s].mean()))
        sf = {"max cross-excitation": np.array(cross),
              "min margin in sequence": -np.array(minmarg),
              "mean margin in sequence": -np.array(meanmarg),
              "min row norm (baseline)": -np.array([norm[s].min() for s in seqs]),
              "max nn-cos (baseline)": np.array([nncos[s].max() for s in seqs])}
        print(f"{'feature':>30} | {'AUC':>7}")
        print("-" * 42)
        sa = {}
        for nm, s in sf.items():
            sa[nm] = auc(s, sy)
            print(f"{nm:>30} | {sa[nm]:7.3f}")
    else:
        sa = {}
        print("(degenerate failure count; no AUC)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "n_probe": len(ids), "fail_rate": float(y.mean()),
               "auc_single": aucs, "sub_top1": hit1 / max(scored, 1),
               "sub_top5": hit5 / max(scored, 1), "n_scored": scored,
               "examples": [{"tok": w, "emitted": g, "predicted": p}
                            for w, g, p in shown],
               "seq_fail_rate": float(sy.mean()), "auc_seq": sa},
              open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
