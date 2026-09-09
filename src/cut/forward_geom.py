"""Does a glitch token's IDENTITY survive the forward pass, and does its
CONTINUATION survive independently of that?

Every detector in the literature -- and everything in this project so far --
scores a token as a row of a matrix. Field reports are about interaction: the
same token behaves differently depending on context, distinct glitch tokens get
confused with each other, and models emit unrelated fragments when asked what
they are looking at. Those are claims about the residual stream.

Token ids are fed DIRECTLY, never through the tokenizer on a concatenated
string. segmentation.py shows a leading space re-segments ~90% of Pythia tokens
regardless of health, so text-built prompts would measure the tokenizer instead.

Two capacities are measured separately, because the qualitative logit lens
suggests they dissociate -- 'idepress' is a verified glitch token whose next
token the model predicts perfectly ('ant'), while it cannot say what the token
itself is:

  IDENTITY RETRIEVAL.  Take h_l at the token's position in context c, and in a
      different context c'. Centre within each context to remove the context
      offset, then ask: among all probed tokens, is the nearest neighbour of
      h_l(t,c) equal to h_l(t,c')?  Top-1 accuracy over context pairs is a
      readout-free measure of how far up the network the token's own identity
      is still recoverable. It needs no unembedding, no probe training, and no
      assumption about what the model does with the identity.

  CONTINUATION.  H(next | c, t). Whether the model knows what FOLLOWS the token
      -- the capacity a purely predictive model needs, and the one that
      pretraining on running text actually rewards.

A token that keeps continuation but loses identity is not "untrained". It is
trained only in the direction the pretraining objective ever asked for. Copying,
quoting, spelling and reasoning-about-a-token are all self-reference, which is
why they are what breaks.

The predecessor sweep re-runs identity retrieval against a no-predecessor
reference, so "does a leading space break it" becomes a number, with the
tokenizer held out of the picture by construction.
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONTEXTS = [
    "The word",
    " In 1994, the",
    " He said that",
    " According to the",
    " Please repeat this exactly:",
    " Q: what does this mean?\nA:",
    " The city of",
    "\n\n",
]

NORM_PATHS = ("gpt_neox.final_layer_norm", "model.norm", "transformer.ln_f",
              "model.transformer.norm", "model.final_layernorm", "model.model.norm")


def final_norm(model):
    for path in NORM_PATHS:
        o, ok = model, True
        for p in path.split("."):
            if not hasattr(o, p):
                ok = False; break
            o = getattr(o, p)
        if ok:
            return o
    return torch.nn.Identity()


@torch.no_grad()
def hidden_last(model, seqs, dev, batch=32):
    """(N, L+1, d) hidden states at the FINAL position of each id sequence."""
    outs = []
    for s in range(0, len(seqs), batch):
        x = torch.tensor(seqs[s:s + batch], device=dev)
        hs = model(input_ids=x, output_hidden_states=True).hidden_states
        outs.append(torch.stack([h[:, -1, :].float() for h in hs], 1).cpu())
    return torch.cat(outs)


def retrieval(A, B):
    """Rank of the correct match: where does B[i] sit in A[i]'s neighbour list?

    Rank rather than top-1 accuracy, because with a few dozen candidates top-1
    saturates at 1.000 for every group and measures nothing. Rank keeps
    resolving once the pool is large.

    Both are centred first -- otherwise the shared context component dominates
    every cosine and the match is decided by nothing.
    """
    A = A - A.mean(0, keepdim=True)
    B = B - B.mean(0, keepdim=True)
    A = A / (A.norm(dim=-1, keepdim=True) + 1e-9)
    B = B / (B.norm(dim=-1, keepdim=True) + 1e-9)
    S = A @ B.T
    own = S.diag().unsqueeze(1)
    return (S > own).sum(1).numpy() + 1          # 1 = perfect


@torch.no_grad()
def entropy_at(model, seqs, dev, batch=32):
    out = []
    for s in range(0, len(seqs), batch):
        x = torch.tensor(seqs[s:s + batch], device=dev)
        lp = torch.log_softmax(model(input_ids=x).logits[:, -1].float(), -1)
        out.append((-(lp.exp() * lp).sum(-1)).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def lens_top(model, nrm, H, dev):
    W = model.get_output_embeddings()
    dt = next(model.parameters()).dtype
    return np.stack([W(nrm(H[i].to(dev).to(dt))).float().argmax(1).cpu().numpy()
                     for i in range(H.shape[0])])


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
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--ext", default="external/pythia_6_9b.jsonl.gz")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--pool", type=int, default=1500,
                    help="healthy candidates in the retrieval pool; top-1 over a "
                         "few dozen saturates and resolves nothing")
    ap.add_argument("--out", default="results/forward_geom.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    nrm = final_norm(model)
    V = model.get_input_embeddings().weight.shape[0]
    ver, tested = load_verified(a.ext, V)

    rng = np.random.default_rng(0)
    pos = [int(i) for i in np.where(ver == 1)[0]]
    rng.shuffle(pos); pos = pos[:a.n]
    negpool = np.where((ver == 0) & tested)[0]
    # match the group sizes: unequal n makes every centred similarity measure
    # depend on the imbalance rather than on the tokens.
    neg = [int(i) for i in rng.choice(negpool, size=min(a.pool, len(negpool)),
                                      replace=False)]
    allids = pos + neg
    N = len(allids)
    grp = {"glitch": np.arange(len(pos)), "healthy": np.arange(len(pos), N)}
    print(f"{a.model}: {len(pos)} glitch vs {len(neg)} healthy in a shared "
          f"retrieval pool of {N}, {len(CONTEXTS)} contexts, {dev}")
    print("  glitch sample:", [repr(tok.decode([t]))[:12] for t in pos[:6]])

    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    L = model.config.num_hidden_layers

    # ---------- collect hidden states in every context ----------
    Hs, ent = [], []
    for c in CONTEXTS:
        pre = enc(c)
        seqs = [pre + [t] for t in allids]
        Hs.append(hidden_last(model, seqs, dev).half())
        ent.append(entropy_at(model, seqs, dev))
    ent = np.mean(ent, 0)

    # ---------- identity retrieval, layer by layer ----------
    pairs = [(i, j) for i in range(len(CONTEXTS)) for j in range(len(CONTEXTS)) if i != j]
    acc = {g: np.zeros(L + 1) for g in grp}
    rnk = {g: np.zeros(L + 1) for g in grp}
    for l in range(L + 1):
        R = np.stack([retrieval(Hs[i][:, l], Hs[j][:, l]) for i, j in pairs])
        hit = (R == 1).mean(0)
        med = np.median(R, 0)
        for g, idx in grp.items():
            acc[g][l] = hit[idx].mean()
            rnk[g][l] = np.median(med[idx])

    print()
    print("=" * 74)
    print("IDENTITY RETRIEVAL BY LAYER  (%d candidates, chance top-1 = %.4f)"
          % (N, 1 / N))
    print("can the token be identified from its own hidden state at layer l?")
    print("=" * 74)
    print(f"{'layer':>6} | {'glitch':>8} | {'healthy':>8} | {'gap':>7}")
    print("-" * 38)
    rows = []
    for l in range(L + 1):
        r = {"layer": l, "glitch": float(acc["glitch"][l]),
             "healthy": float(acc["healthy"][l]),
             "gap": float(acc["healthy"][l] - acc["glitch"][l])}
        rows.append(r)
        if l <= 4 or l % max(1, L // 8) == 0 or l == L:
            print(f"{l:6d} | {r['glitch']:8.3f} | {r['healthy']:8.3f} | {r['gap']:7.3f}")

    half_g = next((r["layer"] for r in rows if r["glitch"] < 0.5 * rows[0]["glitch"]), None)
    half_h = next((r["layer"] for r in rows if r["healthy"] < 0.5 * rows[0]["healthy"]), None)
    print(f"\nlayer at which retrieval falls below half its layer-0 value:"
          f"  glitch {half_g}   healthy {half_h}")

    # ---------- the dissociation ----------
    print()
    print("=" * 74)
    print("DISSOCIATION: identity vs continuation")
    print("=" * 74)
    mid = L // 2
    print(f"{'group':>10} | {'identity @L0':>12} | {'identity @L' + str(mid):>12} | "
          f"{'next-tok entropy':>16}")
    print("-" * 60)
    diss = {}
    for g, idx in grp.items():
        diss[g] = {"id0": float(acc[g][0]), "idmid": float(acc[g][mid]),
                   "entropy": float(ent[idx].mean())}
        print(f"{g:>10} | {acc[g][0]:12.3f} | {acc[g][mid]:12.3f} | "
              f"{ent[idx].mean():16.3f}")

    # per-token split: which glitch tokens keep continuation but lose identity?
    per = (np.stack([retrieval(Hs[i][:, mid], Hs[j][:, mid])
                     for i, j in pairs]) == 1).mean(0)
    gi = grp["glitch"]
    keep = [allids[i] for i in gi if per[i] >= 0.5]
    lost = [allids[i] for i in gi if per[i] < 0.5]
    print(f"\nverified glitch tokens with identity retained at L{mid}: {len(keep)}/{len(gi)}")
    print("  retained:", [repr(tok.decode([t]))[:14] for t in keep[:8]])
    print("  lost    :", [repr(tok.decode([t]))[:14] for t in lost[:8]])
    if keep and lost:
        ek = ent[[i for i in gi if per[i] >= 0.5]].mean()
        el = ent[[i for i in gi if per[i] < 0.5]].mean()
        print(f"  next-token entropy: retained {ek:.3f}  lost {el:.3f}")

    # ---------- logit lens specimens ----------
    print()
    print("=" * 74)
    print("LOGIT LENS AT THE TOKEN'S POSITION (context 'The word') -- what the")
    print("model predicts comes NEXT. Continuation, not identity.")
    print("=" * 74)
    tp = lens_top(model, nrm, Hs[0], dev)
    probe = sorted(set([0, 1, 2, max(3, L // 4), L // 2, 3 * L // 4, L]))
    print(f"{'token':>20} | {'idL%d' % mid:>5} | " +
          " ".join(f"L{l:<2d}".ljust(12) for l in probe))
    print("-" * (30 + 13 * len(probe)))
    spec = []
    for label, idx in [("GLITCH", grp["glitch"][:10]), ("HEALTHY", grp["healthy"][:5])]:
        print(f"--- {label}")
        for i in idx:
            t = allids[i]
            cells = [repr(tok.decode([int(tp[i, l])]))[:11].ljust(12) for l in probe]
            print(f"{repr(tok.decode([t]))[:20]:>20} | {per[i]:5.2f} | " + " ".join(cells))
            spec.append({"group": label, "id": int(t), "tok": tok.decode([t]),
                         "id_retrieval": float(per[i]),
                         "readout": {int(l): tok.decode([int(tp[i, l])]) for l in probe}})

    # ---------- predecessor sweep ----------
    print()
    print("=" * 74)
    print("PREDECESSOR SWEEP -- identity retrieval against a no-predecessor")
    print("reference, ids fed directly so no re-segmentation is possible")
    print("=" * 74)
    base = enc("The word")
    ref = hidden_last(model, [base + [t] for t in allids], dev)
    preds = [("space", enc(" ")), ("newline", enc("\n")), ("tab", enc("\t")),
             ("dquote", enc('"')), ("lparen", enc("(")), ("word ' the'", enc(" the")),
             ("self (doubled)", None)]
    print(f"{'predecessor':>16} | {'glitch':>8} | {'healthy':>8} | {'gap':>7}")
    print("-" * 48)
    pred_rows = []
    for pname, pids in preds:
        seqs = ([base + [t, t] for t in allids] if pids is None
                else [base + pids + [t] for t in allids])
        H = hidden_last(model, seqs, dev)
        hit = (retrieval(H[:, mid], ref[:, mid]) == 1).astype(float)
        r = {"pred": pname, "glitch": float(hit[grp["glitch"]].mean()),
             "healthy": float(hit[grp["healthy"]].mean())}
        r["gap"] = r["healthy"] - r["glitch"]
        pred_rows.append(r)
        print(f"{pname:>16} | {r['glitch']:8.3f} | {r['healthy']:8.3f} | {r['gap']:7.3f}")
    print("\n(a predecessor that mattered would move one group and not the other;")
    print(" a flat column means the field-report prefix pattern is tokenizer-only)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "n_layers": L, "n_per_group": len(pos),
               "layers": rows, "dissociation": diss, "specimens": spec,
               "predecessor": pred_rows,
               "identity_retained": [tok.decode([t]) for t in keep],
               "identity_lost": [tok.decode([t]) for t in lost]},
              open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
