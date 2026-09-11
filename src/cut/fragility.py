"""Fragility: how often does a token fail to copy across RANDOM contexts, and
how much of that is token, how much context, how much token x context?

The interaction screen (interaction_screen.py) established that copy failures
among individually-clean tokens are not pairwise: they factor as a fragile
token times a hostile surrounding context, and which contexts are hostile did
not reproduce across filler sequences. Every existing glitch detector -- and the
standard single probe in this repo -- measures a token in ONE context. A token
that copies perfectly there but fails in 15% of ordinary contexts is invisible
to all of them, and it is exactly the hazard for a system that assembles
prompts programmatically.

This builds the object directly: a fixed bank of C random contexts (common
clean filler words; lengths 8/16/32/64; random interior slot), every token in
every context, cell = teacher-forced copy logprob at the slot. N x C, not N^2.

From the matrix:

  DECOMPOSITION   lp(t,c) = mu + token(t) + context(c) + residual(t,c).
                  The residual's share of variance IS the context dependence,
                  quantified. Reported on the logprob and on the fail indicator.

  FRAGILITY       frag(t) = fraction of contexts with lp < -0.5 (p < 0.61).
                  Reported for all tokens, for Magikarp-verified glitch tokens
                  as a reference (should be ~1), and -- the target -- for tokens
                  that PASS the standard single probe (lp_single > -0.1).

Slot hidden states are stored (Text-span slot at layers L/2 and L, Copy-span
slot at L) so fragility_predict.py can test whether the token's evolving
representation on half the contexts predicts its failures on the other half.

Contexts are the SAME for every token, so context effects are shared and any
token difference in fragility is a token difference.
"""
from __future__ import annotations
import argparse, gzip, json, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ngram_identify import reproduce
from .copyprompt import copy_logprob_ids
from .interaction_screen import common_word_ids, FEWSHOT
from .loadmodel import add_model_args, load_from_args, describe
from .copyprompt import copy_prompt_parts
from .copyprompt import STYLE as PROMPT_STYLE

FAIL = -0.5


def load_verified(path, n):
    ver = np.zeros(n, int); tested = np.zeros(n, bool)
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line); i = int(r["i"])
        if i < n and "magikarp" in r:
            tested[i] = True
            if "verified" in r["magikarp"]:
                ver[i] = 1
    return ver, tested


@torch.no_grad()
def score_context(model, tok, dev, ctx, slot, tokens, layers, batch=24):
    """All tokens in one context. Same length throughout -> no padding.
    Returns lp at the copy-span slot (N,) and slot states."""
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    head = e("Repeat the text exactly.\n")
    for d in FEWSHOT:
        head += e("Text:") + e(d) + e("\nCopy:") + e(d) + e("\n")
    pre_t, pre_c = e("Text:"), e("\nCopy:")
    head, pre_t, pre_c = copy_prompt_parts(tok, head, FEWSHOT)   # raw or chat framing (GLITCH_PROMPT_STYLE)
    K = len(ctx)
    text_start = len(head) + len(pre_t)
    copy_start = text_start + K + len(pre_c)
    base = head + pre_t + list(ctx) + pre_c + list(ctx)
    tpos, cpos = text_start + slot, copy_start + slot
    N = len(tokens)
    lp_out = np.zeros(N, np.float32)
    st = {"text_mid": [], "text_last": [], "copy_last": []}
    for s in range(0, N, batch):
        ch = tokens[s:s + batch]
        seqs = []
        for t in ch:
            q = list(base); q[tpos] = int(t); q[cpos] = int(t); seqs.append(q)
        ids = torch.tensor(seqs, device=dev)
        out = model(input_ids=ids, output_hidden_states=True)
        lp = torch.log_softmax(out.logits[:, cpos - 1].float(), -1)
        lp_out[s:s + len(ch)] = lp.gather(-1, ids[:, cpos].unsqueeze(-1)).squeeze(-1).cpu().numpy()
        st["text_mid"].append(out.hidden_states[layers["mid"]][:, tpos].half().cpu())
        st["text_last"].append(out.hidden_states[layers["last"]][:, tpos].half().cpu())
        st["copy_last"].append(out.hidden_states[layers["last"]][:, cpos].half().cpu())
    return lp_out, {k: torch.cat(v) for k, v in st.items()}


def decompose(M):
    """Two-way additive decomposition; returns variance shares."""
    mu = M.mean()
    tok_eff = M.mean(1, keepdims=True) - mu
    ctx_eff = M.mean(0, keepdims=True) - mu
    resid = M - mu - tok_eff - ctx_eff
    ss = lambda x: float((x ** 2).sum())
    tot = ss(M - mu)
    if M.size == 0 or tot == 0:          # empty or constant group: shares undefined
        nan = float("nan")
        return {"token": nan, "context": nan, "interaction": nan}, resid
    return {"token": ss(np.broadcast_to(tok_eff, M.shape)) / tot,
            "context": ss(np.broadcast_to(ctx_eff, M.shape)) / tot,
            "interaction": ss(resid) / tot}, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--ext", default="external/allenai_OLMo_2_1124_7B.jsonl.gz")
    ap.add_argument("--n-random", type=int, default=2000)
    ap.add_argument("--n-random-base", type=int, default=4200,
                    help="draw this many first (the scale ladder used 4200), then extend from the remainder, "
                         "so a larger --n-random keeps the smaller run's token ids as a subset")
    ap.add_argument("--n-glitch", type=int, default=100)
    ap.add_argument("--per-len", type=int, default=12)
    ap.add_argument("--lens", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--extra-tokens", default=None,
                    help="json with a 'token_ids' list (e.g. results/entities.json) whose "
                         "tokens are added to the matrix so downstream analyses have "
                         "fragility for exactly the tokens they use")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/fragility.json")
    ap.add_argument("--pt", default="results/fragility.pt")
    add_model_args(ap)
    a = ap.parse_args()

    model, tok, dev = load_from_args(a)
    V = model.get_input_embeddings().weight.shape[0]
    L = model.config.num_hidden_layers
    layers = {"mid": L // 2, "last": L}
    # labels are optional: the fragility measurement itself needs none. Without
    # them there is no verified-glitch reference class, and the run says so.
    if a.ext and Path(a.ext).exists():
        ver, tested = load_verified(a.ext, V)
    else:
        print(f"  no label file ({a.ext}); running label-free, no glitch reference class")
        ver, tested = np.zeros(V, int), np.zeros(V, bool)
    rng = random.Random(a.seed)
    special = set(tok.all_special_ids)
    batch = a.batch or 24

    # ---------- tokens: stratified over id terciles (uniform ids skew rare) ----------
    def ok(t):
        s = tok.decode([t])
        return t not in special and s.strip() != "" and s.isprintable()
    thirds = [range(0, V // 3), range(V // 3, 2 * V // 3), range(2 * V // 3, V)]
    base = min(a.n_random, a.n_random_base)
    toks = []
    for rg in thirds:
        c = [t for t in rng.sample(list(rg), min(len(rg), base)) if ok(t)]
        toks += c[:base // 3]
    if a.n_random > base:
        # extension: the first `base` draws above are byte-identical to a run with --n-random base,
        # so the smaller sample is a subset; the context bank drawn afterwards differs.
        have = set(toks); more = a.n_random - base
        for rg in thirds:
            rest = [t for t in rg if t not in have]
            c = [t for t in rng.sample(rest, min(len(rest), more)) if ok(t)]
            toks += c[:more // 3]
        print(f"  token sample extended {base} -> {a.n_random} (first {base} ids unchanged)", flush=True)
    n_extra = 0
    if a.extra_tokens:
        extra = [int(t) for t in json.load(open(a.extra_tokens))["token_ids"] if ok(int(t))]
        n_extra = len(set(extra) - set(toks))
        toks = toks + extra
    toks = sorted(set(toks))
    glitch = [int(i) for i in np.where(ver == 1)[0] if ok(int(i)) and int(i) not in set(toks)]
    rng.shuffle(glitch); glitch = glitch[:a.n_glitch]
    tokens = toks + glitch
    is_glitch = np.array([0] * len(toks) + [1] * len(glitch))
    N = len(tokens)
    print(f"{a.model}: {len(toks)} random/entity tokens ({n_extra} added from "
          f"{a.extra_tokens}) + {len(glitch)} verified glitch = {N}", flush=True)

    # ---------- filler and context bank ----------
    cw = [t for t in common_word_ids(tok, V) if t not in set(tokens)]
    exw, _, _ = reproduce(model, tok, dev, [[t] for t in cw])
    fill = [t for t, ok_ in zip(cw, exw) if ok_]
    print(f"  filler pool: {len(fill)} clean common words")
    bank = []
    for K in a.lens:
        for _ in range(a.per_len):
            ctx = rng.sample(fill, K)
            slot = rng.randint(1, K - 2)
            bank.append((ctx, slot))
    C = len(bank)
    print(f"  context bank: {C} contexts, lengths {a.lens} x {a.per_len}, random interior slot")

    # ---------- standard single probe: what existing detectors see ----------
    single = copy_logprob_ids(model, tok, dev, np.array(tokens))
    clean_look = single > -0.1
    print(f"  standard single probe: {int(clean_look.sum())}/{N} tokens look clean (lp > -0.1); "
          f"glitch tokens looking clean: {int((clean_look & (is_glitch == 1)).sum())}")

    # ---------- score N x C ----------
    M = np.zeros((N, C), np.float32)
    states = {k: [] for k in ("text_mid", "text_last", "copy_last")}
    for c, (ctx, slot) in enumerate(bank):
        lp, st = score_context(model, tok, dev, ctx, slot, tokens, layers, batch=batch)
        M[:, c] = lp
        for k in states:
            states[k].append(st[k])
        if c % 8 == 0:
            print(f"    context {c + 1}/{C}  K={len(ctx)}  slot={slot}  "
                  f"mean lp {lp.mean():.3f}  fail rate {(lp < FAIL).mean():.3f}", flush=True)
    states = {k: torch.stack(v, 1) for k, v in states.items()}          # (N, C, d)
    F = (M < FAIL)
    # save the matrix before any statistic, so a failure below cannot lose the run
    Path(a.pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": tokens, "is_glitch": is_glitch, "single": single, "M": M,
                "bank": bank, "states": states, "layers": layers, "fail_thr": FAIL}, a.pt)

    # ---------- decomposition ----------
    shares_lp, resid = decompose(M.astype(np.float64))
    shares_f, _ = decompose(F.astype(np.float64))
    # among clean-looking non-glitch tokens only -- the population that matters
    m_cl = clean_look & (is_glitch == 0)
    shares_cl, _ = decompose(M[m_cl].astype(np.float64))
    print()
    print("=" * 78)
    print("VARIANCE DECOMPOSITION of copy logprob:  token + context + token x context")
    print("=" * 78)
    for nm, sh in (("all tokens, logprob", shares_lp), ("all tokens, fail indicator", shares_f),
                   ("clean-looking tokens, logprob", shares_cl)):
        print(f"  {nm:>32}:  token {sh['token']:.3f}   context {sh['context']:.3f}   "
              f"interaction {sh['interaction']:.3f}")

    # ---------- fragility ----------
    frag = F.mean(1); mu = M.mean(1)
    ctx_host = F.mean(0)
    print()
    print("=" * 78)
    print(f"FRAGILITY  frag(t) = fraction of {C} contexts with slot lp < {FAIL}")
    print("=" * 78)
    for nm, m in (("all random tokens", is_glitch == 0), ("verified glitch (reference)", is_glitch == 1),
                  ("clean-looking random tokens", m_cl)):
        if m.sum() == 0:
            print(f"  {nm:>28}: (none)"); continue
        f = frag[m]
        print(f"  {nm:>28} (n={m.sum():4d}): mean frag {f.mean():.3f}   "
              f"frag>=0.10: {(f >= 0.10).mean():.3f}   frag>=0.25: {(f >= 0.25).mean():.3f}   "
              f"frag==0: {(f == 0).mean():.3f}")
    print(f"\n  context hostility (fraction of tokens failing): min {ctx_host.min():.3f}  "
          f"median {np.median(ctx_host):.3f}  max {ctx_host.max():.3f}")
    byK = {}
    for K in a.lens:
        idx = [c for c, (ctx, _) in enumerate(bank) if len(ctx) == K]
        byK[K] = float(F[m_cl][:, idx].mean())
    print("  clean-looking fail rate by context length: "
          + "  ".join(f"K={K}: {v:.3f}" for K, v in byK.items()))

    dec = [tok.decode([t]) for t in tokens]
    print("\n  most fragile CLEAN-LOOKING tokens (pass the single probe, fail in context):")
    order = [i for i in np.argsort(-frag) if m_cl[i]][:12]
    for i in order:
        worst_c = int(np.argmin(M[i]))
        ctx, slot = bank[worst_c]
        snippet = tok.decode(ctx[max(0, slot - 3):slot]) + " [" + dec[i] + "] " + \
            tok.decode(ctx[slot + 1:slot + 4])
        print(f"    {dec[i]!r:>18}  single {single[i]:+.3f}  frag {frag[i]:.2f}  "
              f"mean lp {mu[i]:+.3f}   worst: lp {M[i, worst_c]:+.2f} in ...{snippet!r}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "model_info": describe(model, tok, a.model), "prompt_style": PROMPT_STYLE,
               "N": N, "C": C, "lens": a.lens, "fail_thr": FAIL,
               "n_clean_look": int(clean_look.sum()),
               "shares": {"logprob": shares_lp, "fail": shares_f, "clean_logprob": shares_cl},
               "frag_summary": {nm: {"mean": float(frag[m].mean()),
                                     "ge10": float((frag[m] >= 0.1).mean()),
                                     "ge25": float((frag[m] >= 0.25).mean())}
                                for nm, m in (("random", is_glitch == 0),
                                              ("glitch", is_glitch == 1),
                                              ("clean_looking", m_cl))},
               "fail_by_K": byK,
               "ctx_hostility": ctx_host.tolist(),
               "top_fragile_clean": [{"tok": dec[i], "id": int(tokens[i]),
                                      "single": float(single[i]), "frag": float(frag[i])}
                                     for i in order]}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}, {a.pt}")


if __name__ == "__main__":
    main()
