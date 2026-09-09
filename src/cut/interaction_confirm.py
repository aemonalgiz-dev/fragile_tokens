"""Do the pairs that survive the copy-logprob screen misbehave in REASONING mode?

The screen (interaction_screen.py) selects pairs with a cheap continuous score.
A pair selected that way must not be validated on the same data, and the user's
actual interest is not copy-logprob but what a reasoning model does when asked
to identify and discuss the text. So this stage uses

  - a DIFFERENT measure: exact reproduction of the context inside a chat-format
    "repeat this text exactly, then explain it" response, and substitution
    (the response reproduces the carrier but swaps a slot token);
  - a DIFFERENT filler sequence (F3), unused by the screen;
  - two distances per pair: the distance at which its screen effect was largest,
    and d=8 as a common reference;
  - MATCHED CONTROLS: each surviving pair (a,b) is compared with (a,b') and
    (a',b), where the new partner has |I| in the bottom quartile of its row or
    column. Token identity and slot are held fixed; only the pairing differs.

If survivors fail no more often than their matched controls, the screen's
interaction is real in copy-logprob but is not glitch behaviour in the sense
that motivated the experiment.

--attention (repetition diagonal only): reload with eager attention and measure
attention mass from the second occurrence of a token to its first in the Text
span. This is the induction / duplicate-token signature -- the mechanistic
candidate for why REPEATING a token at distance would matter.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .stats import paired_boot
from .analyze_trajectory import auc
from .copyprompt import copy_prompt_parts


def build(tok, ctx_ids):
    """Chat-format identification prompt, context ids spliced in directly."""
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    user = ("Here is a short text. First repeat the text exactly, word for word. "
            "Then explain step by step what it means.\n\nText:")
    try:
        s = tok.apply_chat_template([{"role": "user", "content": "\x00"}],
                                    tokenize=False, add_generation_prompt=True)
        head, tail = s.split("\x00")
        return e(head + user) + list(ctx_ids) + e(tail + "The text is:")
    except Exception:
        return e(user) + list(ctx_ids) + e("\nAnswer: The text is:")


@torch.no_grad()
def generate(model, tok, dev, prompts, max_new, batch=8):
    """Left-padded greedy generation. EOS is NOT suppressed: we want the model's
    natural response, and reproduction is checked on its prefix."""
    pad = tok.pad_token_id or tok.eos_token_id or 0
    outs = []
    for s in range(0, len(prompts), batch):
        ch = prompts[s:s + batch]
        mx = max(len(p) for p in ch)
        ids = torch.tensor([[pad] * (mx - len(p)) + list(p) for p in ch], device=dev)
        att = torch.tensor([[0] * (mx - len(p)) + [1] * len(p) for p in ch], device=dev)
        g = model.generate(ids, attention_mask=att, max_new_tokens=max_new,
                           do_sample=False, pad_token_id=pad)
        outs += [tok.decode(g[k, mx:].tolist(), skip_special_tokens=True)
                 for k in range(len(ch))]
    return outs


def _norm(s):
    """Lowercase, drop everything but letters/digits/spaces, collapse spaces.
    Reasoning-mode responses quote the text back lowercased and with punctuation
    dropped ('.Ver' -> 'ver'); a strict match calls that a failure when it is
    normalisation, not substitution."""
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in s).split())


def score(tok, ctx, slots, out):
    """(reproduced_strict, reproduced_normalised, substituted).
    Strict: verbatim. Normalised: case/punctuation-insensitive -- the PRIMARY
    failure criterion. Substituted: carrier words present, a slot token missing
    under the normalised comparison."""
    want = tok.decode(ctx, skip_special_tokens=True).strip()
    got = out.strip()
    strict = want in got
    nw, ng = _norm(want), _norm(got)
    repro = strict or (nw != "" and nw in ng)
    i, j = slots
    fill = [_norm(tok.decode([t])) for k, t in enumerate(ctx) if k not in (i, j)]
    fill = [f for f in fill if f]
    fill_ok = sum(f in ng for f in fill) >= max(1, int(0.75 * len(fill)))
    a_in = _norm(tok.decode([ctx[i]])) in ng
    b_in = _norm(tok.decode([ctx[j]])) in ng
    return strict, repro, (not repro) and fill_ok and not (a_in and b_in)


def matched_controls(I, ai, bi, rng, k=1):
    """Partners with |I| in the bottom quartile of row ai / column bi."""
    N = I.shape[0]; absI = np.abs(I)
    row = [b for b in range(N) if b not in (ai, bi) and absI[ai, b] <= np.percentile(absI[ai], 25)]
    col = [a for a in range(N) if a not in (ai, bi) and absI[a, bi] <= np.percentile(absI[:, bi], 25)]
    rng.shuffle(row); rng.shuffle(col)
    return [(ai, b) for b in row[:k]] + [(a, bi) for a in col[:k]]


@torch.no_grad()
def induction_attention(model, tok, dev, ctxs, slot):
    """Mean over heads, per layer, of attention from slot j to slot i in the
    Text span of the copy prompt. Requires eager attention."""
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    head = e("Repeat the text exactly.\n")
    for d in [" apple pie is good", " the quick brown fox"]:
        head += e("Text:") + e(d) + e("\nCopy:") + e(d) + e("\n")
    pre_t, pre_c = e("Text:"), e("\nCopy:")
    head, pre_t, pre_c = copy_prompt_parts(tok, head, FEWSHOT)   # raw or chat framing (GLITCH_PROMPT_STYLE)
    ts = len(head) + len(pre_t); i, j = slot
    out = []
    for c in ctxs:
        ids = torch.tensor(head + pre_t + list(c) + pre_c + list(c), device=dev).unsqueeze(0)
        o = model(input_ids=ids, output_attentions=True)
        out.append([float(A[0, :, ts + j, ts + i].mean()) for A in o.attentions])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--screen", default="results/interaction_screen.json")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--ref-d", type=int, default=8)
    ap.add_argument("--attention", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/interaction_confirm.json")
    a = ap.parse_args()

    sc = json.load(open(a.screen))
    pool = sc["pool"]; N = sc["N"]; F3 = sc["filler3"]
    I = np.array(sc["I"]); rej = np.array(sc["rej"], bool)
    I_by_d = {int(k): np.array(v) for k, v in sc["I_by_dist"].items()}
    rng = random.Random(a.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32,
        **({"attn_implementation": "eager"} if a.attention else {})).to(dev).eval()

    cand = sorted([(ai, bi) for ai in range(N) for bi in range(N) if rej[ai, bi]],
                  key=lambda ab: I[ab])
    surv = cand[:a.top_k]
    exploratory = len(surv) < 5
    if exploratory:
        print(f"only {len(surv)} FDR-significant pairs; running top-{a.top_k} by I as "
              f"EXPLORATORY (cannot confirm anything)")
        surv = [divmod(int(k), N) for k in np.argsort(I.ravel())[:a.top_k]]
    print(f"{len(surv)} pairs, {sum(1 for x, y in surv if x == y)} on the repetition "
          f"diagonal, {'exploratory' if exploratory else 'FDR-significant'}")

    def ctx_at(d, ta, tb):
        c = list(F3[:d + 4]); c[2] = ta; c[2 + d] = tb
        return c, (2, 2 + d)

    items = []                                   # (group, pair_idx, ctx, slots)
    for k, (ai, bi) in enumerate(surv):
        best_d = min(I_by_d, key=lambda d: I_by_d[d][ai, bi])
        ds = sorted({best_d, a.ref_d})
        ctrls = matched_controls(I, ai, bi, rng)
        for d in ds:
            c, sl = ctx_at(d, pool[ai], pool[bi]); items.append(("surv", k, c, sl))
            for (x, y) in ctrls:
                c, sl = ctx_at(d, pool[x], pool[y]); items.append(("ctrl", k, c, sl))
    print(f"{len(items)} reasoning prompts...", flush=True)
    # group by length so generation batches are not dominated by padding
    order = sorted(range(len(items)), key=lambda k: len(items[k][2]))
    outs = [None] * len(items)
    for s in range(0, len(order), 8):
        idx = order[s:s + 8]
        mx = max(len(items[k][2]) for k in idx) + 16
        res = generate(model, tok, dev, [build(tok, items[k][2]) for k in idx], mx)
        for k, o in zip(idx, res):
            outs[k] = o

    fail = {"surv": {}, "ctrl": {}}; strictf = {"surv": {}, "ctrl": {}}
    subst = {"surv": {}, "ctrl": {}}; samples = []; allout = []
    for (grp, k, c, sl), o in zip(items, outs):
        st_, rp, sb = score(tok, c, sl, o)
        fail[grp].setdefault(k, []).append(0.0 if rp else 1.0)
        strictf[grp].setdefault(k, []).append(0.0 if st_ else 1.0)
        subst[grp].setdefault(k, []).append(1.0 if sb else 0.0)
        allout.append({"group": grp, "pair": k, "ctx": tok.decode(c), "out": o[:300],
                       "strict": bool(st_), "repro": bool(rp), "subst": bool(sb)})
        if grp == "surv" and not rp and len(samples) < 8:
            samples.append({"ctx": tok.decode(c), "out": o[:200]})
    agg = lambda D, g: np.array([np.mean(D[g][k]) for k in range(len(surv))])
    fs, fc = agg(fail, "surv"), agg(fail, "ctrl")
    fss, fsc = agg(strictf, "surv"), agg(strictf, "ctrl")
    ss, scn = agg(subst, "surv"), agg(subst, "ctrl")

    print()
    print("=" * 74)
    print("REASONING-MODE CONFIRMATION  (survivors vs matched-partner controls)")
    print("=" * 74)
    print(f"{'':>26} | {'survivor':>9} | {'matched ctrl':>12}")
    print("-" * 54)
    print(f"{'failure (strict match)':>26} | {fss.mean():9.3f} | {fsc.mean():12.3f}")
    print(f"{'failure (case/punct-insens.)':>26} | {fs.mean():9.3f} | {fc.mean():12.3f}"
          "   <- primary")
    print(f"{'substitution':>26} | {ss.mean():9.3f} | {scn.mean():12.3f}")
    d, lo, hi = paired_boot(fs, fc)
    ds_, los, his = paired_boot(ss, scn)
    print(f"\n  paired difference, failure     : {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"{'   *' if lo > 0 else ''}")
    print(f"  paired difference, substitution: {ds_:+.3f}  95% CI [{los:+.3f}, {his:+.3f}]"
          f"{'   *' if los > 0 else ''}")
    transfers = (lo > 0) and not exploratory
    print(f"\n  PRE-REGISTERED: screen transfers to reasoning-mode behaviour: {transfers}")
    if exploratory:
        print("  (exploratory: nothing survived FDR, so this cannot confirm anything)")
    print("\n  sample survivor failures:")
    for s_ in samples[:6]:
        print(f"    ctx {s_['ctx']!r}\n    -> {s_['out']!r}")

    attn = None
    if a.attention:
        diag = [x for x, y in surv if x == y]
        rest = [x for x in range(N) if not rej[x, x] and x not in diag]
        rng.shuffle(rest)
        if len(diag) >= 3:
            print()
            print("=" * 74)
            print(f"INDUCTION ATTENTION on the repetition diagonal (d={a.ref_d}, 2nd -> 1st)")
            print("=" * 74)
            toks = [(pool[x], 1) for x in diag] + [(pool[x], 0) for x in rest[:len(diag)]]
            ctxs = [ctx_at(a.ref_d, t, t)[0] for t, _ in toks]
            A = induction_attention(model, tok, dev, ctxs, (2, 2 + a.ref_d))
            y = np.array([lab for _, lab in toks])
            best = max(range(A.shape[1]), key=lambda l: abs(auc(A[:, l], y) - 0.5))
            print(f"  mean attention j->i: significant {A[y==1].mean():.4f} vs "
                  f"not {A[y==0].mean():.4f}")
            print(f"  summed over layers AUC {auc(A.sum(1), y):.3f}; best layer {best} "
                  f"AUC {auc(A[:, best], y):.3f}")
            attn = {"n_sig": int(y.sum()), "auc_sum": auc(A.sum(1), y),
                    "best_layer": int(best), "auc_best": auc(A[:, best], y)}
        else:
            print(f"\n({len(diag)} significant repetition pairs; attention check needs 3)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "n_pairs": len(surv), "exploratory": exploratory,
               "pairs": [[pool[x], pool[y], float(I[x, y])] for x, y in surv],
               "fail_surv": float(fs.mean()), "fail_ctrl": float(fc.mean()),
               "fail_strict_surv": float(fss.mean()), "fail_strict_ctrl": float(fsc.mean()),
               "outputs": allout,
               "subst_surv": float(ss.mean()), "subst_ctrl": float(scn.mean()),
               "fail_diff": [d, lo, hi], "subst_diff": [ds_, los, his],
               "transfers": bool(transfers), "samples": samples, "attention": attn},
              open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
