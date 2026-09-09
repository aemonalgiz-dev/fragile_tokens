"""Are there token GROUPINGS that behave like glitch tokens, and is it
predictable which ones?

Every published detector, and everything else in this repo, scores one token at
a time. The phenomenon this targets is different: a sequence of individually
healthy tokens that jointly produces glitch-like behaviour. Single-token probes
on short completions are exactly the regime where it is invisible.

THE PREDICTOR. BPE makes some adjacencies unreachable. If

    encode(decode(a) + decode(b)) != [a, b]

then no string encodes to a followed immediately by b -- a merge rule would have
combined them or split them differently. That adjacency therefore occurs in
training data essentially never (only across a document boundary or a
pre-tokenizer split), while remaining perfectly feedable as ids. It is the
compositional analogue of an unreachable token, and unlike every embedding-based
indicator it is computable from the TOKENIZER ALONE -- no model, no checkpoints,
no forward pass. If it predicts damage, prompt assembly can be checked statically.

This matters because programmatic prompt assembly manufactures such adjacencies
routinely: concatenating strings, splicing retrieved passages, interpolating
tool output. A human typing a sentence almost never does.

THE CONTROL. Sampling unreachable and reachable pairs separately risks
confounding adjacency with token rarity. So the primary comparison is the ORDER
SWAP: pairs where (a,b) is unreachable but (b,a) is reachable. Identical tokens,
identical individual health, identical individual copy scores -- only the
adjacency differs. Anything that survives that control is compositional.

THE MEASURES. Sequence-level analogues of the single-token criteria, plus one
aimed at the agentic failure mode:

  copy      per-token logprob of reproducing the whole sequence. The established
            criterion, generalised from a token to an n-gram.
  interact  copy(sequence) minus the mean of copy(each token alone). This is the
            number that matters: it is the damage NOT attributable to the parts.
            Negative = the grouping is worse than its members.
  derail    a trivially answerable question is placed after the sequence. Does
            the model still answer it? This is the agentic failure: not "the
            model says something odd about the token" but "the model stops doing
            the task".
  entropy   mean next-token entropy over a continuation.
  repeat    degenerate-repetition rate in a long continuation.

REGIME. Everything is measured twice, short and long, because the claim under
test is that the effect is masked at short generation length and compounds when
the model conditions on its own output.
"""
from __future__ import annotations
import argparse, gzip, json, random, unicodedata
from .copyprompt import copy_prompt_parts
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FEWSHOT = [" apple pie is good", " the quick brown fox"]
# a question with a short, unambiguous, easily-detected answer
DERAIL_Q = "\nQuestion: What is 2+2? Answer with a single number.\nAnswer:"
DERAIL_OK = ("4", "four", "Four")


def load_verified(path, n):
    ver = np.zeros(n, int); tested = np.zeros(n, bool)
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line); i = int(r["i"])
        if i < n and "magikarp" in r:
            tested[i] = True
            if "verified" in r["magikarp"]:
                ver[i] = 1
    return ver, tested


def reachable(tok, seq):
    """Does any string encode to exactly this id sequence?"""
    try:
        s = tok.decode(seq)
        return tok(s, add_special_tokens=False)["input_ids"] == list(seq)
    except Exception:
        return False


@torch.no_grad()
def copy_lp(model, tok, dev, seqs, batch=16):
    """Per-token logprob of reproducing each id sequence after a copy prompt.

    Right padding is safe here: scoring is teacher-forced and the target
    positions are read by index, so trailing pads are never attended to by any
    position that matters under a causal mask.
    """
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    head = e("Repeat the text exactly.\n")
    for a in FEWSHOT:
        head += e("Text:") + e(a) + e("\nCopy:") + e(a) + e("\n")
    pre_t, pre_c = e("Text:"), e("\nCopy:")
    head, pre_t, pre_c = copy_prompt_parts(tok, head, FEWSHOT)   # raw or chat framing (GLITCH_PROMPT_STYLE)
    pad = tok.pad_token_id or tok.eos_token_id or 0
    out = []
    for s in range(0, len(seqs), batch):
        ch = seqs[s:s + batch]
        built = [head + pre_t + list(q) + pre_c + list(q) for q in ch]
        mx = max(len(b) for b in built)
        ids = torch.tensor([b + [pad] * (mx - len(b)) for b in built], device=dev)
        logits = model(input_ids=ids).logits[:, :-1, :].float()
        lp = torch.log_softmax(logits, -1)
        for k, q in enumerate(ch):
            n = len(built[k])
            # the copied span is the last len(q) tokens of built[k]
            pos = range(n - len(q) - 1, n - 1)
            v = [lp[k, p, built[k][p + 1]].item() for p in pos]
            out.append(float(np.mean(v)))
    return np.array(out)


@torch.no_grad()
def generate(model, tok, dev, prompts, max_new, batch=16):
    """Left-padded batched greedy generation. Left padding is REQUIRED here:
    with right padding the last position is a pad and generation continues from
    garbage."""
    pad = tok.pad_token_id or tok.eos_token_id or 0
    outs = []
    for s in range(0, len(prompts), batch):
        ch = prompts[s:s + batch]
        mx = max(len(p) for p in ch)
        ids = torch.tensor([[pad] * (mx - len(p)) + list(p) for p in ch], device=dev)
        att = torch.tensor([[0] * (mx - len(p)) + [1] * len(p) for p in ch], device=dev)
        g = model.generate(ids, attention_mask=att, max_new_tokens=max_new,
                           min_new_tokens=max_new, do_sample=False, pad_token_id=pad,
                           suppress_tokens=[tok.eos_token_id]
                           if tok.eos_token_id is not None else None)
        outs += [g[k, mx:].tolist() for k in range(len(ch))]
    return outs


@torch.no_grad()
def mean_entropy(model, tok, dev, prompts, gens, batch=8):
    pad = tok.pad_token_id or tok.eos_token_id or 0
    out = []
    for s in range(0, len(prompts), batch):
        chp, chg = prompts[s:s + batch], gens[s:s + batch]
        full = [list(p) + list(g) for p, g in zip(chp, chg)]
        mx = max(len(f) for f in full)
        ids = torch.tensor([[pad] * (mx - len(f)) + f for f in full], device=dev)
        lg = model(input_ids=ids).logits.float()
        lp = torch.log_softmax(lg, -1)
        H = -(lp.exp() * lp).sum(-1)
        for k, g in enumerate(chg):
            out.append(float(H[k, -len(g) - 1:-1].mean()))
    return np.array(out)


def rep_rate(g):
    return 1.0 - len(set(g)) / max(len(g), 1)


def n_scripts(s):
    out = set()
    for c in s:
        if c.isalpha():
            try:
                out.add(unicodedata.name(c).split()[0])
            except ValueError:
                out.add("UNNAMED")
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--ext", default="external/allenai_OLMo_2_1124_7B.jsonl.gz")
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--long", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ngram_glitch.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    V = model.get_input_embeddings().weight.shape[0]
    ver, tested = load_verified(a.ext, V)
    rng = random.Random(a.seed)

    # healthy pool: tested by Magikarp and NOT flagged, printable, non-whitespace
    pool = [int(i) for i in np.where((ver == 0) & tested)[0]
            if tok.decode([int(i)]).strip() and tok.decode([int(i)]).isprintable()]
    rng.shuffle(pool)
    print(f"{a.model}: healthy pool {len(pool)}, {dev}")

    # ---------- build the order-swap set ----------
    swap = []                      # (a, b) unreachable, (b, a) reachable
    both_ok, both_bad = [], []
    seen = set()
    while len(swap) < a.n_pairs and len(seen) < 400000:
        x, y = rng.choice(pool), rng.choice(pool)
        if x == y or (x, y) in seen:
            continue
        seen.add((x, y))
        f, r = reachable(tok, [x, y]), reachable(tok, [y, x])
        if not f and r:
            swap.append((x, y))
        elif f and r and len(both_ok) < a.n_pairs:
            both_ok.append((x, y))
        elif not f and not r and len(both_bad) < a.n_pairs:
            both_bad.append((x, y))
    print(f"sampled {len(seen)} random healthy pairs ->")
    print(f"  order-swap usable (one direction unreachable): {len(swap)}")
    print(f"  both directions reachable                    : {len(both_ok)}")
    print(f"  both directions unreachable                  : {len(both_bad)}")
    if len(swap) < 20:
        print("too few swap pairs; aborting"); return

    # ---------- single-token baselines ----------
    toks = sorted({t for p in swap for t in p})
    single = dict(zip(toks, copy_lp(model, tok, dev, [[t] for t in toks])))

    unreach = [list(p) for p in swap]
    reach = [[p[1], p[0]] for p in swap]

    print()
    print("=" * 78)
    print("ORDER-SWAP TEST -- identical token pairs, only the adjacency differs")
    print("=" * 78)
    res = {}
    for nm, seqs in [("UNREACHABLE (a,b)", unreach), ("REACHABLE (b,a)", reach)]:
        c = copy_lp(model, tok, dev, seqs)
        parts = np.array([np.mean([single[t] for t in s]) for s in seqs])
        res[nm] = {"copy": c, "interact": c - parts}
    ua, ra = res["UNREACHABLE (a,b)"], res["REACHABLE (b,a)"]
    print(f"{'condition':>20} | {'copy lp/tok':>12} | {'interaction':>12}")
    print("-" * 52)
    for nm in ("UNREACHABLE (a,b)", "REACHABLE (b,a)"):
        print(f"{nm:>20} | {res[nm]['copy'].mean():12.4f} | "
              f"{res[nm]['interact'].mean():12.4f}")
    d = ua["copy"] - ra["copy"]
    boot = np.array([np.mean(rng.choices(list(d), k=len(d))) for _ in range(2000)])
    print(f"\npaired difference (unreachable - reachable): {d.mean():+.4f}")
    print(f"  bootstrap 95% CI [{np.percentile(boot,2.5):+.4f}, "
          f"{np.percentile(boot,97.5):+.4f}]")
    print(f"  pairs where unreachable is worse: {(d<0).mean():.3f}")

    # ---------- regime contrast + derailment ----------
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    pre = e("Consider the following text: ")
    qids = e(DERAIL_Q)
    print()
    print("=" * 78)
    print("DERAILMENT AND REGIME  (short = 8 new tokens, long = %d)" % a.long)
    print("=" * 78)
    print(f"{'condition':>20} | {'derail ok':>9} | {'entropy':>8} | "
          f"{'repeat':>7} | {'scripts':>7}")
    print("-" * 62)
    summ = {}
    for nm, seqs in [("UNREACHABLE (a,b)", unreach), ("REACHABLE (b,a)", reach)]:
        dp = [pre + s + qids for s in seqs]
        dg = generate(model, tok, dev, dp, 8)
        ok = np.array([any(w in tok.decode(g) for w in DERAIL_OK) for g in dg])
        lp_ = [pre + s for s in seqs]
        lg = generate(model, tok, dev, lp_, a.long)
        H = mean_entropy(model, tok, dev, lp_, lg)
        rr = np.array([rep_rate(g) for g in lg])
        sc = np.array([n_scripts(tok.decode(g)) for g in lg])
        summ[nm] = {"derail": float(ok.mean()), "entropy": float(H.mean()),
                    "repeat": float(rr.mean()), "scripts": float(sc.mean()),
                    "H": H, "rr": rr, "ok": ok, "gen": lg}
        print(f"{nm:>20} | {ok.mean():9.3f} | {H.mean():8.3f} | "
              f"{rr.mean():7.3f} | {sc.mean():7.2f}")

    u, r = summ["UNREACHABLE (a,b)"], summ["REACHABLE (b,a)"]
    dH = u["H"] - r["H"]
    bH = np.array([np.mean(rng.choices(list(dH), k=len(dH))) for _ in range(2000)])
    print(f"\npaired entropy difference: {dH.mean():+.4f}  "
          f"95% CI [{np.percentile(bH,2.5):+.4f}, {np.percentile(bH,97.5):+.4f}]")
    print(f"task-answer rate: unreachable {u['derail']:.3f} vs "
          f"reachable {r['derail']:.3f}")

    # ---------- worst offenders ----------
    print()
    print("=" * 78)
    print("MOST DAMAGED UNREACHABLE PAIRS (by paired copy difference)")
    print("=" * 78)
    order = np.argsort(d)[:12]
    worst = []
    for i in order:
        x, y = swap[i]
        print(f"  {tok.decode([x])!r:>16} + {tok.decode([y])!r:<16} "
              f"d={d[i]:+.3f}  ids=({x},{y})")
        print(f"      -> {tok.decode(u['gen'][i])[:110]!r}")
        worst.append({"a": tok.decode([x]), "b": tok.decode([y]),
                      "ids": [x, y], "delta": float(d[i]),
                      "gen": tok.decode(u["gen"][i])[:300]})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "n_swap": len(swap),
               "copy": {k: float(v["copy"].mean()) for k, v in res.items()},
               "interact": {k: float(v["interact"].mean()) for k, v in res.items()},
               "paired_copy_delta": float(d.mean()),
               "paired_copy_ci": [float(np.percentile(boot, 2.5)),
                                  float(np.percentile(boot, 97.5))],
               "frac_unreachable_worse": float((d < 0).mean()),
               "behaviour": {k: {kk: vv for kk, vv in v.items()
                                 if kk in ("derail", "entropy", "repeat", "scripts")}
                             for k, v in summ.items()},
               "paired_entropy_delta": float(dH.mean()),
               "worst": worst}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
