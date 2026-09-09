"""Does a model's ability to REPRODUCE a token sequence collapse as the sequence
grows, and is the collapse predictable?

Three earlier hypotheses in this project are now falsified and shape this one:

  - Long-generation entropy amplification: no. The glitch/healthy entropy gap
    wobbles around zero and never grows (reasoning_drift.py).
  - A shared attractor basin for glitch-seeded chains: no. Chain alignment
    decays to 1e-4, identical to healthy (reasoning_drift.py).
  - BPE-unreachable adjacency as a predictor of damage: no. Unreachable pairs
    were EASIER to copy than the same two tokens in the reachable order, paired
    difference +0.94 [+0.61, +1.26] (ngram_glitch.py).

What survived is a single robust observation: the failure mode is CONFIDENT
MISIDENTIFICATION. Asked to identify a glitch token, the model names a different
token and reasons fluently about that one instead -- '.XtraLayout' becomes
'layui', ' Hexatrigesimal' becomes 'Calculus' -- at LOW entropy, and it never
stops doing the task. Derailment was 1.000 in both arms of the pair experiment.
That is why entropy-based detectors miss it and why it is an agentic hazard:
the work completes, on substituted content, with no uncertainty signal.

So the target metric here is reproduction, not likelihood, and the question is
compositional:

  Q1  How does exact-reproduction accuracy fall with n, for sequences of
      INDIVIDUALLY HEALTHY tokens? If a 6-token sequence of healthy tokens is
      misreproduced at a high rate, glitch-like behaviour is compositional and
      no per-token audit can catch it.

  Q2  Is failure predictable from properties computable WITHOUT running the
      model -- sequence length, how many adjacencies are BPE-unreachable,
      whether the decoded string re-encodes to a different id sequence, and the
      weakest individual token in the sequence?

  Q3  When it fails, is it confident? Confident-wrong is the dangerous cell;
      hedged-wrong is recoverable by a guardrail.

Reproduction is scored on the DECODED STRING, not on ids. A model that emits a
different id sequence decoding to the same text has not failed at the task
anyone cares about, and scoring ids would count that as an error.
"""
from __future__ import annotations
import argparse, gzip, json, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEMOS = [" apple pie is good", " the quick brown fox", " 42 blue houses"]


def load_verified(path, n):
    ver = np.zeros(n, int); tested = np.zeros(n, bool)
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line); i = int(r["i"])
        if i < n and "magikarp" in r:
            tested[i] = True
            if "verified" in r["magikarp"]:
                ver[i] = 1
    return ver, tested


def unreach_count(tok, seq):
    """How many adjacent positions in seq are BPE-unreachable?"""
    c = 0
    for i in range(len(seq) - 1):
        pair = [seq[i], seq[i + 1]]
        try:
            if tok(tok.decode(pair), add_special_tokens=False)["input_ids"] != pair:
                c += 1
        except Exception:
            c += 1
    return c


def resegments(tok, seq):
    try:
        return tok(tok.decode(seq), add_special_tokens=False)["input_ids"] != list(seq)
    except Exception:
        return True


@torch.no_grad()
def reproduce(model, tok, dev, seqs, batch=12, slack=6):
    """Greedy-generate a copy of each sequence; return (exact, entropy, output).

    Left padding is required for batched generation -- with right padding the
    final position is a pad and the model continues from it.
    """
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    head = e("Repeat the text exactly.\n")
    for d in DEMOS:
        head += e("Text:") + e(d) + e("\nCopy:") + e(d) + e("\n")
    pre_t, pre_c = e("Text:"), e("\nCopy:")
    pad = tok.pad_token_id or tok.eos_token_id or 0
    exact, ents, outs = [], [], []
    for s in range(0, len(seqs), batch):
        ch = seqs[s:s + batch]
        pr = [head + pre_t + list(q) + pre_c for q in ch]
        mx = max(len(p) for p in pr)
        ids = torch.tensor([[pad] * (mx - len(p)) + p for p in pr], device=dev)
        att = torch.tensor([[0] * (mx - len(p)) + [1] * len(p) for p in pr], device=dev)
        mn = max(len(q) for q in ch) + slack
        g = model.generate(ids, attention_mask=att, max_new_tokens=mn,
                           do_sample=False, pad_token_id=pad)
        o = model(input_ids=g, attention_mask=torch.cat(
            [att, torch.ones(len(ch), g.shape[1] - mx, dtype=att.dtype, device=dev)], 1))
        lp = torch.log_softmax(o.logits[:, mx - 1:-1].float(), -1)
        H = -(lp.exp() * lp).sum(-1)
        for k, q in enumerate(ch):
            # skip_special_tokens matters: without it EOS decodes into the string
            # as literal '<|endoftext|>' and a perfect copy scores as a failure.
            gen = tok.decode(g[k, mx:].tolist(), skip_special_tokens=True)
            want = tok.decode(list(q), skip_special_tokens=True)
            # the copy is finished at the first newline the model emits
            got = gen.split("\n")[0]
            exact.append(got.startswith(want) or want.strip() == got.strip())
            ents.append(float(H[k, :max(1, len(q))].mean()))
            outs.append(got)
    return np.array(exact), np.array(ents), outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--ext", default="external/allenai_OLMo_2_1124_7B.jsonl.gz")
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--per-n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-mode", choices=["tested", "common"], default="tested",
                    help="'tested' = Magikarp-tested rare tail; 'common' = ordinary "
                         "high-frequency English words, the control for whether the "
                         "length effect is about rarity rather than composition")
    ap.add_argument("--out", default="results/ngram_identify.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    V = model.get_input_embeddings().weight.shape[0]
    ver, tested = load_verified(a.ext, V)
    rng = random.Random(a.seed)

    if a.pool_mode == "tested":
        # Magikarp only TESTED the rare tail, so this pool skews hard to code
        # identifiers. Failure here could just mean "runs of rare identifiers
        # are hard to copy", which is not a compositional claim at all.
        pool = [int(i) for i in np.where((ver == 0) & tested)[0]
                if tok.decode([int(i)]).strip() and tok.decode([int(i)]).isprintable()]
    else:
        # ordinary high-frequency English: low ids are the earliest merges, which
        # in a BPE vocabulary are the most frequent pieces.
        pool = [i for i in range(2, 12000)
                if tok.decode([i]).strip().isalpha() and tok.decode([i]).isascii()
                and len(tok.decode([i]).strip()) > 1]
        rng.shuffle(pool); pool = pool[:2400]
    glitch = [int(i) for i in np.where(ver == 1)[0]]
    print(f"{a.model}: healthy pool {len(pool)}, verified glitch {len(glitch)}, {dev}")

    # single-token reproduction, used as the per-token health baseline
    base_ex, _, _ = reproduce(model, tok, dev, [[t] for t in pool])
    health = dict(zip(pool, base_ex.astype(float)))
    print(f"single healthy tokens reproduced exactly: {base_ex.mean():.3f}")
    gl_ex, gl_H, _ = reproduce(model, tok, dev, [[t] for t in glitch])
    print(f"single verified-glitch tokens reproduced : {gl_ex.mean():.3f}")
    # restrict the pool to tokens the model demonstrably CAN reproduce alone, so
    # any failure at n>1 is compositional and not inherited from a member
    clean = [t for t in pool if health[t] == 1.0]
    print(f"pool restricted to individually-reproducible tokens: {len(clean)}")

    print()
    print("=" * 78)
    print("Q1  REPRODUCTION vs SEQUENCE LENGTH (all members individually clean)")
    print("=" * 78)
    print(f"{'n':>3} | {'exact':>7} | {'conf-wrong':>10} | {'entropy':>8} | "
          f"{'mean unreach adj':>16}")
    print("-" * 60)
    rows, store = [], {}
    for n in a.ns:
        seqs = [[rng.choice(clean) for _ in range(n)] for _ in range(a.per_n)]
        ex, H, outs = reproduce(model, tok, dev, seqs)
        ur = np.array([unreach_count(tok, s) for s in seqs])
        rs = np.array([resegments(tok, s) for s in seqs])
        # confident-wrong: failed the copy AND was not uncertain about it
        med = np.median(H)
        cw = float(((~ex) & (H <= med)).mean())
        r = {"n": n, "exact": float(ex.mean()), "conf_wrong": cw,
             "entropy": float(H.mean()), "unreach": float(ur.mean()),
             "resegment": float(rs.mean())}
        rows.append(r); store[n] = (seqs, ex, H, outs, ur, rs)
        print(f"{n:3d} | {r['exact']:7.3f} | {cw:10.3f} | {r['entropy']:8.3f} | "
              f"{r['unreach']:16.2f}")

    print()
    print("=" * 78)
    print("Q2  IS FAILURE PREDICTABLE WITHOUT RUNNING THE MODEL?")
    print("AUC for predicting reproduction FAILURE, within each n")
    print("=" * 78)
    from src.cut.analyze_trajectory import auc
    print(f"{'n':>3} | {'n_fail':>6} | {'unreach adj':>11} | {'re-segments':>11} | "
          f"{'seq char len':>12}")
    print("-" * 56)
    pred = []
    for n in a.ns:
        seqs, ex, H, outs, ur, rs = store[n]
        y = (~ex).astype(int)
        if y.sum() < 5 or y.sum() == len(y):
            print(f"{n:3d} | {y.sum():6d} | (degenerate)"); continue
        cl = np.array([len(tok.decode(s)) for s in seqs])
        p = {"n": n, "n_fail": int(y.sum()), "unreach": auc(ur, y),
             "resegment": auc(rs.astype(float), y), "charlen": auc(cl, y)}
        pred.append(p)
        print(f"{n:3d} | {p['n_fail']:6d} | {p['unreach']:11.3f} | "
              f"{p['resegment']:11.3f} | {p['charlen']:12.3f}")

    print()
    print("=" * 78)
    print("Q3  WHAT DOES FAILURE LOOK LIKE? (n=%d)" % a.ns[-1])
    print("=" * 78)
    seqs, ex, H, outs, ur, rs = store[a.ns[-1]]
    bad = [i for i in range(len(seqs)) if not ex[i]][:10]
    for i in bad:
        print(f"  want {tok.decode(seqs[i])[:60]!r}")
        print(f"  got  {outs[i][:60]!r}   H={H[i]:.2f} unreach={ur[i]}")
    ok = [i for i in range(len(seqs)) if ex[i]][:3]
    for i in ok:
        print(f"  OK   {tok.decode(seqs[i])[:60]!r}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "single_healthy": float(base_ex.mean()),
               "single_glitch": float(gl_ex.mean()), "rows": rows, "pred": pred,
               "failures": [{"want": tok.decode(seqs[i]), "got": outs[i],
                             "entropy": float(H[i]), "unreach": int(ur[i])}
                            for i in bad]}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
