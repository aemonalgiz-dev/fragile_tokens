"""Behavioral ground truth: copy-probe every token in the vocabulary.

Day 1 validated this probe -- it separates known-glitch from healthy tokens by
3.18 nats and 90 exact-match points. So "glitch" here is a measured behavioural
property of the final model, not an embedding heuristic. That distinction is the
whole point: the trajectory study then asks whether embedding geometry at step t
predicts BEHAVIOUR at step 143000.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from .modelio import load
from .repetition import build_prefix, score_batch

FRAME = " she said that the value of"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/behav_gt_1.4b.pt")
    a = ap.parse_args()

    model, tok, device = load(a.model)
    f = tok(FRAME, add_special_tokens=False).input_ids
    V = model.get_input_embeddings().weight.shape[0]
    n = min(V, a.limit) if a.limit else V

    seqs = [[f[0], f[1], t, f[2], f[3]] for t in range(n)]
    out = []
    for s in tqdm(range(0, n, a.batch_size), desc="copy-probe"):
        out.append(score_batch(model, tok, device, seqs[s:s + a.batch_size],
                               batch_size=a.batch_size))
    lp = torch.cat(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": a.model, "frame": FRAME, "copy_logprob": lp, "n": n}, a.out)

    q = torch.tensor([.001, .01, .05, .25, .5, .75])
    print(f"\ncopy-logprob over {n} tokens")
    for qq in q:
        print(f"  q{float(qq):<6}: {lp.kthvalue(max(1,int(qq*n))).values:+.3f}")
    print(f"  worst 1% mean : {lp.sort().values[:n//100].mean():+.3f}")
    print(f"  median        : {lp.median():+.3f}")
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
