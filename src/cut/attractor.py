"""Output-side failure mode: are glitch tokens OVER-PRODUCED in open slots?

Motivated by the ' 恒一' case (o200k_base token 198478, verified: the only
vocabulary entry containing that name, single-token only in its space-prefixed
form). There the reported symptom is not that the model mishandles the token on
input -- it is that the model EMITS it, capturing the given-name slot while the
surname varies.

Every existing glitch detector, mine included, is input-side: feed the token in,
see if the model copes. None asks whether a token is produced more often than it
should be. This measures that.

Design:
  SLOT contexts   -- the next token is an open-class choice (a name, an
                     identifier, a filename, a domain). Nothing grammatical
                     constrains WHICH one.
  CLOSED contexts -- the next token is grammatically or semantically forced.
  BASE contexts   -- broad natural text, used as the model's own unigram
                     estimate so we can ask "more than it should be" at all.

  attractor score A(t) = log p_slot(t) - log p_base(t)

A(t) is compared for behaviourally-glitched tokens vs healthy tokens MATCHED ON
p_base -- without that matching, rare tokens win by construction and the result
is meaningless.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from .modelio import load, next_logits
from .stats import paired_boot

SLOT = [
    "His name was", "Her name was", "The author's name is", "I'd like you to meet",
    "The character was called", "Dr.", "Please welcome", "signed,",
    "The file was saved as", "Chapter 1:", "The project is named",
    "import ", "class ", "def ", "https://www.", "The variable is called",
    "My name is", "The village of", "a man named", "the company, called",
]
CLOSED = [
    "The cat sat on the", "She opened the door and walked", "Two plus two equals",
    "The capital of France is", "He could not believe his", "It was raining, so I took an",
    "The sun rises in the", "I have been waiting for a long", "Once upon a",
    "In order to succeed you must work", "The opposite of hot is", "Thank you very",
    "As a matter of", "on the other", "in spite of the", "for the first",
    "at the end of the", "one of the most", "according to the", "as soon as",
]
BASE = [
    "The", "In 1994, the", "Scientists have discovered that", "\n\n",
    "According to the report,", "def process(data):\n    ", "He said,",
    "The results indicate that", "# Introduction\n\n", "Q: What is it? A:",
    "https://", "1. ", "The company announced", "She wrote that",
    "In this paper we", "The following table shows", "It is important to",
    "After the war,", "The city of", "A new study",
]


@torch.no_grad()
def mean_logp(model, tok, device, contexts):
    """Mean over contexts of log p(token | context), for every token. -> (V,)"""
    acc = None
    for c in contexts:
        ids = tok(c, add_special_tokens=False).input_ids
        if not ids:
            continue
        lp = torch.log_softmax(
            next_logits(model, torch.tensor(ids, device=device).unsqueeze(0)), -1)[0]
        acc = lp.clone() if acc is None else acc + lp
    return (acc / len(contexts)).float().cpu().numpy()


def matched_compare(A, base, labels, n_bins=20, seed=0):
    """Compare A[glitch] vs A[healthy] within bins of equal p_base."""
    rng = np.random.default_rng(seed)
    edges = np.quantile(base, np.linspace(0, 1, n_bins + 1))
    g_all, h_all = [], []
    for i in range(n_bins):
        m = (base >= edges[i]) & (base < edges[i + 1] if i < n_bins - 1 else base <= edges[i + 1])
        g = np.where(m & (labels == 1))[0]
        h = np.where(m & (labels == 0))[0]
        if len(g) == 0 or len(h) == 0:
            continue
        take = rng.choice(h, size=len(g), replace=len(h) < len(g))
        g_all.append(A[g]); h_all.append(A[take])
    return np.concatenate(g_all), np.concatenate(h_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--gt", default="results/behav_gt_1.4b.pt")
    ap.add_argument("--glitch-pct", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out", default="results/attractor.json")
    a = ap.parse_args()

    model, tok, device = load(a.model)
    gt = torch.load(a.gt, weights_only=False)
    lp_copy = gt["copy_logprob"].numpy(); n = len(lp_copy)
    labels = (lp_copy <= np.percentile(lp_copy, a.glitch_pct)).astype(int)

    slot = mean_logp(model, tok, device, SLOT)[:n]
    closed = mean_logp(model, tok, device, CLOSED)[:n]
    base = mean_logp(model, tok, device, BASE)[:n]
    A_slot, A_closed = slot - base, closed - base

    print(f"model {a.model}   glitch = worst {a.glitch_pct}% copy-logprob "
          f"({labels.sum()} tokens)\n")

    for name, A in [("OPEN SLOTS", A_slot), ("CLOSED contexts", A_closed)]:
        g, h = matched_compare(A, base, labels)
        d, lo, hi = paired_boot(g, h)
        print(f"{name:16s} over-production vs frequency-matched healthy:")
        print(f"                 glitch {g.mean():+7.3f}   healthy {h.mean():+7.3f}   "
              f"diff {d:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  (n={len(g)} matched pairs)")

    # absolute intrusion: do glitch tokens actually reach the top of a slot?
    print(f"\ntop-{a.topk} intrusion (share of slot top-k occupied by glitch tokens):")
    share = labels.mean()
    for name, ctxs in [("slot", SLOT), ("closed", CLOSED)]:
        hits = tot = 0
        for c in ctxs:
            ids = tok(c, add_special_tokens=False).input_ids
            if not ids: continue
            with torch.no_grad():
                l = next_logits(model, torch.tensor(ids, device=device).unsqueeze(0))[0]
            top = l.topk(a.topk).indices.cpu().numpy()
            top = top[top < n]
            hits += labels[top].sum(); tot += len(top)
        print(f"  {name:7s} {hits}/{tot} = {hits/tot:.4%}   "
              f"vs {share:.4%} vocabulary share   enrichment {(hits/tot)/share:5.2f}x")

    order = np.argsort(-A_slot)
    print(f"\nstrongest slot attractors overall (highest log p_slot - log p_base):")
    for i in order[:15]:
        print(f"  A={A_slot[i]:+6.2f}  copy_lp={lp_copy[i]:+7.2f}  "
              f"{'GLITCH' if labels[i] else '      '}  {tok.decode([int(i)])!r}")

    gi = [int(i) for i in order if labels[i]][:12]
    print(f"\nstrongest attractors that are ALSO behaviourally glitched:")
    for i in gi:
        print(f"  A={A_slot[i]:+6.2f}  copy_lp={lp_copy[i]:+7.2f}  {tok.decode([i])!r}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "n_glitch": int(labels.sum()),
               "A_slot": A_slot.tolist(), "A_closed": A_closed.tolist(),
               "base": base.tolist(), "labels": labels.tolist()},
              open(a.out, "w"))
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
