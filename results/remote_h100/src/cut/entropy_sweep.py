"""Stage 1 (G1): context-invariant commitment sweep over the full vocabulary.

A token is a *welded prefix* if, regardless of context, the model is near-certain
about what comes next. We measure this as the MAX entropy of p(next | ctx, t)
across a bank of diverse contexts -- max, not mean, so that a token only counts
as welded if its commitment survives every context we throw at it.
"""
from __future__ import annotations
import json, argparse, math
from pathlib import Path
import torch
from tqdm import tqdm
from .modelio import load, next_logits

# Diverse neutral-ish contexts. Spread across register (prose, code, web, list,
# dialogue) so that a low MAX entropy really means context-invariant commitment.
CONTEXT_BANK = [
    "",
    "The",
    "\n\n",
    " In the beginning,",
    "Q: What do you think? A:",
    "def process(",
    "https://www.",
    "\n1. ",
    " He said that",
    "import numpy as np\n",
]


@torch.no_grad()
def sweep(model, tok, device, batch_size: int = 128, limit: int | None = None):
    V = model.get_input_embeddings().weight.shape[0]
    n_tok = min(V, limit) if limit else V
    ids = torch.arange(n_tok, device=device)

    ent = torch.zeros(len(CONTEXT_BANK), n_tok)
    top1 = torch.zeros(len(CONTEXT_BANK), n_tok, dtype=torch.long)
    top2 = torch.zeros(len(CONTEXT_BANK), n_tok, dtype=torch.long)

    for ci, ctx in enumerate(CONTEXT_BANK):
        ctx_ids = tok(ctx, return_tensors="pt").input_ids[0].to(device) if ctx else \
                  torch.empty(0, dtype=torch.long, device=device)
        for s in tqdm(range(0, n_tok, batch_size), desc=f"ctx[{ci}]", leave=False):
            chunk = ids[s:s + batch_size]
            B = chunk.shape[0]
            inp = torch.cat(
                [ctx_ids.unsqueeze(0).expand(B, -1), chunk.unsqueeze(1)], dim=1
            ) if ctx_ids.numel() else chunk.unsqueeze(1)
            logits = next_logits(model, inp)
            logp = torch.log_softmax(logits, dim=-1)
            p = logp.exp()
            ent[ci, s:s + B] = (-(p * logp).sum(-1)).cpu()
            t2 = logits.topk(2, dim=-1).indices
            top1[ci, s:s + B] = t2[:, 0].cpu()
            top2[ci, s:s + B] = t2[:, 1].cpu()

    max_ent = ent.max(dim=0).values
    mean_ent = ent.mean(dim=0)
    # agreement: do all contexts predict the same next token?
    agree = (top1 == top1[0:1]).all(dim=0)
    return {
        "max_ent": max_ent, "mean_ent": mean_ent, "agree": agree,
        "top1": top1, "top2": top2, "n_tok": n_tok,
    }


@torch.no_grad()
def constituent_health(model):
    """Magikarp-style indicators, used later to EXCLUDE tokens that prior
    single-token detectors would already flag."""
    W = model.get_output_embeddings().weight.detach().float()
    norms = W.norm(dim=-1).cpu()
    mu = W.mean(dim=0, keepdim=True)
    cos = torch.nn.functional.cosine_similarity(W, mu, dim=-1).cpu()
    return {"unemb_norm": norms, "unemb_cos_mean": cos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/sweep.pt")
    a = ap.parse_args()

    model, tok, device = load(a.model)
    res = sweep(model, tok, device, a.batch_size, a.limit)
    res.update(constituent_health(model))
    res["model"] = a.model
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, a.out)

    me = res["max_ent"]; ag = res["agree"]
    welded = ((me < 1.0) & ag).sum().item()
    print(f"\nvocab swept      : {res['n_tok']}")
    print(f"max-ent  median  : {me.median():.3f} nats")
    print(f"max-ent  p1      : {me.kthvalue(max(1,int(0.01*len(me)))).values:.3f} nats")
    print(f"welded (H<1,agree): {welded}  ({100*welded/res['n_tok']:.2f}%)")
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
