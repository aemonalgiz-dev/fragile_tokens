"""Does post-hoc kNN repair work on a REAL model with REAL glitch tokens?

The synthetic rig said neighbour-based repair removes 1.245 nats of damage at
+0.0012 val loss, across 6 seeds. That was a 23M model on manufactured glitch
tokens. This is the test that matters: pythia-1.4b, glitch tokens verified
behaviourally on two models, and a perplexity guard-rail on real held-out text.

Full pipeline, end to end:
  1. score every token with the late detector (embedding trajectory features)
  2. repair the top-N by interpolating toward their k nearest healthy neighbours
  3. re-run the repetition test on verified-glitch / our-confirmed / control sets
  4. measure perplexity before and after -- a repair that fixes tokens by
     degrading the model is worthless

Repairing the detector's top-N rather than a hand-picked list is deliberate: it
tests the whole pipeline, including the ~50% false positives, which is what
anyone actually deploying this would face.
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .external_gt import load_labels, IND
from .verify_candidates import TEMPLATES


@torch.no_grad()
def knn_repair(model, doomed_ids, k=8, alpha=1.0):
    """Interpolate each doomed row toward its k nearest HEALTHY rows."""
    dev = model.get_input_embeddings().weight.device
    V = model.get_input_embeddings().weight.shape[0]
    mask = torch.zeros(V, dtype=torch.bool, device=dev)
    mask[torch.tensor(doomed_ids, device=dev)] = True
    for W in (model.get_input_embeddings().weight,
              model.get_output_embeddings().weight):
        H = W[~mask].float()
        D = W[mask].float()
        idx = (F.normalize(D, dim=-1) @ F.normalize(H, dim=-1).T).topk(k, dim=-1).indices
        tgt = H[idx].mean(1)
        W[mask] = ((1 - alpha) * D + alpha * tgt).to(W.dtype)
    return int(mask.sum())


@torch.no_grad()
def repetition_rate(model, tok, dev, ids_list):
    """Fraction of tokens the model can copy back (majority of 3 templates)."""
    hits = []
    for t in ids_list:
        ok = 0
        for tname in TEMPLATES:
            demos, tail = TEMPLATES[tname]
            s = ""
            for a, b in demos:
                s += tail.format(t=a) + b + ("\"\n" if tname == "quoted" else "\n")
            head, sep = tail.split("{t}")
            prompt = (tok(s, add_special_tokens=False).input_ids
                      + tok(head, add_special_tokens=False).input_ids + [t]
                      + tok(sep, add_special_tokens=False).input_ids)
            p = torch.tensor(prompt, device=dev).unsqueeze(0)
            out = model.generate(p, max_new_tokens=6, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0, p.shape[1]:].tolist())
            target = tok.decode([t])
            ok += len(target) > 0 and gen.startswith(target)
        hits.append(ok >= 2)
    return sum(hits) / max(len(hits), 1)


@torch.no_grad()
def perplexity(model, tok, dev, text_path, n_chunks=40, seq=512):
    """Held-out perplexity -- the guard-rail. Same chunks before and after."""
    raw = open(text_path, encoding="utf-8").read(4_000_000)
    ids = tok(raw, add_special_tokens=False).input_ids
    step = max(1, (len(ids) - seq - 1) // n_chunks)
    tot, cnt = 0.0, 0
    for i in range(0, min(len(ids) - seq - 1, step * n_chunks), step):
        x = torch.tensor(ids[i:i + seq], device=dev).unsqueeze(0)
        y = torch.tensor(ids[i + 1:i + 1 + seq], device=dev).unsqueeze(0)
        logits = model(input_ids=x).logits.float()
        tot += F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1)).item()
        cnt += 1
    return float(np.exp(tot / max(cnt, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--traj", default="results/traj_1.4b.pt")
    ap.add_argument("--ext", default="external/pythia_6_9b.jsonl.gz")
    ap.add_argument("--cands", default="results/candidates.json")
    ap.add_argument("--text", default="data/model_en.txt")
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--behav-gt", default=None,
                    help="copy-probe scores. When given, a token is repaired only "
                         "if it is BOTH detector-flagged AND behaviourally impaired. "
                         "Without this guard the detector's top-N is 97% false "
                         "positives and includes ' and', ' of', ' to' -- repairing "
                         "those destroys the model.")
    ap.add_argument("--behav-pct", type=float, default=5.0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="results/repair_real.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
    model.eval()

    tr = torch.load(a.traj, weights_only=False)
    n = tr["unemb_cnorm"].shape[1]
    ver, tested, cat, dec = load_labels(a.ext, n)
    C = json.load(open(a.cands))

    # --- 1. score with the late detector, fit on the realistic task ---
    i = tr["steps"].index(143000)
    X = np.stack([tr[k][i].numpy() for k in IND], 1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(ver, dtype=torch.float32)
    ok_mask = np.array([cat.get(j) == "OK" for j in range(n)]) | (ver == 1)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.1)
    pw = torch.tensor(float((ver[ok_mask] == 0).sum() / max(ver[ok_mask].sum(), 1)))
    for _ in range(400):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(
            Xt[ok_mask] @ w + b, yt[ok_mask], pos_weight=pw).backward()
        opt.step()
    score = (Xt @ w + b).detach().numpy()
    doomed = [int(j) for j in np.argsort(-score)[:a.top_n]]
    if a.behav_gt:
        blp = torch.load(a.behav_gt, weights_only=False)["copy_logprob"].numpy()
        thr = np.percentile(blp, a.behav_pct)
        kept = [j for j in doomed if blp[j] <= thr]
        print(f"behavioural guard: {len(doomed)} flagged -> {len(kept)} also impaired "
              f"(copy-logprob <= {thr:.2f}, the worst {a.behav_pct}%)")
        doomed = kept

    groups = {
        "verified glitch (magikarp)": [j for j in range(n) if ver[j] == 1][:20],
        "confirmed glitch (ours)": C["candidates"][:20],
        "control (normal tokens)": C["controls"][:20],
    }
    covered = {g: sum(1 for t in ids if t in set(doomed)) for g, ids in groups.items()}

    print(f"model {a.model}   repairing top-{a.top_n} by late-detector score, k={a.k}")
    print("coverage of the repair set: " +
          ", ".join(f"{g.split()[0]} {c}/{len(ids)}"
                    for (g, ids), c in zip(groups.items(), covered.values())))

    before = {g: repetition_rate(model, tok, dev, ids) for g, ids in groups.items()}
    ppl_before = perplexity(model, tok, dev, a.text)

    n_rep = knn_repair(model, doomed, k=a.k)

    after = {g: repetition_rate(model, tok, dev, ids) for g, ids in groups.items()}
    ppl_after = perplexity(model, tok, dev, a.text)

    print()
    print(f"{'group':>28} | {'before':>8} | {'after':>8} | {'delta':>8}")
    print("-" * 62)
    for g in groups:
        print(f"{g:>28} | {before[g]:7.1%} | {after[g]:7.1%} | {after[g]-before[g]:+7.1%}")
    print()
    print(f"  repaired rows      : {n_rep}")
    print(f"  held-out perplexity: {ppl_before:.4f} -> {ppl_after:.4f} "
          f"({100*(ppl_after/ppl_before-1):+.3f}%)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "top_n": a.top_n, "k": a.k, "repaired": n_rep,
               "before": before, "after": after, "coverage": covered,
               "ppl_before": ppl_before, "ppl_after": ppl_after},
              open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
