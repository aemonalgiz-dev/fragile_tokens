"""Behavioural evaluation of a trained arm, scored against TRUE corpus counts.

The Pythia work defined "glitch" behaviourally and could only guess at frequency.
Here we own the corpus, so we can measure the exact dose-response between a
token's true training frequency and how badly the model handles it.

THE METRIC: copy uplift, not copy logprob.

A 23M model cannot do few-shot copying -- raw copy-logprob sits at -13 nats for
every token, and it correlates 0.405 with log frequency, so it is partly just
re-reading the unigram prior. Predicting that from embeddings would be circular.

Uplift removes the prior exactly:

    uplift(t) = log p(t | prompt showing t) - log p(t | prompt showing a decoy)

Both prompts are identical in length and structure; only the token in the "Text:"
slot differs. The second term is the model's baseline propensity to emit t at that
position -- which is precisely the unigram prior we want gone. What survives is
"does the model actually USE the token it was shown", which is the copy behaviour
glitch tokens fail at.

The decoy prompt is the same for every t, so it costs ONE extra forward pass total.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .train import make_model, Data

FEWSHOT = [" apple pie is good", " the quick brown fox"]
HEAD = "Repeat the text exactly.\n"


def build_prefix(tok, target_id):
    def enc(s):
        return tok(s, add_special_tokens=False)["input_ids"]
    ids = enc(HEAD)
    for a in FEWSHOT:
        ids += enc("Text:") + enc(a) + enc("\nCopy:") + enc(a) + enc("\n")
    ids += enc("Text:") + [target_id] + enc("\nCopy:")
    return ids


@torch.no_grad()
def copy_uplift(model, tok, device, vocab, decoy, batch=64):
    """log p(t | prompt showing t) - log p(t | prompt showing decoy)."""
    dec_ids = torch.tensor(build_prefix(tok, decoy), device=device).unsqueeze(0)
    base = torch.log_softmax(model(input_ids=dec_ids).logits[0, -1].float(), -1)

    self_lp = torch.zeros(vocab)
    for s in range(0, vocab, batch):
        chunk = list(range(s, min(s + batch, vocab)))
        seqs = [build_prefix(tok, t) for t in chunk]
        ids = torch.tensor(seqs, device=device)
        lp = torch.log_softmax(model(input_ids=ids).logits[:, -1].float(), -1)
        self_lp[s:s + len(chunk)] = lp[torch.arange(len(chunk)), torch.tensor(chunk)].cpu()
    return (self_lp - base.cpu()).numpy(), self_lp.numpy()


ENT_CTX = ["The", " In 1994, the", " He said that", "\n\n", " According to the",
           " She wrote that", " The city of", " It is important to",
           " After the war,", " A new study"]


@torch.no_grad()
def pred_entropy(model, tok, device, vocab, batch=128):
    """Mean H(next | context, t) over a context bank.

    This is GlitchMiner's (AAAI'26) signal: a token whose representation is
    unlearned leaves the model unable to predict what follows, so entropy spikes.
    Unlike copy-uplift it needs no in-context-copying ability, so a 23M model can
    actually express it -- which copy-based probes at this scale cannot.
    """
    acc = torch.zeros(vocab)
    for c in ENT_CTX:
        pre = tok(c, add_special_tokens=False)["input_ids"]
        for s in range(0, vocab, batch):
            chunk = list(range(s, min(s + batch, vocab)))
            ids = torch.tensor([pre + [t] for t in chunk], device=device)
            lp = torch.log_softmax(model(input_ids=ids).logits[:, -1].float(), -1)
            acc[s:s + len(chunk)] += (-(lp.exp() * lp).sum(-1)).cpu()
    return (acc / len(ENT_CTX)).numpy()


@torch.no_grad()
def val_loss(model, data, device, vocab, batch=16, iters=40, mask=None):
    """Re-measured on the arm's ACTUAL final weights -- never inherited. A repair
    that fixes glitch tokens by degrading the model is worthless, so this is the
    guard-rail metric for every arm."""
    ev = []
    g = torch.Generator().manual_seed(7)
    for _ in range(iters):
        x, y = data.batch(batch, device, "val", g)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            l = model(input_ids=x).logits
        fl = l.view(-1, vocab).float()
        if mask is not None:
            fl = fl + mask
        ev.append(F.cross_entropy(fl, y.reshape(-1)).item())
    return float(np.mean(ev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="baseline")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tok", default="data/tok")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--masked", action="store_true",
                    help="arm trained with the doomed-token softmax mask; "
                         "evaluate it the way it would deploy")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.tok)
    counts = np.load(Path(a.tok) / "counts.npy")
    V = len(counts)

    model = make_model(V).to(dev)
    ck = torch.load(Path(a.runs) / a.arm / "final.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    decoy = int(np.argsort(-counts)[50])       # a common, unambiguously healthy token
    up, raw = copy_uplift(model, tok, dev, V, decoy, a.batch)
    ent = pred_entropy(model, tok, dev, V)
    np.save(Path(a.runs) / a.arm / "copy_uplift.npy", up)
    np.save(Path(a.runs) / a.arm / "pred_entropy.npy", ent)

    dec = [tok.decode([i]) for i in range(V)]
    multi = np.array([len(d) >= 2 for d in dec])
    pos = counts > 0
    print(f"arm={a.arm}   decoy={tok.decode([decoy])!r}")
    print(f"  corr(uplift, log count) = "
          f"{np.corrcoef(up[pos], np.log(counts[pos]))[0,1]:+.3f}   "
          f"(raw logprob was +0.405 -- lower is better, the prior is gone)")
    print()
    print(f"  corr(entropy, log count) = "
          f"{np.corrcoef(ent[pos], np.log(counts[pos]))[0,1]:+.3f}")
    print()
    print("DOSE-RESPONSE: true training frequency vs behavioural damage")
    print(f"{'true count':>16} | {'n':>6} | {'median entropy':>14} | {'median uplift':>14}")
    print("-" * 62)
    bands = [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 10000),
             (10000, 100000), (100000, 10 ** 12)]
    rows = []
    for lo, hi in bands:
        m = (counts >= lo) & (counts < hi) & multi
        if m.sum() == 0:
            continue
        med = float(np.median(up[m])); me = float(np.median(ent[m]))
        lbl = f"{lo}" if hi == lo + 1 else f"{lo}-{hi-1}"
        print(f"{lbl:>16} | {m.sum():6d} | {me:14.3f} | {med:14.3f}")
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                     "median_uplift": med, "median_entropy": me})

    healthy = (counts >= 10000) & multi
    doomed = (counts < 100) & multi
    sep = float(np.median(up[healthy]) - np.median(up[doomed]))
    sep_e = float(np.median(ent[doomed]) - np.median(ent[healthy]))
    print()
    print(f"  doomed  (count<100,  >=2ch): n={doomed.sum():5d}  median uplift {np.median(up[doomed]):+.3f}")
    print(f"  healthy (count>=10k, >=2ch): n={healthy.sum():5d}  median uplift {np.median(up[healthy]):+.3f}")
    print(f"  uplift separation : {sep:+.3f} nats")
    print(f"  ENTROPY separation: {sep_e:+.3f} nats  "
          f"(doomed {np.median(ent[doomed]):.3f} vs healthy {np.median(ent[healthy]):.3f})")

    data = Data("data/tokens.bin", 512)
    mask = None
    if a.masked:
        mask = torch.zeros(V, device=dev)
        mask[torch.tensor(counts == 0, device=dev)] = float("-inf")
    vl = val_loss(model, data, dev, V, mask=mask)

    summ = json.load(open(Path(a.runs) / a.arm / "summary.json"))
    summ.update({"val_loss": vl, "dose_response": rows,
                 "median_uplift_doomed": float(np.median(up[doomed])),
                 "median_uplift_healthy": float(np.median(up[healthy])),
                 "separation": sep, "separation_entropy": sep_e,
                 "median_entropy_doomed": float(np.median(ent[doomed])),
                 "median_entropy_healthy": float(np.median(ent[healthy]))})
    json.dump(summ, open(Path(a.runs) / a.arm / "summary.json", "w"), indent=1)
    print(f"\nval loss {vl:.4f}   saved -> {a.runs}/{a.arm}/")


if __name__ == "__main__":
    main()
