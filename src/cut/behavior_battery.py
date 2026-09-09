"""Three independent behavioural criteria, because one is not enough.

Every glitch label in this project so far comes from a copy/repetition probe. If
that probe is idiosyncratic, every downstream result inherits its bias. This runs
two additional criteria that do NOT depend on in-context copying, and measures
how much they agree.

  copy      -- reproduce the token when shown it (Magikarp's criterion, and ours)
  entropy   -- H(next | ctx, t) averaged over a context bank. GlitchMiner's
               (AAAI'26) signal: an unlearned row leaves the model unable to
               predict what follows. Needs no copying ability at all.
  spelling  -- reproduce the token's characters separated by spaces. GlitchQuiz
               (USENIX Sec'26) uses spelling as one of its eight templates; it
               probes whether the row carries usable character-level identity.

The interesting number is not any single criterion but their CONCORDANCE, and
whether the paper's conclusions survive relabelling under each.
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .analyze_trajectory import auc

ENT_CTX = ["The", " In 1994, the", " He said that", "\n\n", " According to the",
           " She wrote that", " The city of", " It is important to",
           " After the war,", " A new study"]


@torch.no_grad()
def entropy_score(model, tok, dev, ids, batch=64):
    """Mean H(next | ctx, t) over a context bank. Higher = glitchier."""
    acc = torch.zeros(len(ids))
    for c in ENT_CTX:
        pre = tok(c, add_special_tokens=False).input_ids
        for s in range(0, len(ids), batch):
            ch = ids[s:s + batch]
            x = torch.tensor([pre + [t] for t in ch], device=dev)
            lp = torch.log_softmax(model(input_ids=x).logits[:, -1].float(), -1)
            acc[s:s + len(ch)] += (-(lp.exp() * lp).sum(-1)).cpu()
    return (acc / len(ENT_CTX)).numpy()


@torch.no_grad()
def spelling_score(model, tok, dev, ids, max_new=24):
    """Can the model spell the token out, character by character?"""
    demos = ("Spell each word letter by letter.\n"
             "Word: cat\nLetters: c a t\n"
             "Word: blue\nLetters: b l u e\n")
    out = []
    for t in ids:
        s = tok(demos + "Word:", add_special_tokens=False).input_ids
        prompt = s + [t] + tok("\nLetters:", add_special_tokens=False).input_ids
        p = torch.tensor(prompt, device=dev).unsqueeze(0)
        g = model.generate(p, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        gen = tok.decode(g[0, p.shape[1]:].tolist())
        target = tok.decode([t]).strip()
        letters = [c for c in gen if not c.isspace()]
        want = list(target)
        # fraction of the token's characters produced in order
        i = 0
        for c in letters:
            if i < len(want) and c == want[i]:
                i += 1
        out.append(i / max(len(want), 1))
    return np.array(out)


def load_ver(path, n):
    v = np.zeros(n, int); tested = np.zeros(n, bool)
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line); i = int(r["i"])
        if i < n and "magikarp" in r:
            tested[i] = True
            v[i] = 1 if "verified" in r["magikarp"] else 0
    return v, tested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--ext", default="external/pythia_6_9b.jsonl.gz")
    ap.add_argument("--copy-gt", default="results/behav_gt_1.4b.pt")
    ap.add_argument("--sample", type=int, default=1500,
                    help="healthy tokens to sample as the comparison pool")
    ap.add_argument("--out", default="results/battery.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
    model.eval()
    V = model.get_input_embeddings().weight.shape[0]
    ver, tested = load_ver(a.ext, V)
    copy_lp = torch.load(a.copy_gt, weights_only=False)["copy_logprob"].numpy()[:V]

    rng = np.random.default_rng(0)
    pos = [int(i) for i in np.where(ver == 1)[0]]
    neg = [int(i) for i in rng.choice(np.where((ver == 0) & tested)[0],
                                      size=min(a.sample, int(((ver == 0) & tested).sum())),
                                      replace=False)]
    ids = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    print(f"{a.model}: {len(pos)} verified glitch vs {len(neg)} tested-and-rejected\n")

    ent = entropy_score(model, tok, dev, ids)
    spell = spelling_score(model, tok, dev, ids)
    cp = copy_lp[ids]

    print("each criterion, as a discriminator of Magikarp's verified set:")
    print(f"{'criterion':>12} | {'AUC':>7} | {'glitch mean':>12} | {'healthy mean':>13}")
    print("-" * 52)
    scores = {"copy": -cp, "entropy": ent, "spelling": -spell}
    for nm, s in scores.items():
        print(f"{nm:>12} | {auc(s, y):7.3f} | {s[y==1].mean():12.3f} | {s[y==0].mean():13.3f}")

    print()
    print("CONCORDANCE -- do the criteria pick the same tokens?")
    print("(top-k by each criterion within this pool, Jaccard overlap)")
    k = len(pos)
    top = {nm: set(np.argsort(-s)[:k].tolist()) for nm, s in scores.items()}
    names = list(top)
    print(f"{'':>12} | " + " | ".join(f"{n:>9}" for n in names))
    print("-" * 46)
    for n1 in names:
        row = []
        for n2 in names:
            j = len(top[n1] & top[n2]) / len(top[n1] | top[n2])
            row.append(f"{j:9.2f}")
        print(f"{n1:>12} | " + " | ".join(row))

    print()
    print("ROBUSTNESS -- relabel using each criterion, then score OUR detector")
    tr = torch.load("results/traj_1.4b.pt", weights_only=False)
    n = tr["unemb_cnorm"].shape[1]
    i2 = tr["steps"].index(2)
    ilast = tr["steps"].index(143000)
    print(f"{'labels from':>12} | {'resid_i @step2':>15} | {'unemb_cos @final':>17}")
    print("-" * 50)
    res = {}
    for nm, s in scores.items():
        lab = np.zeros(len(ids), int)
        lab[np.argsort(-s)[:k]] = 1
        sub = np.array(ids)
        e = auc(-tr["resid_i"][i2].numpy()[sub], lab)
        l = auc(tr["unemb_cos"][ilast].numpy()[sub], lab)
        res[nm] = {"early": e, "late": l}
        print(f"{nm:>12} | {e:15.3f} | {l:17.3f}")
    e = auc(-tr["resid_i"][i2].numpy()[np.array(ids)], y)
    l = auc(tr["unemb_cos"][ilast].numpy()[np.array(ids)], y)
    print(f"{'magikarp':>12} | {e:15.3f} | {l:17.3f}   <- reference")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "n_pos": len(pos), "n_neg": len(neg),
               "auc": {nm: auc(s, y) for nm, s in scores.items()},
               "relabel": res}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
