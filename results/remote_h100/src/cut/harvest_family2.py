"""Second-wave families: LLaMA-architecture models, and a weight-tying scope test.

Everything tested so far is GPT-NeoX-derived (Pythia, OLMo, OLMoE share tokenizer
lineage) and untied. Two gaps:

  LLM360/Amber -- LlamaForCausalLM, SentencePiece 32k vocab, UNTIED. A genuinely
      different lineage, so a clean replication target for residual suppression.

  HuggingFaceFW ablation suite -- eight models with IDENTICAL architecture and
      tokenizer trained on different corpora. Ideal for isolating corpus... except
      they are TIED (tie_word_embeddings=True). Under tying there is one matrix,
      so resid_i == resid_o and the gradient-sparsity signal vanishes: a rare row
      still receives dense gradient every step as a softmax negative. These are
      therefore a SCOPE TEST (does the mechanism survive tying?), not a
      replication. Reported separately for that reason.

Labels are derived per-model with the copy probe on that model's own final
checkpoint -- no external label set exists for these families, and the mechanism
claim is a relation between two measurements on one model, so it does not need one.
"""
from __future__ import annotations
import argparse, gc, json, re, subprocess, urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def branches(model):
    d = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/models/{model}/refs", timeout=40))
    return [b["name"] for b in d.get("branches", []) if b["name"] != "main"]


def pick(model, n_want):
    """Order checkpoints by their embedded number, take n_want spread evenly."""
    br = branches(model)
    keyed = []
    for b in br:
        m = re.search(r"(?:ckpt_|step-?)(\d+)", b)
        if m:
            keyed.append((int(m.group(1)), b))
    keyed.sort()
    if not keyed:
        return []
    # LOG spacing, not linear: the effect lives early, and linear spacing spends
    # every sample on late training where the ratios have already converged.
    idx = np.unique(np.round(np.geomspace(1, len(keyed), n_want)).astype(int) - 1)
    idx = np.clip(idx, 0, len(keyed) - 1)
    return [keyed[i] for i in idx]


def decompose(W, prev):
    if prev is None:
        z = torch.zeros(W.shape[0]); return z, z, z
    d = (W - prev).float()
    g = d.mean(0); gh = g / (g.norm() + 1e-12)
    sh = d @ gh
    return d.norm(dim=-1).cpu(), sh.cpu(), (d - sh.unsqueeze(1) * gh.unsqueeze(0)).norm(dim=-1).cpu()


FEWSHOT = [" apple pie is good", " the quick brown fox"]


@torch.no_grad()
def copy_logprob(model, tok, dev, V, batch=48):
    """Teacher-forced logprob of copying each token back. Defines the glitch set."""
    def enc(s): return tok(s, add_special_tokens=False)["input_ids"]
    head = enc("Repeat the text exactly.\n")
    for a in FEWSHOT:
        head += enc("Text:") + enc(a) + enc("\nCopy:") + enc(a) + enc("\n")
    pre_t, pre_c = enc("Text:"), enc("\nCopy:")
    out = []
    for s in range(0, V, batch):
        ch = list(range(s, min(s + batch, V)))
        seqs = [head + pre_t + [t] + pre_c + [t] for t in ch]
        ids = torch.tensor(seqs, device=dev)
        logits = model(input_ids=ids).logits[:, :-1, :].float()
        tgt = ids[:, 1:]
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        out.append(lp[:, -1].cpu())          # the copied token's own logprob
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-ckpt", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glitch-pct", type=float, default=1.0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    revs = pick(a.model, a.n_ckpt)
    if not revs:
        print(f"{a.model}: no parseable checkpoints"); return
    print(f"{a.model}: {len(revs)} checkpoints, {revs[0][1]} .. {revs[-1][1]}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    cache = Path.home() / ".cache/huggingface/hub"

    rec = {"steps": [], "d": [], "sh": [], "rs": []}
    prev = None
    final_model = None
    for num, rev in revs:
        try:
            m = AutoModelForCausalLM.from_pretrained(a.model, revision=rev,
                                                     dtype=torch.float32)
        except Exception as e:
            print(f"  {rev}: FAILED {type(e).__name__} {str(e)[:70]}", flush=True); continue
        W = m.get_output_embeddings().weight.detach()
        d, sh, rs = decompose(W, prev)
        rec["steps"].append(num); rec["d"].append(d); rec["sh"].append(sh); rec["rs"].append(rs)
        prev = W.clone()
        print(f"  {rev:24s} |W| {W.float().norm(dim=-1).mean():.4f}  d {d.mean():.5f}", flush=True)
        del m; gc.collect()
        blobs = cache / ("models--" + a.model.replace("/", "--")) / "blobs"
        if blobs.exists():
            subprocess.run(["find", str(blobs), "-size", "+50M", "-delete"], check=False)

    if len(rec["steps"]) < 3:
        print("insufficient checkpoints"); return
    # the copy probe runs on `main` -- the released final model. Some repos have a
    # final step-branch with a broken config, and main is the canonical endpoint.
    final_model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    V = final_model.get_input_embeddings().weight.shape[0]
    print(f"\ndefining glitch set on the final checkpoint (copy probe over {V} tokens)...",
          flush=True)
    lp = copy_logprob(final_model, tok, dev, V)
    thr = np.percentile(lp, a.glitch_pct)
    lab = (lp <= thr).astype(int)
    dec = [tok.decode([i]) for i in range(V)]
    nonws = np.array([dec[i].strip() != "" for i in range(V)])
    print(f"glitch = worst {a.glitch_pct}% copy-logprob -> {lab.sum()} tokens")
    print("  examples:", [repr(dec[int(i)])[:14] for i in np.where(lab == 1)[0][:10]])

    print(f"\n{'ckpt':>10} | {'total':>8} | {'shared':>8} | {'residual':>9} | "
          f"{'resid (non-ws)':>14}")
    print("-" * 62)
    rows = []
    for i, st in enumerate(rec["steps"]):
        if i == 0: continue
        g = lambda k: rec[k][i].numpy()[:V]
        r = {"ckpt": st}
        for nm, k in [("total", "d"), ("shared", "sh"), ("resid", "rs")]:
            v = g(k); h = v[lab == 0].mean()
            r[nm] = float(v[lab == 1].mean() / h) if h else float("nan")
        v = g("rs"); m1 = (lab == 1) & nonws; m0 = (lab == 0) & nonws
        r["resid_nonws"] = float(v[m1].mean() / v[m0].mean()) if m0.any() and m1.any() else float("nan")
        rows.append(r)
        print(f"{st:10d} | {r['total']:8.3f} | {r['shared']:8.3f} | {r['resid']:9.3f} | "
              f"{r['resid_nonws']:14.3f}")
    mid = rows[:max(1, len(rows) // 2)]
    print(f"\n-> early-half mean residual ratio {np.mean([r['resid'] for r in mid]):.3f}"
          f"   (non-whitespace only {np.nanmean([r['resid_nonws'] for r in mid]):.3f})")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "n_glitch": int(lab.sum()), "rows": rows},
              open(a.out, "w"), indent=1)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
