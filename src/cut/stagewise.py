"""Is under-training stage-relative? Do SFT/DPO/RLVR create NEW glitch tokens?

The mechanism in this project is gradient sparsity: an input-embedding row only
receives a token-specific update on steps where that token is actually in the
batch. Pretraining sees trillions of tokens, so almost every row gets touched
sometimes. Post-training sees a few orders of magnitude fewer -- and instruction
and preference data are far narrower in distribution than a web crawl.

That makes a sharp prediction nobody in the glitch-token literature has tested:
a token can be adequately trained WITH RESPECT TO PRETRAINING and abandoned WITH
RESPECT TO POST-TRAINING. "Glitch token" is not a property of a model, it is a
property of a model AND a training stage. Behaviour elicited by post-trained
capabilities would then be more fragile on these tokens than base-model
behaviour, even though the base weights are fine.

OLMo-2 publishes base -> SFT -> DPO -> RLVR-Instruct as separate checkpoints of
one model, which is the only fully open stage series with untied embeddings.

Two measurements:

  1. COVERAGE. For each stage transition, decompose the embedding delta into the
     shared drift and the token-specific residual (same decomposition used on
     pretraining checkpoints). Count rows whose token-specific residual is
     essentially zero -- rows the stage never actually saw. If this fraction is
     large, post-training abandonment is real by construction.

  2. DAMAGE. Run the copy probe at every stage. A token that copies at base and
     stops copying at Instruct has been broken BY post-training. Then ask the
     question that ties 1 to 2: does zero token-specific update during a stage
     predict which tokens degrade across it?

Note on 2: post-training changes formatting behaviour globally, so absolute copy
scores drift for every token. The comparison is therefore always glitch-vs-
healthy WITHIN a stage, and the headline number is the AUC in step 2, which is
invariant to any monotone per-stage shift.
"""
from __future__ import annotations
import argparse, gc, json, subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .analyze_trajectory import auc
from .harvest_family2 import FEWSHOT
from .copyprompt import copy_prompt_parts


@torch.no_grad()
def copy_logprob_ids(model, tok, dev, ids, batch=48):
    """Teacher-forced logprob of copying each listed token back.

    Takes an explicit id list rather than a vocab size so large vocabularies can
    be subsampled: the claims here are rates and AUCs, which a uniform sample
    estimates fine, and a full 152k-row probe on four 8B stages is hours of GPU
    for no extra resolution.
    """
    def enc(s):
        return tok(s, add_special_tokens=False)["input_ids"]
    head = enc("Repeat the text exactly.\n")
    for a in FEWSHOT:
        head += enc("Text:") + enc(a) + enc("\nCopy:") + enc(a) + enc("\n")
    pre_t, pre_c = enc("Text:"), enc("\nCopy:")
    head, pre_t, pre_c = copy_prompt_parts(tok, head, FEWSHOT)   # raw or chat framing (GLITCH_PROMPT_STYLE)
    out = []
    for s in range(0, len(ids), batch):
        ch = ids[s:s + batch]
        seqs = [head + pre_t + [int(t)] + pre_c + [int(t)] for t in ch]
        x = torch.tensor(seqs, device=dev)
        logits = model(input_ids=x).logits[:, :-1, :].float()
        lp = torch.log_softmax(logits, -1).gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        out.append(lp[:, -1].cpu())
    return torch.cat(out).numpy()


def purge_cache(name):
    """Delete the downloaded weights for one repo. Four stages of an 8B model is
    ~60GB; without this the third family fills the disk."""
    blobs = (Path.home() / ".cache/huggingface/hub" /
             ("models--" + name.replace("/", "--")) / "blobs")
    if blobs.exists():
        subprocess.run(["find", str(blobs), "-size", "+50M", "-delete"], check=False)


def decompose(W, prev):
    d = (W - prev).float()
    g = d.mean(0)
    gh = g / (g.norm() + 1e-12)
    sh = d @ gh
    return (d.norm(dim=-1).cpu().numpy(),
            sh.cpu().numpy(),
            (d - sh.unsqueeze(1) * gh.unsqueeze(0)).norm(dim=-1).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=[
        "allenai/OLMo-2-0425-1B",
        "allenai/OLMo-2-0425-1B-SFT",
        "allenai/OLMo-2-0425-1B-DPO",
        "allenai/OLMo-2-0425-1B-Instruct"])
    ap.add_argument("--glitch-pct", type=float, default=1.0)
    ap.add_argument("--vocab-sample", type=int, default=0,
                    help="probe a uniform sample of this many rows instead of the "
                         "whole vocabulary (0 = full). Coverage is always full-vocab.")
    ap.add_argument("--out", default="results/stagewise.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.stages[0])

    W_in, lp = [], []
    probe_ids = None
    for name in a.stages:
        m = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
        tied = m.config.tie_word_embeddings
        Ein = m.get_input_embeddings().weight.detach()
        Vfull = Ein.shape[0]
        if probe_ids is None:
            if a.vocab_sample and a.vocab_sample < Vfull:
                rs = np.random.default_rng(0)
                probe_ids = np.sort(rs.choice(Vfull, a.vocab_sample, replace=False))
            else:
                probe_ids = np.arange(Vfull)
        print(f"{name}: V={Vfull} tied={tied} probe={len(probe_ids)} "
              f"|E_in| {Ein.float().norm(dim=-1).mean():.4f}", flush=True)
        # only the INPUT embedding is used; keeping the output matrix for every
        # stage costs gigabytes of host RAM for nothing.
        W_in.append(Ein.float().cpu())
        print("  copy probe...", flush=True)
        lp.append(copy_logprob_ids(m, tok, dev, probe_ids))
        del m, Ein; gc.collect(); torch.cuda.empty_cache()
        purge_cache(name)

    V = len(probe_ids)
    dec = [tok.decode([int(i)]) for i in probe_ids]
    nonws = np.array([d.strip() != "" for d in dec])

    print()
    print("=" * 78)
    print("1. COVERAGE -- how much of the vocabulary does each stage actually touch?")
    print("=" * 78)
    print(f"{'transition':>34} | {'zero-resid rows':>15} | {'frac':>7} | "
          f"{'median resid':>12}")
    print("-" * 78)
    # coverage is measured over the FULL vocabulary -- it is embedding-only and
    # costs nothing, so there is no reason to subsample it.
    Vfull = W_in[0].shape[0]
    cov = []
    for i in range(1, len(a.stages)):
        tot, sh, rs = decompose(W_in[i], W_in[i - 1])
        eps = max(1e-9, 1e-4 * float(np.median(rs[rs > 0])) if (rs > 0).any() else 1e-9)
        z = int((rs <= eps).sum())
        nm = f"{a.stages[i-1].split('-')[-1]} -> {a.stages[i].split('-')[-1]}"
        cov.append({"transition": nm, "zero_rows": z, "frac": z / Vfull,
                    "median_resid": float(np.median(rs)),
                    "resid": rs, "total": tot})
        print(f"{nm:>34} | {z:15d} | {z/Vfull:7.4f} | {np.median(rs):12.3e}")

    print()
    print("=" * 78)
    print("2. DAMAGE -- glitch sets per stage, defined by the copy probe")
    print("=" * 78)
    thr = np.percentile(lp[0], a.glitch_pct)      # threshold FIXED from the base model
    print(f"threshold = base-model {a.glitch_pct}th pct of copy-logprob = {thr:.3f}")
    print(f"{'stage':>34} | {'n below thr':>11} | {'mean lp':>9} | "
          f"{'new vs base':>11} | {'healed':>7}")
    print("-" * 82)
    base_bad = lp[0] <= thr
    dmg = []
    for i, name in enumerate(a.stages):
        bad = lp[i] <= thr
        new = int((bad & ~base_bad).sum()); heal = int((~bad & base_bad).sum())
        dmg.append({"stage": name, "n_bad": int(bad.sum()), "mean_lp": float(lp[i].mean()),
                    "new": new, "healed": heal})
        print(f"{name:>34} | {int(bad.sum()):11d} | {lp[i].mean():9.3f} | "
              f"{new:11d} | {heal:7d}")

    # A FIXED threshold is contaminated by the global formatting shift that
    # instruction tuning induces (mean copy-logprob moves for every token). The
    # rank-based set is the honest one: worst 1% within each stage, so any
    # monotone per-stage shift cancels.
    rank_bad = [lp[i] <= np.percentile(lp[i], a.glitch_pct) for i in range(len(lp))]
    print(f"\nrank-based (worst {a.glitch_pct}% WITHIN each stage, shift-invariant):")
    print(f"{'stage':>34} | {'overlap w/ base':>15} | {'new':>6} | {'jaccard':>8}")
    print("-" * 72)
    for i, name in enumerate(a.stages):
        ov = int((rank_bad[i] & rank_bad[0]).sum())
        un = int((rank_bad[i] | rank_bad[0]).sum())
        print(f"{name:>34} | {ov:15d} | {int((rank_bad[i] & ~rank_bad[0]).sum()):6d} | "
              f"{ov/un:8.3f}")
    final_bad = lp[-1] <= thr
    newly = rank_bad[-1] & ~rank_bad[0]
    ex_r = [i for i in np.where(newly)[0] if dec[i].strip()][:15]
    print("\n  rank-based newly-glitched examples:", [repr(dec[i])[:16] for i in ex_r])

    # CONTROL. A worst-1% set turning over could just be tokens jittering across
    # the threshold. Real movement means the newly-glitched sat far outside the
    # tail in the base model. k = the cut, so rank ~k is the boundary itself.
    k = int(rank_bad[0].sum())
    r0 = np.argsort(np.argsort(lp[0]))          # 0 = worst in base
    rF = np.argsort(np.argsort(lp[-1]))
    if newly.sum():
        b = r0[newly]
        print(f"\n  base-model rank of the {int(newly.sum())} newly-glitched "
              f"(threshold at rank {k}, vocab {V}):")
        print(f"    median {int(np.median(b))}   quartiles "
              f"{int(np.percentile(b,25))}/{int(np.percentile(b,75))}   "
              f"max {int(b.max())}")
        for mult, lab in [(2, "2x"), (5, "5x"), (10, "10x")]:
            print(f"    beyond {mult}x the threshold rank: "
                  f"{float((b > mult*k).mean()):.3f}")
    healed = rank_bad[0] & ~rank_bad[-1]
    if healed.sum():
        h = rF[healed]
        print(f"  final-model rank of the {int(healed.sum())} healed: "
              f"median {int(np.median(h))}, beyond 5x threshold "
              f"{float((h > 5*k).mean()):.3f}")
    print(f"\ntokens healthy at base but below threshold after post-training: {int(newly.sum())}")
    ex = [i for i in np.where(newly)[0] if dec[i].strip()][:15]
    print("  examples:", [repr(dec[i])[:16] for i in ex])

    print()
    print("=" * 78)
    print("3. DOES POST-TRAINING GRADIENT SPARSITY PREDICT THE DAMAGE?")
    print("AUC of (low token-specific residual during post-training) vs (degraded)")
    print("=" * 78)
    # degradation as a continuous target avoids threshold games
    delta = lp[-1] - lp[0]
    # restrict the full-vocab residuals to the probed rows so they line up with
    # the behavioural labels
    tot_rs = np.zeros(V)
    for c in cov:
        tot_rs += c["resid"][probe_ids]
    print(f"{'predictor':>34} | {'AUC vs newly-glitched':>22} | "
          f"{'spearman vs delta lp':>21}")
    print("-" * 82)
    from scipy.stats import spearmanr
    preds = {"total post-train residual": -tot_rs,
             "base copy-logprob": -lp[0],
             "residual, non-ws only": -tot_rs}
    for nm, s in preds.items():
        mask = nonws if "non-ws" in nm else np.ones(V, bool)
        if newly[mask].sum() < 5:
            print(f"{nm:>34} | (only {int(newly[mask].sum())} positives)"); continue
        A = auc(s[mask], newly[mask].astype(int))
        r = float(spearmanr(s[mask], -delta[mask])[0])
        print(f"{nm:>34} | {A:22.3f} | {r:21.3f}")

    print("\nCONTROL: how much do glitch rows move during post-training at all?")
    print(f"{'transition':>34} | {'total ratio':>11} | {'resid ratio':>11}")
    print("-" * 62)
    for c in cov:
        g, h = base_bad, ~base_bad
        ct, cr = c["total"][probe_ids], c["resid"][probe_ids]
        print(f"{c['transition']:>34} | "
              f"{ct[g].mean()/ct[h].mean():11.3f} | "
              f"{cr[g].mean()/cr[h].mean():11.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"stages": a.stages,
               "coverage": [{k: v for k, v in c.items() if k not in ("resid", "total")}
                            for c in cov],
               "damage": dmg,
               "n_newly_glitched": int(newly.sum()),
               "examples": [dec[i] for i in ex],
               "rank_based": [{"stage": a.stages[i],
                               "overlap": int((rank_bad[i] & rank_bad[0]).sum()),
                               "new": int((rank_bad[i] & ~rank_bad[0]).sum()),
                               "jaccard": float((rank_bad[i] & rank_bad[0]).sum() /
                                                max(1, (rank_bad[i] | rank_bad[0]).sum()))}
                              for i in range(len(a.stages))],
               "newly_base_rank_median": (int(np.median(r0[newly]))
                                          if newly.sum() else None),
               "newly_beyond_5x": (float((r0[newly] > 5 * k).mean())
                                   if newly.sum() else None),
               "threshold_rank": int(k), "n_probed": int(V),
               "newly_examples": [dec[i] for i in ex_r]},
              open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
