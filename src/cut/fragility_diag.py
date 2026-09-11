"""Are the fragile clean-looking tokens canonical, or Unicode/duplicate variants?

The most fragile tokens that pass the single probe are accented foreign words
and one ordinary English word at -22 logprob in context. That is what a
NON-CANONICAL token id looks like: a decomposed-accent or duplicate encoding
whose decoded string re-encodes to a DIFFERENT id. Such a token copies after
"Copy:" (induction has one candidate) but in a natural context the model's prior
favours the canonical twin and the variant's logprob collapses.

If fragility is mostly this, then (a) it is reachable in the wild only via
unnormalised text (e.g. NFD from macOS filesystems) or programmatic id
injection, and (b) the geometric predictors that "work" are detecting a
never-emitted OUTPUT row, not a subtle property of healthy tokens. Both matter
for how the result is stated.
"""
from __future__ import annotations
import argparse, gzip, json, unicodedata
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .stats import auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--ext", default="external/allenai_OLMo_2_1124_7B.jsonl.gz")
    ap.add_argument("--pt", default="results/fragility.pt")
    a = ap.parse_args()

    d = torch.load(a.pt, weights_only=False)
    tokens = np.array(d["tokens"]); is_glitch = d["is_glitch"]; single = d["single"]
    M = d["M"]; F = M < d["fail_thr"]; frag = F.mean(1)
    tok = AutoTokenizer.from_pretrained(a.model)
    clean = (single > -0.1) & (is_glitch == 0)

    cat = {}
    for line in gzip.open(a.ext, "rt", encoding="utf-8"):
        r = json.loads(line); cat[int(r["i"])] = r.get("category", "?")

    canon = np.zeros(len(tokens), bool); nfc = np.zeros(len(tokens), bool)
    twin = {}
    for k, t in enumerate(tokens):
        s = tok.decode([int(t)])
        ids = tok(s, add_special_tokens=False)["input_ids"]
        canon[k] = ids == [int(t)]
        nfc[k] = unicodedata.normalize("NFC", s) == s
        if not canon[k]:
            twin[k] = ids
    cats = np.array([cat.get(int(t), "?") for t in tokens])

    ci = np.where(clean)[0]
    frg = frag[ci] >= 0.10
    print(f"clean-looking tokens: {len(ci)}; fragile (frag>=0.10): {frg.sum()}")
    print("\ncanonical (decode->encode round-trips to the same id) among clean-looking:")
    for lab, m in (("fragile", frg), ("not fragile", ~frg)):
        c_ = canon[ci][m]; n_ = nfc[ci][m]
        print(f"  {lab:>12}: canonical {c_.mean():.3f}   NFC-normal {n_.mean():.3f}   n={m.sum()}")
    print("\nMagikarp category among clean-looking:")
    for lab, m in (("fragile", frg), ("not fragile", ~frg)):
        u, n = np.unique(cats[ci][m], return_counts=True)
        print(f"  {lab:>12}: " + ", ".join(f"{x}={y}" for x, y in zip(u, n)))

    print("\nthe fragile clean-looking tokens:")
    for k in ci[frg][np.argsort(-frag[ci][frg])]:
        s = tok.decode([int(tokens[k])])
        tw = " -> re-encodes as " + str([tok.decode([i]) for i in twin[k]]) if k in twin else ""
        print(f"  {s!r:>20} id {int(tokens[k]):6d} frag {frag[k]:.2f} single {single[k]:+.3f} "
              f"cat={cats[k]:<24} canonical={canon[k]}{tw}")

    # does fragility survive restricting to CANONICAL clean-looking tokens?
    cc = ci[canon[ci]]
    fr_c = frag[cc]
    print(f"\ncanonical clean-looking tokens: {len(cc)}; fragile>=0.10: {(fr_c >= 0.10).sum()}; "
          f"any failure: {(fr_c > 0).sum()}; mean frag {fr_c.mean():.4f}")

    # static output-row features, canonical-only, against 'any failure' (more positives)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16)
    E_in = model.get_input_embeddings().weight.detach().float()
    E_out = model.get_output_embeddings().weight.detach().float()
    del model
    sel = torch.tensor(tokens[cc])
    feats = {"cdist_out": (E_out[sel] - E_out.mean(0)).norm(dim=-1).numpy(),
             "norm_out": E_out[sel].norm(dim=-1).numpy(),
             "norm_in": E_in[sel].norm(dim=-1).numpy(),
             "cdist_in": (E_in[sel] - E_in.mean(0)).norm(dim=-1).numpy(),
             "id": tokens[cc].astype(float)}
    for tgt_name, y in (("frag>=0.10", (fr_c >= 0.10).astype(int)),
                        ("any failure", (fr_c > 0).astype(int))):
        if y.sum() < 5 or y.sum() == len(y):
            print(f"\n{tgt_name}: {y.sum()} positives -- too few"); continue
        print(f"\nCANONICAL-ONLY, target {tgt_name} ({y.sum()} pos / {len(y)}):")
        for nm, v in feats.items():
            A = auc(v, y); A = max(A, 1 - A)
            print(f"   {nm:>10}: AUC {A:.3f}")


if __name__ == "__main__":
    main()
