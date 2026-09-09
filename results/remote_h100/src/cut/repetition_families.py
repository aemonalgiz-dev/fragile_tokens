"""n-TOKEN glitches: repetition ladders in the vocabulary.

Jeff's reframing, and the data supports it over my earlier "weld" hypothesis.

BPE builds ladders of repeated units -- ' ', '  ', '    ', '        ' -- so the
same base unit occupies several vocabulary slots at different multiplicities.
Two consequences:

  1. Glitchiness varies ALONG a ladder. 'AAAA' can be broken while 'A' and
     'AAAAAAAA' are fine, which no pure frequency account explains.
  2. Writing an arbitrary count requires COMPOSING rungs. 6 copies might tokenize
     as 4+2. That composition is a genuine multi-token object built from
     individually-present units -- the n-token case, with a real mechanism
     (the model must track a count across token boundaries) rather than an
     invented one.

Stage 1 maps the ladders. Stage 2 walks each one, testing copy behaviour as a
function of BOTH the repetition count and the number of tokens it takes to write.
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .verify_candidates import build, TEMPLATES


def find_ladders(tok, vocab_size, max_base=4, min_reps=2):
    """Vocabulary entries that are a base unit repeated k>=2 times."""
    fam = defaultdict(dict)
    for i in range(vocab_size):
        s = tok.decode([i])
        if not s or len(s) < 2:
            continue
        for b in range(1, min(max_base, len(s) // min_reps) + 1):
            base = s[:b]
            if len(s) % b == 0 and base * (len(s) // b) == s:
                fam[base][len(s) // b] = i
                break
    return {b: v for b, v in fam.items() if len(v) >= 2}


@torch.no_grad()
def can_copy(model, tok, device, ids, max_new=None):
    """Majority vote over the 3 templates: can the model reproduce this string?"""
    target = tok.decode(ids)
    ok = 0
    for tname in TEMPLATES:
        demos, tail = TEMPLATES[tname]
        s = ""
        for a, b in demos:
            s += tail.format(t=a) + b + ("\"\n" if tname == "quoted" else "\n")
        head, sep = tail.split("{t}")
        prompt = (tok(s, add_special_tokens=False).input_ids
                  + tok(head, add_special_tokens=False).input_ids + list(ids)
                  + tok(sep, add_special_tokens=False).input_ids)
        p = torch.tensor(prompt, device=device).unsqueeze(0)
        out = model.generate(p, max_new_tokens=max_new or (len(ids) + 4),
                             do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, p.shape[1]:].tolist())
        # raw prefix match, NOT stripped: stripping makes every pure-whitespace
        # target compare equal to the empty string and auto-fail.
        ok += len(target) > 0 and gen.startswith(target)
    return ok >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--max-reps", type=int, default=24)
    ap.add_argument("--families", type=int, default=8)
    ap.add_argument("--out", default="results/repetition_ladders.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
    model.eval()
    V = model.get_input_embeddings().weight.shape[0]

    lad = find_ladders(tok, min(V, len(tok)))
    ranked = sorted(lad.items(), key=lambda kv: -len(kv[1]))
    print(f"repetition ladders found: {len(lad)}")
    for b, v in ranked[:14]:
        print(f"  base={b!r:12s} rungs at k = {sorted(v)}")

    def gaps(v, upto):
        rungs = set(v)
        return sum(1 for m in range(2, upto + 1) if m not in rungs)
    # exclude bases that collide with the prompt templates -- ':' and '"' appear
    # in "Copy:"/"Text:" and quoted-repeat, so those fail for parsing reasons at
    # every m, including m=1, which is a test artifact and not a glitch.
    BAD = set([":", chr(34), chr(10), chr(9)])
    ok_fam = {b: v for b, v in lad.items()
              if len(v) >= 5 and not (set(b) & BAD)}
    gapped = sorted(ok_fam.items(), key=lambda kv: -gaps(kv[1], a.max_reps))
    picked = [b for b, v in gapped[:a.families]]
    print()
    print(f"families selected for GAP-richness (gaps force multi-token writes):")
    for b in picked:
        print(f"  {b!r:10s} rungs<= {a.max_reps}: {sorted(k for k in lad[b] if k <= a.max_reps)}")
    out = {"model": a.model, "ladders": {}, "walks": {}}
    for b, v in ranked:
        out["ladders"][b] = {str(k): int(i) for k, i in v.items()}

    print()
    print("WALKING EACH LADDER: copy success vs repetition count")
    print("(n_tok = how many tokens the tokenizer needs to write base*m)")
    for b in picked:
        print(f"\nbase {b!r}  rungs at k={sorted(lad[b])}")
        print(f"  {'m':>4} | {'n_tok':>5} | {'single?':>7} | {'copies?':>7}")
        rows = []
        for m in range(1, a.max_reps + 1):
            ids = tok(b * m, add_special_tokens=False).input_ids
            if not ids:
                continue
            ok = can_copy(model, tok, dev, ids)
            single = len(ids) == 1
            rows.append({"m": m, "n_tok": len(ids), "single": single, "copies": bool(ok)})
            print(f"  {m:4d} | {len(ids):5d} | {'yes' if single else 'no':>7} | "
                  f"{'OK' if ok else 'FAIL':>7}")
        out["walks"][b] = rows

    # the n-token question: do MULTI-token repetitions fail more than single-token ones?
    print()
    print("=== n-TOKEN EFFECT ===")
    s_ok = s_n = m_ok = m_n = 0
    for b, rows in out["walks"].items():
        for r in rows:
            if r["single"]:
                s_n += 1; s_ok += r["copies"]
            else:
                m_n += 1; m_ok += r["copies"]
    print(f"  single-token repetitions : {s_ok}/{s_n} copy correctly "
          f"({s_ok/max(s_n,1):.0%})")
    print(f"  MULTI-token repetitions  : {m_ok}/{m_n} copy correctly "
          f"({m_ok/max(m_n,1):.0%})")
    out["ntoken_effect"] = {"single_ok": s_ok, "single_n": s_n,
                            "multi_ok": m_ok, "multi_n": m_n}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
