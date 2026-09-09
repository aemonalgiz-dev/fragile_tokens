"""Is the n-token effect real, or just "longer strings are harder"?

The ladder walk found multi-token repetitions copy at 6% vs 30% for single-token
ones. But multi-token cases occur at higher repetition counts, so that gap is
confounded with raw string length: a 24-character target is harder than a
2-character one whatever it contains.

This isolates the variable. For every repetitive target we test a NON-repetitive
target of the same character length, and compare copy rates as a function of
length. If repetition matters, the two curves separate. If they track each other,
the earlier result was length all along and there is no n-token effect.
"""
from __future__ import annotations
import argparse, json, random, string
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .repetition_families import can_copy, find_ladders

ALPHABET = string.ascii_lowercase + string.digits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--max-reps", type=int, default=24)
    ap.add_argument("--families", type=int, default=5)
    ap.add_argument("--reps-per-len", type=int, default=3)
    ap.add_argument("--out", default="results/repetition_control.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
    model.eval()
    V = model.get_input_embeddings().weight.shape[0]
    rng = random.Random(0)

    lad = find_ladders(tok, min(V, len(tok)))
    BAD = set([":", chr(34), chr(10), chr(9)])
    fams = {b: v for b, v in lad.items() if len(v) >= 5 and not (set(b) & BAD)}
    def gaps(v):
        return sum(1 for m in range(2, a.max_reps + 1) if m not in v)
    picked = [b for b, _ in sorted(fams.items(), key=lambda kv: -gaps(kv[1]))[:a.families]]
    print(f"families: {[repr(b) for b in picked]}\n")

    rows = []
    for b in picked:
        for m in range(1, a.max_reps + 1):
            s = b * m
            ids = tok(s, add_special_tokens=False).input_ids
            if not ids:
                continue
            rep_ok = can_copy(model, tok, dev, ids)
            # length-matched non-repetitive controls
            ctrl = []
            for _ in range(a.reps_per_len):
                c = "".join(rng.choice(ALPHABET) for _ in range(len(s)))
                cids = tok(c, add_special_tokens=False).input_ids
                ctrl.append((can_copy(model, tok, dev, cids), len(cids)))
            rows.append({"base": b, "m": m, "chars": len(s), "n_tok": len(ids),
                         "rep_ok": bool(rep_ok),
                         "ctrl_ok": sum(c[0] for c in ctrl) / len(ctrl),
                         "ctrl_ntok": sum(c[1] for c in ctrl) / len(ctrl)})
        print(f"  {b!r} done")

    print()
    print("COPY SUCCESS by target character length")
    print(f"{'chars':>10} | {'n':>4} | {'repetitive':>11} | {'non-repetitive':>15} | {'gap':>7}")
    print("-" * 60)
    bands = [(1, 2), (3, 4), (5, 8), (9, 12), (13, 18), (19, 24), (25, 100)]
    summary = []
    for lo, hi in bands:
        sel = [r for r in rows if lo <= r["chars"] <= hi]
        if not sel:
            continue
        rr = sum(r["rep_ok"] for r in sel) / len(sel)
        cc = sum(r["ctrl_ok"] for r in sel) / len(sel)
        print(f"{lo:4d}-{hi:<5d} | {len(sel):4d} | {rr:10.0%} | {cc:14.0%} | {rr-cc:+6.0%}")
        summary.append({"lo": lo, "hi": hi, "n": len(sel), "rep": rr, "ctrl": cc})

    allrep = sum(r["rep_ok"] for r in rows) / len(rows)
    allctrl = sum(r["ctrl_ok"] for r in rows) / len(rows)
    print(f"\n  overall  repetitive {allrep:.0%}   non-repetitive {allctrl:.0%}   "
          f"gap {allrep-allctrl:+.0%}")

    # and the n-token cut, now WITHIN length bands
    print()
    print("n-TOKEN effect within matched character-length bands (repetitive only):")
    print(f"{'chars':>10} | {'single-tok':>11} | {'multi-tok':>11}")
    print("-" * 38)
    for lo, hi in bands:
        sel = [r for r in rows if lo <= r["chars"] <= hi]
        s1 = [r for r in sel if r["n_tok"] == 1]
        sm = [r for r in sel if r["n_tok"] > 1]
        if not s1 or not sm:
            continue
        print(f"{lo:4d}-{hi:<5d} | {sum(r['rep_ok'] for r in s1)/len(s1):10.0%} "
              f"({len(s1):2d}) | {sum(r['rep_ok'] for r in sm)/len(sm):10.0%} ({len(sm):2d})")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "rows": rows, "bands": summary}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
