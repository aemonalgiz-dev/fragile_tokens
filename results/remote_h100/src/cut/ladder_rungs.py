"""Within-ladder variation: the part that survives length matching.

The n-token composition effect did not survive a length-matched control -- copy
failure tracked target length, not token count. But one thing did survive, and it
is length-controlled by construction: on a DENSE ladder every count k is its own
single token of length k, so adjacent rungs differ by exactly one character.
Any flip between k and k+1 cannot be a length effect.

Base '-' on pythia-1.4b: OK at 1-6, FAIL at 7, OK at 8, FAIL at 9-11, OK at 12-15,
FAIL at 16. This measures that properly and asks whether the failures land on
counts that are rare in text -- "round" formatting widths (8, 10, 12, 16, 20)
being common and awkward ones (7, 9, 11, 13) being rare.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .repetition_families import can_copy, find_ladders

ROUND = {2, 4, 5, 8, 10, 12, 15, 16, 20, 24, 25, 30, 32, 40, 48, 50, 60, 64, 80, 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--bases", default="-,=,*,.,#,_")
    ap.add_argument("--max-k", type=int, default=32)
    ap.add_argument("--out", default="results/ladder_rungs.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
    model.eval()
    V = model.get_input_embeddings().weight.shape[0]
    lad = find_ladders(tok, min(V, len(tok)))

    rows = []
    for b in a.bases.split(","):
        if b not in lad:
            print(f"  base {b!r} has no ladder, skipping")
            continue
        ks = sorted(k for k in lad[b] if k <= a.max_k)
        print(f"\nbase {b!r}: single-token rungs at k = {ks}")
        line = []
        for k in ks:
            ids = [lad[b][k]]
            ok = can_copy(model, tok, dev, ids)
            rows.append({"base": b, "k": k, "ok": bool(ok), "round": k in ROUND})
            line.append(f"{k}{'+' if ok else '-'}")
        print("  " + " ".join(line) + "     (+ copies, - fails)")

    print()
    print("=== is failure explained by 'round' repetition counts? ===")
    r_ok = [r for r in rows if r["round"]]
    o_ok = [r for r in rows if not r["round"]]
    print(f"  round counts (2,4,8,10,12,16,20,...) : "
          f"{sum(r['ok'] for r in r_ok)}/{len(r_ok)} copy "
          f"({sum(r['ok'] for r in r_ok)/max(len(r_ok),1):.0%})")
    print(f"  awkward counts (7,9,11,13,...)       : "
          f"{sum(r['ok'] for r in o_ok)}/{len(o_ok)} copy "
          f"({sum(r['ok'] for r in o_ok)/max(len(o_ok),1):.0%})")

    # adjacent-rung flips: the cleanest length control there is
    flips = 0; pairs = 0
    for b in set(r["base"] for r in rows):
        rs = sorted([r for r in rows if r["base"] == b], key=lambda r: r["k"])
        for x, y in zip(rs, rs[1:]):
            if y["k"] == x["k"] + 1:
                pairs += 1
                flips += x["ok"] != y["ok"]
    print(f"\n  adjacent rungs differing by ONE character: {flips}/{pairs} flip verdict "
          f"({flips/max(pairs,1):.0%})")
    print("  (a length account predicts ~0 flips between k and k+1)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "rows": rows,
               "round_ok": sum(r["ok"] for r in r_ok), "round_n": len(r_ok),
               "odd_ok": sum(r["ok"] for r in o_ok), "odd_n": len(o_ok),
               "flips": flips, "pairs": pairs}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
