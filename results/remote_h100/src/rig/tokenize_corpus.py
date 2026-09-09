"""One pass over the model corpus: write a uint16 memmap AND the exact per-token
counts. Those counts are the ground truth this whole rig exists to produce --
for every vocabulary entry we know precisely how often the model will see it."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="data/tok")
    ap.add_argument("--text", default="data/model_en.txt")
    ap.add_argument("--out", default="data/tokens.bin")
    ap.add_argument("--chunk", type=int, default=4000)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tok)
    V = len(tok)
    assert V <= 65536, "uint16 memmap requires vocab <= 65536"
    counts = np.zeros(V, dtype=np.int64)
    total = 0
    buf, out = [], open(a.out, "wb")

    def flush(buf):
        nonlocal total
        if not buf:
            return
        for ids in tok(buf, add_special_tokens=False)["input_ids"]:
            arr = np.asarray(ids, dtype=np.uint16)
            arr.tofile(out)
            np.add.at(counts, ids, 1)
            total += len(ids)

    with open(a.text, encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            if len(buf) >= a.chunk:
                flush(buf); buf = []
                print(f"  {total/1e6:7.1f}M tokens", end="\r")
    flush(buf); out.close()

    np.save(Path(a.tok) / "counts.npy", counts)
    z = int((counts == 0).sum())
    print(f"\n\nmodel corpus: {total/1e6:.1f}M tokens, vocab {V}")
    print(f"  zero-count      : {z:6d}  ({100*z/V:5.2f}% of vocab)")
    for thr in [10, 100, 1000, 10000]:
        c = int((counts < thr).sum())
        print(f"  count < {thr:<6d}  : {c:6d}  ({100*c/V:5.2f}%)")
    json.dump({"total_tokens": int(total), "vocab": V, "zero": z},
              open(Path(a.tok) / "corpus_stats.json", "w"), indent=1)
    print(f"saved -> {a.out} and {a.tok}/counts.npy")


if __name__ == "__main__":
    main()
