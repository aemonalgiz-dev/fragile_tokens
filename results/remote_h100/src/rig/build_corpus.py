"""Manufacture glitch tokens by construction, with exact ground truth.

The documented root cause of glitch tokens is a mismatch between the corpus the
TOKENIZER was trained on and the corpus the MODEL is trained on: a string appears
often enough in the former to earn a merge, but rarely enough in the latter to
learn an embedding.

So we build that on purpose:
  tokenizer corpus = English + code + German   (a "broad mix" vocabulary)
  model corpus     = English only

Every merge that exists only because of the code/German portion becomes an
under-trained token whose TRUE frequency in the model's data we can count exactly.
That gives real labels instead of a behavioural proxy -- the thing we could never
have with Pythia.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datasets import load_dataset

SOURCES = {
    "en":   ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text"),
    "code": ("codeparrot/codeparrot-clean-valid", None, "train", "content"),
    "de":   ("wikimedia/wikipedia", "20231101.de", "train", "text"),
}


def stream_to(path: Path, key: str, max_mb: float):
    repo, cfg, split, col = SOURCES[key]
    ds = load_dataset(repo, cfg, split=split, streaming=True) if cfg else \
         load_dataset(repo, split=split, streaming=True)
    budget = int(max_mb * 1024 * 1024)
    written = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in ds:
            t = (row[col] or "").strip()
            if len(t) < 40:
                continue
            b = len(t.encode("utf-8")) + 1
            if written + b > budget:
                break
            f.write(t + "\n")
            written += b
    print(f"  {key:5s} -> {path.name:24s} {written/1024/1024:7.1f} MB")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--model-mb", type=float, default=480,
                    help="English text the MODEL trains on")
    ap.add_argument("--tok-en-mb", type=float, default=40)
    ap.add_argument("--tok-code-mb", type=float, default=5)
    ap.add_argument("--tok-de-mb", type=float, default=5)
    a = ap.parse_args()
    out = Path(a.out)

    print("model corpus (English only):")
    stream_to(out / "model_en.txt", "en", a.model_mb)
    print("tokenizer corpus (broad mix -- 80/10/10):")
    stream_to(out / "tok_en.txt", "en", a.tok_en_mb)
    stream_to(out / "tok_code.txt", "code", a.tok_code_mb)
    stream_to(out / "tok_de.txt", "de", a.tok_de_mb)
    print("\ndone")


if __name__ == "__main__":
    main()
