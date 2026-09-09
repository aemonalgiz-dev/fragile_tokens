"""Train a byte-level BPE on the BROAD mix.

The count array is the ground truth the whole rig exists to produce: for each of
the 16,384 vocabulary entries we know exactly how many times the model will ever
see it. Zero-count and near-zero-count tokens are glitch tokens by construction.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--out", default="data/tok")
    a = ap.parse_args()
    d = Path(a.data)

    files = [str(d / f) for f in ["tok_en.txt", "tok_code.txt", "tok_de.txt"]]
    print("training BPE on:", [Path(f).name for f in files])
    tk = ByteLevelBPETokenizer()
    tk.train(files=files, vocab_size=a.vocab_size, min_frequency=2,
             special_tokens=["<|endoftext|>"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tk.save_model(a.out)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk._tokenizer if hasattr(tk, "_tokenizer") else None,
        unk_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    fast.save_pretrained(a.out)
    print(f"vocab: {fast.vocab_size}  -> {a.out}")



if __name__ == "__main__":
    main()
