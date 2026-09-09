"""Compact a results .pt for archiving: drop hidden-state tensors, keep everything else.

The fragility and interaction-screen runs save slot hidden states alongside the
log-probability matrices (gigabytes); the matrices, token lists, banks and every
statistic are a few megabytes. Anything holding more than --max-elems numbers
(tensor, array, or a list/dict of them) is dropped and named in the printout, so
the compact file's contents are explicit. The compacts under results/compact/ for
the 7B were made the same way by hand.

    python -m src.cut.compact_pt results/fragility_qwen3_32b.pt --out results/compact/fragility_qwen3_32b_compact.pt
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import torch


def n_elems(v) -> int:
    if isinstance(v, torch.Tensor):
        return v.numel()
    if isinstance(v, np.ndarray):
        return v.size
    if isinstance(v, dict):
        return sum(n_elems(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return sum(n_elems(x) for x in v) if v and not isinstance(v[0], (int, float, str)) else len(v)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt")
    ap.add_argument("--out")
    ap.add_argument("--max-elems", type=float, default=5e6)
    a = ap.parse_args()
    out = a.out or str(Path("results/compact") / (Path(a.pt).stem + "_compact.pt"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    d = torch.load(a.pt, weights_only=False, map_location="cpu")
    keep, drop = {}, []
    for k, v in d.items():
        n = n_elems(v)
        if n > a.max_elems:
            drop.append((k, n))
        else:
            keep[k] = v
    torch.save(keep, out)
    print(f"{a.pt} ({os.path.getsize(a.pt) / 1e9:.2f} GB) -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print("  kept:", ", ".join(keep))
    print("  dropped:", ", ".join(f"{k} ({n / 1e6:.0f}M elems)" for k, n in drop) or "nothing")


if __name__ == "__main__":
    main()
