"""Post-hoc repair of a finished model -- the bar the early intervention must beat.

This is the GlitchEdit family (USENIX Sec '26): detect glitch tokens in a trained
model, then patch their embeddings by interpolating toward healthy ones.

Two variants:
  centroid -- park doomed rows at the healthy centroid. Deliberately the SAME
              operation the early arm applies at step 256, so "early vs late" is
              a clean one-variable comparison.
  knn      -- interpolate toward the k nearest healthy rows in embedding space,
              which is closer to what GlitchEdit actually does.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import torch

from .train import make_model


def repair(model, counts, thresh, mode, k=8, alpha=1.0, device="cuda"):
    doomed_np = counts < thresh
    doomed = torch.tensor(doomed_np, device=device)
    healthy = ~doomed
    n = int(doomed.sum())
    with torch.no_grad():
        for W in (model.get_input_embeddings().weight,
                  model.get_output_embeddings().weight):
            H = W[healthy].float()
            if mode == "centroid":
                tgt = H.mean(0, keepdim=True).expand(n, -1)
            else:
                D = W[doomed].float()
                Hn = torch.nn.functional.normalize(H, dim=-1)
                Dn = torch.nn.functional.normalize(D, dim=-1)
                idx = (Dn @ Hn.T).topk(k, dim=-1).indices
                tgt = H[idx].mean(1)
            W[doomed] = ((1 - alpha) * W[doomed].float() + alpha * tgt).to(W.dtype)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="baseline")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tok", default="data/tok")
    ap.add_argument("--mode", default="centroid", choices=["centroid", "knn"])
    ap.add_argument("--thresh", type=int, default=100)
    ap.add_argument("--arm", default=None)
    a = ap.parse_args()

    arm = a.arm or f"posthoc_{a.mode}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    counts = np.load(Path(a.tok) / "counts.npy")
    V = len(counts)

    model = make_model(V).to(dev)
    ck = torch.load(Path(a.runs) / a.src / "final.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    n = repair(model, counts, a.thresh, a.mode, device=dev)

    out = Path(a.runs) / arm
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": ck["step"]}, out / "final.pt")

    src_summ = json.load(open(Path(a.runs) / a.src / "summary.json"))
    json.dump({"arm": arm, "val_loss": src_summ["val_loss"], "steps": ck["step"],
               "intervene": f"posthoc_{a.mode}", "thresh": a.thresh,
               "repaired_rows": n, "params": src_summ.get("params"),
               "note": "val_loss inherited from source arm before repair; "
                       "re-measured by evaluate.py if needed"},
              open(out / "summary.json", "w"), indent=1)
    print(f"{arm}: repaired {n} rows from {a.src} -> {out}")


if __name__ == "__main__":
    main()
