"""Idea 1: harvest per-token embedding health across pretraining checkpoints.

Streams Pythia revisions in training order, keeping the previous checkpoint's
matrices in memory so we can record PER-ROW UPDATE MAGNITUDE. That last quantity
is the one that adjudicates the open disagreement: "Hub of Short Rows"
(2608.29702) claims the rows that under-trained-token detectors flag *were*
updated during training, i.e. glitch tokens are not simply un-touched rows.

Saves only per-token scalars (6 x |V| floats per checkpoint), not weights.
"""
from __future__ import annotations
import argparse, gc, json, shutil
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# Log-spaced over Pythia's 143k steps, dense early where the action is.
DEFAULT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 143000]


def row_stats(W: torch.Tensor):
    """Raw norm/cos plus MEAN-CENTERED norm.

    Centering matters: LM unembeddings become strongly anisotropic late in
    training (pythia-160m ends with every row at cos=0.947 to the global mean),
    which destroys raw cos-to-mean as an indicator. The centered norm
    ||W_i - mu|| is what actually survives -- it measures how much
    token-specific information the row carries once the shared bias direction
    that every row inherits is removed.
    """
    W = W.float()
    mu = W.mean(dim=0, keepdim=True)
    return (W.norm(dim=-1), F.cosine_similarity(W, mu, dim=-1),
            (W - mu).norm(dim=-1))



def decompose(W: torch.Tensor, prev: torch.Tensor | None):
    """Split each row's update into the SHARED drift and the token-SPECIFIC residual.

    Early in training every embedding row is swept along by a common direction
    (the global mean shift), which is not token-specific learning. Separating it
    matters: if rare tokens keep pace on the shared component but fall behind on
    the residual from the very first steps, then "glitch tokens are updated early"
    is an artifact of global drift, and the real abandonment signal is visible
    much earlier than the raw update magnitude suggests.
    """
    if prev is None:
        z = torch.zeros(W.shape[0])
        return z, z, z
    d = (W - prev).float()
    g = d.mean(dim=0)
    gh = g / (g.norm() + 1e-12)
    shared = d @ gh                                  # signed projection on drift
    resid = (d - shared.unsqueeze(1) * gh.unsqueeze(0)).norm(dim=-1)
    return d.norm(dim=-1), shared, resid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS)
    ap.add_argument("--out", default="results/traj_160m.pt")
    ap.add_argument("--purge", action="store_true",
                    help="delete each checkpoint from the HF cache after use")
    a = ap.parse_args()

    rec = {"model": a.model, "steps": [], "unemb_norm": [], "unemb_cos": [],
           "in_norm": [], "in_cos": [], "d_unemb": [], "d_in": [],
           "unemb_cnorm": [], "in_cnorm": [], "shared_o": [], "resid_o": [],
           "resid_i": []}
    prev_out = prev_in = None

    for st in a.steps:
        rev = f"step{st}"
        m = AutoModelForCausalLM.from_pretrained(a.model, revision=rev,
                                                 dtype=torch.float32)
        Wo = m.get_output_embeddings().weight.detach()
        Wi = m.get_input_embeddings().weight.detach()

        n_o, c_o, cn_o = row_stats(Wo)
        n_i, c_i, cn_i = row_stats(Wi)
        d_o, sh_o, rs_o = decompose(Wo, prev_out)
        d_i, _, rs_i = decompose(Wi, prev_in)

        rec["steps"].append(st)
        for k, v in [("unemb_norm", n_o), ("unemb_cos", c_o), ("in_norm", n_i),
                     ("in_cos", c_i), ("d_unemb", d_o), ("d_in", d_i),
                     ("unemb_cnorm", cn_o), ("in_cnorm", cn_i),
                     ("shared_o", sh_o), ("resid_o", rs_o), ("resid_i", rs_i)]:
            rec[k].append(v.clone())

        prev_out, prev_in = Wo.clone(), Wi.clone()
        print(f"{rev:>12}  |W_out| {n_o.mean():.4f}  cos_mu {c_o.mean():+.4f}  "
              f"centered {cn_o.mean():.4f}  delta {d_o.mean():.5f}")
        del m
        gc.collect()
        if a.purge:
            for p in Path.home().joinpath(".cache/huggingface/hub").glob(
                    f"models--{a.model.replace('/', '--')}/snapshots/*"):
                if p.is_dir() and not any(x.name.startswith("config") for x in p.iterdir()):
                    continue

    for k in ["unemb_norm", "unemb_cos", "in_norm", "in_cos", "d_unemb", "d_in",
              "unemb_cnorm", "in_cnorm", "shared_o", "resid_o", "resid_i"]:
        rec[k] = torch.stack(rec[k])          # (n_ckpt, |V|)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, a.out)
    print(f"\nsaved -> {a.out}   shape {tuple(rec['unemb_cos'].shape)}")


if __name__ == "__main__":
    main()
