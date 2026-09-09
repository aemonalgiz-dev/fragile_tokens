"""Write results/emb_<tag>_sample.npz for the paper's geometry figure: the unembedding rows of the
fragility sample, read from the checkpoint shards (no model instantiation).  Run on the node that
holds the shards, after the compact stage:   python3 scratch/emb_sample.py <tag> <hf id>"""
import sys
import numpy as np
import torch
from src.cut.loadmodel import load_embeddings

tag, name = sys.argv[1], sys.argv[2]
d = torch.load(f"results/compact/fragility_{tag}_compact.pt", weights_only=False)
tok = np.asarray(d["tokens"])
E_in, E_out, gain, tied = load_embeddings(name)
print(tag, "embeddings", tuple(E_in.shape), tuple(E_out.shape), "tied", tied, flush=True)
idx = torch.as_tensor(tok, dtype=torch.long)
Eo_full = E_out * gain.unsqueeze(0)
np.savez_compressed(f"results/emb_{tag}_sample.npz", tokens=tok,
                    E_in=E_in[idx].float().numpy().astype(np.float32),
                    E_out=Eo_full[idx].float().numpy().astype(np.float32),
                    mu_in=E_in.float().mean(0).numpy(), mu_out=Eo_full.float().mean(0).numpy(),
                    single=np.asarray(d["single"], np.float32), M=np.asarray(d["M"], np.float32),
                    fail_thr=float(d["fail_thr"]), tied=bool(tied))
print("saved results/emb_%s_sample.npz" % tag)
