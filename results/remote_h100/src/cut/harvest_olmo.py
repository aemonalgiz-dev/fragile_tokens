"""Cross-family verification: does the abandonment signature hold outside Pythia?

Every result so far shares one tokenizer, one corpus (the Pile) and one
architecture family. If the step-32 residual collapse is a Pile artifact, the
central finding is wrong. OLMo is genuinely independent: Dolma corpus, its own
tokenizer, different architecture -- and Magikarp published 438 verified
under-trained tokens for it, twelve times Pythia's 36.

LIMITATION, stated up front: OLMo's earliest public checkpoint is step 1000.
The step-2-to-8 early-detection claim is NOT testable here, because those
checkpoints do not exist. What is testable is the mechanism -- whether doomed
rows show total-magnitude parity while their token-specific residual is
suppressed -- and late-checkpoint detection against external labels.

Checkpoints are purged after use; at ~4.7GB each the full ladder would otherwise
be ~60GB.
"""
from __future__ import annotations
import argparse, gc, json, re, shutil, subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def row_stats(W):
    W = W.float()
    mu = W.mean(0, keepdim=True)
    return (W.norm(dim=-1).cpu(),
            F.cosine_similarity(W, mu, dim=-1).cpu(),
            (W - mu).norm(dim=-1).cpu())


def decompose(W, prev):
    if prev is None:
        z = torch.zeros(W.shape[0]); return z, z, z
    d = (W - prev).float()
    g = d.mean(0); gh = g / (g.norm() + 1e-12)
    sh = d @ gh
    return d.norm(dim=-1).cpu(), sh.cpu(), (d - sh.unsqueeze(1) * gh.unsqueeze(0)).norm(dim=-1).cpu()


def pick_revisions(model, wanted):
    import urllib.request
    url = f"https://huggingface.co/api/models/{model}/refs"
    d = json.load(urllib.request.urlopen(url))
    steps = []
    for b in d.get("branches", []):
        m = re.match(r"^step(\d+)-", b["name"])
        if m:
            steps.append((int(m.group(1)), b["name"]))
    steps.sort()
    out, avail = [], dict(steps)
    for w in wanted:
        best = min(steps, key=lambda s: abs(s[0] - w))
        if best not in out:
            out.append(best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-1B-hf")
    ap.add_argument("--steps", type=int, nargs="*",
                    default=[1000, 2000, 3000, 5000, 10000, 20000, 40000,
                             80000, 160000, 320000, 480000, 640000, 738020])
    ap.add_argument("--out", default="results/traj_olmo1b.pt")
    ap.add_argument("--purge", action="store_true", default=True)
    a = ap.parse_args()

    revs = pick_revisions(a.model, a.steps)
    print(f"{a.model}: {len(revs)} checkpoints")
    for s, r in revs:
        print(f"   step {s:<8} {r}")

    keys = ["unemb_norm", "unemb_cos", "unemb_cnorm", "in_cnorm",
            "d_unemb", "shared_o", "resid_o", "resid_i"]
    rec = {k: [] for k in keys}
    rec["steps"] = []
    prev_o = prev_i = None
    cache = Path.home() / ".cache/huggingface/hub"

    for st, rev in revs:
        m = AutoModelForCausalLM.from_pretrained(a.model, revision=rev,
                                                 dtype=torch.float32)
        Wo = m.get_output_embeddings().weight.detach()
        Wi = m.get_input_embeddings().weight.detach()
        n_o, c_o, cn_o = row_stats(Wo)
        _, _, cn_i = row_stats(Wi)
        d_o, sh_o, rs_o = decompose(Wo, prev_o)
        _, _, rs_i = decompose(Wi, prev_i)
        for k, v in zip(keys, [n_o, c_o, cn_o, cn_i, d_o, sh_o, rs_o, rs_i]):
            rec[k].append(v.clone())
        rec["steps"].append(st)
        prev_o, prev_i = Wo.clone(), Wi.clone()
        print(f"  step {st:<8} |W_out| {n_o.mean():.4f}  centered {cn_o.mean():.4f}  "
              f"delta {d_o.mean():.5f}", flush=True)
        del m
        gc.collect()
        if a.purge:
            d = cache / ("models--" + a.model.replace("/", "--")) / "blobs"
            if d.exists():
                subprocess.run(["find", str(d), "-size", "+100M", "-delete"], check=False)

    for k in keys:
        rec[k] = torch.stack(rec[k])
    rec["model"] = a.model
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, a.out)
    print(f"\nsaved -> {a.out}  shape {tuple(rec['unemb_cnorm'].shape)}")


if __name__ == "__main__":
    main()
