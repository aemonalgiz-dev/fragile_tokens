"""Cross-family trajectory harvest, with geometric structure.

Every observational result so far shares one tokenizer, one corpus (the Pile) and
one architecture family. If the step-32 residual collapse is a Pile artifact the
central claim is wrong, so this runs the same measurement across families.

Beyond the scalar stats, this records TWO GEOMETRIC quantities per token:

  hubness (k-occurrence N_k)
      How often a token appears in other tokens' k-nearest-neighbour lists. In
      high dimensions a few points become "hubs" that are near everything and
      therefore distinguish nothing. This is the quantity our intervention result
      implicitly measured: parking a row at the healthy centroid made it WORSE
      than leaving it alone, and "maximally confusable" is precisely a hub. It
      also connects to Hub of Short Rows (2608.29702), which found a hub of short
      rows near the origin inflating intrinsic-dimension estimates.

  local intrinsic dimension (TwoNN, per token)
      mu = r2/r1 over the two nearest neighbours; ID is estimated from the
      distribution of log mu. Computed per-token here so it can be compared
      between glitch and healthy rows rather than reported for the table as a
      whole.

Both are computed on the GPU in chunks -- a full 50k x 50k similarity matrix
never exists at once.
"""
from __future__ import annotations
import argparse, gc, json, re, subprocess, urllib.request
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


@torch.no_grad()
def geometry(W, k=10, chunk=2048, device="cuda"):
    """k-occurrence (hubness) and per-token TwoNN local intrinsic dimension."""
    X = F.normalize(W.float().to(device), dim=-1)
    V = X.shape[0]
    occ = torch.zeros(V, device=device)
    r1 = torch.zeros(V, device=device)
    r2 = torch.zeros(V, device=device)
    for s in range(0, V, chunk):
        sim = X[s:s + chunk] @ X.T                      # cosine similarity
        idx = torch.arange(s, min(s + chunk, V), device=device)
        sim[torch.arange(len(idx), device=device), idx] = -2.0   # exclude self
        top = sim.topk(k, dim=-1)
        occ.scatter_add_(0, top.indices.reshape(-1),
                         torch.ones(top.indices.numel(), device=device))
        # distances from cosine similarity, for TwoNN
        d = (2 - 2 * top.values[:, :2]).clamp(min=1e-12).sqrt()
        r1[idx] = d[:, 0]
        r2[idx] = d[:, 1]
    mu = (r2 / r1.clamp(min=1e-12)).clamp(min=1 + 1e-9)
    return occ.cpu(), mu.log().cpu()


def pick_revisions(model, wanted, pattern):
    d = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/models/{model}/refs", timeout=40))
    steps = []
    for b in d.get("branches", []):
        if pattern and not re.match(pattern, b["name"]):
            continue
        m = re.search(r"step[-_]?(\d+)", b["name"], re.I)
        if m:
            steps.append((int(m.group(1)), b["name"]))
    steps.sort()
    out = []
    for w in wanted:
        best = min(steps, key=lambda s: abs(s[0] - w))
        if best not in out:
            out.append(best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, nargs="*",
                    default=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000,
                             2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000])
    ap.add_argument("--pattern", default=None,
                    help="regex a branch name must match (e.g. '^stage1-' for OLMo-2)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-purge", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    revs = pick_revisions(a.model, a.steps, a.pattern)
    print(f"{a.model}: {len(revs)} checkpoints", flush=True)
    for s, r in revs:
        print(f"   step {s:<9} {r}", flush=True)

    keys = ["unemb_norm", "unemb_cos", "unemb_cnorm", "in_cnorm", "d_unemb",
            "shared_o", "resid_o", "resid_i", "hub_o", "logmu_o", "hub_i"]
    rec = {k: [] for k in keys}
    rec["steps"] = []
    prev_o = prev_i = None
    cache = Path.home() / ".cache/huggingface/hub"

    for st, rev in revs:
        try:
            m = AutoModelForCausalLM.from_pretrained(a.model, revision=rev,
                                                     dtype=torch.float32)
        except Exception as e:
            print(f"  step {st}: LOAD FAILED {type(e).__name__} {str(e)[:80]}", flush=True)
            continue
        Wo = m.get_output_embeddings().weight.detach()
        Wi = m.get_input_embeddings().weight.detach()
        n_o, c_o, cn_o = row_stats(Wo)
        _, _, cn_i = row_stats(Wi)
        d_o, sh_o, rs_o = decompose(Wo, prev_o)
        _, _, rs_i = decompose(Wi, prev_i)
        hub_o, lmu_o = geometry(Wo, a.k, device=dev)
        hub_i, _ = geometry(Wi, a.k, device=dev)
        for k, v in zip(keys, [n_o, c_o, cn_o, cn_i, d_o, sh_o, rs_o, rs_i,
                               hub_o, lmu_o, hub_i]):
            rec[k].append(v.clone())
        rec["steps"].append(st)
        prev_o, prev_i = Wo.clone(), Wi.clone()
        print(f"  step {st:<9} |W| {n_o.mean():.4f} centered {cn_o.mean():.4f} "
              f"delta {d_o.mean():.5f} hub_max {hub_o.max():.0f}", flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()
        if not a.no_purge:
            blobs = cache / ("models--" + a.model.replace("/", "--")) / "blobs"
            if blobs.exists():
                subprocess.run(["find", str(blobs), "-size", "+50M", "-delete"], check=False)

    if not rec["steps"]:
        print("no checkpoints harvested"); return
    for k in keys:
        rec[k] = torch.stack(rec[k])
    rec["model"] = a.model
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, a.out)
    print(f"\nsaved -> {a.out}  shape {tuple(rec['unemb_cnorm'].shape)}", flush=True)


if __name__ == "__main__":
    main()
