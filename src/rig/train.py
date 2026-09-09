"""From-scratch training rig with a Pythia-shaped checkpoint ladder.

Architecture is GPTNeoX so the dynamics are directly comparable to the Pythia
measurements -- crucially it has UNTIED input/output embeddings, which the whole
shared-drift-vs-residual analysis depends on.

All intervention arms BRANCH from one saved checkpoint, so arms are identical up
to the moment of intervention. That makes the comparison paired and removes seed
noise between arms entirely.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

LADDER = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000,
          6000, 8000, 10000, 12000, 14000]


def make_model(vocab, hidden=384, layers=6, heads=6, seq=512):
    cfg = GPTNeoXConfig(vocab_size=vocab, hidden_size=hidden,
                        num_hidden_layers=layers, num_attention_heads=heads,
                        intermediate_size=hidden * 4, max_position_embeddings=seq,
                        rotary_pct=0.25, use_parallel_residual=True,
                        tie_word_embeddings=False)
    return GPTNeoXForCausalLM(cfg)


def row_stats(W):
    W = W.float()
    mu = W.mean(0, keepdim=True)
    return (W.norm(dim=-1).cpu(),
            F.cosine_similarity(W, mu, dim=-1).cpu(),
            (W - mu).norm(dim=-1).cpu())


def decompose(W, prev):
    """Split each row's update into the shared drift and the token-specific residual."""
    if prev is None:
        z = torch.zeros(W.shape[0])
        return z, z, z
    d = (W - prev).float()
    g = d.mean(0)
    gh = g / (g.norm() + 1e-12)
    sh = d @ gh
    resid = (d - sh.unsqueeze(1) * gh.unsqueeze(0)).norm(dim=-1)
    return d.norm(dim=-1).cpu(), sh.cpu(), resid.cpu()


class Data:
    def __init__(self, path, seq, holdout=0.01):
        self.a = np.memmap(path, dtype=np.uint16, mode="r")
        n = len(self.a)
        self.split = int(n * (1 - holdout))
        self.seq = seq

    def batch(self, bs, dev, split="train", gen=None):
        if split == "train":
            lo, hi = 0, self.split - self.seq - 1
        else:
            lo, hi = self.split, len(self.a) - self.seq - 1
        ix = torch.randint(lo, hi, (bs,), generator=gen)
        x = torch.stack([torch.from_numpy(self.a[i:i + self.seq].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.a[i + 1:i + 1 + self.seq].astype(np.int64)) for i in ix])
        return x.to(dev), y.to(dev)


def intervene(model, mode, counts, thresh, device):
    """Applied once, at the branch step. Returns a description string."""
    if mode == "none":
        return "no intervention"
    Wi = model.get_input_embeddings().weight
    Wo = model.get_output_embeddings().weight
    doomed = torch.tensor(counts < thresh, device=device)
    healthy = ~doomed
    n = int(doomed.sum())
    with torch.no_grad():
        for W in (Wi, Wo):
            mu = W[healthy].mean(0, keepdim=True)
            if mode == "reinit":
                sd = W[healthy].std(0, keepdim=True)
                W[doomed] = mu + 0.1 * sd * torch.randn_like(W[doomed])
            elif mode == "shrink":
                W[doomed] = 0.5 * W[doomed] + 0.5 * mu
    return f"{mode}: {n} rows adjusted (true count < {thresh})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tokens.bin")
    ap.add_argument("--tok", default="data/tok")
    ap.add_argument("--steps", type=int, default=14000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--branch-step", type=int, default=256)
    ap.add_argument("--resume", default=None, help="branch checkpoint to start from")
    ap.add_argument("--intervene", default="none", choices=["none", "reinit", "shrink"])
    ap.add_argument("--freeze-doomed", action="store_true",
                    help="after intervening, hold the doomed rows fixed for the rest "
                         "of training. Direct causal test: early repair failed, and "
                         "the suspected cause is that shared drift (measured at 13.8x "
                         "stronger for doomed rows) re-corrupts them over the "
                         "remaining steps. If that is the mechanism, freezing fixes it.")
    ap.add_argument("--intervene-thresh", type=int, default=100)
    ap.add_argument("--mask-doomed", action="store_true",
                    help="exclude never-occurring tokens from the output softmax "
                         "from the branch step onward (reclaims the vocabulary slot "
                         "and makes them unemittable). Restricted to count==0 "
                         "tokens: masking a token that DOES occur as a target would "
                         "make its loss infinite.")
    ap.add_argument("--seed", type=int, default=0,
                    help="controls BOTH model init and data order, so a seed "
                         "is a fully independent replicate")
    ap.add_argument("--arm", default="baseline")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    counts = np.load(Path(a.tok) / "counts.npy")
    V = len(counts)
    data = Data(a.data, a.seq)
    outdir = Path(a.out) / a.arm
    outdir.mkdir(parents=True, exist_ok=True)

    model = make_model(V, seq=a.seq).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        start = ck["step"]
        print(f"resumed from {a.resume} at step {start}")
    print(f"arm={a.arm}  params={nparam/1e6:.1f}M  vocab={V}  device={dev}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def sched(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        p = min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))
        return 0.1 + 0.45 * (1 + math.cos(math.pi * p))

    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    if a.resume and a.intervene != "none":
        print("  " + intervene(model, a.intervene, counts, a.intervene_thresh, dev), flush=True)

    frozen = None
    if a.freeze_doomed:
        dm = torch.tensor(counts < a.intervene_thresh, device=dev)
        frozen = (dm,
                  model.get_input_embeddings().weight.detach()[dm].clone(),
                  model.get_output_embeddings().weight.detach()[dm].clone())
        print(f"  freeze: {int(dm.sum())} rows held fixed for the rest of training",
              flush=True)

    logit_mask = None
    if a.mask_doomed:
        never = torch.tensor(counts == 0, device=dev)
        logit_mask = torch.zeros(V, device=dev)
        logit_mask[never] = float("-inf")
        print(f"  mask: {int(never.sum())} never-occurring tokens removed from softmax",
              flush=True)

    keys = ["unemb_norm", "unemb_cos", "unemb_cnorm", "in_cnorm",
            "d_unemb", "shared_o", "resid_o", "resid_i"]
    rec = {k: [] for k in keys}
    rec["steps"] = []
    prev_o = prev_i = None
    gen = torch.Generator().manual_seed(1234 + a.seed)
    t0 = time.time()
    losses = []

    for step in range(start, a.steps + 1):
        if step in LADDER or step == a.steps:
            Wo = model.get_output_embeddings().weight.detach()
            Wi = model.get_input_embeddings().weight.detach()
            n_o, c_o, cn_o = row_stats(Wo)
            _, _, cn_i = row_stats(Wi)
            d_o, sh_o, rs_o = decompose(Wo, prev_o)
            _, _, rs_i = decompose(Wi, prev_i)
            vals = [n_o, c_o, cn_o, cn_i, d_o, sh_o, rs_o, rs_i]
            for k, v in zip(keys, vals):
                rec[k].append(v.clone())
            rec["steps"].append(step)
            prev_o, prev_i = Wo.clone(), Wi.clone()
            if step == a.branch_step and not a.resume:
                torch.save({"model": model.state_dict(), "step": step},
                           Path(a.out) / f"branch_{step}.pt")
                print(f"  saved branch checkpoint at step {step}", flush=True)
        if step == a.steps:
            break

        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        x, y = data.batch(a.batch, dev, "train", gen)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            logits = model(input_ids=x).logits
        flat = logits.view(-1, V).float()
        if logit_mask is not None:
            flat = flat + logit_mask
        loss = F.cross_entropy(flat, y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if frozen is not None:      # undo any update to the frozen rows
            dm, wi, wo = frozen
            with torch.no_grad():
                model.get_input_embeddings().weight[dm] = wi
                model.get_output_embeddings().weight[dm] = wo
        losses.append(loss.item())
        if step % 500 == 0:
            print(f"  step {step:6d}  loss {np.mean(losses[-200:]):6.3f}  "
                  f"{(time.time()-t0)/60:5.1f} min", flush=True)

    model.eval()
    ev = []
    with torch.no_grad():
        g2 = torch.Generator().manual_seed(7)
        for _ in range(40):
            x, y = data.batch(a.batch, dev, "val", g2)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                l = model(input_ids=x).logits
            fl = l.view(-1, V).float()
            if logit_mask is not None:      # evaluate the model as it would deploy
                fl = fl + logit_mask
            ev.append(F.cross_entropy(fl, y.reshape(-1)).item())
    val = float(np.mean(ev))
    print(f"\narm={a.arm}  val loss {val:.4f}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    for k in keys:
        rec[k] = torch.stack(rec[k])
    rec["val_loss"] = val
    rec["arm"] = a.arm
    rec["intervene"] = a.intervene
    torch.save(rec, outdir / "traj.pt")
    torch.save({"model": model.state_dict(), "step": a.steps}, outdir / "final.pt")
    json.dump({"arm": a.arm, "val_loss": val, "steps": a.steps,
               "intervene": a.intervene, "thresh": a.intervene_thresh,
               "params": nparam}, open(outdir / "summary.json", "w"), indent=1)
    print(f"saved -> {outdir}")


if __name__ == "__main__":
    main()
