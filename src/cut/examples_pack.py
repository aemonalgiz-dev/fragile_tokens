"""Collect concrete specimens for the paper: named tokens, real generations,
per-token trajectories. Aggregates alone do not let a reader see the phenomenon.
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cut.behavior_battery import entropy_score
from src.cut.verify_candidates import TEMPLATES


@torch.no_grad()
def gen_copy(model, tok, dev, t, template="fewshot"):
    demos, tail = TEMPLATES[template]
    s = ""
    for a, b in demos:
        s += tail.format(t=a) + b + "\n"
    head, sep = tail.split("{t}")
    prompt = (tok(s, add_special_tokens=False).input_ids
              + tok(head, add_special_tokens=False).input_ids + [t]
              + tok(sep, add_special_tokens=False).input_ids)
    p = torch.tensor(prompt, device=dev).unsqueeze(0)
    out = model.generate(p, max_new_tokens=6, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, p.shape[1]:].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--traj", default="results/traj_1.4b.pt")
    ap.add_argument("--ext", default="external/pythia_6_9b.jsonl.gz")
    ap.add_argument("--out", default="results/examples.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    V = model.get_input_embeddings().weight.shape[0]

    ver = np.zeros(V, int); cat = {}; dec = {}
    for l in gzip.open(a.ext, "rt", encoding="utf-8"):
        r = json.loads(l); i = int(r["i"])
        if i < V:
            cat[i] = r.get("category"); dec[i] = r.get("decoded")
            if "verified" in r.get("magikarp", ""): ver[i] = 1

    tr = torch.load(a.traj, weights_only=False)
    n = tr["unemb_cnorm"].shape[1]
    i2, i64 = tr["steps"].index(2), tr["steps"].index(64)
    r2 = tr["resid_i"][i2].numpy(); r64 = tr["resid_i"][i64].numpy()

    rng = np.random.default_rng(0)
    abandoned = [i for i in range(V) if ver[i] == 1 and dec.get(i)
                 and dec[i][:1].isalpha() and not dec[i].startswith(" ")][:10]
    stillborn = [i for i in range(V) if cat.get(i) == "UNREACHABLE_MULTI_TOKEN"][:10]
    healthy = [int(x) for x in rng.choice(
        [i for i in range(V) if cat.get(i) == "OK" and ver[i] == 0], 10, replace=False)]

    out = {}
    print("=" * 74)
    print("SPECIMENS: token | entropy | zero-update@2 | zero-update@64 | copy output")
    print("=" * 74)
    for gname, ids in [("ABANDONED (verified glitch, bare fragments)", abandoned),
                       ("STILLBORN (unreachable multi-token)", stillborn),
                       ("HEALTHY (random OK)", healthy)]:
        ent = entropy_score(model, tok, dev, ids)
        print(f"\n--- {gname} ---")
        rows = []
        for t, e in zip(ids, ent):
            g = gen_copy(model, tok, dev, t)
            z2 = "yes" if r2[t] < 1e-8 else "no"
            z64 = "yes" if r64[t] < 1e-8 else "no"
            print(f"  {dec.get(t)!r:22s} H={e:5.2f}  zero@2={z2:3s} @64={z64:3s}  "
                  f"copy-> {g[:30]!r}")
            rows.append({"id": int(t), "tok": dec.get(t), "entropy": float(e),
                         "zero2": z2, "zero64": z64, "gen": g})
        out[gname] = rows

    # per-token trajectory case study
    print()
    print("=" * 74)
    print("TRAJECTORY CASE STUDY: glitch/healthy ratio is an aggregate; these are")
    print("the raw per-token update magnitudes for three named tokens.")
    print("=" * 74)
    picks = [("abandoned", abandoned[0]), ("stillborn", stillborn[0]),
             ("healthy", healthy[0])]
    print(f"{'step':>8} | " + " | ".join(
        f"{k}:{dec.get(t)!r}"[:26].ljust(26) for k, t in picks))
    print("-" * 92)
    traj = {}
    for si, st in enumerate(tr["steps"]):
        if st not in (2, 8, 32, 64, 256, 1000, 16000, 143000):
            continue
        cells = []
        for k, t in picks:
            tot = tr["d_unemb"][si].numpy()[t]
            res = tr["resid_o"][si].numpy()[t]
            cells.append(f"tot {tot:6.3f} res {res:6.3f}".ljust(26))
            traj.setdefault(k, []).append({"step": st, "total": float(tot),
                                           "resid": float(res)})
        print(f"{st:8d} | " + " | ".join(cells))
    out["trajectory"] = {"picks": {k: dec.get(t) for k, t in picks}, "rows": traj}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
