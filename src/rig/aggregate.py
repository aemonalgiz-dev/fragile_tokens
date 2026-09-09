"""Pool the seeded arms and test each intervention against baseline.

Seeds are paired: arm X at seed S and baseline at seed S share the step-256
checkpoint and the data order, so the per-seed difference is the intervention's
effect with initialisation and data-order noise differenced out.

Reports doomed entropy and healthy entropy SEPARATELY, never just their gap.
The `mask` arm on the local rig shrank the gap by making HEALTHY tokens worse,
which the gap alone reports as an improvement. The gap is a gameable metric.
"""
from __future__ import annotations
import argparse, json, glob, os
from pathlib import Path
import numpy as np

from ..cut.stats import paired_boot

ARMS = ["baseline", "reinit", "freeze", "mask", "posthoc_centroid", "posthoc_knn"]


def load(runs):
    data = {}
    for f in sorted(glob.glob(os.path.join(runs, "s*", "*", "summary.json"))):
        p = Path(f)
        seed = int(p.parent.parent.name[1:])
        d = json.load(open(f))
        if "median_entropy_doomed" not in d:
            continue
        data.setdefault(d["arm"], {})[seed] = d
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/rig_aggregate.json")
    a = ap.parse_args()

    data = load(a.runs)
    if "baseline" not in data:
        print("no baseline arms found"); return
    seeds = sorted(data["baseline"])
    print(f"seeds: {seeds}   arms: {[k for k in ARMS if k in data]}\n")

    g = lambda arm, k: np.array([data[arm][s][k] for s in seeds if s in data[arm]])
    base_d = g("baseline", "median_entropy_doomed")
    base_h = g("baseline", "median_entropy_healthy")
    base_v = g("baseline", "val_loss")

    print(f"{'arm':>17} | {'doomed entropy':>20} | {'healthy entropy':>20} | {'val loss':>18}")
    print("-" * 86)
    out = {}
    for arm in ARMS:
        if arm not in data:
            continue
        n = len([s for s in seeds if s in data[arm]])
        d, h, v = g(arm, "median_entropy_doomed"), g(arm, "median_entropy_healthy"), g(arm, "val_loss")
        row = {"n": n, "doomed": d.tolist(), "healthy": h.tolist(), "val": v.tolist()}
        if arm == "baseline":
            print(f"{arm:>17} | {d.mean():8.3f} (ref)      | {h.mean():8.3f} (ref)      | "
                  f"{v.mean():8.4f} (ref)")
        else:
            m = min(len(d), len(base_d))
            dd, ld, hd = paired_boot(d[:m], base_d[:m])
            dh, lh, hh = paired_boot(h[:m], base_h[:m])
            dv, lv, hv = paired_boot(v[:m], base_v[:m])
            row.update({"d_doomed": [dd, ld, hd], "d_healthy": [dh, lh, hh],
                        "d_val": [dv, lv, hv]})
            flag = ""
            if hd < 0 and lh > -99 and dh > 0.05:
                flag = "  <-- HEALTHY DEGRADED"
            print(f"{arm:>17} | {d.mean():8.3f} {dd:+7.3f}     | {h.mean():8.3f} {dh:+7.3f}     | "
                  f"{v.mean():8.4f} {dv:+.4f}{flag}")
        out[arm] = row

    print()
    print("paired vs baseline, 95% CI on the per-seed difference (negative doomed = repair):")
    print(f"{'arm':>17} | {'delta doomed entropy':>30} | {'delta val loss':>24}")
    print("-" * 78)
    for arm in ARMS:
        if arm == "baseline" or arm not in out or "d_doomed" not in out[arm]:
            continue
        dd, ld, hd = out[arm]["d_doomed"]
        dv, lv, hv = out[arm]["d_val"]
        sig = "*" if (ld < 0 and hd < 0) or (ld > 0 and hd > 0) else " "
        print(f"{arm:>17} | {dd:+8.3f} [{ld:+7.3f},{hd:+7.3f}]{sig}  | "
              f"{dv:+8.4f} [{lv:+.4f},{hv:+.4f}]")
    print("\n* = 95% CI excludes zero")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"seeds": seeds, "arms": out}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
