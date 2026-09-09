"""Aggregate the stage-relative abandonment sweep across families.

The headline is turnover: how much of the worst-1% copy-probe set is DIFFERENT
after post-training. The number that decides whether it means anything is the
boundary control -- where the newly-glitched tokens sat in the base model's own
ranking. If their median base rank is near the threshold, the turnover is
tokens jittering across a cut and the claim is empty. If they sat far outside
the tail, post-training genuinely broke them.
"""
from __future__ import annotations
import json
from pathlib import Path

ORDER = ["amber", "olmoe", "olmo2_1b", "olmo2_7b", "tulu3_8b",
         "qwen25_7b", "qwen3_8b", "qwen25_1_5b_TIED", "olmo2_13b"]
LINEAGE = {"amber": "LLaMA-1", "olmoe": "OLMo/MoE", "olmo2_1b": "OLMo-2",
           "olmo2_7b": "OLMo-2", "olmo2_13b": "OLMo-2", "tulu3_8b": "Llama-3.1",
           "qwen25_7b": "Qwen2.5", "qwen3_8b": "Qwen3",
           "qwen25_1_5b_TIED": "Qwen2.5 (TIED)"}


def main():
    rows = []
    for tag in ORDER:
        p = Path(f"results/sw_{tag}.json")
        if not p.exists():
            print(f"(missing {tag})")
            continue
        d = json.load(open(p))
        rb = d.get("rank_based") or []
        final = rb[-1] if rb else {}
        rows.append({"tag": tag, "lin": LINEAGE.get(tag, "?"),
                     "stages": len(d["stages"]), "probed": d.get("n_probed"),
                     "thr": d.get("threshold_rank"),
                     "jac": final.get("jaccard"), "new": final.get("new"),
                     "med": d.get("newly_base_rank_median"),
                     "b5": d.get("newly_beyond_5x"),
                     "cov": d.get("coverage", []),
                     "ex": d.get("newly_examples", [])})

    print("=" * 92)
    print("TURNOVER OF THE WORST-1% SET, base -> final post-trained stage")
    print("=" * 92)
    print(f"{'family':>18} {'lineage':>15} {'st':>3} {'probed':>7} "
          f"{'jaccard':>8} {'new':>6} | {'median base rank':>16} {'>5x thr':>8}")
    print("-" * 92)
    for r in rows:
        med = f"{r['med']}" if r["med"] is not None else "-"
        b5 = f"{r['b5']:.3f}" if r["b5"] is not None else "-"
        jac = f"{r['jac']:.3f}" if r["jac"] is not None else "-"
        print(f"{r['tag']:>18} {r['lin']:>15} {r['stages']:3d} {r['probed']:7d} "
              f"{jac:>8} {r['new']:6d} | {med:>16} {b5:>8}")

    print()
    print("interpretation of the control: threshold sits at rank ~probed/100, so a")
    print("median base rank far above that is real movement, not boundary jitter.")
    print()
    print("=" * 92)
    print("COVERAGE -- rows receiving zero token-specific update in a stage")
    print("=" * 92)
    for r in rows:
        cs = " | ".join(f"{c['transition']}: {c['frac']:.4f}" for c in r["cov"])
        print(f"{r['tag']:>18}  {cs}")

    print()
    print("=" * 92)
    print("WHAT POST-TRAINING BREAKS (newly-glitched examples)")
    print("=" * 92)
    for r in rows:
        print(f"{r['tag']:>18}  {[e[:14] for e in r['ex'][:10]]}")


if __name__ == "__main__":
    main()
