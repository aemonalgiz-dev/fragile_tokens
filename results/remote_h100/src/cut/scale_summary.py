"""One table across the scale ladder: does each pre-registered quantity hold?

Reads every results/fragility_<tag>.json, fragility_predict_<tag>.json,
reasoning_<tag>.json (and _think), interaction_screen_<tag>.json that exists,
and prints the numbers the plan (docs/plan_scale.md) fixed in advance, with the
7B reference row first. Runs anywhere; no model needed.
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path


def load(p):
    return json.load(open(p, encoding="utf-8")) if Path(p).exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = a.results
    ref_pred = load(f"{R}/fragility_predict_L_fixed.json") or load(f"{R}/fragility_predict_L.json")
    rows = [("olmo2_7b (reference)", load(f"{R}/fragility_L.json"), ref_pred,
             load(f"{R}/reasoning_drift.json"), None, load(f"{R}/interaction_screen_mixed.json"))]
    for p in sorted(glob.glob(f"{R}/fragility_*.json")):
        tag = re.sub(r".*fragility_(.+)\.json$", r"\1", p)
        if tag in ("L", "predict", "predict_L", "predict_t05") or tag.startswith("predict"):
            continue
        # a _v2 reasoning file (rerun with the corrected thinking switch) supersedes the original
        rows.append((tag, load(p), load(f"{R}/fragility_predict_{tag}.json"),
                     load(f"{R}/reasoning_{tag}_v2.json") or load(f"{R}/reasoning_{tag}.json"),
                     load(f"{R}/reasoning_{tag}_think_v2.json") or load(f"{R}/reasoning_{tag}_think.json"),
                     load(f"{R}/interaction_screen_{tag}.json")))

    # calibration-free gate (p_alone > 0.5) from gate_summary.py, where available
    gates = {"olmo2_7b (reference)": load(f"{R}/gate_L.json")}
    for tag, *_ in rows[1:]:
        gates[tag] = load(f"{R}/gate_{tag}.json")
    greedy = {"olmo2_7b (reference)": load(f"{R}/fragility_predict_L_greedy.json")}
    for tag, *_ in rows[1:]:
        greedy[tag] = load(f"{R}/fragility_predict_{tag}_greedy.json")
    print("=" * 118)
    print("GREEDY GATE (p_alone > 0.5): fragility and geometry comparable across calibrations")
    print("=" * 118)
    print(f"{'model':>22} | {'n clean':>7} {'single p50':>10} {'ctx med':>8} | "
          f"{'frag>=.10 (p<.61)':>17} {'(p<.5)':>7} {'never':>6} {'half-corr':>9} | {'ctx var':>7} | "
          f"{'n pos':>5} {'static':>6} {'surf':>6} {'Δ paired':>20}")
    for tag, *_ in rows:
        g = gates.get(tag)
        if not g:
            print(f"{tag:>22} | {'-':>7}"); continue
        sh = g.get("shares_clean") or {}
        pr = greedy.get(tag)
        if pr:
            st = pr["cv"]["static geometry"]; su = pr["cv"]["surface only"]; vs = st.get("vs_surface")
            geo = (f"{pr['n_fragile']:5d} {st['auc']:6.3f} {su['auc']:6.3f} "
                   + (f"{vs[0]:+.3f} [{vs[1]:+.2f},{vs[2]:+.2f}]" if vs else "   -").rjust(20))
        else:
            geo = f"{'-':>5}"
        print(f"{tag:>22} | {g['n_clean']:7d} {g['single_pct']['50']:10.3f} {g['ctx_median_lp']:8.4f} | "
              f"{g['fail_p61']['ge10']:17.3f} {g['fail_p50']['ge10']:7.3f} {g['fail_p61']['never']:6.3f} "
              f"{g['fail_p61']['half_corr'] if g['fail_p61']['half_corr'] is not None else float('nan'):9.3f} | "
              f"{sh.get('context', float('nan')):7.3f} | {geo}")
    print("(tied-embedding models: the static tier is reported, not held to the Δ>0 bar)")
    print()
    print("=" * 118)
    print("SCALE LADDER  -- pre-registered quantities (docs/plan_scale.md)  [absolute gate lp > -0.1]")
    print("=" * 118)
    hdr = (f"{'model':>22} | {'params':>6} {'tied':>5} | {'frag>=.10':>9} {'ctx var':>7} | "
           f"{'static':>6} {'surf':>6} {'Δ paired':>18} | {'repro g/h':>10} {'think g/h':>10} | {'I ww':>8} {'sig':>4}")
    print(hdr); print("-" * len(hdr))
    for tag, fr, pr, rs, rt, sc in rows:
        mi = (fr or {}).get("model_info", {}) if fr else {}
        params = "" ; tied = str(mi.get("tie_word_embeddings", ""))[:5]
        frag = f"{fr['frag_summary']['clean_looking']['ge10']:.3f}" if fr and fr.get("frag_summary", {}).get("clean_looking") else "  -"
        ctx = f"{fr['shares']['clean_logprob']['context']:.3f}" if fr else "  -"
        if pr:
            st = pr["cv"]["static geometry"]; su = pr["cv"]["surface only"]
            vs = st.get("vs_surface")
            static = f"{st['auc']:.3f}"; surf = f"{su['auc']:.3f}"
            delta = f"{vs[0]:+.3f} [{vs[1]:+.2f},{vs[2]:+.2f}]" if vs else "   -"
        else:
            static = surf = "  -"; delta = "   -"
        repro = f"{rs['reasoning']['self'][0]:.2f}/{rs['reasoning']['self'][1]:.2f}" if rs else "  -"
        think = f"{rt['reasoning']['self'][0]:.2f}/{rt['reasoning']['self'][1]:.2f}" if rt else "  -"
        if sc:
            ct = sc.get("celltype", {}).get("weakxweak")
            iww = f"{ct['mean_I']:+.4f}" if ct else "  -"
            sig = str(sc.get("n_sig_offdiag", "-"))
        else:
            iww = "  -"; sig = "-"
        print(f"{tag:>22} | {params:>6} {tied:>5} | {frag:>9} {ctx:>7} | {static:>6} {surf:>6} "
              f"{delta:>18} | {repro:>10} {think:>10} | {iww:>8} {sig:>4}")
    print()
    print("verdict per column (from the plan): frag>=.10 in 0.5-3%; ctx var < 0.01; static > surf with")
    print("paired Δ excluding 0; reasoning reproduction glitch << healthy; I(weak x weak) within ±0.01")
    print("and sig pairs <= matched null. Where a column is '-', that stage has not run for the model.")
    print("\nfragile tokens shared across models sharing a tokenizer, and top fragile per model:")
    for tag, fr, *_ in rows:
        if fr and fr.get("top_fragile_clean"):
            print(f"  {tag:>22}: " + ", ".join(f"{t['tok']!r}({t['frag']:.2f})"
                                               for t in fr["top_fragile_clean"][:6]))


if __name__ == "__main__":
    main()
