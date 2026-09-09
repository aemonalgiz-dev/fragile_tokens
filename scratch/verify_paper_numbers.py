"""Print, for each number in paper v5 that came from an earlier session, the value the result files hold.
Paper values are typed in from docs/paper_fragile_tokens_v5.md; a mismatch prints MISMATCH."""
import json, os, math, statistics, collections
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np, torch

def J(p):
    return json.load(open(p, encoding="utf-8"))

def check(label, paper, data, tol=0.006):
    try:
        ok = abs(float(paper) - float(data)) <= tol
    except Exception:
        ok = str(paper) == str(data)
    print(f"{'ok      ' if ok else 'MISMATCH'} {label:60s} paper {paper}  data {data}")

def keys(d, depth=0, maxk=40):
    return {k: (type(v).__name__ if isinstance(v, (dict, list)) else v) for k, v in list(d.items())[:maxk]}

print("=== Table 4: fragility on shared ids (recomputed from compact matrices, greedy gate)")
def load(tag):
    d = torch.load(f"results/compact/fragility_{tag}_compact.pt", weights_only=False, map_location="cpu")
    tok = np.asarray(d["tokens"]); s = np.asarray(d["single"], float); M = np.asarray(d["M"], float)
    frag = (M < float(d["fail_thr"])).mean(1); gate = s > math.log(0.5)
    return tok, gate, frag
big = load("qwen3_235b")
for tag, paper_n, paper_small, paper_big in (("qwen3_1_7b", 1592, 2.0, 6.2), ("qwen3_32b", 1569, 0.8, 4.3)):
    t, g, f = load(tag)
    tb, gb, fb = big
    common = np.intersect1d(t[g], tb[gb])
    fs = {int(x): y for x, y in zip(t, f)}; fbig = {int(x): y for x, y in zip(tb, fb)}
    ps = np.mean([fs[i] >= 0.10 for i in common]) * 100; pb = np.mean([fbig[i] >= 0.10 for i in common]) * 100
    check(f"shared ids {tag} x 235B: n", paper_n, len(common), 0)
    check(f"shared ids {tag}: fragile small", paper_small, round(ps, 1), 0.15)
    check(f"shared ids {tag}: fragile 235B", paper_big, round(pb, 1), 0.15)

print("\n=== Table 2 / §3.2: OLMo first bank variance shares, by-length rates, held-out AUCs")
for f in ("results/fragility.json", "results/fragility_L.json"):
    d = J(f); print(f, "keys:", list(d.keys())[:25])
    for k in ("shares", "shares_clean", "fail_by_K", "heldout", "ctx_hostility", "n_clean_look", "frag_summary"):
        if k in d:
            v = d[k]; print("   ", k, json.dumps(v)[:300] if not isinstance(v, list) else (str([round(x, 3) for x in v])[:200] if all(isinstance(x, (int, float)) for x in v) else str(v)[:200]))

print("\n=== §3.3: according split-half, glitch mean fragility")
for f in ("results/gate_L.json",):
    d = J(f); print(f, "fail_p61 subset:", {k: v for k, v in d["fail_p61"].items() if k != "top"}, "| fail_p50 never/half:", d["fail_p50"].get("never"), d["fail_p50"].get("half_corr"))

print("\n=== §4.1 reasoning on OLMo (reasoning_drift.json)")
r = J("results/reasoning_drift.json"); R = r["reasoning"]; rows = R["rows"]
print("   steps:", [(x["step"], round(x["gap"], 2), round(x["pcos_g"], 4), round(x["pcos_h"], 4)) for x in rows][:12])
print("   self (exact id) worst/controls:", R["self"])

print("\n=== §4.3 Pythia identity (verification.json / trajectory)")
for f in ("results/verification.json", "results/trajectory_1.4b.json"):
    if os.path.exists(f):
        d = J(f); print(f, "keys:", list(d.keys())[:20])
        for k in ("layers", "by_layer", "final", "match", "summary", "n_glitch", "n_control"):
            if k in d: print("   ", k, json.dumps(d[k])[:300])

print("\n=== §8.1 segmentation (segmentation.json)")
if os.path.exists("results/segmentation.json"):
    d = J("results/segmentation.json"); print("   keys:", list(d.keys())[:20]); print("   ", json.dumps(d)[:400])

print("\n=== §8.5 rescore and rerun")
for f in ("results/rescore_qwen3_32b.json", "results/rescore_qwen25_72b.json"):
    d = J(f); print(f, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if not isinstance(v, (list, dict))})
print("   32B rerun: fragility corr over shared ids and 12 of 15 -> results/gate_qwen3_32b_r2.json / findings")
for f in ("results/gate_qwen3_32b_r2.json", "results/fragility_qwen3_32b_r2.json"):
    if os.path.exists(f):
        d = J(f); print(f, {k: v for k, v in d.items() if k in ("corr_with_original", "rerun", "n_clean", "N")})

print("\n=== §8.3: 72B worst-by-probe copy in 48 of 80 cells; probe range")
t, g, f = load("qwen25_72b")
d = torch.load("results/compact/fragility_qwen25_72b_compact.pt", weights_only=False, map_location="cpu")
s = np.asarray(d["single"], float); print("   72B single-probe lp of 3 worst:", sorted(s)[:3])

print("\n=== §9 hallucination bridge")
for f in ("results/hallucination_bridge_rescored.json", "results/hallucination_bridge.json"):
    if os.path.exists(f):
        d = J(f); print(f, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if not isinstance(v, (list, dict))})
        break

print("\n=== §6 text: five tokens 47% / 57%, calibration (interaction_screen*.json)")
for f in ("results/interaction_screen.json", "results/interaction_screen_qwen3_32b.json"):
    d = J(f); print(f, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if not isinstance(v, (list, dict)) and any(x in k for x in ("share", "top", "five", "recur", "both", "power", "diag", "n_tok", "n_pool", "n_car"))})
    print("   keys:", [k for k in d.keys()][:40])

print("\n=== Table 12 cell means (interaction_screen_mixed / 32b / 235b)")
for f in ("results/interaction_screen_mixed.json", "results/interaction_screen_qwen3_32b.json", "results/interaction_screen_qwen3_235b.json"):
    d = J(f)
    for k in ("by_type", "cells", "cell_types", "types", "by_cell"):
        if k in d: print(f, k, json.dumps(d[k])[:500])
    for k in ("diag", "diagonal", "diag_by_d", "repetition"):
        if k in d: print(f, k, json.dumps(d[k])[:300])
