"""Cross-model summary of the task probe (results/task_probe_<tag>.json) and, when present,
the agent probe (results/agent_probe_<tag>.json).

Every table compares the three token classes side by side, in the order glitch | healthy |
fragile, on identical tasks; the effect of interest is the gap to the healthy controls,
reported with a bootstrap 95% interval over episodes. Writes docs/task_probe.md.
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np

ORDER = ["qwen3_1_7b", "qwen25_7b", "olmo2_7b", "qwen38_27b", "qwen38_27b_think", "qwen3_32b", "qwen3_32b_think", "qwen25_72b", "qwen3_235b", "qwen38_flash"]
NAMES = {"qwen3_1_7b": "Qwen3-1.7B", "qwen25_7b": "Qwen2.5-7B", "olmo2_7b": "OLMo-2-7B", "qwen3_32b": "Qwen3-32B",
         "qwen3_32b_think": "Qwen3-32B, thinking", "qwen25_72b": "Qwen2.5-72B", "qwen3_235b": "Qwen3-235B",
         "qwen38_27b": "Qwen3.8-27B", "qwen38_27b_think": "Qwen3.8-27B, thinking", "qwen38_flash": "Qwen3.8-Flash-Next"}
GROUPS = [("glitch", "glitch"), ("stable", "healthy"), ("fragile", "fragile")]


def rate(rs, key):
    return float(np.mean([bool(r[key]) for r in rs])) if rs else float("nan")


def gap_ci(a, b, n=4000, seed=0):
    """difference of proportions mean(a) - mean(b) with a bootstrap 95% interval"""
    a = np.array([bool(x) for x in a], float); b = np.array([bool(x) for x in b], float)
    if not len(a) or not len(b):
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def fmt_gap(g):
    d, (lo, hi) = g
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]" if d == d else "n/a"


def load_runs(pattern):
    runs = {}
    for p in glob.glob(pattern):
        tag = re.sub(r".*_probe_(.+)\.json$", r"\1", p)
        runs[tag] = json.load(open(p, encoding="utf-8"))
    return [t for t in ORDER if t in runs] + sorted(t for t in runs if t not in ORDER), runs


def three_way(L, tags, runs, measure, label, higher_is_better=True):
    L += [f"### {label}", "", "| model | glitch | healthy | fragile | fragile − healthy | glitch − healthy |", "|---|---|---|---|---|---|"]
    for t in tags:
        rs = runs[t]["records"]
        by = {g: [r for r in rs if r["group"] == g] for g, _ in GROUPS}
        cells = [f"{rate(by[g], measure):.2f}" for g, _ in GROUPS]
        gf = gap_ci([r[measure] for r in by["fragile"]], [r[measure] for r in by["stable"]])
        gg = gap_ci([r[measure] for r in by["glitch"]], [r[measure] for r in by["stable"]])
        L.append(f"| {NAMES.get(t, t)} | " + " | ".join(cells) + f" | {fmt_gap(gf)} | {fmt_gap(gg)} |")
    L.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--md", default="docs/task_probe.md")
    a = ap.parse_args()
    L = ["# Tokens inside tasks: glitch, healthy, fragile", "",
         "Three token classes per model, ten tokens each, on identical tasks: **glitch** (verified on the 7B; label-free on the "
         "others: worst by the single probe and failing in at least 90% of contexts), **healthy** (passes the greedy gate and never "
         "fails in any context; the control), **fragile** (passes the gate, fails in at least 10% of contexts). The token is spliced by "
         "id into the user turn under the model's chat template; greedy decoding. Gaps are differences of proportions with a "
         "bootstrap 95% interval over episodes.", ""]

    tags, runs = load_runs(f"{a.results}/task_probe_*.json")
    if tags:
        L += ["## Simple tasks (five drawn from a bank of ten; the answer must contain the token)", ""]
        three_way(L, tags, runs, "string_in_output", "Token string returned in the answer")
        three_way(L, tags, runs, "task_done", "Task completed regardless of the token")
        L += ["### Returned rate by task, glitch / healthy / fragile", ""]
        tasks = sorted({k for t in tags for k in runs[t]["by_task"]})
        L += ["| task | " + " | ".join(NAMES.get(t, t) for t in tags) + " |", "|---|" + "---|" * len(tags)]
        for task in tasks:
            cells = []
            for t in tags:
                rs = [r for r in runs[t]["records"] if r["task"] == task]
                cells.append(" / ".join(f"{rate([r for r in rs if r['group'] == g], 'string_in_output'):.2f}" for g, _ in GROUPS) if rs else "")
            L.append(f"| {task} | " + " | ".join(cells) + " |")
        L.append("")

    atags, aruns = load_runs(f"{a.results}/agent_probe_*.json")
    if atags:
        L += ["## Agentic tasks with mocked tools (research, investigate, decide)", "",
              "The mocked world knows the token the model was given: a search whose query contains it returns relevant hits; "
              "anything else returns no results. *Fabricated*: the final answer states a version number although no tool call "
              "ever retrieved one.", ""]
        three_way(L, atags, aruns, "first_query_has_token", "First search query contains the token")
        three_way(L, atags, aruns, "any_query_has_token", "Any search query contains the token")
        three_way(L, atags, aruns, "final_mentions_token", "Final answer mentions the token")
        three_way(L, atags, aruns, "finished", "Episode reached a final answer")
        three_way(L, atags, aruns, "fabricated", "Fabricated facts (version claimed, zero retrievals)")
        # looping, computed from the stored transcripts: an identical tool call repeated, and the round cap hit
        for t in atags:
            for r in aruns[t]["records"]:
                calls = [(x["name"], json.dumps(x["args"], sort_keys=True)) for x in r["transcript"] if x.get("role") == "tool"]
                r["repeated_call"] = len(calls) != len(set(calls))
                r["hit_round_cap"] = not r["finished"]
        three_way(L, atags, aruns, "repeated_call", "Repeated an identical tool call (looping)")
        three_way(L, atags, aruns, "hit_round_cap", "Hit the round cap without a final answer")
        L += ["### Tool calls per episode, glitch / healthy / fragile", "", "| model | glitch | healthy | fragile |", "|---|---|---|---|"]
        for t in atags:
            rs = aruns[t]["records"]
            L.append(f"| {NAMES.get(t, t)} | " + " | ".join(f"{np.mean([r['n_tool_calls'] for r in rs if r['group'] == g]):.2f}" for g, _ in GROUPS) + " |")
        L.append("")

    L += ["## What the model wrote where the token belonged (simple tasks)", "",
          "Fragile and glitch tokens that did not come back, with the read-back of the slot where the task has one.", ""]
    for t in tags:
        L += [f"### {NAMES.get(t, t)}", "", "| task | group | token | fragility | wrote instead | output (start) |", "|---|---|---|---|---|---|"]
        for r in runs[t]["records"]:
            if r["group"] in ("fragile", "glitch") and not r["string_in_output"]:
                out = r["output"].replace("\n", " ").replace("|", "\\|")[:110]
                rb = (r["readback"] or "").replace("|", "\\|")[:40]
                L.append(f"| {r['task']} | {r['group']} | `{r['string']!r}` | {r['fragility']:.2f} | `{rb!r}` | {out} |")
        L.append("")
    if atags:
        L += ["## Agentic transcripts: what was searched for, and what was concluded", ""]
        for t in atags:
            L += [f"### {NAMES.get(t, t)}", "", "| task | group | token | first search query | tool calls | world hits | final answer (start) |", "|---|---|---|---|---|---|---|"]
            for r in aruns[t]["records"]:
                if r["group"] in ("fragile", "glitch", "stable"):
                    fin = (r["final"] or "").replace("\n", " ").replace("|", "\\|")[:120]
                    q = (r["first_query"] or "").replace("|", "\\|")[:50]
                    L.append(f"| {r['task']} | {GROUPS[[g for g,_ in GROUPS].index(r['group'])][1]} | `{r['string']!r}` | `{q}` | {r['n_tool_calls']} | {r['world_hits']} | {fin} |")
            L.append("")

    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    open(a.md, "w", encoding="utf-8", newline="\n").write("\n".join(L))
    # console: the three-way tables only
    stop = next((i for i, l in enumerate(L) if l.startswith("## What the model wrote")), len(L))
    print("\n".join(L[:stop]))
    print(f"\nwrote {a.md}")


if __name__ == "__main__":
    main()
