#!/bin/bash
# The node runs the pre-rewrite ladder driver (tarball predated the rewrite), which lacks the
# greedy-gate stages and the compacts. This runs them once the ladder prints ALL DONE:
# 235B CPU stages, 235B greedy specimens (model still cached unless the old driver's
# cleanup removed it -- it re-downloads if so), then the 72B greedy specimens (re-download).
set -u
cd ~/glitch || exit 1
export PATH=$HOME/venv/bin:$PATH
export HF_XET_HIGH_PERFORMANCE=1
G="-0.693"
while ! grep -q "ALL DONE" logs/scale.log 2>/dev/null; do sleep 60; done
echo "ladder done; post-stages start $(date -u)" >> logs/post_ladder.log
run () { nm="$1"; shift; out="$1"; shift
  if [ -f "$out" ]; then echo "== $nm already done" >> logs/post_ladder.log; return 0; fi
  echo "== $nm ($(date -u +%H:%M:%S))" >> logs/post_ladder.log; "$@" > "logs/$nm.log" 2>&1
  echo "   exit $? ($(date -u +%H:%M:%S))" >> logs/post_ladder.log; }

T=qwen3_235b; HF=Qwen/Qwen3-235B-A22B-FP8
run ${T}_compact results/compact/fragility_${T}_compact.pt python3 -m src.cut.compact_pt results/fragility_${T}.pt --out results/compact/fragility_${T}_compact.pt
run ${T}_screen_compact results/compact/interaction_screen_${T}_compact.pt python3 -m src.cut.compact_pt results/interaction_screen_${T}.pt --out results/compact/interaction_screen_${T}_compact.pt
run ${T}_gate results/gate_${T}.json python3 -m src.cut.gate_summary --pt results/fragility_${T}.pt --out results/gate_${T}.json
run ${T}_predict_greedy results/fragility_predict_${T}_greedy.json env GLITCH_FORCE_CPU=1 python3 -m src.cut.fragility_predict --model $HF --pt results/fragility_${T}.pt --gate-lp $G --out results/fragility_predict_${T}_greedy.json
run ${T}_specimens_greedy results/specimens_${T}_greedy.jsonl python3 -m src.cut.specimens --model $HF --ext "" --pt results/fragility_${T}.pt --gate-lp $G --out results/specimens_${T}_greedy.jsonl --md docs/specimens_${T}_greedy.md

T=qwen25_72b; HF=Qwen/Qwen2.5-72B-Instruct
run ${T}_specimens_greedy results/specimens_${T}_greedy.jsonl python3 -m src.cut.specimens --model $HF --ext "" --pt results/fragility_${T}.pt --gate-lp $G --out results/specimens_${T}_greedy.jsonl --md docs/specimens_${T}_greedy.md
# rerun with the patched reasoning_drift (string-level reproduction + all 64 generations saved)
run ${T}_reasoning_v2 results/reasoning_${T}_v2.json python3 -m src.cut.reasoning_drift --model $HF --pt results/fragility_${T}.pt --out results/reasoning_${T}_v2.json
echo "POST LADDER DONE $(date -u)" >> logs/post_ladder.log
