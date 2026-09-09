#!/bin/bash
# Small node: the reasoning prompt (token as the whole user turn) seeded with each model's FRAGILE
# clean-looking tokens, for the models this node can hold. Waits for the agent-probe rerun to finish.
set -u
cd ~/glitch || exit 1
export PATH=$HOME/venv/bin:$PATH TOKENIZERS_PARALLELISM=false HF_XET_HIGH_PERFORMANCE=1
while ! grep -q "QWEN38 AGENT DONE" logs/scale.log 2>/dev/null; do sleep 60; done
echo "#### fragile-seeded reasoning  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
run() {  # tag hf pt [extra flags]
  local T=$1 HF=$2 PT=$3; shift 3
  [ -f results/reasoning_fragile_${T}.json ] && return 0
  local STYLE=raw; case "$T" in qwen38_*) STYLE=chat;; esac
  GLITCH_PROMPT_STYLE=$STYLE python3 -m src.cut.reasoning_drift --model "$HF" --pt "$PT" --fragile "$@" \
      --out results/reasoning_fragile_${T}.json > logs/reasoning_fragile_${T}.log 2>&1
  echo "   reasoning_fragile ${T} exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
}
run qwen3_1_7b Qwen/Qwen3-1.7B results/compact/fragility_qwen3_1_7b_compact.pt
run qwen25_7b Qwen/Qwen2.5-7B-Instruct results/compact/fragility_qwen25_7b_compact.pt
run L allenai/OLMo-2-1124-7B-Instruct results/compact/fragility_L_compact.pt
run qwen38_27b Qwen/Qwen3.8-27B results/compact/fragility_qwen38_27b_compact.pt
run qwen3_32b Qwen/Qwen3-32B results/compact/fragility_qwen3_32b_compact.pt
echo "FRAGILE REASONING DONE $(date -u)" | tee -a logs/scale.log
