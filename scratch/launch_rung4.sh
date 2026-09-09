#!/bin/bash
# Replacement 4xH100, in this order:
#   1. FP8 pre-flight on Qwen3-0.6B-FP8
#   2. Qwen2.5-7B-Instruct, full rung at N_TOK=8400  -- the family control for the 72B
#   3. Qwen2.5-72B: batch-size-1 re-score of a slice of the stored matrix (checks the run),
#      then the 72B leftovers (greedy specimens, reasoning with the string metric)
#   4. Qwen3-235B-A22B-FP8, full rung with thinking + interaction pool
# Everything is resumable: each stage is skipped when its output exists.
set -u
cd ~/glitch || exit 1
export PATH=$HOME/venv/bin:$PATH
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs results results/compact docs
G="-0.693"

echo "#### 1. pre-flight: FP8 smoke on Qwen/Qwen3-0.6B-FP8  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
python3 -m src.cut.fragility --model Qwen/Qwen3-0.6B-FP8 --ext "" --n-random 30 --n-glitch 0 \
    --lens 8 --per-len 1 --out results/smoke_fp8.json --pt results/smoke_fp8.pt > logs/smoke_fp8.log 2>&1
echo "   smoke exit $?" | tee -a logs/scale.log; grep -E "saved|Traceback|Error" logs/smoke_fp8.log | tail -2

echo "#### 2. Qwen2.5-7B-Instruct control rung, N_TOK=8400  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
MODELS="Qwen/Qwen2.5-7B-Instruct:qwen25_7b" N_TOK=8400 bash src/cut/run_scale.sh >> logs/scale.log 2>&1
echo "   control rung exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log

echo "#### 3. Qwen2.5-72B: re-score slice at batch 1, then leftovers  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
T=qwen25_72b; HF=Qwen/Qwen2.5-72B-Instruct
if [ -f results/fragility_${T}.pt ]; then
  [ -f results/rescore_${T}.json ] || python3 -m src.cut.rescore_slice --model $HF --pt results/fragility_${T}.pt \
      --n-fragile 120 --n-clean 120 --n-ref 30 --out results/rescore_${T}.json > logs/${T}_rescore.log 2>&1
  echo "   rescore exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log; grep -E "agreement|corr|mean fragility" logs/${T}_rescore.log
  [ -f results/specimens_${T}_greedy.jsonl ] || python3 -m src.cut.specimens --model $HF --ext "" --pt results/fragility_${T}.pt \
      --gate-lp $G --out results/specimens_${T}_greedy.jsonl --md docs/specimens_${T}_greedy.md > logs/${T}_specimens_greedy.log 2>&1
  echo "   greedy specimens exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
  [ -f results/reasoning_${T}_v2.json ] || python3 -m src.cut.reasoning_drift --model $HF --pt results/fragility_${T}.pt \
      --out results/reasoning_${T}_v2.json > logs/${T}_reasoning_v2.log 2>&1
  echo "   reasoning v2 exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
else
  echo "   results/fragility_qwen25_72b.pt not on this node; 72B stage skipped" | tee -a logs/scale.log
fi
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct

echo "#### 4. Qwen3-235B-A22B-FP8 rung, N_TOK=8400  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
MODELS="Qwen/Qwen3-235B-A22B-FP8:qwen3_235b:thinking:screen" N_TOK=8400 KEEP_CACHE=1 \
    bash src/cut/run_scale.sh >> logs/scale.log 2>&1
echo "   235B rung exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
echo "RUNG4 SCRIPT DONE $(date -u)" | tee -a logs/scale.log
