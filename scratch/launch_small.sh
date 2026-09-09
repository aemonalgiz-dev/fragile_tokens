#!/bin/bash
# The cheaper single-GPU node (>= 80 GB: the 32B in bf16 needs 64 GB). In this order:
#   1. Qwen3-32B: batch-size-1 re-score of a slice of the STORED matrix (the run Jeff suspects)
#   2. Qwen2.5-7B-Instruct control rung at N_TOK=8400 (family control for the 72B)
#   3. Qwen3-32B fresh full rung at N_TOK=8400 under a new tag (qwen3_32b_r2): reproducibility
#      on the first 4,200 ids, plus more geometry positives; thinking on; no interaction pool
# Everything is resumable: each stage is skipped when its output exists.
set -u
cd ~/glitch || exit 1
export PATH=$HOME/venv/bin:$PATH
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs results results/compact docs
G="-0.693"

echo "#### 1. Qwen3-32B: re-score slice at batch 1  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
if [ -f results/fragility_qwen3_32b.pt ]; then
  [ -f results/rescore_qwen3_32b.json ] || python3 -m src.cut.rescore_slice --model Qwen/Qwen3-32B \
      --pt results/fragility_qwen3_32b.pt --n-fragile 120 --n-clean 120 --n-ref 30 \
      --out results/rescore_qwen3_32b.json > logs/qwen3_32b_rescore.log 2>&1
  echo "   rescore exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
  grep -E "agreement|corr|mean fragility" logs/qwen3_32b_rescore.log | tee -a logs/scale.log
else
  echo "   results/fragility_qwen3_32b.pt missing; rescore skipped" | tee -a logs/scale.log
fi

echo "#### 2. Qwen2.5-7B-Instruct control rung, N_TOK=8400  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
MODELS="Qwen/Qwen2.5-7B-Instruct:qwen25_7b" N_TOK=8400 bash src/cut/run_scale.sh >> logs/scale.log 2>&1
echo "   control rung exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log

echo "#### 3. Qwen3-32B fresh rung (tag qwen3_32b_r2), N_TOK=8400  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
MODELS="Qwen/Qwen3-32B:qwen3_32b_r2:thinking" N_TOK=8400 bash src/cut/run_scale.sh >> logs/scale.log 2>&1
echo "   32B rerun exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
echo "SMALL NODE DONE $(date -u)" | tee -a logs/scale.log
