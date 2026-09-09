#!/bin/bash
# Large node (4x H100), after the 235B rung has released the cards:
#   Qwen3.8-Flash-Next-FP8 (512-expert MoE, official FP8, 186 GB), chat framing of the copy prompt,
#   TEACHER-FORCED stages only (fragility, gate, geometry, compacts): under transformers' FP8 MoE
#   path generation runs at seconds per token (measured ~7 s/token on the 235B), so the reasoning,
#   specimen and probe stages are not run for FP8 MoE models on this stack.
set -u
cd ~/glitch || exit 1
export PATH=$HOME/venv/bin:$PATH HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLITCH_PROMPT_STYLE=chat
mkdir -p logs results results/compact docs
while ! grep -q "RUNG4B DONE" logs/scale.log 2>/dev/null; do sleep 60; done
sleep 30
timeout 900 python3 -m pip install -q flash-linear-attention causal-conv1d > logs/kernels_install_big.log 2>&1 || true

T=qwen38_flash; HF=Qwen/Qwen3.8-Flash-Next-FP8
echo "#### 7. Qwen3.8-Flash-Next-FP8 rung, N_TOK=8400, chat framing, teacher-forced stages only  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
MODELS="$HF:$T" N_TOK=8400 KEEP_CACHE=1 SKIP_GEN=1 bash src/cut/run_scale.sh >> logs/scale.log 2>&1
echo "   Flash-Next rung exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
grep -E "filler pool|single probe" logs/${T}_fragility.log | tee -a logs/scale.log
if [ -f results/fragility_${T}.pt ]; then
  # emission specimens without the greedy continuation (one forward per cell): modes, top-5, entropy
  [ -f results/specimens_${T}.jsonl ] || python3 -m src.cut.specimens --model $HF --ext "" --pt results/fragility_${T}.pt --no-continuation \
      --out results/specimens_${T}.jsonl --md docs/specimens_${T}.md > logs/${T}_specimens.log 2>&1
  echo "   specimens (no continuation) exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
  [ -f results/specimens_${T}_greedy.jsonl ] || python3 -m src.cut.specimens --model $HF --ext "" --pt results/fragility_${T}.pt --no-continuation --gate-lp -0.693 \
      --out results/specimens_${T}_greedy.jsonl --md docs/specimens_${T}_greedy.md > logs/${T}_specimens_greedy.log 2>&1
  echo "   greedy specimens (no continuation) exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
fi
echo "QWEN38 FLASH DONE $(date -u)" | tee -a logs/scale.log
