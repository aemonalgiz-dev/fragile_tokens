#!/bin/bash
# Large node (4x H100): Qwen3.8-Flash-Next-FP8, chat framing, teacher-forced stages only (SKIP_GEN=1).
# v4: the v1 load died of CUDA OOM on GPU 1 during transformers' fused-expert conversion (device_map=auto
# packed one card to 79 GB; the "MergeModulelist ... Ckpt contains: 1" line was the downstream symptom).
# Cap every card so the conversion has headroom: 186 GB of FP8 weights over 4 x 56 GiB = 224 GiB.
set -u
cd ~/glitch || exit 1
export PATH=$HOME/venv/bin:$PATH HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLITCH_PROMPT_STYLE=chat GLITCH_MAX_GPU_MEM=56GiB
mkdir -p logs results results/compact docs
# fla is pure Triton; causal-conv1d needs a source build that failed, and the torch fallback is fine for teacher forcing
timeout 600 python3 -m pip install -q flash-linear-attention > logs/fla_install_big.log 2>&1 && echo "fla installed" || echo "fla install failed, torch fallback"
T=qwen38_flash; HF=Qwen/Qwen3.8-Flash-Next-FP8
rm -f results/fragility_${T}.pt results/fragility_${T}.json
echo "#### 7c. Qwen3.8-Flash-Next-FP8 rung (v5, 56GiB/card cap + fp8 embedding rescale), N_TOK=8400, chat framing, teacher-forced only  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
MODELS="$HF:$T" N_TOK=8400 KEEP_CACHE=1 SKIP_GEN=1 bash src/cut/run_scale.sh >> logs/scale.log 2>&1
echo "   Flash-Next rung exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
grep -E "fp8 embedding|filler pool|single probe" logs/${T}_fragility.log | tee -a logs/scale.log
if [ -f results/fragility_${T}.pt ]; then
  [ -f results/specimens_${T}.jsonl ] || python3 -m src.cut.specimens --model $HF --ext "" --pt results/fragility_${T}.pt --no-continuation \
      --out results/specimens_${T}.jsonl --md docs/specimens_${T}.md > logs/${T}_specimens.log 2>&1
  echo "   specimens (no continuation) exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
  [ -f results/specimens_${T}_greedy.jsonl ] || python3 -m src.cut.specimens --model $HF --ext "" --pt results/fragility_${T}.pt --no-continuation --gate-lp -0.693 \
      --out results/specimens_${T}_greedy.jsonl --md docs/specimens_${T}_greedy.md > logs/${T}_specimens_greedy.log 2>&1
  echo "   greedy specimens (no continuation) exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
fi
echo "QWEN38 FLASH DONE3 $(date -u)" | tee -a logs/scale.log
