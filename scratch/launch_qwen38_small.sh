#!/bin/bash
# Small node (1x H100 80GB): Qwen3.8-27B in bf16 (56 GB), CHAT framing of the copy prompt
# (GLITCH_PROMPT_STYLE=chat: instruction + demos + "Text:" in the user turn, "Copy:" in the
# assistant turn). Under the raw framing this family answers "Copy:" with a blank line and
# drops the leading space of the copied token (filler pool 3 of ~2,000).
#   1. Qwen3.8-27B: random-context pipeline via the ladder driver (fragility, gate, geometry,
#      reasoning plain + thinking, specimens, compacts), N_TOK=8400
#   2. simple-task probe, agentic probe plain, agentic probe thinking
set -u
cd ~/glitch || exit 1
export PATH=$HOME/venv/bin:$PATH HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLITCH_PROMPT_STYLE=chat
mkdir -p logs results results/compact docs
rm -f /tmp/glitch_scale.lock 2>/dev/null; rmdir /tmp/glitch_scale.lock 2>/dev/null

echo "#### Qwen3.8-27B rung, N_TOK=8400, prompt framing = chat  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
MODELS="Qwen/Qwen3.8-27B:qwen38_27b:thinking" N_TOK=8400 KEEP_CACHE=1 bash src/cut/run_scale.sh >> logs/scale.log 2>&1
echo "   27B rung exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log

if [ -f results/fragility_qwen38_27b.pt ]; then
  echo "#### probes  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
  T=qwen38_27b; HF=Qwen/Qwen3.8-27B
  [ -f results/task_probe_${T}.json ] || python3 -m src.cut.task_probe --model $HF --pt results/fragility_${T}.pt \
      --out results/task_probe_${T}.json > logs/task_probe_${T}.log 2>&1
  echo "   task_probe exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
  [ -f results/agent_probe_${T}.json ] || python3 -m src.cut.agent_probe --model $HF --pt results/fragility_${T}.pt \
      --out results/agent_probe_${T}.json > logs/agent_probe_${T}.log 2>&1
  echo "   agent_probe exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
  [ -f results/agent_probe_${T}_think.json ] || python3 -m src.cut.agent_probe --model $HF --pt results/fragility_${T}.pt \
      --thinking --max-new 640 --out results/agent_probe_${T}_think.json > logs/agent_probe_${T}_think.log 2>&1
  echo "   agent_probe thinking exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
fi
echo "QWEN38 CHAT DONE $(date -u)" | tee -a logs/scale.log
