#!/bin/bash
# Small node: Qwen3.8-27B agentic probe in thinking mode (parser-fixed), after the fragile-seeded reasoning runs.
set -u
cd ~/glitch || exit 1
export PATH=$HOME/venv/bin:$PATH TOKENIZERS_PARALLELISM=false GLITCH_PROMPT_STYLE=chat
while ! grep -q "FRAGILE REASONING DONE" logs/scale.log 2>/dev/null; do sleep 60; done
T=qwen38_27b; HF=Qwen/Qwen3.8-27B
echo "#### agent probe thinking, 27B (rerun after stop)  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
python3 -m src.cut.agent_probe --model $HF --pt results/fragility_${T}.pt --thinking --max-new 640 --out results/agent_probe_${T}_think.json > logs/agent_probe_${T}_think_v3.log 2>&1
echo "   agent_probe thinking exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
echo "QWEN38 AGENT THINK DONE $(date -u)" | tee -a logs/scale.log
