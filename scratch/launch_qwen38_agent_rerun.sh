#!/bin/bash
# Small node: rerun the Qwen3.8-27B agent probes (plain, thinking) after the tool-call parser fix
# (Qwen3.8 emits <function=...><parameter=...> calls; the first run parsed none). Waits for the
# first launcher's marker so the two never share the card.
set -u
cd ~/glitch || exit 1
export PATH=$HOME/venv/bin:$PATH TOKENIZERS_PARALLELISM=false GLITCH_PROMPT_STYLE=chat
while ! grep -q "QWEN38 CHAT DONE" logs/scale.log 2>/dev/null; do sleep 30; done
T=qwen38_27b; HF=Qwen/Qwen3.8-27B
mkdir -p results/agent_v1
mv -f results/agent_probe_${T}.json results/agent_v1/ 2>/dev/null; mv -f results/agent_probe_${T}_think.json results/agent_v1/ 2>/dev/null
echo "#### agent probe rerun (parser fix)  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
python3 -m src.cut.agent_probe --model $HF --pt results/fragility_${T}.pt --out results/agent_probe_${T}.json > logs/agent_probe_${T}_v2.log 2>&1
echo "   agent_probe exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
python3 -m src.cut.agent_probe --model $HF --pt results/fragility_${T}.pt --thinking --max-new 640 --out results/agent_probe_${T}_think.json > logs/agent_probe_${T}_think_v2.log 2>&1
echo "   agent_probe thinking exit $?  ($(date -u +%H:%M:%S))" | tee -a logs/scale.log
echo "QWEN38 AGENT DONE $(date -u)" | tee -a logs/scale.log
