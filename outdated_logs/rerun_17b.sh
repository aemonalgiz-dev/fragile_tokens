#!/bin/bash
# deferred: rerun the 1.7B reasoning stages with the corrected thinking switch and
# the printable glitch-set filter, once the ladder segment has finished and the
# GPU is free. Outputs keep _v2 so the originals remain for comparison.
cd ~/glitch
while ! grep -q "ALL DONE" logs/scale.log; do sleep 30; done
export PATH=$HOME/venv/bin:$PATH
python3 -m src.cut.reasoning_drift --model Qwen/Qwen3-1.7B --pt results/fragility_qwen3_1_7b.pt --out results/reasoning_qwen3_1_7b_v2.json > logs/qwen3_1_7b_reasoning_v2.log 2>&1
python3 -m src.cut.reasoning_drift --model Qwen/Qwen3-1.7B --pt results/fragility_qwen3_1_7b.pt --thinking --max-new 512 --out results/reasoning_qwen3_1_7b_think_v2.json > logs/qwen3_1_7b_reasoning_think_v2.log 2>&1
echo "RERUN DONE $(date -u)" >> logs/rerun_17b.log
