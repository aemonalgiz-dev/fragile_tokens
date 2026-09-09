#!/bin/bash
# after the deferred 1.7B reruns: specimen stores under the greedy gate (p_alone > 0.5),
# so the 32B -- where the absolute -0.1 gate admits no fragile token -- has a fragile-clean group.
cd ~/glitch
while ! grep -q "RERUN DONE" logs/rerun_17b.log 2>/dev/null; do sleep 20; done
export PATH=$HOME/venv/bin:$PATH
python3 -m src.cut.specimens --model Qwen/Qwen3-1.7B --ext "" --pt results/fragility_qwen3_1_7b.pt --gate-lp -0.693 \
  --out results/specimens_qwen3_1_7b_greedy.jsonl --md docs/specimens_qwen3_1_7b_greedy.md > logs/qwen3_1_7b_specimens_greedy.log 2>&1
python3 -m src.cut.specimens --model Qwen/Qwen3-32B --ext "" --pt results/fragility_qwen3_32b.pt --gate-lp -0.693 \
  --out results/specimens_qwen3_32b_greedy.jsonl --md docs/specimens_qwen3_32b_greedy.md > logs/qwen3_32b_specimens_greedy.log 2>&1
echo "GREEDY SPECIMENS DONE $(date -u)" >> logs/greedy_specimens.log
