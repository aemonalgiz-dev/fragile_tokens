#!/bin/bash
# once the 72B weights are loaded (its download no longer competes for bandwidth):
# pre-download the 235B FP8 checkpoint, then smoke-test the FP8 MoE forward path on
# Qwen3-30B-A3B-FP8 alongside the running 72B rung (8 GB per card), so a failure
# in the 235B load path is known hours before the ladder reaches it.
set -u
cd ~/glitch || exit 1
export PATH=$HOME/venv/bin:$PATH
export HF_XET_HIGH_PERFORMANCE=1
while ! grep -qE "filler pool|context 1/" logs/qwen25_72b_fragility.log 2>/dev/null; do sleep 30; done
echo "72B loaded; pre-downloading 235B  $(date -u)" >> logs/deferred_235b.log
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-235B-A22B-FP8', max_workers=16)" > logs/predownload_235b.log 2>&1
echo "235B download exit $?  $(date -u)  $(du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-FP8 | cut -f1)" >> logs/deferred_235b.log
python3 -m src.cut.fragility --model Qwen/Qwen3-30B-A3B-FP8 --ext "" --n-random 30 --n-glitch 0 \
    --lens 8 --per-len 1 --out results/smoke_fp8_moe.json --pt results/smoke_fp8_moe.pt > logs/smoke_fp8_moe.log 2>&1
echo "MoE FP8 smoke exit $?  $(date -u)" >> logs/deferred_235b.log
grep -E "saved|Traceback|Error" logs/smoke_fp8_moe.log | tail -2 >> logs/deferred_235b.log
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8
echo "DEFERRED DONE $(date -u)" >> logs/deferred_235b.log
