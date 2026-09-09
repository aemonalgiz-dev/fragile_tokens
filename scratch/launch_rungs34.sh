#!/bin/bash
# rungs 3-4 on the 4xH100 node: FP8 pre-flight on a small official FP8 release,
# pre-download of the 235B while the 72B runs, then the ladder at N_TOK=8400.
set -u
cd ~/glitch || exit 1
export PATH=$HOME/venv/bin:$PATH
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs results results/compact docs

echo "#### pre-flight: FP8 load + fragility smoke on Qwen/Qwen3-0.6B-FP8  ($(date -u +%H:%M:%S))"
python3 -m src.cut.fragility --model Qwen/Qwen3-0.6B-FP8 --ext "" --n-random 30 --n-glitch 0 \
    --lens 8 --per-len 1 --out results/smoke_fp8.json --pt results/smoke_fp8.pt > logs/smoke_fp8.log 2>&1
echo "   fragility smoke exit $?"
python3 - > logs/smoke_fp8_embed.log 2>&1 <<'PY'
from src.cut.loadmodel import load_embeddings
E_in, E_out, gain, tied = load_embeddings("Qwen/Qwen3-0.6B-FP8")
print("embeddings read from FP8 shards:", tuple(E_in.shape), tuple(E_out.shape), "gain", tuple(gain.shape), "tied", tied, E_in.dtype)
PY
echo "   embedding-reader smoke exit $?"; tail -1 logs/smoke_fp8_embed.log
grep -E "saved|Traceback|Error" logs/smoke_fp8.log | tail -3

echo "#### pre-download Qwen3-235B-A22B-FP8 in the background  ($(date -u +%H:%M:%S))"
( setsid nohup python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-235B-A22B-FP8', max_workers=16)" > logs/predownload_235b.log 2>&1 & )

echo "#### ladder  ($(date -u +%H:%M:%S))"
MODELS="Qwen/Qwen2.5-72B-Instruct:qwen25_72b Qwen/Qwen3-235B-A22B-FP8:qwen3_235b:thinking:screen" \
N_TOK=8400 bash src/cut/run_scale.sh > logs/scale.log 2>&1
echo "#### ladder finished  ($(date -u +%H:%M:%S))"; tail -3 logs/scale.log
