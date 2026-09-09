#!/usr/bin/env bash
# Cross-family harvest. Sequential: each 7B checkpoint is ~27GB and gets purged
# after use, so running these in parallel would thrash disk and bandwidth for no
# gain -- the bottleneck is download, not compute.
#
# Ordered by value:
#   pythia-6.9b   Magikarp labelled THIS EXACT model, so no label transfer needed
#   OLMo-1B       different corpus (Dolma v1.5), labels transfer from OLMo-7B
#   OLMo-2-0425   different data mix again, and has a step-0 checkpoint
#   OLMoE         MoE architecture -- the strongest architectural contrast
#   OLMo-2-1124   Magikarp labelled, largest OLMo-2
#   pythia-410m   cheap extra scale point within a known family
set -u
cd "$(dirname "$0")/../.." || exit 1
LOCK=".families.lock"
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
mkdir -p logs results

PY="0 1 2 4 8 16 32 64 128 256 512 1000 2000 4000 8000 16000 32000 64000 100000 143000"

run () {  # name model out steps [pattern]
  local nm=$1 model=$2 out=$3 steps=$4 pat=${5:-}
  echo "=== $nm  $(date -u +%H:%M) ==="
  local extra=""
  [ -n "$pat" ] && extra="--pattern $pat"
  python3 -m src.cut.harvest_generic --model "$model" --out "$out" \
      --steps $steps $extra > "logs/harvest_${nm}.log" 2>&1
  echo "    exit=$? -> $out"
  df -h / | tail -1
}

run pythia69   EleutherAI/pythia-6.9b        results/traj_pythia6.9b.pt "$PY"
run olmo1b     allenai/OLMo-1B-hf            results/traj_olmo1b.pt \
    "1000 2000 3000 5000 10000 20000 40000 80000 160000 320000 480000 640000 738020"
run olmo2_1b   allenai/OLMo-2-0425-1B        results/traj_olmo2_1b.pt \
    "0 300 10000 20000 30000 50000 100000 200000 400000 800000 1200000 1600000 1907359" '^stage1-'
run olmoe      allenai/OLMoE-1B-7B-0924      results/traj_olmoe.pt \
    "5000 10000 20000 40000 80000 160000 320000 640000 1000000 1220000"
run olmo2_7b   allenai/OLMo-2-1124-7B        results/traj_olmo2_7b.pt \
    "150 1000 5000 10000 50000 100000 200000 400000 600000 928646"
run pythia410  EleutherAI/pythia-410m        results/traj_pythia410m.pt "$PY"

echo "FAMILIES COMPLETE $(date -u)"
