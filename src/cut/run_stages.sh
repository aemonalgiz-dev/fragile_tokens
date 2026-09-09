#!/bin/bash
# Stage-relative abandonment across model families.
#
# Sequential, not parallel: each stage loads a 7-13B model and a second job on
# the same GPU is how this box OOM'd before. The mkdir lock is atomic, so a
# second invocation cannot race past it.
#
# Ordered cheapest-and-most-different first, so breadth arrives early: if the
# effect is an OLMo artifact, Amber and Qwen say so before the 13B run starts.
set -u
cd "$(dirname "$0")/../.." || exit 1

LOCK=/tmp/glitch_stages.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "already running (lock $LOCK held); refusing to double-launch"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

mkdir -p logs results

run () {                       # run <tag> <vocab_sample> <stages...>
  tag="$1"; shift
  samp="$1"; shift
  if [ -f "results/sw_${tag}.json" ]; then
    echo "== $tag already done, skipping"; return
  fi
  echo "== $tag  ($(date -u +%H:%M:%S)) : $*"
  python3 -m src.cut.stagewise --stages "$@" \
      --vocab-sample "$samp" --out "results/sw_${tag}.json" \
      > "logs/sw_${tag}.log" 2>&1
  echo "   exit $?  ($(date -u +%H:%M:%S))"
  # keep the disk from filling across ~20 model downloads
  find "$HOME/.cache/huggingface/hub" -size +50M -delete 2>/dev/null
}

# different lineage, small vocab -- cheapest independent replication
run amber 0 LLM360/Amber LLM360/AmberChat LLM360/AmberSafe

# MoE, different vocab
run olmoe 0 allenai/OLMoE-1B-7B-0924 allenai/OLMoE-1B-7B-0924-SFT \
                allenai/OLMoE-1B-7B-0924-Instruct

# the reference series, full 4 stages
run olmo2_1b 0 allenai/OLMo-2-0425-1B allenai/OLMo-2-0425-1B-SFT \
                allenai/OLMo-2-0425-1B-DPO allenai/OLMo-2-0425-1B-Instruct

# Qwen lineage, 152k vocab -- subsampled, rates and AUCs do not need all of it
run qwen25_7b 40000 Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct

# Llama-3.1 lineage (base itself is gated; Tulu-3 exposes SFT/DPO/RLVR)
run tulu3_8b 40000 allenai/Llama-3.1-Tulu-3-8B-SFT \
                allenai/Llama-3.1-Tulu-3-8B-DPO allenai/Llama-3.1-Tulu-3-8B

# scale check within the reference lineage
run olmo2_7b 40000 allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-SFT \
                allenai/OLMo-2-1124-7B-DPO allenai/OLMo-2-1124-7B-Instruct

run qwen3_8b 40000 Qwen/Qwen3-8B-Base Qwen/Qwen3-8B

# TIED embeddings: scope test, not a replication. A tied row gets dense
# softmax-negative gradient every step, which is the sparsity being measured.
run qwen25_1_5b_TIED 40000 Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct

# most expensive, last
run olmo2_13b 40000 allenai/OLMo-2-1124-13B allenai/OLMo-2-1124-13B-SFT \
                allenai/OLMo-2-1124-13B-DPO allenai/OLMo-2-1124-13B-Instruct

echo "ALL DONE $(date -u)"
