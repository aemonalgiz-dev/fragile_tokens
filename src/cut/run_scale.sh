#!/bin/bash
# The scale ladder: does fragility -- and the rest -- hold on larger and more
# capable models? One model at a time under an atomic lock; each stage skipped
# if its output exists, so the ladder resumes after an interruption.
#
#   bash src/cut/run_scale.sh                      # the default ladder
#   MODELS="Qwen/Qwen3-32B:qwen3_32b:thinking" bash src/cut/run_scale.sh
#   N_TOK=8400 MODELS="..." bash src/cut/run_scale.sh   # rungs 3-4 (ids of the 4200 run kept as a subset)
#
# Entry format  <hf id>:<tag>[:thinking][:4bit][:screen]
#   thinking  run the reasoning stage with the model's thinking mode on (as well as off)
#   4bit      load with bitsandbytes NF4 (70B on one 80GB card); a caveat, not a fix
#   screen    also run one interaction-screen pool (N=48) -- expensive, so opt-in
set -u
cd "$(dirname "$0")/../.." || exit 1
LOCK=/tmp/glitch_scale.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running (lock $LOCK)"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
mkdir -p logs results results/compact docs

# Four rungs, one tokenizer (Qwen2.5 and Qwen3 share it), bf16 or official FP8
# throughout -- no int4, so no quantisation control is needed. The OLMo-2-7B
# result already in results/ is the cross-family reference row.
# Qwen3-1.7B is tied (E_out = E_in): its static-geometry tier is reported but
# not held to the bar. Qwen2.5-72B has no official FP8, so it runs in bf16
# (145 GB) on the 4-card node. Qwen3-235B-A22B-FP8 is Qwen's own release.
DEFAULT="Qwen/Qwen3-1.7B:qwen3_1_7b:thinking \
Qwen/Qwen3-32B:qwen3_32b:thinking:screen \
Qwen/Qwen2.5-72B-Instruct:qwen25_72b \
Qwen/Qwen3-235B-A22B-FP8:qwen3_235b:thinking:screen"
MODELS="${MODELS:-$DEFAULT}"
N_TOK="${N_TOK:-4200}"; PER_LEN="${PER_LEN:-6}"
# the calibration-free gate (p_alone > 0.5) that made the 1.7B/32B/7B comparable
GREEDY_LP="-0.693"
# leave the downloaded weights in place after a rung (default deletes that model's cache dir)
KEEP_CACHE="${KEEP_CACHE:-0}"
# SKIP_GEN=1 runs only the teacher-forced stages (fragility, gate, geometry, screen, compacts).
# Needed for FP8 MoE checkpoints under transformers' FP8 path, where generation runs at
# seconds per token (Qwen3-235B-A22B-FP8: ~7 s/token) while batched scoring is fine.
SKIP_GEN="${SKIP_GEN:-0}"

stage () { nm="$1"; shift; out="$1"; shift
  if [ -f "$out" ]; then echo "== $nm already done"; return 0; fi
  echo "== $nm  ($(date -u +%H:%M:%S))"; "$@" > "logs/$nm.log" 2>&1
  rc=$?; echo "   exit $rc  ($(date -u +%H:%M:%S))"; return $rc; }

for entry in $MODELS; do
  IFS=: read -r hf tag f1 f2 f3 <<< "$entry"
  flags=""; think=0; screen=0
  for f in "$f1" "$f2" "$f3"; do
    case "$f" in thinking) think=1;; 4bit) flags="$flags --load-4bit";; screen) screen=1;; esac
  done
  # OLMo-2 family shares its tokenizer with the labelled 7B; everyone else is label-free
  case "$hf" in allenai/OLMo-2-*) ext="external/allenai_OLMo_2_1124_7B.jsonl.gz";; *) ext="";; esac
  echo "#### $tag  ($hf)  thinking=$think screen=$screen flags='$flags' N_TOK=$N_TOK"

  stage "${tag}_fragility" "results/fragility_${tag}.json" python3 -m src.cut.fragility \
      --model "$hf" --ext "$ext" --n-random "$N_TOK" --per-len "$PER_LEN" $flags \
      --out "results/fragility_${tag}.json" --pt "results/fragility_${tag}.pt" || continue
  # the matrix is safe first: compact copy (no hidden states) for the archive
  stage "${tag}_compact" "results/compact/fragility_${tag}_compact.pt" python3 -m src.cut.compact_pt \
      "results/fragility_${tag}.pt" --out "results/compact/fragility_${tag}_compact.pt"
  # cross-model gate and both geometry runs (paper gate, greedy gate); CPU only
  stage "${tag}_gate" "results/gate_${tag}.json" python3 -m src.cut.gate_summary \
      --pt "results/fragility_${tag}.pt" --out "results/gate_${tag}.json"
  stage "${tag}_predict" "results/fragility_predict_${tag}.json" python3 -m src.cut.fragility_predict \
      --model "$hf" --pt "results/fragility_${tag}.pt" --out "results/fragility_predict_${tag}.json"
  stage "${tag}_predict_greedy" "results/fragility_predict_${tag}_greedy.json" python3 -m src.cut.fragility_predict \
      --model "$hf" --pt "results/fragility_${tag}.pt" --gate-lp "$GREEDY_LP" \
      --out "results/fragility_predict_${tag}_greedy.json"
  if [ "$SKIP_GEN" != 1 ]; then
  stage "${tag}_reasoning" "results/reasoning_${tag}.json" python3 -m src.cut.reasoning_drift \
      --model "$hf" --pt "results/fragility_${tag}.pt" $flags --out "results/reasoning_${tag}.json"
  if [ "$think" = 1 ]; then
    stage "${tag}_reasoning_think" "results/reasoning_${tag}_think.json" python3 -m src.cut.reasoning_drift \
        --model "$hf" --pt "results/fragility_${tag}.pt" --thinking --max-new 512 $flags \
        --out "results/reasoning_${tag}_think.json"
  fi
  stage "${tag}_specimens" "results/specimens_${tag}.jsonl" python3 -m src.cut.specimens \
      --model "$hf" --ext "$ext" --pt "results/fragility_${tag}.pt" $flags \
      --out "results/specimens_${tag}.jsonl" --md "docs/specimens_${tag}.md"
  stage "${tag}_specimens_greedy" "results/specimens_${tag}_greedy.jsonl" python3 -m src.cut.specimens \
      --model "$hf" --ext "$ext" --pt "results/fragility_${tag}.pt" --gate-lp "$GREEDY_LP" $flags \
      --out "results/specimens_${tag}_greedy.jsonl" --md "docs/specimens_${tag}_greedy.md"
  else
    echo "== ${tag}: generation stages skipped (SKIP_GEN=1)"
  fi
  if [ "$screen" = 1 ]; then
    stage "${tag}_screen" "results/interaction_screen_${tag}.json" python3 -m src.cut.interaction_screen \
        --model "$hf" --ext "$ext" --n 48 --pool-mode mixed $flags \
        --out "results/interaction_screen_${tag}.json" --pt "results/interaction_screen_${tag}.pt"
    stage "${tag}_screen_compact" "results/compact/interaction_screen_${tag}_compact.pt" python3 -m src.cut.compact_pt \
        "results/interaction_screen_${tag}.pt" --out "results/compact/interaction_screen_${tag}_compact.pt"
  fi
  # keep the disk clear for the next model -- this model's weights only, so a
  # pre-download of the next rung running alongside is left alone
  if [ "$KEEP_CACHE" != 1 ]; then
    rm -rf "$HOME/.cache/huggingface/hub/models--${hf//\//--}" 2>/dev/null
  fi
done
echo "ALL DONE $(date -u)"
