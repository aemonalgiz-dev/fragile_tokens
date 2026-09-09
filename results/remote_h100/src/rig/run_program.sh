#!/usr/bin/env bash
# Full seeded intervention program.
#
# Every arm within a seed BRANCHES from that seed's own step-256 checkpoint, so
# arms are paired: identical weights AND identical data order up to the moment of
# intervention. Across seeds, both model init and data order change, so a seed is
# a genuinely independent replicate.
#
# The local rig ran every arm at n=1, which cannot separate a real intervention
# effect from one lucky run. This exists to fix that.
set -u
cd "$(dirname "$0")/../.." || exit 1

# single-instance guard: two concurrent copies of this script oversubscribe the
# GPU and every run dies with OOM (learned the hard way).
LOCK=".program.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another run_program.sh holds $LOCK -- refusing to start a second copy"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

SEEDS="${SEEDS:-0 1 2 3 4 5}"
STEPS="${STEPS:-14000}"
PAR="${PAR:-4}"          # ~4.7GB per run; 4 concurrent leaves generous headroom
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="${OUT:-runs}"
LOG="logs"; mkdir -p "$LOG"

run() { echo "=== $* ==="; "$@"; }

# ---- stage 1: baselines (each writes its seed's branch checkpoint) ----
echo "STAGE 1: ${SEEDS} baselines, ${PAR}-way concurrency"
for S in $SEEDS; do
  echo "python3 -m src.rig.train --steps $STEPS --branch-step 256 --seed $S \
        --arm baseline --out $OUT/s$S > $LOG/s${S}_baseline.log 2>&1"
done | xargs -P "$PAR" -I{} bash -c "{}"

# ---- stage 2: training arms, branched ----
echo "STAGE 2: intervention arms"
for S in $SEEDS; do
  B="$OUT/s$S/branch_256.pt"
  echo "python3 -m src.rig.train --steps $STEPS --resume $B --seed $S \
        --intervene reinit --intervene-thresh 100 --arm reinit --out $OUT/s$S \
        > $LOG/s${S}_reinit.log 2>&1"
  echo "python3 -m src.rig.train --steps $STEPS --resume $B --seed $S \
        --intervene reinit --intervene-thresh 100 --freeze-doomed --arm freeze --out $OUT/s$S \
        > $LOG/s${S}_freeze.log 2>&1"
  echo "python3 -m src.rig.train --steps $STEPS --resume $B --seed $S \
        --mask-doomed --arm mask --out $OUT/s$S > $LOG/s${S}_mask.log 2>&1"
done | xargs -P "$PAR" -I{} bash -c "{}"

# ---- stage 3: post-hoc repair (weight surgery, no training) ----
echo "STAGE 3: post-hoc repair"
for S in $SEEDS; do
  for M in centroid knn; do
    run python3 -m src.rig.posthoc --src baseline --mode "$M" --runs "$OUT/s$S" \
        >> "$LOG/s${S}_posthoc.log" 2>&1
  done
done

# ---- stage 4: evaluate every arm ----
echo "STAGE 4: evaluation"
for S in $SEEDS; do
  for A in baseline reinit freeze mask posthoc_centroid posthoc_knn; do
    EXTRA=""; [ "$A" = "mask" ] && EXTRA="--masked"
    echo "python3 -m src.rig.evaluate --arm $A --runs $OUT/s$S $EXTRA \
          > $LOG/s${S}_eval_${A}.log 2>&1"
  done
done | xargs -P 3 -I{} bash -c "{}"

echo "PROGRAM COMPLETE"
