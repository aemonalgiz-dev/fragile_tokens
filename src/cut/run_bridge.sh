#!/bin/bash
# Fragility at scale (with every entity's constituent tokens in the matrix), the
# geometry question with ~50 positives, then the hallucination bridge. One GPU
# job at a time under an atomic lock.
set -u
cd "$(dirname "$0")/../.." || exit 1
LOCK=/tmp/glitch_bridge.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "already running (lock $LOCK held); refusing to double-launch"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
mkdir -p logs results
stage () { nm="$1"; shift; out="$1"; shift
  if [ -f "$out" ]; then echo "== $nm already done"; return 0; fi
  echo "== $nm  ($(date -u +%H:%M:%S))"; "$@" > "logs/$nm.log" 2>&1
  rc=$?; echo "   exit $rc  ($(date -u +%H:%M:%S))"; return $rc; }
stage entities    results/entities.json           python3 -m src.cut.entities || exit 1
stage fragility_L results/fragility_L.json        python3 -m src.cut.fragility \
    --n-random 4200 --per-len 6 --extra-tokens results/entities.json \
    --out results/fragility_L.json --pt results/fragility_L.pt || exit 1
stage predict_L   results/fragility_predict_L.json python3 -m src.cut.fragility_predict \
    --pt results/fragility_L.pt --out results/fragility_predict_L.json
stage bridge      results/hallucination_bridge.json python3 -m src.cut.hallucination_bridge
# the specimen store: what was emitted, how confidently, with model provenance
stage specimens   results/specimens_confident_substitution.jsonl python3 -m src.cut.specimens
echo "ALL DONE $(date -u)"
