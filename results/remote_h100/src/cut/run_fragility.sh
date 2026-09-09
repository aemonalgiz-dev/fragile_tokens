#!/bin/bash
# Fragility matrix (N tokens x C contexts), then geometric prediction. One GPU job
# at a time under an atomic lock; this box has OOM'd on a double launch before.
set -u
cd "$(dirname "$0")/../.." || exit 1
LOCK=/tmp/glitch_fragility.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "already running (lock $LOCK held); refusing to double-launch"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
mkdir -p logs results
stage () { nm="$1"; shift; out="$1"; shift
  if [ -f "$out" ]; then echo "== $nm already done"; return 0; fi
  echo "== $nm  ($(date -u +%H:%M:%S))"; "$@" > "logs/$nm.log" 2>&1
  rc=$?; echo "   exit $rc  ($(date -u +%H:%M:%S))"; return $rc; }
stage fragility         results/fragility.json         python3 -m src.cut.fragility "$@" || exit 1
stage fragility_predict results/fragility_predict.json python3 -m src.cut.fragility_predict
echo "ALL DONE $(date -u)"
