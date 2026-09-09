#!/bin/bash
# Non-contiguous interaction: screen -> confirm -> geometry, one GPU job at a time.
# Atomic mkdir lock: this box has OOM'd on a double launch before.
set -u
cd "$(dirname "$0")/../.." || exit 1
LOCK=/tmp/glitch_interaction.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "already running (lock $LOCK held); refusing to double-launch"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
mkdir -p logs results

stage () {                   # stage <name> <output> <cmd...>
  nm="$1"; shift; out="$1"; shift
  if [ -f "$out" ]; then echo "== $nm already done"; return 0; fi
  echo "== $nm  ($(date -u +%H:%M:%S))"
  "$@" > "logs/interaction_${nm}.log" 2>&1
  rc=$?; echo "   exit $rc  ($(date -u +%H:%M:%S))"; return $rc
}

# TAG (env) suffixes every output so variants (e.g. TAG=_mixed) do not collide
# with the primary run; extra args go to the screen only.
T="${TAG:-}"
stage screen$T   "results/interaction_screen$T.json"   python3 -m src.cut.interaction_screen \
    --out "results/interaction_screen$T.json" --pt "results/interaction_screen$T.pt" "$@" || exit 1
stage confirm$T  "results/interaction_confirm$T.json"  python3 -m src.cut.interaction_confirm --attention \
    --screen "results/interaction_screen$T.json" --out "results/interaction_confirm$T.json"
stage geometry$T "results/interaction_geometry$T.json" python3 -m src.cut.interaction_geometry \
    --pt "results/interaction_screen$T.pt" --out "results/interaction_geometry$T.json"
echo "ALL DONE $(date -u)"
