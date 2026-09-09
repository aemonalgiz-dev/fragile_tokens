#!/bin/bash
# Local: pull every new small output (json, jsonl, md, log, compact .pt) from the node
# as soon as it appears, so a node loss costs at most the stage in progress.
#   bash scratch/pull_loop.sh [seconds between pulls, default 120] [max minutes, default 600]
# Reads the host from lambda_details.txt on every iteration; stops when the remote
# log contains "RUNG4 SCRIPT DONE" or "POST LADDER DONE" or the time limit passes.
set -u
EVERY="${1:-120}"; MAXMIN="${2:-600}"; D="${3:-lambda_details.txt}"
KEY=.lambda_key; [ "$D" = "lambda_details_small.txt" ] && [ -f .lambda_key_small ] && KEY=.lambda_key_small
STAMP=".last_pull_$(basename "$D" .txt)"
# the two nodes' driver logs would overwrite each other locally: skip them for the small node
EXCL=""; [ "$D" = "lambda_details_small.txt" ] && EXCL="-not -name scale.log -not -name launch.log"
end=$(( $(date +%s) + MAXMIN*60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  H=$(grep -oE "[A-Za-z0-9_-]+@[0-9.]+" "$D" | head -1)
  P=$(grep -oiE "port: *[0-9]+" "$D" | grep -oE "[0-9]+" | head -1); P="${P:-22}"
  if ssh -i "$KEY" -p "$P" -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes "$H" \
       "cd ~/glitch && find results docs logs -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.log' -o -path 'results/compact/*.pt' \) $EXCL -newer $STAMP 2>/dev/null | tar czf - -T - 2>/dev/null; touch $STAMP" 2>/dev/null \
       > scratch/.pull.tgz; then
    if [ -s scratch/.pull.tgz ]; then
      n=$(tar tzf scratch/.pull.tgz 2>/dev/null | wc -l)
      tar xzf scratch/.pull.tgz -C . 2>/dev/null && echo "$(date -u +%H:%M:%S) pulled $n file(s): $(tar tzf scratch/.pull.tgz | grep -vE '\.log$' | tr '\n' ' ' | cut -c1-200)"
    fi
    if grep -qE "${STOP_MARKER:-RUNG4B DONE|SMALL NODE DONE}" logs/scale.log 2>/dev/null; then echo "remote finished; last pull done $(date -u)"; exit 0; fi
  else
    echo "$(date -u +%H:%M:%S) unreachable ($H)"
  fi
  sleep "$EVERY"
done
echo "pull loop time limit reached $(date -u)"
