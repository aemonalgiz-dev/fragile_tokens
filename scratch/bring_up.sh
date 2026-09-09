#!/bin/bash
# Local, one shot: bring a fresh 4xH100 from bare Lambda image to "rung 4 running".
# Host is read from lambda_details.txt. Idempotent enough to re-run.
set -u
# lambda_details.txt:  "ssh: user@host"  and optionally  "port: NNNN"  (RunPod / Vast use non-22 ports)
H=$(grep -oE "[A-Za-z0-9_-]+@[0-9.]+" lambda_details.txt | head -1)
P=$(grep -oiE "port: *[0-9]+" lambda_details.txt | grep -oE "[0-9]+" | head -1); P="${P:-22}"
S="ssh -i .lambda_key -p $P -o StrictHostKeyChecking=no -o ConnectTimeout=20"
C="scp -i .lambda_key -P $P -o StrictHostKeyChecking=no -q"
echo "== node $H  $(date -u +%H:%M:%S)"
$S $H 'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c; df -h / | tail -1' || { echo "cannot reach $H"; exit 1; }

echo "== upload code, scripts, 72B compact matrix"
$C scratch/glitch_sync.tgz scratch/bootstrap.sh scratch/launch_rung4.sh $H:~/ || exit 1
$S $H 'sed -i "s/\r$//" ~/bootstrap.sh ~/launch_rung4.sh; chmod +x ~/bootstrap.sh ~/launch_rung4.sh'

echo "== bootstrap (venv, torch, transformers, kernels, FP8 quantizer guard)  $(date -u +%H:%M:%S)"
$S $H 'bash ~/bootstrap.sh 2>&1 | tail -8' || exit 1
$S $H 'mkdir -p ~/glitch/results/compact'
$C results/compact/fragility_qwen25_72b_compact.pt $H:~/glitch/results/fragility_qwen25_72b.pt

echo "== launch rung 4 detached  $(date -u +%H:%M:%S)"
$S $H 'cd ~/glitch && ( setsid nohup bash ~/launch_rung4.sh > logs/launch.log 2>&1 & ); sleep 3; pgrep -af "launch_rung[4]" | cut -c1-60; tail -2 logs/launch.log'
echo "== done; start the pull loop with:  bash scratch/pull_loop.sh 120 720"
