#!/bin/bash
# Local, one shot: bring the cheaper single-GPU node up and launch scratch/launch_small.sh.
# Reads host/port from lambda_details_small.txt ("ssh: user@host", optional "port: N").
# Key: .lambda_key_small if present, else .lambda_key.
set -u
D=lambda_details_small.txt
H=$(grep -oE "[A-Za-z0-9_-]+@[0-9.]+" $D | head -1)
P=$(grep -oiE "port: *[0-9]+" $D | grep -oE "[0-9]+" | head -1); P="${P:-22}"
K=.lambda_key; [ -f .lambda_key_small ] && K=.lambda_key_small
S="ssh -i $K -p $P -o StrictHostKeyChecking=no -o ConnectTimeout=20"
C="scp -i $K -P $P -o StrictHostKeyChecking=no -q"
echo "== node $H port $P key $K  $(date -u +%H:%M:%S)"
$S $H 'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c; df -h / | tail -1' || { echo "cannot reach $H"; exit 1; }
echo "== upload code, scripts, the stored 32B matrix (compact)"
$C scratch/glitch_sync.tgz scratch/bootstrap.sh scratch/launch_small.sh $H:~/ || exit 1
$S $H 'sed -i "s/\r$//" ~/bootstrap.sh ~/launch_small.sh; chmod +x ~/bootstrap.sh ~/launch_small.sh'
echo "== bootstrap  $(date -u +%H:%M:%S)"
$S $H 'bash ~/bootstrap.sh 2>&1 | tail -6' || exit 1
$S $H 'mkdir -p ~/glitch/results/compact'
$C results/compact/fragility_qwen3_32b_compact.pt $H:~/glitch/results/fragility_qwen3_32b.pt
# the Qwen2.5-7B matrix, gate and geometry already ran on the 4xH100: ship them so the driver
# skips those stages and runs only the 7B's reasoning and specimen stages here
$C results/compact/fragility_qwen25_7b_compact.pt $H:~/glitch/results/fragility_qwen25_7b.pt
$C results/fragility_qwen25_7b.json results/gate_qwen25_7b.json results/fragility_predict_qwen25_7b.json \
   results/fragility_predict_qwen25_7b_greedy.json $H:~/glitch/results/
$C results/compact/fragility_qwen25_7b_compact.pt $H:~/glitch/results/compact/
echo "== launch detached  $(date -u +%H:%M:%S)"
$S $H 'cd ~/glitch && ( setsid nohup bash ~/launch_small.sh > logs/launch.log 2>&1 & ); sleep 3; pgrep -af "launch_smal[l]" | cut -c1-60; tail -2 logs/scale.log'
echo "== done; pull with:  bash scratch/pull_loop.sh 120 720 lambda_details_small.txt"
