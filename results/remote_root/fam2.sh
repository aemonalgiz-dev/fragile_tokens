set -u
cd ~/glitch
mkdir -p logs
for spec in "amber:LLM360/Amber:10" \
            "pile:HuggingFaceFW/ablation-model-the-pile:10" \
            "c4:HuggingFaceFW/ablation-model-c4:10" \
            "fineweb:HuggingFaceFW/ablation-model-fineweb-v1:10" \
            "refinedweb:HuggingFaceFW/ablation-model-refinedweb:10"; do
  nm="${spec%%:*}"; rest="${spec#*:}"; model="${rest%%:*}"; n="${rest##*:}"
  echo "=== $nm  $(date -u +%H:%M) ==="
  python3 -m src.cut.harvest_family2 --model "$model" --n-ckpt "$n" \
      --out "results/f2_${nm}.json" > "logs/f2_${nm}.log" 2>&1
  echo "    exit=$?"; tail -6 "logs/f2_${nm}.log" | grep -E "residual|early-half|glitch =" || true
  df -h / | tail -1
done
echo "FAMILY2 COMPLETE $(date -u)"
