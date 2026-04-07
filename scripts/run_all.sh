#!/bin/bash
set -e

echo "============================================"
echo "  AgentShield Full Experiment Suite"
echo "============================================"

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: Set ANTHROPIC_API_KEY first"
    echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
    exit 1
fi

mkdir -p results/full

# Quick validation (10 min, ~$5)
echo ""
echo "[Quick] Running small-scale validation..."
python3 scripts/full_experiment.py \
    --seed 42 --n-agents 5 --n-scenarios 10 \
    --output-dir results/quick

# Medium run (1-2 hrs, ~$30)
echo ""
echo "[Medium] Running medium-scale experiment..."
for seed in 42 123 456; do
    echo "  Seed: $seed"
    python3 scripts/full_experiment.py \
        --seed $seed --n-agents 5 --n-scenarios 50 \
        --output-dir results/medium
done

# Full run (8-12 hrs, ~$150) — uncomment when ready
# echo ""
# echo "[Full] Running full-scale experiment..."
# for seed in 42 123 456; do
#     echo "  Seed: $seed"
#     python3 scripts/full_experiment.py \
#         --seed $seed --n-agents 5 --n-scenarios 500 \
#         --output-dir results/full
# done

echo ""
echo "============================================"
echo "  Done! Results in results/"
echo "============================================"
