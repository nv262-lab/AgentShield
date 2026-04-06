#!/bin/bash
set -e
echo "============================================"
echo "  AgentShield - Full Reproduction"
echo "============================================"

pip install -r requirements.txt --quiet
mkdir -p results

echo "[1/4] Running main experiment..."
for seed in 42 123 456; do
    python scripts/run_main_experiment.py --seed $seed --output results/seed_${seed}.json
done

echo "[2/4] Running ablation..."
python scripts/run_ablation.py 2>/dev/null || echo "Skipped"

echo "[3/4] Running BFT validation..."
python scripts/run_bft_validation.py 2>/dev/null || echo "Skipped"

echo "[4/4] Generating figures..."
python scripts/generate_figures.py 2>/dev/null || echo "Skipped"

echo "Done! Results in results/"
