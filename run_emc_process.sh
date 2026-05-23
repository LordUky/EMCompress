#!/bin/bash
# Stage 1 — EMC process (ReSimplifyIt simple + full baselines) on all 7 datasets.
# Produces screened timestamps + rewritten queries that feed Stage 2 downstream
# VideoQA (run_emc_guided_inference.sh) and the efficiency tables in the paper
# (analyze_efficiency.py).
#
# Usage: bash run_emc_process.sh [num_threads]
#
# Results saved to:
#   results_simple_baseline/simple_{dataset}.json       (results)
#   results_simple_baseline/simple_{dataset}_log.json   (full trajectory logs)
#   results_full_baseline/full_{dataset}.json           (results)
#   results_full_baseline/full_{dataset}_log.json       (full trajectory logs)

THREADS=${1:-50}
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "EMC Baseline Runner"
echo "Threads: $THREADS"
echo "Working dir: $WORKDIR"
echo "============================================"

# Run simple baseline on all 7 datasets
echo ""
echo ">>> Starting Simple Baseline (all 7 datasets)..."
python "$WORKDIR/run_emc_simple_baseline.py" --dataset all --num_threads "$THREADS"

echo ""
echo ">>> Starting Full Baseline (all 7 datasets)..."
python "$WORKDIR/run_emc_full_baseline.py" --dataset all --num_threads "$THREADS"

echo ""
echo "============================================"
echo "All done!"
echo "============================================"
