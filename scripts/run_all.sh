#!/usr/bin/env bash
# Run the full benchmark + profile + plot. Copy results/ back afterward.
set -euo pipefail
mkdir -p results

echo "== profiling one decode step (kernel count + trace) =="
python -m bench.profile --batch-size 1

echo "== sweeping batch sizes: eager vs CUDA graph =="
python -m bench.benchmark --batch-sizes 1 8 32 64 --out results/sweep.csv

echo "== plotting =="
python -m bench.plot --csv results/sweep.csv --out results/latency.png

echo "Done. results/ has sweep.csv, latency.png, one_step_trace.json."
