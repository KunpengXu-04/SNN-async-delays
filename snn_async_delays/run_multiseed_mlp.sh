#!/bin/bash
# Plan D + MLP readout sweep: h=50, K=1~6, seeds 42 and 0
# Purpose: test whether nonlinear readout breaks the K=2 ceiling
set -e
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

cd "$(dirname "$0")"

for K in 1 2 3 4 5 6; do
  for SEED in 42 0; do
    echo "=== MLP K=${K} seed=${SEED} ==="
    python -m scripts.run_step2_sequential \
      --config configs/step2_sequential.yaml \
      --K $K --seed $SEED --n_hidden 50 \
      --readout_type mlp --device cuda
  done
done
echo "=== MLP SWEEP DONE ==="
