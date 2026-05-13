#!/bin/bash
# Control: MLP readout + d=0 (weights_only), Plan D, K=3
# Tests whether MLP alone (without trainable delays) can crack K=3
set -e
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

cd "$(dirname "$0")"

echo "=== CONTROL: MLP readout + d=0 (weights_only), K=2, seed=42 ==="
python -m scripts.run_step2_sequential \
  --config configs/step2_sequential.yaml \
  --K 2 --seed 42 --n_hidden 50 \
  --readout_type mlp --train_mode weights_only --device cuda

echo "=== CONTROL: MLP readout + d=0 (weights_only), K=3, seed=42 ==="
python -m scripts.run_step2_sequential \
  --config configs/step2_sequential.yaml \
  --K 3 --seed 42 --n_hidden 50 \
  --readout_type mlp --train_mode weights_only --device cuda

echo "=== CONTROL: MLP readout + d=0 (weights_only), K=3, seed=0 ==="
python -m scripts.run_step2_sequential \
  --config configs/step2_sequential.yaml \
  --K 3 --seed 0 --n_hidden 50 \
  --readout_type mlp --train_mode weights_only --device cuda

echo "=== CONTROL DONE ==="
