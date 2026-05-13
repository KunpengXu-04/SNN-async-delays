#!/bin/bash
# Multi-seed validation: Plan D, K=1,2,3, seeds=0,1,2, h=20 and h=50
set -e
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

cd "$(dirname "$0")"

echo "=== h=20 seed=0 ==="
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 1 --seed 0 --n_hidden 20 --device cuda
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 2 --seed 0 --n_hidden 20 --device cuda
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 3 --seed 0 --n_hidden 20 --device cuda

echo "=== h=20 seed=1 ==="
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 1 --seed 1 --n_hidden 20 --device cuda
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 2 --seed 1 --n_hidden 20 --device cuda
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 3 --seed 1 --n_hidden 20 --device cuda

echo "=== h=20 seed=2 ==="
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 1 --seed 2 --n_hidden 20 --device cuda
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 2 --seed 2 --n_hidden 20 --device cuda
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 3 --seed 2 --n_hidden 20 --device cuda

echo "=== h=50 seed=0 ==="
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 2 --seed 0 --n_hidden 50 --device cuda
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 3 --seed 0 --n_hidden 50 --device cuda

echo "=== h=50 seed=1 ==="
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 2 --seed 1 --n_hidden 50 --device cuda
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 3 --seed 1 --n_hidden 50 --device cuda

echo "=== h=50 seed=2 ==="
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 2 --seed 2 --n_hidden 50 --device cuda
python -m scripts.run_step2_sequential --config configs/step2_sequential.yaml --K 3 --seed 2 --n_hidden 50 --device cuda

echo "=== ALL DONE ==="
