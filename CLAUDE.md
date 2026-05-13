# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands are run from inside `snn_async_delays/`.

```bash
# Create and activate conda environment
conda env create -f environment.yaml
conda activate snn_async

# Single run (Step 1, one op)
python -m scripts.run_step1 --config configs/step1_singleop.yaml --op NAND --train_mode weights_and_delays

# Full sweep (Step 1: all ops × 3 modes × 3 hidden sizes)
python -m scripts.run_step1 --config configs/step1_singleop.yaml --sweep

# Step 2 single run (K=4 queries, same op)
python -m scripts.run_step2 --config configs/step2_multiquery_sameop.yaml --K 4 --train_mode weights_and_delays

# Step 2 sweep (K=1..8, all modes)
python -m scripts.run_step2 --config configs/step2_multiquery_sameop.yaml --sweep

# Step 3 sweep (mixed ops)
python -m scripts.run_step3 --config configs/step3_multiquery_multiop.yaml --sweep

# GPU support: add --device cuda
# Override any config key: --epochs 100 --hidden_size 20 --seed 0
```

Results are saved to `runs/<run_name>/`:
- `best_model.pt`, `train_log.csv`, `eval_results.json`, `config.json`
- `plots/training_curves.png`, `plots/delays_ih.png`

Sweep summaries go to `runs/step{1,2,3}_sweep_summary.json`.

---

## Project Overview

This project investigates whether **synaptic delays in SNNs (Spiking Neural Networks) provide computational advantages beyond accuracy improvements** — specifically, whether asynchrony enables temporal multiplexing ("fitting more computation into the same neuron budget"). The research question: given a fixed network size and energy budget, can a delay-enabled SNN process more parallel queries (higher K) than a weight-only baseline?

The spec is in `snn_async_project_plan.md` (Chinese/English). The three experimental steps are:
- **Step 1**: Single boolean op, compare weight-only vs delay-only vs weight+delay baselines.
- **Step 2**: One op (e.g., NAND), vary K (number of queries per trial), measure max K at ≥95% accuracy.
- **Step 3**: Mixed ops per trial, same throughput analysis.

## Intended Project Structure

```
snn_async_delays/
  configs/          # YAML configs for each step (step1/step2/step3)
  snn/              # Core SNN: neurons.py, synapses.py, model.py
  data/             # boolean_dataset.py, encoding.py
  train/            # trainer.py, eval.py
  scripts/          # run_step1.py, run_step2.py, run_step3.py
  utils/            # seed.py, logging.py, viz.py
```

## Key Architectural Decisions

**Simulation parameters**: `dt=1ms`, `T=50ms`, LIF neurons with refractory period. Discrete delays `d_ij ∈ [0, d_max]` (e.g., `d_max=50ms`).

**Three training modes** (must be a config switch `train_mode`):
- `weights_only` — delays fixed at 0 or constant `d0`
- `delays_only` — weights frozen after random init
- `weights_and_delays` — both trained

**Multi-query (Step 2/3) uses temporal slots (Plan A)**: one trial is divided into K input windows; each window injects one (A, B) pair; readout is taken per-window. This is the mechanism that tests "asynchronous temporal multiplexing." Do NOT use spatial parallel channels (Plan B) as the primary experiment.

**Input encoding**: rate coding — A=1 → `r_on` Hz (e.g., 200–500 Hz), A=0 → `r_off` Hz (0–20 Hz). For Step 3 add one-hot op channels.

**Output decoding**: spike count or mean membrane over the readout window → BCEWithLogitsLoss.

**Delay parameterization**: use sigmoid mapping to `[0, d_max]` or straight-through estimator (STE) for discrete rounding. Make this switchable via `delay_param_type` config.

**Separate learning rates**: `lr_w` for weights, `lr_d` for delays.

## Core Evaluation Metrics

The primary claims require these specific metrics (must be in eval output):
- **Max K @ accuracy ≥ τ** — under fixed hidden neuron count and fixed trial length
- **Energy-normalized throughput**: `K / total_hidden_spikes`
- **Latency per query**: time from input-window end to readout decision

Comparison dimensions: `weight-only` vs `delay-only` vs `weight+delay`; hidden size sweep (10, 20, 50, 100).

## Critical Implementation Constraints

1. Multi-query experiments **must multiplex K queries within a single trial** — not across separate trials. This is the entire research claim.
2. Readout must be **per-window**, not aggregated at trial end (otherwise the network trivially delays all computation).
3. `delay-only` mode tends to collapse (all-silent or all-burst) — needs careful initialization and possibly light regularization.
4. Every run must save: best checkpoint, training curves, final metrics as JSON/CSV, and a full config dump for reproducibility.

## Known Bugs / Fixes

- **Windows GBK encoding**: `load_cfg()` in all three `scripts/run_step*.py` must open YAML files with `encoding="utf-8"` (Windows defaults to GBK which fails on UTF-8 YAML comments). Fixed in all three scripts.

---

## Step 1 Results (Single-op Solvability Baseline)

**Config**: 8 ops × 3 modes × hidden ∈ {10, 20, 50}, 200 epochs, seed=42, GPU.
**Summary file**: `runs/step1_sweep_summary.json`

### Accuracy by Mode (mean over all ops and hidden sizes)

| Train mode          | Mean acc | Min  | Max  | Runs ≥ 95% |
|---------------------|----------|------|------|------------|
| `weights_only`      | 0.878    | 0.769| 0.940| 0 / 24     |
| `delays_only`       | 0.737    | 0.517| 0.892| 0 / 24     |
| `weights_and_delays`| **0.954**| 0.846| 0.989| **15 / 24**|

### Accuracy by Mode × Hidden Size (mean over all ops)

| Hidden | `weights_only` | `delays_only` | `weights_and_delays` |
|--------|---------------|---------------|----------------------|
| h=10   | 0.853         | 0.706         | **0.924**            |
| h=20   | 0.888         | 0.753         | **0.962**            |
| h=50   | 0.893         | 0.753         | **0.974**            |

### Per-op Accuracy at h=50 (`weights_and_delays`)

| Op       | Accuracy | ≥ 95%? |
|----------|----------|--------|
| AND      | 0.988    | PASS   |
| OR       | 0.986    | PASS   |
| NAND     | 0.989    | PASS   |
| NOR      | 0.985    | PASS   |
| A_IMP_B  | 0.985    | PASS   |
| B_IMP_A  | 0.978    | PASS   |
| XNOR     | 0.952    | PASS   |
| XOR      | 0.932    | **FAIL** |

### Energy (mean hidden spikes/trial) at h=50

| Mode                | Spikes/trial |
|---------------------|--------------|
| `weights_only`      | 34.1         |
| `weights_and_delays`| 27.7         |
| `delays_only`       | **2.5**      |

### Key Findings

1. **`weights_and_delays` dominates**: +7–8% accuracy gain over `weights_only` at every hidden size; the gap widens slightly with larger networks.
2. **`delays_only` collapses on XOR/XNOR**: average accuracy 0.584 (near chance) vs 0.800 on simple ops (AND/OR/NAND/NOR). Confirms the architecture note — delay-only mode needs regularization or special init for hard ops.
3. **No mode reaches ≥95% without delays on weight training**: `weights_only` tops out at 0.940 (AND, h=50).
4. **XOR is the hardest op**: only `weights_and_delays` at h=50 comes close (0.932, still below 95% threshold). This sets a floor for Step 2 experiments — XOR will likely need h≥50.
5. **Energy sparsity**: `delays_only` fires only ~2.5 spikes/trial at h=50 (14× sparser than `weights_only`), but at severe accuracy cost. `weights_and_delays` achieves high accuracy with ~19% fewer spikes than `weights_only`.

---

## Step 2 Results (Multi-Query Temporal Multiplexing)

Full experiment log: `docs/EXPERIMENT_LOG.md` (Sections 3–12)

### Four Experimental Designs

| Plan | Input channels | Readout | T grows with K? | What it tests |
|------|---------------|---------|-----------------|---------------|
| A (serial slots) | 2 shared | K separate windows | Yes (K×35) | Baseline; delays needed for alignment |
| C (simultaneous) | 2K dedicated | 1 shared window | No (fixed 30) | Spatial vs temporal separation |
| D (sequential)   | 2 shared | 1 shared window | Yes (K×10+10) | **Pure temporal routing** |

### Plan A Key Finding (NAND, h=20, K=1~20)

`weights_and_delays` achieves ≥95% for all K=1~20. `weights_only_d0` stuck at ~80% even at K=1.  
**Mechanism**: delays perform *time alignment* (shift input into readout window), not true multiplexing — LIF decay (τ=10, slot_len=35) provides natural slot isolation (~3% residual).

### Plan C Key Finding (NAND, h=20, K=1~12)

Alignment effect (d=20 vs d=0): **+11.8%**. Capacity effect (w_and_d vs d=20): **+2.7%**.  
Delays' main role is alignment; spatial channels (2K) already handle query separation.

### Plan D Key Finding (NAND, h=20/50, sub_win=10, K=1~12)

**Design**: K queries share 2 channels, injected sequentially in sub-windows; single readout decodes all K simultaneously. Weights structurally cannot distinguish queries — only delays can.

**h=20 results** (w_and_d):

| K | acc | gap vs d0 |
|---|-----|-----------|
| 1 | 94.9% | +19.3% |
| 2 | 91.2% | +14.8% |
| 3 | 83.5% | ~+7% |
| ≥4 | ~80% | ~+4% |

Max K @ 90%: **K=2** (h=20). Bottleneck: h=20 capacity (~6.7 neurons/query at K=3).

**h=50 results, linear readout** (w_and_d, 4 seeds): K=1: 95.05±0.21%, K=2: 92.20±0.75%, K=3: 87.52±1.04%. Max K@90% = **K=2** (linear readout). Increasing neurons by 150% yields only +1~3% — bottleneck is the shared single-readout decoder (Linear(h,K)), not neuron count.

**h=50 results, MLP readout** (w_and_d, seeds 42+0): K=1: 95.95%, K=2: 93.50%, K=3: **92.68%**, K=4: 89.85%, K=5: 86.29%, K=6: 83.84%. Max K@90% = **K=3** (MLP readout). K=3 accuracy jumps from 87.52% (linear) to 92.68% (MLP), confirming the K=2 ceiling under linear readout was a *decoder* limitation, not a *representation* limitation.

### Overall Conclusion

Trainable delays are a **structurally necessary** mechanism for shared-channel temporal multiplexing:  
- d=0 is a structural failure (not a learning failure) in Plan D  
- The primary benefit of delays is **time alignment** (+11–19% vs d=0)  
- True temporal routing capacity exists: **K=3 @ >92%** with MLP readout (h=50)  
- The K=2 ceiling under linear readout reflects decoder expressiveness, not SNN representation capacity
