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

**h=50 results, MLP readout** (w_and_d, seeds 42+0): K=1: 95.95%, K=2: 93.50%, K=3: **92.68%**, K=4: 89.85%, K=5: 86.29%, K=6: 83.84%. Max K@90% = **K=3** (MLP readout).

**Ablation control — MLP + d=0** (seeds 42+0): K=2: 78.15%, K=3: ~77%. Essentially identical to linear+d0. Confirms: **it is the delay, not MLP**, that enables temporal routing. MLP adds +5.2% on top of delay representations, but contributes nothing without delays.

**Full 2×2 ablation (h=50):**

| Readout | Delay | K=3 acc | Max K@90% |
|---------|-------|---------|----------|
| Linear  | d=0   | ~76%    | 0        |
| Linear  | trainable | 87.52% | **2** |
| MLP     | d=0   | ~77%    | 0        |
| MLP     | trainable | **92.68%** | **3** |

### Overall Conclusion

Trainable delays are the **sole necessary mechanism** for shared-channel temporal multiplexing:  
- d=0 fails regardless of readout type (linear or MLP): both ~77% at K=3  
- The primary benefit of delays is **time alignment** (+11–19% vs d=0)  
- Delays create **non-linearly separable** temporal representations: Max K@90% = 2 (linear) / 3 (MLP)  
- Full experiment log: `docs/EXPERIMENT_LOG.md` Sections 3–12

---

## Depth Ablation Results (2-Layer SNN, Plan D)

**Config**: 4 models × K∈{2,3,4} × 2 seeds = 24 runs, 200 epochs, NAND, h=50 total neurons.  
**Summary**: `runs/depth_ablation/depth_ablation_summary.csv`  
**Log**: `docs/EXPERIMENT_LOG.md` Sections 13–14

### Complete 2×2 Ablation Matrix (depth × readout, all with trainable delays unless noted)

| | Linear readout | MLP readout |
|---|---|---|
| **L1-h50** | K@90%=2, K=3→87.3% | K@90%=**3**, K=3→92.7% |
| **L2-h25h25** | K@90%=2, K=3→88.6% | K@90%=**2**, K=3→88.3% |
| **L2-h25h25-d0** | K@90%=**0**, K=3→77.4% | — |

### Accuracy Results (mean over 2 seeds)

| Model | K=2 | K=3 | K=4 | Max K@90% |
|-------|-----|-----|-----|-----------|
| L1-h50-linear | **92.4%** | 87.3% | 84.9% | **2** |
| L1-h50-MLP | **95.9%** | **92.7%** | **89.9%** | **3** |
| L2-h25h25-linear | 91.4% | 88.6% | 88.2% | **2** |
| L2-h25h25-MLP | 91.6% | 88.3% | 86.5% | **2** |
| L2-h25h25-d0 | 77.7% | 77.4% | 76.9% | **0** |

### Energy-Normalised Throughput K/spk (mean over 2 seeds)

| Model | K=2 | K=3 | K=4 |
|-------|-----|-----|-----|
| L1-h50-linear | **0.144** | **0.142** | **0.168** |
| L1-h50-MLP | ~0.10 | ~0.11 | ~0.10 |
| L2-h25h25-linear | 0.089 | 0.066 | 0.069 |
| L2-h25h25-MLP | 0.073 | 0.084 | 0.083 |
| L2-h25h25-d0 | 0.107 | 0.122 | 0.122 |

### Key Findings

1. **MLP readout is the only intervention that improves Max K@90%**: L1+MLP is the sole model reaching K@90%=3. Neither depth (L2+linear) nor their combination (L2+MLP) breaks the threshold.
2. **Depth and MLP gains are not additive**: L2+MLP ≈ L2+linear (88.3% vs 88.6% at K=3). MLP provides no benefit at h2=25 because the readout network (25→25→K) is too small — its capacity is constrained by the readout input dimension, not the MLP depth.
3. **MLP benefit scales with readout input size**: L1+MLP (h=50 input) gains +5.4pp over L1+linear at K=3. L2+MLP (h2=25 input) gains only +0pp. The MLP must receive rich enough representations to add nonlinear decoding value.
4. **Depth gives modest raw accuracy gains at K=3,4 but at ~50% energy cost**: L2+linear is +1–3pp over L1+linear, but K/spk drops from 0.14 to 0.07. Not energy-efficient.
5. **Delays remain necessary regardless of architecture**: L2-d0 ≈ 77% = L1-d0, confirming delays are the primary and irreplaceable mechanism for temporal routing.

### Practical Recommendations

- **Best accuracy at K=3**: L1-h50 + MLP (92.7%), Max K@90%=3
- **Best efficiency**: L1-h50 + linear (K/spk=0.14), Max K@90%=2
- **Max K@90%=3 appears to be a hard ceiling**: neither architecture (depth/MLP, see above) nor timing parameters (below) can break it.

---

## Timing-Parameter Ablation Results (Direction B)

**Config**: `configs/step2_timing_ablation.yaml`, runner `scripts/run_timing_ablation.py`.
**Hypothesis**: `sub_win == lif_tau_m == 10ms`, so query Q0's membrane signal decays by
~1/e (≈63%) before the readout window opens at `t = win_len = K*sub_win` — this timing
mismatch was suspected to cause the Max K@90%=3 ceiling. Three candidate fixes tested
(one-at-a-time + combined), all on the best-known architecture (`L1-h50 + MLP + trainable
delays`), K∈{3,4}, 2 seeds, 200 epochs:

| Condition | sub_win | read_len | τ_m | K=3 acc | K=4 acc | Max K@90% |
|---|---|---|---|---|---|---|
| `baseline`     | 10 | 10 | 10.0 | **92.7%** | **89.9%** | **3** |
| `read_len20`   | 10 | 20 | 10.0 | 91.3%     | 88.4%     | 3 |
| `subwin5`      | 5  | 10 | 10.0 | 85.2%     | 83.7%     | 0 |
| `tau20`        | 10 | 10 | 20.0 | 88.6%     | 86.1%     | 0 |
| `combined`     | 5  | 20 | 20.0 | 82.6%     | 81.3%     | 0 |

**Hypothesis REFUTED — every intervention performs at or below baseline; none crosses
90% at K=4.** Mechanism (revealed by `mean_hidden_spikes`, which collapses from ~28→40 in
baseline down to ~7→10 in `combined`): increasing `τ_m` makes LIF neurons fire *less*
often (slower membrane charge/discharge ⇒ fewer threshold crossings ⇒ less information
throughput), and shrinking `sub_win` gives rate-coded inputs less time to generate spikes
in the first place — both reduce, not preserve, transmitted information. Worse, slower
decay (`τ_m ↑`) actively **weakens the "natural slot isolation"** that Plan A identified
as the mechanism separating queries (LIF decay at τ=10/slot_len=35 gave ~3% residual
cross-talk) — Q0's signal now leaks further into Q1/Q2's processing windows, *increasing*
cross-query interference rather than reducing information loss.

**Conclusion**: Max K@90%=3 is not a simple membrane-decay timing artifact fixable by
retuning `τ_m`/`read_len`/`sub_win` — these knobs trade off information throughput and
slot isolation in ways that net to zero or negative. The ceiling more likely reflects a
fundamental **representational-interference** limit on how many simultaneously-active
queries a shared 50-dim hidden layer can keep separable, independent of both architecture
(depth/readout, Sections 13–14) and timing constants (this section). Full log:
`docs/EXPERIMENT_LOG.md` Section 15.

---

## Step 3 Results (Mixed-Op Temporal Multiplexing, Plan D)

**Config**: L1-h50 + MLP readout, Plan D sequential, n_input=10 (2 A/B + 8 one-hot op), K=1~4, seeds 42+0, 200 epochs, 8 ops uniformly sampled per query, n_train=4000.
**Summary**: `runs/step3_planD/step3_planD_summary.csv`
**Log**: `docs/EXPERIMENT_LOG.md` Section 16

### Accuracy by Model × K (mean ± range over 2 seeds)

| K | `w_and_d` | `d0_control` | Delay gap |
|---|-----------|--------------|-----------|
| 1 | 84.3% ± 4.4% | 64.5% ± 1.6% | **+19.8%** |
| 2 | 74.4% ± 0.7% | 60.0% ± 0.5% | **+14.4%** |
| 3 | 68.7% ± 1.9% | 57.6% ± 2.3% | **+11.1%** |
| 4 | 67.6% ± 0.8% | 56.7% ± 0.3% | **+10.9%** |

**Max K@90%**: 0 for both models.

### Key Findings

1. **Delay advantage generalises to mixed ops**: +11–20% gain over d0 at all K, matching Step 2 Plan D magnitude (+14–19%). Temporal routing is insensitive to operation heterogeneity — delays encode temporal-slot identity; one-hot op channels handle operation identity spatially. The two mechanisms are orthogonal.
2. **Absolute accuracy limited by task difficulty, not temporal capacity**: K=1 never reaches 90% — caused by sparse per-op training (~500 samples/op vs. 4000 for NAND in Step 2) and XOR/XNOR dragging down the mean. This is a **data sparsity / multi-function learning ceiling**, not a timing or routing failure.
3. **d0 near-chance at K≥3**: 57–60% for d0 vs. 65–69% for w_and_d at K=3–4. Without delays, spike activity in the shared readout window is negligible (LIF decays erase K=0's sub-window signal) and the network falls back to a weak label prior.
4. **Config note**: initial d0_control used `train_mode: weights_and_delays` (bug — delays were trained, results identical to w_and_d). Fixed to `train_mode: weights_only` + `fixed_delay_value: 0.0` before final analysis.

---

## Step 3 Follow-Up (4 Easy Ops: AND/OR/NAND/NOR)

**Config**: same as Step 3 but `n_ops=4, ops_list=[AND,OR,NAND,NOR]`, n_input=6, n_train=4000 (~1000/op × 4 ops).
**Summary**: `runs/step3_planD_4ops/step3_planD_summary.csv`
**Log**: `docs/EXPERIMENT_LOG.md` Section 17

### Accuracy by Model × K (mean ± range over 2 seeds)

| K | `w_and_d` | `d0_control` | Delay gap | vs 8-op w_and_d |
|---|-----------|--------------|-----------|-----------------|
| 1 | 87.7% ± 1.7% | 62.6% ± 1.4% | **+25.1%** | +3.4% |
| 2 | 80.2% ± 0.5% | 59.4% ± 2.7% | **+20.8%** | +5.8% |
| 3 | 74.9% ± 1.6% | 55.9% ± 0.1% | **+19.0%** | +6.2% |
| 4 | 72.6% ± 0.7% | 53.8% ± 0.4% | **+18.8%** | +5.0% |

**Max K@90%**: 0 for both models.

### Key Findings

1. **Hypothesis refuted**: Removing XOR/XNOR + doubling per-op samples improves w_and_d by +3–6% but K=1 only reaches 87.7% — still 2.3% below the 90% threshold.
2. **Multi-function learning is an independent bottleneck**: Even with 4 easy linearly-separable ops at ~1000 samples/op, accuracy is ~7% below single-op NAND (95% at 4000 samples). The network must simultaneously represent 4 different Boolean computations in shared h=50 neurons — this **multi-function representational competition** is the fundamental ceiling.
3. **Delay advantage larger with easy ops** (+19–25% vs +11–20% for 8 ops): w_and_d gains more from easy ops; d0 is slightly worse (4 ops have more symmetric output distributions → weaker label prior).
4. **Estimated data requirement to reach Max K@90%≥1**: ~16,000 total samples (~4000/op × 4 ops), based on interpolating from Step 2 single-op scaling.
