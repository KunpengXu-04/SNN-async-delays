# Results: XOR dimension-aware delay rescue Level 1B-R R1

**Protocol:** `xor_delay_granularity_rescue_level1br_v1`  
**Date:** 2026-07-16  
**Status:** 50/50 formal R1 cells complete; R2 skipped by the registered gate;
R3 sealed confirmation authorized.

## Result

R1 supports the narrow normalization hypothesis. With the original mean
arrival scaffold (`lambda=.01`), success deteriorates as the number of
independent delay coordinates grows. Multiplying the scaffold weight by the
declared coordinate count restores the analytically intended per-coordinate
teacher strength and yields a 10/10 pass for both higher-dimensional delay
parameterizations.

| Condition | P | Effective lambda | Full pass | Exact interface | Correct initial direction on every coordinate | Full final delay coverage | Final max-error range (step) |
|---|---:|---:|---:|---:|---:|---:|---:|
| global anchor | 1 | .01 | 10/10 | 10/10 | 10/10 | 10/10 | .000093-.005278 |
| per-hidden mean baseline | 16 | .01 | 1/10 | 10/10 | 6/10 | 1/10 | .068483-.808248 |
| per-hidden coordinate-matched | 16 | .16 | 10/10 | 10/10 | 10/10 | 10/10 | .000043-.003391 |
| per-synapse mean baseline | 64 | .01 | 0/10 | 10/10 | 2/10 | 0/10 | .479106-2.547204 |
| per-synapse coordinate-matched | 64 | .64 | 10/10 | 10/10 | 10/10 | 10/10 | .000042-.000657 |

All 50 cells passed the hard-spike interface gate. Consequently, the failed
unscaled cells are not ordinary XOR classification failures; they fail the
registered coordinate-wise timing recovery criterion.

## Mechanism audit

The analytic intervention behaved as predicted:

| Condition | Initial weighted arrival gradient per coordinate |
|---|---:|
| global anchor | .025588 |
| per-hidden mean baseline | .001599 |
| per-hidden coordinate-matched | .025588 |
| per-synapse mean baseline | .0003998 |
| per-synapse coordinate-matched | .025588 |

The ratios are the expected factors of 16 and 64. No cell triggered the
predeclared clipping flag, so the rescue cannot be attributed to global
gradient clipping. The scaled conditions also convert every initial total
delay gradient to a nonzero, target-directed coordinate and converge with
full coordinate coverage from both low and high initial delays.

This makes a normalization-only explanation sufficient for this registered
task: an LR or update-budget rescue is unnecessary. It does not establish that
normalization is the only possible cause in other tasks or heterogeneous
routing regimes.

## Artifact and provenance audit

- Complete formal cell directories: 50/50.
- Missing required artifacts: 0.
- Each cell contains `config.json`, `metrics.json`, `final_model.pt`,
  `training_log.json`, `exhaustive_truth_table_results.json`,
  `resource_ledger.json`, runtime `diagnostic_data.npz`, and runtime
  `diagnostic_panel.png`.
- Aggregate table: `docs/generated/xor_delay_granularity_rescue_level1br_v1/r1/cells.csv`.
- Mechanical decision: `docs/generated/xor_delay_granularity_rescue_level1br_v1/r1/decision.json`.
- Gate plot: `docs/generated/xor_delay_granularity_rescue_level1br_v1/r1/r1_gate_summary.png`.
- The six prior smoke cells remain invalid for claims and are not pooled.
- The test split was not opened.

## Claim boundary

The supported claim is deliberately narrow: under an explicit homogeneous
delay-4 oracle teacher, fixed teacher strength per delay coordinate makes
per-hidden and per-synapse parameterizations optimizable on the K=1
single-event XOR bridge.

R1 does **not** show:

- task labels autonomously discover delay 4;
- learned delays specialize or route simultaneous queries;
- consecutive micro-bursts are robust;
- K>1 multiplexing, WAD superiority, compression, energy savings, or a Pareto
  law;
- equal total supervision across parameterizations. Total oracle-teacher
  strength explicitly grows with P.

## Mechanical next step

Do not run R2. Run the already preregistered 30-cell R3 sealed confirmation:
ten global-anchor cells, ten per-hidden coordinate-matched cells, and ten
per-synapse coordinate-matched cells on the five sealed seeds and both initial
directions. Final rescue success requires global 10/10 and at least one
higher-dimensional recipe 10/10. If both confirm, per-hidden is selected by
the preregistered lower-parameter-count priority.
