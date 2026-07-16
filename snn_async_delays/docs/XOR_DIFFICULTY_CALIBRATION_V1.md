# Preregistration: XOR difficulty calibration v1

This is Phase 1A of `PUBLICATION_ROADMAP.md`. It is a validation-only search
for a non-saturated regime, not a comparison of WAD and d0 and not a test-set
experiment.

## Design

- Balanced deterministic-burst XOR, shared two-channel sequential input.
- Fixed-subwindow/growing-duration regime: `sub_win=10`, `read_len=10`, so
  total `T=10K+10` for `K={2,3,4}`.
- `N={20,35,50}`, paired seeds `{0,1,42}`, WAD and d0: 54 cells.
- All-time observation and linear readout only.
- 2,000 training and 500 validation examples; the test split is not evaluated.
- Worst-query and exact-trial validation accuracy are primary; balanced
  accuracy and the complete resource vector are required.

The validation set is also used for best-checkpoint selection. Its reported
performance is therefore optimistic and only suitable for choosing the next
protocol's difficulty.

## Locked selection rule

Prefer adjacent settings in which both delay conditions have mean validation
worst-query accuracy in `[0.65, 0.95]`, neither crosses the hard floor/ceiling
`[0.55, 0.98]`, and seed SD is at most `0.12`. If nothing qualifies, this
protocol records a failed calibration. A replacement protocol must be written;
the test set must remain unopened.

## Diagnostic panels — execution amendment

At the researcher's request on 2026-07-12, every completed cell immediately
saves `plots/diagnostic_data.npz` and `plots/diagnostic_panel.png` using trace
seed 999. The first completed cells are backfilled with the identical routine.
This changes artifact retention only, not factors, metrics, checkpoints, or the
difficulty-selection rule. Panels are illustrative and cannot override the
full validation table.

## Execution

```text
python -m scripts.run_xor_difficulty_calibration --device cuda
```

The experiment is stored under
`runs/exploratory/xor_difficulty_calibration_v1/`. Each cell writes
`validation_results.json`, deliberately not `eval_results.json`. Diagnostic
artifacts are written during each cell rather than deferred until sweep end.
