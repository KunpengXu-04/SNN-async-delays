# Results: simultaneous output-interface calibration v2

## Decision

The protocol is complete: nine new temporal-scaffold cells, nine reused
spatial-v1 cells, all required checkpoints, logs, validation results,
exhaustive 64-pattern results, NPZ files and panels are present. The test split
remained sealed.

The preregistered rule selects output threshold **0.2**. Thresholds 0.2 and 0.3
pass the declared arm-level viability gates; 0.5 fails the temporal silent/tie
gate. Among viable thresholds, pooled output spikes per operation/window are
closest to one at 0.2 (`0.9668`, distance `0.0332`) rather than 0.3 (`0.9381`,
distance `0.0619`). Accuracy was not used as the tie-breaker.

| threshold | spatial worst-B | temporal worst-B | spatial silent | temporal silent | pooled spikes/window | viable |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.2 | 0.742 | 0.500 | 0.148 | 0.291 | 0.967 | yes |
| 0.3 | 0.677 | 0.500 | 0.141 | 0.269 | 0.938 | yes |
| 0.5 | 0.545 | 0.500 | 0.175 | 0.699 | 0.667 | no |

## Critical interpretation

This is an excitability/interface calibration success, not a temporal-routing
success. At threshold 0.2, temporal per-window balanced accuracies for seeds
`0/1/42` are:

- XOR window: `0.500 / 0.500 / 0.500`;
- NAND window: `0.807 / 0.679 / 0.579`;
- NOR window: `0.500 / 0.500 / 0.500`.

Random-validation exact-trial accuracies are `.248/.216/.182` (mean `.215`).
The exhaustive truth table confirms the same pattern: worst-window balanced
accuracy is `.500` for every seed, with exact-trial `.281/.234/.188`.

Thus the scaffold and membrane curriculum made the output pair fire and reduced
silence, but the shared pair learned useful discrimination only in the middle
NAND window. XOR and NOR remain compatible with constant or prevalence-driven
decisions. The arm-level gate allowed exactly-chance worst balanced accuracy;
passing it must not be rewritten as evidence that the temporal interface has
learned all three operations.

Threshold 0.3 sometimes improves the middle window further (seed 0 reaches
1.0 balanced accuracy) and has slightly lower mean silence, but it does not
improve the worst window and loses the locked one-spike tie-breaker. Selecting
0.3 post hoc would violate the protocol. Threshold 0.5 is clearly too silent.

## Consequences

- Freeze output threshold 0.2 for both simultaneous configurations.
- The spatial-control pilot is eligible to proceed under its existing locked
  protocol.
- Keep the temporal 36-cell matrix locked until a small, versioned viability
  preflight verifies matched membrane-to-spike training for d0, scalar, fixed
  heterogeneous and WAD conditions.
- Treat d0 temporal silence as a structural lower bound. The primary temporal
  comparison must be WAD against the strongest delay-enabled non-learned
  control, not WAD against d0 alone.
- The calibration timing scaffold is never a formal pilot condition and cannot
  support a delay-learning claim.

## Artifacts

- Decision: `docs/generated/simultaneous_output_interface_calibration_v2/decision.json`
- Cell table: `docs/generated/simultaneous_output_interface_calibration_v2/cells.csv`
- Aggregate figure: `docs/generated/simultaneous_output_interface_calibration_v2/threshold_summary.png`
- Formal runs: `runs/exploratory/simultaneous_output_interface_calibration_v2/`
