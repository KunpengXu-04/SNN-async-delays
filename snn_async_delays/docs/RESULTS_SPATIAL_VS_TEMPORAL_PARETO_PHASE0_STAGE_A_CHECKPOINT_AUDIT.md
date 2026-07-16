# Results: Phase-0 Stage-A checkpoint mechanism audit

## Scope and provenance

This is a read-only diagnostic audit of all 15 formal Stage-A cells. Both
`best_model.pt` and `last_model.pt` were replayed on the four exact XOR patterns,
for 30 checkpoints and 120 checkpoint-pattern forward passes. No training,
checkpoint modification, test access, or Stage-B construction occurred.

The audit extracts true output spikes, target-time pre-reset membrane voltage,
output current, hidden activity, realized delays, and best-to-final weight
drift. Counterfactual output-threshold probes are explicitly exploratory and
cannot overturn the formal Stage-A decision.

## Delay clarification

Every input-to-hidden and hidden-to-output realized delay is exactly zero in
all 30 replayed checkpoints. Stage A was an all-d0 spatial baseline. The stored
`delay_placement=input_to_hidden_only` string was inaccurate metadata inherited
from the planned Stage-B intervention locus; the operative settings were
`train_mode=weights_only`, `fixed_delay_value=0`, and `output_delay_mode=d0`.
The diagnostic heatmaps are therefore correct.

## Final-checkpoint failure decomposition

The 15 cells divide into:

- 6 successful cells;
- 7 cells with at least one silent truth pattern;
- 2 cells with opponent collisions.

Across the silent cells there are 10 silent truth patterns. Seven have zero
hidden spikes, so their failure originates before the output layer. The other
three have hidden spikes but fail to produce a correct output event:

- h=8/seed107 reaches `0.02999948` against threshold `.03`, only
  `5.2e-7` below firing;
- h=16/seeds107 and 211 have hidden activity but insufficient or adverse
  hidden-to-output drive for pattern `10`.

Thus silence has at least two causes: absent hidden representation and output
threshold/weight conversion failure. It is not valid to explain every silent
cell by one global output threshold.

There are three collided truth patterns across h=12/seed211 and h=24/seed211.
In every case the incorrect opponent spikes early at `t=10`, while the correct
opponent spikes at the declared target `t=11`. The collision is an all-time
count tie, not simultaneous firing at the target. Since the two-spike input
micro-burst occupies consecutive timesteps, this is consistent with early-event
leakage through the first d0 feedforward path, but causal attribution requires
a new encoding/loss control.

## Best versus final checkpoint

Accuracy-selected `best_model.pt` is not a rescue:

- only 4/15 best checkpoints have a valid one-target-spike interface on all
  four patterns;
- 6/15 final checkpoints do;
- no cell loses valid-interface pattern count from best to final.

Four cells lose one truth-table classification by the final epoch, but this is
not equivalent to losing valid output semantics. A zero signed count defaults
to class 0, so accuracy can improve or deteriorate while silence/collision
semantics change differently. Selecting checkpoints by accuracy alone is
therefore misaligned with the intended spiking interface.

## Exploratory threshold sensitivity

The final checkpoints were replayed, without retraining, at output thresholds
`.015, .0225, .03, .0375, .045, .06`.

| threshold | cells with full truth + interface | valid / silent / collision patterns |
|---:|---:|---:|
| .015 | 7/15 | 48 / 9 / 3 |
| .0225 | 7/15 | 48 / 9 / 3 |
| .030 | 6/15 | 47 / 10 / 3 |
| .0375 | 3/15 | 40 / 17 / 3 |
| .045 | 1/15 | 35 / 23 / 2 |
| .060 | 1/15 | 32 / 26 / 1, plus one wrong-class pattern |

No tested threshold produces any hidden width that succeeds in all three
seeds. Lowering the threshold recovers only one additional cell and does not
remove collision. Raising it suppresses at most two collided patterns while
creating much more silence. Threshold `.03` is locally brittle in individual
cases, but threshold choice alone is not the global explanation or solution.

## Scientific conclusion

Stage A failed because the present training/interface construction does not
reliably enforce both:

1. a usable hidden representation for every truth pattern; and
2. exactly one opponent spike at the declared time with no early competitor.

The audit weakens a pure checkpoint-selection explanation and rules out a
single universal output-threshold repair over the tested range. It motivates a
new, versioned interface-stability calibration rather than Stage B.

The next protocol should separate three factors with new calibration and
held-out initialization seeds: exact truth-table-balanced training, explicit
global one-target/no-early-spike supervision, and input micro-burst structure.
It must select checkpoints using interface semantics and stability, not
accuracy alone. h=12 or h=24 may not be adopted from the failed experiment.

## Artifacts

- `docs/generated/spatial_vs_temporal_pareto_phase0/stage_a_checkpoint_audit/audit_decision.json`
- `checkpoint_pattern_audit.csv`
- `checkpoint_summary.csv`
- `best_final_drift.csv`
- `threshold_sensitivity_by_cell.csv`
- `threshold_sensitivity_summary.csv`
- `stage_a_checkpoint_mechanism_audit.png`
- `threshold_sensitivity.png`

