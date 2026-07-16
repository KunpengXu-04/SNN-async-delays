# XOR task bridge Level 1A v1 — Stage-I result

**Status:** complete; formal Stage I passed. Stage II is authorized but has not
started. The test split remains closed.

## Completion and provenance

All 90 preregistered cells completed. Every cell has its config, final metrics,
final checkpoint, full training log, exhaustive truth-table record, resource
ledger, runtime NPZ and runtime diagnostic panel. The aggregate cell table,
decision JSON and interface-gate heatmap are complete.

The first interactive process was interrupted after 47 complete cells when a
new chat message terminated the attached terminal. The runner's preregistered
resume behavior reads cells with an existing `metrics.json` and reruns cells
without it. The interrupted 48th cell had only a config file, so it was rerun
from initialization; the 47 complete cells were not changed. The resumed
process completed 90/90 and exited with code 0. No hyperparameter, seed, loss,
gate or selection rule changed.

## Locked decision

| voltage-envelope weight | weight LR | passing cells / 10 |
|---:|---:|---:|
| 0 | .001 | 0/10 |
| 0 | .003 | 8/10 |
| 0 | .01 | **10/10** |
| .1 | .001 | 0/10 |
| .1 | .003 | 4/10 |
| .1 | .01 | 7/10 |
| 1 | .001 | 0/10 |
| 1 | .003 | 0/10 |
| 1 | .01 | 8/10 |

The unique passing candidate is therefore:

- voltage-envelope weight `eta=0`;
- weight learning rate `lr_w=.01`.

It passes all five d0 cells and all five fixed-delay-4 cells. In every selected
cell, balanced accuracy is 1, all four hard output spike trains exactly equal
their targets, silence and collision rates are zero, target-time rate is 1,
and every pattern has exactly one output spike and nonzero hidden activity.

## Interpretation

Stage I establishes a seed-robust K=1 XOR hard-spike interface under both the
causal d0 and fixed interior-delay schedules. It also rejects the need for the
registered voltage-envelope auxiliary in this interface: the lowest permitted
weight, zero, is the only robust candidate. This does not show that voltage
envelopes are generally harmful; failures at positive eta include additional
collisions and are conditional on the frozen 500-update grid.

No delay was learned in Stage I. The result therefore does not establish joint
task/delay optimization, WAD, timing discovery, temporal multiplexing,
compression, generalization or a Pareto advantage. It authorizes only the
preregistered 85-cell Stage II with `eta=0` and `lr_w=.01` frozen.

## Artifacts

- Decision: `docs/generated/xor_task_bridge_level1a_v1/stage_i/decision.json`
- Cells: `docs/generated/xor_task_bridge_level1a_v1/stage_i/cells.csv`
- Heatmap: `docs/generated/xor_task_bridge_level1a_v1/stage_i/interface_gate_heatmap.png`
- Runs: `runs/exploratory/xor_task_bridge_level1a_v1/stage_i/`

