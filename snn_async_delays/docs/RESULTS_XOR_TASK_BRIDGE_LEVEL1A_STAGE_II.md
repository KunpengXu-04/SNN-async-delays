# XOR task bridge Level 1A v1 — Stage-II result

**Status:** complete; Level 1A passed. Level 1B preregistration is authorized.
K greater than one and the test split remain locked.

## Completion

All 85 preregistered Stage-II cells completed. Every cell has its config,
metrics, final checkpoint, full training log, exhaustive truth-table record,
resource ledger, runtime NPZ and runtime diagnostic panel. The aggregate cell
table, decision JSON and learned-delay heatmap are complete.

Stage II read the selected Stage-I interface from its decision file and froze
`eta=0` and `lr_w=.01`. No manual replacement or post-result amendment was
used.

## Locked candidate decision

| arrival weight `lambda` | delay LR | passing learned cells / 10 |
|---:|---:|---:|
| 0 | .01 | 0/10 |
| 0 | .05 | 1/10 |
| .01 | .01 | **10/10** |
| .01 | .05 | 10/10 |
| .1 | .01 | 10/10 |
| .1 | .05 | 10/10 |
| 1 | .01 | 10/10 |
| 1 | .05 | 10/10 |

The preregistered priority rule therefore selects the smallest passing
auxiliary and then the smallest delay learning rate:

- `lambda=.01`;
- `lr_d=.01`.

Across both raw-delay initializations and all five seeds, the selected candidate
passes 10/10. Its final delays range from `3.997372` to `4.001881` steps, with
maximum absolute error `0.002628` step. All ten cells have balanced accuracy 1,
exact hard-spike truth-table completion, correct target timing, zero silence and
collision, correct nonzero initial total-delay direction, and active hidden
representations for every pattern.

## Controls and mechanism

The fixed-d0/delayed-target control fails 0/5. Its correct-target-time rate is
zero in every seed. One seed reaches balanced accuracy 1 but still fails the
hard-spike interface because it emits at the wrong time and collides. This
confirms that ordinary class accuracy cannot substitute for exact temporal
evaluation and that the target-at-15 endpoint is not trivially solved by the
d0 weights in this protocol.

Task-only delay learning is not reliable. With `lambda=0`, the `.01` delay-LR
arm passes 0/10 and the `.05` arm passes only 1/10. The `.05` arm reaches exact
spike-train completion in 8/10 cells, but most still fail the delay-recovery or
initial-gradient gate. Initial task-gradient direction is correct in only 5/10
cells at either tested delay LR.

For the selected `lambda=.01,lr_d=.01` arm, the total initial direction is
correct in 10/10 cells even though task and arrival gradients conflict in 5/10.
Thus the small arrival-centroid term repairs the unreliable task-delay credit;
the result is not evidence that XOR supervision independently discovers the
delay schedule.

## Claim boundary

Level 1A establishes a narrow joint-optimization bridge: one global shared
input-hidden delay can be driven to a prespecified interior schedule while
trainable weights solve the complete K=1 XOR truth table and the evaluated
output remains an exact hard-spike code. The successful method includes an
explicit oracle-derived arrival-centroid timing scaffold.

It does not establish autonomous routing discovery, unique per-neuron or
per-synapse delay learning, robustness to the earlier consecutive micro-burst,
held-out generalization, K scaling, WAD superiority, temporal multiplexing,
resource compression, energy savings or a Pareto law.

## Next authorized gate

Preregister Level 1B at K=1. It may compare global, per-neuron and per-synapse
delay granularity and test the deferred consecutive-micro-burst robustness.
It must preserve the fixed output interface and separately report performance
with and without the explicit timing scaffold. K greater than one remains
locked.

## Artifacts

- Decision: `docs/generated/xor_task_bridge_level1a_v1/stage_ii/decision.json`
- Cells: `docs/generated/xor_task_bridge_level1a_v1/stage_ii/cells.csv`
- Heatmap: `docs/generated/xor_task_bridge_level1a_v1/stage_ii/learned_delay_gate_heatmap.png`
- Runs: `runs/exploratory/xor_task_bridge_level1a_v1/stage_ii/`

