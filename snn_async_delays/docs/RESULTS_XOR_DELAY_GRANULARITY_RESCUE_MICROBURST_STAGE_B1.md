# Results: dimension-aware XOR micro-burst rescue Stage B1

**Protocol:** `xor_delay_granularity_rescue_microburst_v1`  
**Date:** 2026-07-16  
**Status:** learned stage complete, 40/40; dimension-aware micro-burst bridge
passes and per-hidden remains selected.

## Primary result

All three explicitly scaffolded delay parameterizations pass every registered
cell and gate. Per-hidden task-only training passes none.

| Condition | P | Lambda | Full gate | Exact interface | Correct initial direction | Full delay coverage | Max final error range (step) |
|---|---:|---:|---:|---:|---:|---:|---:|
| global anchor | 1 | .01 | 10/10 | 10/10 | 10/10 | 10/10 | .000026-.003762 |
| per-hidden dimension-matched | 16 | .16 | 10/10 | 10/10 | 10/10 | 10/10 | .000038-.000343 |
| per-synapse dimension-matched | 64 | .64 | 10/10 | 10/10 | 10/10 | 10/10 | .000045-.000165 |
| per-hidden task-only | 16 | 0 | 0/10 | 0/10 | 0/10 | 0/10 | 3.150738-3.639009 |

The primary decision therefore passes: fixed d4 is 5/5, global is 10/10 and
the selected per-hidden recipe is 10/10. Per-synapse also passes but is not
selected because it uses 64 rather than 16 independent delay parameters.

## Mechanism audit

Dimension scaling behaves exactly as intended under the consecutive
micro-burst. Initial weighted arrival-gradient magnitude per coordinate is
approximately `.025588` for global, per-hidden and per-synapse in every cell.
No cell triggers the registered gradient-clipping flag.

The task and oracle gradients conflict on average in `.5000`, `.4625` and
`.4828` of valid coordinates for global, per-hidden and per-synapse. The
oracle term nevertheless makes every total initial coordinate gradient
nonzero and target-directed, and every coordinate reaches delay four.

This is not a minor regularization effect: without the teacher, per-hidden
task-only has no full or exact-interface pass. Across its ten cells:

- only 4/10 achieve correct classification on all four patterns;
- only 3 of 40 pattern-level output spike trains are exactly correct;
- mean correct-target-time rate is `.175`;
- mean silence and collision rates are `.325` and `.075`;
- low-initialized cells finish at mean delay `1.963`, while high-initialized
  cells finish at mean delay `5.631`;
- initial task gradients point toward delay four on only `.6875` of low-side
  coordinates and `.3875` of high-side coordinates on average.

Thus task-only from-scratch training does not discover the homogeneous timing
schedule. Any next positive claim must distinguish local task-derived
restoration inside a teacher-created basin from global autonomous discovery.

## Artifact and provenance audit

- Complete learned formal cells: 40/40.
- Missing required artifacts: 0.
- Every cell contains config, checkpoint, complete training log, exhaustive
  truth-table output, resource ledger, runtime NPZ and runtime diagnostic
  panel.
- Test split remains unopened.
- Ten implementation smoke cells remain invalid and are not pooled.

Aggregate artifacts:

- `docs/generated/xor_delay_granularity_rescue_microburst_v1/stage_b1_learned/cells.csv`
- `docs/generated/xor_delay_granularity_rescue_microburst_v1/stage_b1_learned/decision.json`
- `docs/generated/xor_delay_granularity_rescue_microburst_v1/stage_b1_learned/microburst_rescue_gate_summary.png`

## Claim boundary

Supported narrowly: explicit dimension-matched homogeneous delay-four
supervision is seed-robust under the declared K=1 XOR consecutive micro-burst,
and per-hidden tying is sufficient.

Not supported: autonomous task-derived timing, heterogeneous routing, K>1
multiplexing, WAD superiority, compression, energy savings or a Pareto law.

## Mechanical next step

The result authorizes an independent scaffold-withdrawal protocol on fresh
seeds. That protocol must separately test:

1. retention after removing the oracle teacher from an exact scaffolded
   checkpoint; and
2. task-only restoration after a preregistered symmetric timing perturbation.

From-scratch task-only need not be repeated as the primary comparison because
it has already failed 0/10 here. K greater than one remains locked.
