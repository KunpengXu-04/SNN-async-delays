# Results: XOR task-derived timing withdrawal W1

**Protocol:** `xor_task_derived_timing_withdrawal_v1`  
**Date:** 2026-07-16  
**Status:** damage gate passes 10/10; W2 authorized.

## Result

Both symmetric perturbations create a clean timing-only failure.

| Perturbation | Cells | Classification BAcc | Exact interface | Output time | Silence | Collision |
|---:|---:|---:|---:|---:|---:|---:|
| d3 | 5 | 1.0 in 5/5 | 0/5 | step 14 in 20/20 patterns | 0 | 0 |
| d5 | 5 | 1.0 in 5/5 | 0/5 | step 16 in 20/20 patterns | 0 | 0 |

Every trial still emits exactly one output spike and predicts the correct XOR
class. Exact target-time accuracy is zero. Therefore these controls isolate the
timing variable and demonstrate why classification or pooled accuracy alone
would produce a false conclusion of robustness.

## Intervention integrity

- Each branch inherits its matching-seed W0 checkpoint.
- All non-delay tensors are bitwise identical before and after intervention.
- All 16 functional delays equal exactly 3 or 5 steps up to floating-point
  representation.
- Trainable-component lists are empty and update count is zero.
- All nine required branch artifacts are present in 10/10 cells.

## Prospective W2 mechanism audit

Before launching W2, the task gradient was evaluated without taking an update.
This was disclosed and did not change the registered matrix.

| Start | Mean target-directed coordinates | Range | Mean gradient norm |
|---:|---:|---:|---:|
| d3 | .85 | .6875-.9375 | approximately .00473 |
| d5 | .1375 | .0625-.3125 | approximately .00143 |

Every coordinate is nonzero. The loss landscape is nevertheless strongly
asymmetric: from d5, most initial task gradients point away from delay four.
This prospectively predicts that task-delay-only may recover d3 but fail d5.
That prediction must not be used to remove d5, tune its LR, or alter the full
45-cell W2 matrix.

## Decision

`damage_gate_pass=true` and `w2_authorized=true`. W2 must run unchanged:
five retention cells plus oracle-delay-only, task-delay-only, task-joint, and
task-weight-only branches for both perturbations and all five seeds. Success of
joint or weight-only compensation cannot rescue failure of task-delay-only.
