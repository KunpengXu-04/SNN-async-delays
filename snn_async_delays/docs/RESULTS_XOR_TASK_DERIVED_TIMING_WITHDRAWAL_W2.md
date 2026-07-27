# Results: XOR task-derived timing withdrawal W2

**Protocol:** `xor_task_derived_timing_withdrawal_v1`  
**Date:** 2026-07-16  
**Status:** complete 45/45; primary protocol fails.

## Formal decision

| Branch | Direction | Exact interface | All-delay recovery | Registered pass |
|---|---:|---:|---:|---:|
| retention task-delay-only | unperturbed d4 | 5/5 | 5/5 | 5/5 |
| oracle-delay-only | d3 | 5/5 | 5/5 | 5/5 |
| oracle-delay-only | d5 | 5/5 | 5/5 | 5/5 |
| task-delay-only | d3 | 3/5 | 0/5 | 0/5 |
| task-delay-only | d5 | 0/5 | 0/5 | 0/5 |
| task-joint | d3 | 5/5 | 0/5 | 0/5 |
| task-joint | d5 | 3/5 | 0/5 | 0/5 |
| task-weight-only | d3 | 0/5 | not applicable | 0/5 |
| task-weight-only | d5 | 1/5 | not applicable | 1/5 |

The preregistered local-restoration protocol fails because task-delay-only is
0/10. K greater than one remains unauthorized.

## What the result does and does not say

Retention is positive but weak: all five exact W0 checkpoints remain exact,
yet task loss and its delay gradient are zero from the first update. This shows
that the teacher-built point is stationary, not that task loss attracts nearby
states.

The oracle positive control is decisive. Both directions pass 5/5, all delays
finish exactly at four, inherited weights are unchanged, and clipping never
occurs. The model, optimizer, sigmoid parameterization, LR, and 500-update
budget can solve the local problem when the gradient is informative.

Task-only restoration is strongly directional. From d3, mean correct-target-
time rate is `.90` and three seeds recover the exact interface, but only 1 of
80 delay coordinates across the five cells lies within `.1` of four. From d5,
mean correct-target-time rate is `.10`, exact interface is 0/5, and 40 of 80
coordinates finish below `.5` or above `7.5`. No cell clips. This is not a
small-convergence-error result.

Joint training recovers function more often—5/5 at d3 and 3/5 at d5—but never
recovers all delays. Weight-only training reaches the exact interface in only
one d5 seed. Therefore joint success is compensation or a non-unique
distributed solution, not evidence that delays returned to the teacher's
schedule.

## Primary mechanism: integer-boundary gradient pathology

Production delay interpolation uses

\[
d_f=\lfloor \operatorname{stopgrad}(d)\rfloor,\qquad
s(d)=(1-\alpha)s_{d_f}+\alpha s_{d_f+1},\quad \alpha=d-d_f.
\]

At an exact integer the forward map is continuous but not differentiable, and
autograd selects the right-hand interval. A post-result, zero-update audit with
the same W0 weights and a single globally tied delay finds:

| Functional delay | Mean raw task gradient | Descent direction |
|---:|---:|---|
| 4.999 | +.00408 | toward four, correct |
| 5.000 | -.00356 | toward larger delay, incorrect |

The sign flips discontinuously at the registered d5 perturbation. Global tying
does not repair it: all five global-scalar gradients are wrong-directed at d5.
This directly explains the observed late-side divergence and boundary
polarization. Detached spike reset/refractory dynamics and hard-spike surrogate
credit add further forward/backward mismatch, but they are not needed to
explain this specific discontinuity.

## Secondary mechanism: the schedule is not identifiable

Four exhaustive XOR output patterns do not uniquely determine 16 per-hidden
delays. Once the exact output spike trains are reached, filtered hard-spike loss
becomes exactly zero even if most delays remain far from four. The d3 task-only
and task-joint cells demonstrate this directly. Consequently, “all 16 delays
return to four” cannot be expected from task loss alone without parameter tying
or a target-free structural constraint. This is distinct from temporal credit:
even a better gradient may recover function without recovering the oracle's
internal parameter vector.

## Artifact and implementation audit

- Complete formal W2 cells: 45/45.
- Missing mandatory artifacts: 0.
- Frozen-weight and frozen-delay branches obey their declared parameter masks.
- Every branch inherits the matching-seed W0 checkpoint and a fresh optimizer.
- Minimum clip coefficient is one in every cell.
- Test split remains unopened.

The directional endpoint figure is
`docs/generated/xor_task_derived_timing_withdrawal_v1/stage_w2/withdrawal_directional_endpoints.png`.
The post-result mechanism audit is reproducible with
`python -m scripts.audit_xor_withdrawal_w2_mechanism --device cpu` and writes
`mechanism_audit.json`, `integer_boundary_gradient_audit.csv`, and
`integer_boundary_gradient_audit.png` beside the directional figure.

## Evidence-constrained repair order

1. Do not tune LR, threshold, epoch count, or seeds first. They cannot reverse
   the d5 gradient sign, and longer training already drives delays to bounds.
2. Preregister a small integer-boundary temporal-credit preflight. Compare the
   current right-sided backward rule with a symmetric central/soft delay
   backward estimator while preserving the hard forward interface.
3. Require the task-derived gradient condition
   `gradient * (delay - 4) > 0` on both sides and across the full local delay
   grid before any training rescue is allowed.
4. Add a continuous output-level timing objective—current, pre-reset voltage,
   or a smooth event-time transport loss—that uses target output time but no
   hidden-delay oracle. It must be tested for bidirectional gradients first.
5. Separate functional restoration from internal schedule identification. For
   homogeneous calibration, use global tying or a target-free consensus term
   such as `Var(d_j)` only after temporal credit is fixed. For future
   heterogeneous routing, retain functional and causal routing endpoints rather
   than requiring equality to a single oracle vector.
