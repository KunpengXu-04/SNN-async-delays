# Results: XOR task-derived timing withdrawal W0

**Protocol:** `xor_task_derived_timing_withdrawal_v1`  
**Date:** 2026-07-16  
**Status:** W0 complete 5/5; W1 authorized; W2 locked.

## Gate result

All five fresh-seed per-hidden oracle foundations pass the registered exact
interface, coordinate-gradient, and delay-coverage gates.

| Seed | Exact spike trains | Max delay error (step) | Full coverage | Initial direction | Clip flag |
|---:|---:|---:|---:|---:|---:|
| 2503 | 4/4 | .000226 | 16/16 | 16/16 | no |
| 2521 | 4/4 | .000215 | 16/16 | 16/16 | no |
| 2539 | 4/4 | .002032 | 16/16 | 16/16 | no |
| 2551 | 4/4 | .000452 | 16/16 | 16/16 | no |
| 2579 | 4/4 | .000115 | 16/16 | 16/16 | no |

Every cell has balanced accuracy one, correct target time one, zero silence
and collision, one output event per pattern, and hidden activity in all four
patterns. All eight mandatory artifacts are present in every cell.

## Stability and mechanism

Four foundations retain the exact interface for every one of their final 100
updates. Seed 2539 is exact for 99/100: a single transient boundary crossing at
update 436 briefly reduces the exact-pattern count before recovery. This does
not violate final-only checkpoint selection, but it demonstrates that the hard
spike interface is locally discontinuous and should not be described as a
smooth or margin-certified solution. All cells retain full delay coverage for
the entire final 100 updates.

The mean fraction of valid coordinates where task and oracle gradients
initially conflict is `.475`. The oracle term makes every total coordinate
gradient nonzero and target-directed. This reinforces the existing conclusion
that W0 validates the teacher-built starting basin; it does not show that task
loss discovered timing.

## Decision

The formal decision is `foundation_gate_pass=true` and `w1_authorized=true`.
The next allowed action is exactly ten no-update controls: overwrite the 16
functional delays with d3 or d5 while inheriting the matching-seed weights and
verify that both perturbations destroy the exact target spike train. W2 remains
locked until all ten demonstrate genuine damage.
