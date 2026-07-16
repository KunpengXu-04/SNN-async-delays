# Results: XOR Level 1B fixed micro-burst controls

**Protocol:** `xor_delay_granularity_level1b_v1`  
**Date:** 2026-07-16  
**Status:** 10/10 control cells complete; fixed delay 4 passes 5/5; fixed d0
passes 0/5; learned micro-burst matrix is mechanically authorized.

## 1. Question

These controls ask whether the hard-spiking K=1 XOR interface can implement a
declared delayed output under the consecutive micro-burst before any learned
delay result is interpreted. Each selected A/B value channel emits at steps 8
and 9, producing four input events per trial. The required output remains one
and only one correct opponent spike at step 15.

The two controls are:

- fixed input-hidden delay 4: interface feasibility gate; and
- fixed input-hidden d0 with the same target-at-15: timing-specificity negative.

Weights train under the frozen Level-1A interface recipe. Input-hidden delays
are not trainable and hidden-output delays remain d0.

## 2. Completion and integrity

All ten declared cells completed across seeds `{607,709,811,919,1021}`. Every
cell contains config, final checkpoint, strict metrics, full training log,
exhaustive truth table, resource ledger, runtime NPZ and runtime diagnostic
panel. No test split was opened and no cell was excluded.

## 3. Results

| Control | Exact interface pass | Balanced accuracy range | Correct target-time rate | Exact output trains |
|---|---:|---:|---:|---:|
| fixed delay 4 | **5/5** | 1.00--1.00 | 1.00 in every seed | 20/20 patterns |
| fixed d0, target at 15 | **0/5** | .50--.75 | 0.00 in every seed | 0/20 patterns |

Every fixed-delay-4 cell has zero silence, zero collision, one output event per
trial and hidden activity in all four XOR patterns. All 20 outputs occur at
step 15, split evenly between the two opponent output neurons according to the
XOR label.

The d0 control is not merely less accurate. Across its 20 truth-table trials,
all 27 emitted output spikes occur at step 11 and none at the required step 15.
Seed-level failure modes include silence rates from 0 to `.75`, collision rates
from 0 to `.75` and mean output counts from `.25` to `1.75`, but target-time
rate is uniformly zero. All d0 cells retain hidden activity in every pattern,
so their failure is temporal placement/output structure, not absence of hidden
computation.

## 4. Interpretation

The micro-burst interface is feasible when the correct schedule is supplied.
The four-step input-hidden delay plus the simulator's two layer-buffer offsets
places the required opponent output at step 15. A d0 network trained against
the same delayed target instead emits only at step 11, establishing strong
timing specificity under this interface.

This result does not show that delay 4 can be learned from the task, that a
high-dimensional delay model is stable, or that the network performs temporal
multiplexing. The schedule is fixed by the experimenter. It is a feasibility
and negative-control result only.

## 5. Decision

The preregistered fixed-oracle 5/5 gate passes, so the frozen 60-cell learned
micro-burst matrix is mechanically authorized. It has not been launched.

Stage A already showed that the unscaled per-hidden and per-synapse scaffold
recipes fail their 10/10 candidates. Therefore there are now two distinct
next actions that must not be conflated:

1. complete the original frozen Level-1B learned micro-burst matrix, which
   would test the already-failed optimization recipe under a harder encoding;
2. preregister a separate dimension-aware rescue with per-coordinate scaffold
   strength normalized across 1/16/64 delay dimensions.

Changing lambda, delay LR or update budget inside the original learned matrix
would be a post-result protocol violation. A rescue must have a new protocol
identifier and new seeds.

## 6. Claim boundary

Supported narrowly: fixed delay 4 makes the declared consecutive micro-burst
hard-spike interface seed-robust, while fixed d0 cannot imitate its target
timing.

Not supported: learned micro-burst timing, task-derived delay discovery,
per-hidden/per-synapse optimization, K greater than one, routing, WAD
superiority, compression, energy reduction or a Pareto law.

## 7. Artifacts

- cells: `runs/exploratory/xor_delay_granularity_level1b_v1/stage_b_controls/`
- decision: `docs/generated/xor_delay_granularity_level1b_v1/stage_b_controls/decision.json`
- aggregate cells: `docs/generated/xor_delay_granularity_level1b_v1/stage_b_controls/cells.csv`
