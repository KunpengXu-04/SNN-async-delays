# XOR Integer-Boundary Credit Preflight v1

Status: complete; P0 selected Gaussian STE sigma 1.00; P1 failed.

## Why this repair is necessary

W2 showed that the oracle delay-only arm recovered 10/10 cells while the
task-delay-only arm recovered 0/10.  The subsequent read-only audit localized
one concrete implementation pathology: the historical floor/ceil
interpolation chooses the right-hand derivative at an exact integer delay.
For the late perturbation, the mean global raw-delay task gradient changed from
`+0.00407759` at `d=4.999` to `-0.00356313` at `d=5.000`, so gradient descent
moved in the wrong direction exactly at the registered initial point.

This protocol repairs and tests that backward ambiguity.  It does not change
the forward SNN, claim autonomous delay discovery, or authorize K>1.

## Backward estimators

Let the logical delay buffer be `s[d]`, let `k=floor(d)`, and let
`alpha=d-k`.  Every condition retains the same forward value

`s_forward(d) = (1-alpha) s[k] + alpha s[k+1]`.

The historical estimator uses the right slope `s[k+1]-s[k]`.  At exact
integers, `symmetric_integer` substitutes

`(s[k+1]-s[k-1]) / 2`

with one-sided handling at the legal boundaries.  `gaussian_ste` retains the
same hard/linear forward value but uses the derivative of a normalized
Gaussian tap kernel in the backward pass.  Sigma values 0.50, 0.75, and 1.00
are screened as explicitly registered alternatives.

The implementation uses a zero-valued straight-through correction, so any
forward difference is a protocol-invalidating bug.

## P0 screen and decision

P0 contains no optimizer update.  It evaluates five frozen W0 foundations,
two perturbations (`d=3`, `d=5`), two delay granularities (global and
per-hidden), and five estimators: 100 deterministic gradient probes.

Selection is not based on lowest loss.  A candidate must preserve all forward
records, give target-directed and nonzero global gradients for every seed and
side, give a target-directed mean per-hidden gradient for every seed and side,
retain at least 75% nonzero per-hidden coordinates, and improve the late-side
coordinate direction fraction over the historical estimator.  The least
interventional eligible estimator wins.  If none passes, P1 stops and the next
repair target is the task-loss temporal credit path, not the learning rate.

P1, if unlocked, will explicitly separate exact functional recovery from
identification of the oracle delay vector.  The former cannot be used as a
surrogate for the latter.

## P0 decision (observed after registration)

All 100/100 probes completed and every matching forward record was bitwise
identical across estimators.  Only `gaussian_ste` with sigma 1.00 passed every
registered direction gate: global and mean per-hidden gradients were
target-directed for 10/10 seed-side probes, and every coordinate remained
nonzero.  It therefore unlocks P1 without a post-hoc rule change.  This result
does **not** yet show recovery: on the difficult `d=5` side, only 48.75% of
individual per-hidden coordinates were target-directed (legacy: 13.75%).
Thus aggregate credit is repaired, while coordinate-level identification
remains questionable.

## P1 decision (observed after authorization)

P1 passes neither registered endpoint.  Exact functional recovery is 2/10
(d3 2/5, d5 0/5), full 16-coordinate schedule identification is 0/10,
and the joint repair gate is 0/10.  The two functional cells reach zero task
loss with all delays still below the oracle target, while d5 cells again split
toward the legal bounds.  The backward intervention is therefore necessary
for aggregate direction but insufficient for independent-coordinate credit
and schedule identification.  See
[`RESULTS_XOR_INTEGER_BOUNDARY_CREDIT_PREFLIGHT_V1.md`](RESULTS_XOR_INTEGER_BOUNDARY_CREDIT_PREFLIGHT_V1.md).

The machine-readable frozen specification is
[`configs/xor_integer_boundary_credit_preflight_v1.yaml`](../configs/xor_integer_boundary_credit_preflight_v1.yaml).
