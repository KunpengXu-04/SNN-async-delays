# Results: XOR Integer-Boundary Credit Preflight v1

Status: complete; P0 estimator screen passed; conditional P1 recovery failed.

## Executive judgment

The historical exact-integer backward rule was a real bug-like optimization
pathology, and the new backward-only implementation repairs it at the
**aggregate scalar-gradient** level without changing the SNN forward pass.
That repair is not sufficient for the 16-parameter per-hidden model.

The registered P1 endpoint is negative: functional recovery is 2/10, recovery
of all 16 oracle delays is 0/10, and the joint gate is 0/10.  Both functional
successes are on the early `d=3` side; `d=5` remains 0/5.  Therefore this
protocol does not rescue task-derived delay learning and does not unlock K>1.

## Implementation and validity

`DelayedSynapticLayer` now supports three explicit backward modes:

- `right_linear`: unchanged historical default;
- `symmetric_integer`: central subgradient only at exact integer delays;
- `gaussian_ste`: normalized Gaussian-kernel derivative with unchanged
  hard/linear forward values.

The model and Level-1B builder propagate the versioned mode and sigma.  State
dicts remain compatible because the new settings are non-tensor attributes.
Eleven focused tests verify forward identity, integer/fractional gradient
definitions, circular-buffer handling, Gaussian derivative correctness, and
invalid-configuration rejection.  The complete suite passes 120/120.

P0 additionally compares the full input, hidden-spike, output-spike,
output-voltage and output-current records.  All matching records are bitwise
identical across all five estimators in all 20 seed/granularity/perturbation
groups.  Thus the observed differences are backward-credit differences, not a
changed forward model.

## P0: 100-probe gradient screen

The frozen matrix is five W0 checkpoints × two perturbations × global/per-hidden
delay tying × five estimators, with zero optimizer updates.

| estimator | global target-directed | per-hidden mean target-directed | d3 coordinate direction | d5 coordinate direction | eligible |
|---|---:|---:|---:|---:|---:|
| historical right-linear | 5/10 | 5/10 | 0.8500 | 0.1375 | no |
| symmetric integer | 9/10 | 9/10 | 0.8375 | 0.4625 | no |
| Gaussian STE, sigma 0.50 | 9/10 | 9/10 | 0.8375 | 0.4750 | no |
| Gaussian STE, sigma 0.75 | 9/10 | 9/10 | 0.8375 | 0.4750 | no |
| Gaussian STE, sigma 1.00 | 10/10 | 10/10 | 0.8375 | 0.4875 | yes |

Every independent coordinate had a nonzero gradient.  The preregistered rule
therefore selected Gaussian STE with sigma 1.00 and unlocked P1.  This was a
legitimate selection under the frozen rule, but the screen also exposed a
serious warning: at `d=5`, even the winner directs only 48.75% of individual
per-hidden coordinates toward four.  A correct mean gradient is not equivalent
to correct independent-coordinate gradients.

## P1: conditional task-only recovery

P1 freezes both weight matrices, removes the arrival teacher, starts all 16
functional delays at exactly three or five, and applies 500 full-batch task-only
delay updates with the selected backward estimator.

| initial delay | final exact patterns by seed | exact interface | all-delay recovery |
|---:|---|---:|---:|
| 3 | 3, 2, 4, 3, 4 | 2/5 | 0/5 |
| 5 | 0, 3, 0, 0, 1 | 0/5 | 0/5 |
| total | — | 2/10 | 0/10 |

The two d3 functional successes reach all four exact target spike trains at
updates 20 and 27.  Their task loss and task-delay gradient then become exactly
zero, freezing non-oracle delay vectors:

- seed 2539: range `2.4446–3.6040`, mean `3.1828`;
- seed 2579: range `2.3243–3.7292`, mean `3.4104`.

This is direct evidence that the output task does not identify the oracle
vector.  Exact output timing can be implemented by many internal schedules.

The late-side failure is stronger.  Across d5 cells, final coordinates span
as far as `0.2218–7.6572`; no cell has more than one of sixteen coordinates
within 0.1 step of four.  In seed 2503, the task loss worsens from `0.02884` to
`0.06208` while the delay standard deviation grows above three.  Its diagnostic
panel shows the target-directed coordinate fraction decaying toward zero as
the delay vector splits toward both bounds.

## Diagnosis

Three mechanisms must be distinguished:

1. **Exact-integer derivative ambiguity — repaired but not sufficient.**
   Gaussian STE removes the globally wrong late-side direction and preserves
   the forward computation.
2. **Independent-coordinate credit — still broken.** Adam updates 16 delay
   coordinates separately.  A target-directed mean gradient cannot prevent
   roughly half of d5 coordinates from moving the wrong way.
3. **Schedule non-identifiability — structural.** Once the exact four output
   spike trains are reached, the task loss is zero and supplies no reason for
   the internal delays to equal the experimenter-defined oracle value four.

The result therefore rejects the explanation that W2 failed only because of a
one-sided integer subgradient.  It also shows why increasing LR, epochs, or
threshold sweeps is not a principled next repair: none addresses independent
coordinate disagreement or a zero-loss manifold.

## Claim boundary and next admissible discriminator

Supported narrowly: a smooth backward-only estimator can make the aggregate
task gradient bidirectional at exact d3/d5 while preserving hard forward
dynamics.

Not supported: reliable per-hidden recovery, identification of the oracle
delay vector, autonomous timing discovery, heterogeneous routing, K>1,
compression, or a Pareto advantage.

The next minimal causal discriminator should use the selected Gaussian
backward with **one globally tied delay parameter**, the same frozen W0 weights,
the same d3/d5 perturbations, and no oracle teacher.  P0 already shows 10/10
target-directed global gradients.  If global recovery succeeds, the remaining
failure is chiefly overparameterization/identifiability; only then should a
target-free consensus or low-rank delay parameterization be compared with the
global anchor.  If global recovery fails, the task-loss temporal credit itself
must be redesigned (for example a continuous output-voltage timing objective)
before any parameterization rescue.

This next discriminator requires a new preregistration.  It must not reuse P1
as confirmatory evidence or silently weaken schedule recovery to mean delay.

## Artifacts

- P0 summary: `docs/generated/xor_integer_boundary_credit_preflight_v1/stage_p0_gradient_screen/summary.json`
- P0 probes and panel: the adjacent CSV and `diagnostic_panel.png`
- P1 summary: `docs/generated/xor_integer_boundary_credit_preflight_v1/stage_p1_conditional_recovery/summary.json`
- P1 cell metrics and panel: the adjacent CSV and `diagnostic_panel.png`
- Every P1 cell contains a checkpoint, training log, resource ledger,
  exhaustive truth-table result, compressed runtime NPZ, and diagnostic panel.
