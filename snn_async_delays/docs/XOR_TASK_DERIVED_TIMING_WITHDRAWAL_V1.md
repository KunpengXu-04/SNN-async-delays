# XOR task-derived timing scaffold withdrawal v1

Status: complete, 60/60 total formal cells; primary local-restoration claim
fails and K>1 remains locked.

## Why this is the next experiment

The 40-cell learned micro-burst matrix gives two simultaneous facts. With an
explicit arrival teacher, global, per-hidden, and per-synapse delays each pass
10/10. Without that teacher, per-hidden task-only passes 0/10. The former is
not autonomous learning; the latter already rejects reliable from-scratch
schedule discovery under the registered setup.

It would therefore be scientifically wrong to repeat from-scratch task-only
training and label it “withdrawal.” This protocol asks the narrower causal
question that remains open: once a teacher has created the correct basin, does
the task objective contain enough timing information to preserve or locally
restore the schedule?

## Frozen system

The task, exhaustive four-pattern batch, `{8,9}` consecutive micro-burst,
`4 -> 16 -> 2` spiking architecture, hidden/output thresholds `.2/.03` a.u.,
sigmoid input-hidden delays with `dmax=8`, d0 hidden-output delay, filtered
hard-spike loss, Adam learning rates `.01`, 500 updates, and final-checkpoint
selection are unchanged. Per-hidden-neuron tying is frozen because it was the
registered lower-complexity winner: 16 delay values rather than 64.

Five new formal seeds are frozen: `{2503,2521,2539,2551,2579}`. They do not
overlap parent selection, confirmation, or smoke seeds.

## Three mechanically locked stages

### W0: independent teacher-built foundations (5 cells)

Each seed starts at raw delay `-2` and independently rebuilds the selected
per-hidden `.16` scaffolded solution. All weights and delays train. W1 and W2
remain absent unless all five cells pass the exact output-spike interface,
coordinate-gradient, and full-delay-coverage gates.

W0 is complete. All five fresh-seed cells pass. Every final truth table has one
exact output spike at step 15, all 16 delays lie within `.1` step of four, and
no gradient clipping occurs. Maximum coordinate error ranges from `.000115` to
`.002032` step. W1 is therefore mechanically authorized; W2 remains locked.

### W1: perturbation validity (10 cells)

For each W0 checkpoint, the 16 functional delays are overwritten with exactly
3 or 5 steps using

\[
r(d)=\log\frac{d}{d_{\max}-d},\qquad d_{\max}=8.
\]

Weights are inherited and no parameter is updated. All ten cells must lose the
exact target spike train. This is a damage gate: if a perturbation does not
change the functional interface, “recovery” from it is meaningless. The
perturbation magnitude cannot be changed after observing this gate.

W1 is complete and passes 10/10. Every d3 cell emits the otherwise correct
class spike at step 14; every d5 cell emits it at step 16. Thus classification
accuracy remains one in all cells while exact target-time accuracy becomes
zero. Non-delay state is bitwise identical to the matching W0 checkpoint, all
trainable-component lists are empty, and every branch records zero updates.
W2 is mechanically authorized.

A disclosed read-only audit before W2 found strongly asymmetric task credit.
At d3, the mean fraction of target-directed delay coordinates is `.85`; at d5
it is only `.1375`. Every coordinate has a nonzero gradient, but the d5 norm is
also smaller. No W2 training was run and no registered setting or gate was
changed. This asymmetry is now a prospective mechanism prediction, not a
post-hoc explanation.

### W2: retention and recovery (45 cells)

Retention uses the unperturbed W0 checkpoint, removes the arrival teacher,
freezes both weight matrices, and updates only delays under task loss (5 cells).

Each perturbation direction then has four branches per seed:

| Branch | Trainable state | Loss | Inferential role |
|---|---|---|---|
| oracle-delay-only | delays | arrival teacher only, `.16` | positive optimizer control |
| task-delay-only | delays | task only, `lambda=0` | primary local-restoration test |
| task-joint | weights and delays | task only | compensation-aware secondary test |
| task-weight-only | weights | task only | weight-compensation control |

Every branch begins from the same matching-seed W0 weights and receives a
fresh Adam state. Optimizer momentum is never inherited.

## Formal decision

The primary claim requires all of the following:

1. W0 foundation 5/5;
2. W1 genuine interface damage 10/10;
3. retention 5/5;
4. oracle-delay-only recovery 10/10;
5. task-delay-only recovery 10/10, including exact spike trains and all 16
   delays within `.1` step of four.

Initial task-gradient direction and nonzero coverage are mandatory mechanism
diagnostics, not endpoint gates. A nonlinear trajectory can recover after an
initially indirect coordinate step, so making 16/16 initial directions a pass
condition would conflate local geometry with eventual restoration. These
fractions must nevertheless be reported for every cell and used to explain
failures; they cannot be omitted when inconvenient.

Task-joint or weight-only success cannot rescue a failed primary arm. Such a
result would instead show compensation. If the oracle control passes but the
task-delay-only arm fails, the correct conclusion is that this hard-spike task
loss lacks reliable local timing credit under the frozen system.

W2 is complete. Retention passes 5/5 and the oracle positive control passes
10/10. Task-delay-only passes 0/10 under the registered joint functional and
all-coordinate endpoint. Directionally, d3 recovers the exact output interface
in 3/5 cells but never recovers all 16 delays; d5 recovers the exact interface
in 0/5 and drives half of all coordinates near the 0/8 boundaries. Task-joint
recovers the interface in 5/5 d3 and 3/5 d5 cells but never the declared delay
schedule. Weight-only passes only 1/10. The formal protocol therefore fails.

The failure is mechanistically localized. The delay interpolation uses
`floor(d.detach())` and at an exact integer differentiates only through the
right-hand interval. With matching W0 weights and a single globally tied delay,
the mean task gradient changes from `+.00408` at `d=4.999` (correct descent
toward four) to `-.00356` at `d=5.000` (incorrect descent toward larger delay).
Global tying therefore does not solve the late-side failure. Separately, once
an exact spike train is reached, task loss is zero even when the 16 delays are
heterogeneous, so the complete homogeneous schedule is not identifiable from
the four output patterns alone.

## Claim ceiling

Even a full pass establishes only local restoration in a teacher-created
homogeneous basin. It does not overturn the observed 0/10 failure of global
from-scratch discovery. It says nothing about heterogeneous routing, K>1,
temporal multiplexing, WAD superiority, compression, energy, or Pareto laws.

The machine-readable source of truth is
`configs/xor_task_derived_timing_withdrawal_v1.yaml`. The runner is
`scripts/run_xor_task_derived_timing_withdrawal.py`.
