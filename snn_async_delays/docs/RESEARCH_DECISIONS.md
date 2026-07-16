# Research decisions: seven-section guide

This is the compact guidance record requested during the July 2026 reset.  It
does not replace the protocol; it explains why the protocol has its current
shape.

## 1. Scientific objective

Study whether learned delays improve the *matched*
resource--reliability Pareto frontier for `K` temporal queries.  A result is
not a capacity result if latency, output interface, decoder capacity, delay
range, or memory changes with `K` without being counted.

## 2. Encoding policy

Deterministic burst is the primary mechanism probe because event time and
event count are controlled.  Rate and jittered burst are robustness conditions,
not interchangeable continuations of the same experiment.  The jitter encoder
now preserves event count to avoid a timing/SNR confound.

## 3. Plan D task interpretation

Sequential subwindows test retention and late alignment.  They do not alone
establish temporal multiplexing because early query age, total duration,
output dimension, and maximum delay can co-vary with `K`.  Pair Plan D with
simultaneous and fixed-total-duration tasks.

## 4. Readout interface

The final-window count readout is a named late-alignment condition.  It can
erase d0 activity before it is observed.  Compare it with an all-time or
time-binned interface and, when appropriate, a matched spiking-output layer.
Spiking outputs are an ablation, not a default fairness fix: they add neurons,
synapses, delays, thresholds, output events, and another decision window.

## 5. Reliability endpoint

Use worst-query accuracy and exact-trial accuracy as primary endpoints;
report balanced accuracy for imbalanced Boolean operations.  `90%` and `95%`
are predeclared summary thresholds, not a substitute for full curves or
seed-level uncertainty.  Never use pooled accuracy alone to claim Max-K.

## 6. Mathematical cost programme

Report the resource vector before any scalar proxy.  A transparent proxy may
include neuron updates, input/hidden/output events, delay-buffer memory, and
decoder operations.  Compare temporal sharing with replication only under an
equal reliability constraint, using the ratio of minimum feasible costs.

## 7. Diagnostics and reproducibility

Mechanism requires a chain from input event time to effective arrival time,
hidden activity, and readout contribution, plus shuffle/ablation controls.
Rasters must be predeclared examples or success/failure pairs, not a selected
“richest” sample.  Historical artifacts remain immutable; new claims require
registered configs, versioned output, validation-only model selection, and
multi-seed reporting.

## 8. Temporary MLP scaffold decision (2026-07-14)

Use an MLP endpoint temporarily to expose the hidden-dynamics `T x h` surface
for supervisor discussion, but keep it in a separate exploratory protocol.
Independent spatial modules and temporal windows share the same one-output MLP
where structurally possible. Temporal candidates use `windowed_shared`; an
all-time K-output MLP is retained only as the shared-representation control.
All plots must separately charge total hidden neurons, latency, updates, dense
MACs, measured events, decoder operations and delay storage. Success does not
solve the direct-spiking-output problem and cannot unlock formal Stage B.

## 9. Learned-delay mechanism ladder decision (2026-07-15)

Do not diagnose task-level WAD by immediately adding width, K or a new delay
tying scheme. Levels 0A-0D isolate parameter movement, buffer credit, soft
state credit and hard-output compatibility in that order. Level 0D selects
hard filtered-spike loss plus synaptic-current centroid at `lambda=.1` for the
one-event bridge (15/15); pre-reset centroid is rejected (3/15 soft-only, eight
zero-gradient pairs). These numerical settings are not transferable defaults.
The next allowed gate is a newly preregistered K=1 XOR calibration that exposes
task and auxiliary losses separately and audits delay versus weight gradients.
K>1, per-neuron tying and Pareto surfaces remain downstream of that gate.

## 10. Level 1A task-level bridge decision (2026-07-16)

The registered bridge is `xor_task_bridge_level1a_v1`. It is an exhaustive
K=1 XOR optimization calibration, not a generalization or delay-benefit study.
Use one event on each selected one-hot input channel so every pattern has the
same two-event load; defer consecutive micro-bursts to Level 1B. The production
model is `4 -> 16 -> 2` with hard opponent output spikes, input-to-hidden delay
only, and d0 hidden-to-output transmission. The learned delay is deliberately
one global shared scalar broadcast over all input-hidden pairs; per-neuron and
per-synapse granularity are later controlled factors, not Level-1A tuning
choices.

Stage I must first prove that the interface itself works under both causal d0
and fixed delay 4 in every seed. Stage II remains mechanically locked until
that decision passes. Its arrival-centroid term is an explicitly labelled
oracle timing scaffold and is never evidence that task supervision discovered
routing. Preserve exact hard-train matching, silence/collision counts, target
time, hidden activity, component delay gradients, weight-gradient norms, NPZ
state and the runtime diagnostic panel. Stage I selects `eta=0,lr_w=.01` after
a 10/10 exact-interface pass. Stage II is complete and selects
`lambda=.01,lr_d=.01` after a 10/10 learned-delay pass. Task-only training is
not reliable, while the explicit arrival teacher repairs five conflicting
initial task gradients. Treat this as scaffold-assisted compatibility, not
task-derived routing. Level 1B may now preregister K=1 granularity and
micro-burst robustness; K>1, WAD and Pareto work remain locked.

## 11. Level 1B granularity and micro-burst decision (2026-07-16)

Do not jump from one shared scalar directly to K>1 routing. First test whether
the Level-1A scaffold-assisted bridge survives an increase in optimization
dimension at unchanged K=1. The registered input-delay granularities are one
global scalar, one scalar per hidden neuron and one scalar per input-hidden
pair, corresponding to 1, 16 and 64 trainable delay values over the same 64
physical synapses. All coordinates start at the same functional delay; this is
an optimization/credit-assignment comparison, not heterogeneous routing.

The arrival teacher is defined per independent coordinate. A global mean
centroid is prohibited because early and late errors could cancel. Formal
success requires the exact hard-spike interface, maximum delay error at most
`.1` step, delay coverage one, and correct nonzero initial total-gradient
direction for every coordinate across both initial directions and all five
new seeds. Task-only and scaffold results must remain separate.

Consecutive micro-bursts are a second, conditionally locked question. Stage A
must first replicate the global scaffold in 10/10 held-out cells. Fixed delay
4 must then solve the two-event-per-selected-channel interface in 5/5 seeds
before learned micro-burst cells run. One exact output spike at `t=15` remains
the endpoint; extra or early output spikes are failures. No Level-1B outcome
authorizes K>1, resource-frontier or WAD-superiority claims without a new
decision and protocol.

Formal Stage A is now complete. Global scaffold replication passes 10/10, but
the per-hidden and per-synapse candidates pass only 2/10 and 0/10. Per-hidden
solves the exact output interface in every cell, so its failure is specifically
coordinate-wise timing recovery. The mean arrival loss scales each coordinate's
teacher gradient as `1/P`; therefore interpret this as failure of the frozen
Level-1A recipe under increased dimension, not proof that higher-dimensional
delays are intrinsically impossible to train. The only authorized next action
is the ten-cell fixed micro-burst control matrix. Learned micro-burst and K>1
remain locked.

The fixed micro-burst controls are now complete. Fixed delay 4 passes 5/5 with
one exact target spike at `t=15` in all 20 patterns; fixed d0 passes 0/5 and
emits only at `t=11`. The interface is therefore feasible and timing-specific,
and the original 60 learned micro-burst cells clear their mechanical gate.
They have not been launched. Their frozen mean-loss scaling must not be changed
after seeing Stage-A failures. A dimension-normalized repair is a separate
protocol and should use new seeds; completing the original learned matrix and
testing a rescue answer different questions.

## 12. Dimension-aware rescue decision (2026-07-16)

Do not repair Level 1B in place. The versioned rescue is
`xor_delay_granularity_rescue_level1br_v1`, and all Level-1B cells and
candidate decisions remain immutable.

R1 tests one analytic intervention before any LR sweep: retain the mean
per-coordinate arrival loss but set `lambda_P=.01P`. This changes effective
lambda from `.01` to `.16` or `.64` for per-hidden and per-synapse delays so
that the weighted teacher gradient on each coordinate matches the global
anchor. It deliberately changes from fixed total teacher strength to fixed
teacher strength per parameter; any positive result must disclose that total
oracle supervision grows with `P`.

LR or training-budget escalation is conditional. It is allowed only when all
coordinates already have correct nonzero initial total gradients and failure
is therefore plausibly convergence-limited. Wrong-direction cells cannot be
repaired by multiplying LR. Conditional calibration uses new seeds and any
selected recipe must replicate on sealed R3 seeds. Per-hidden is preferred if
both granularities confirm because it uses fewer parameters. No rescue outcome
authorizes micro-burst, K>1 or routing claims without a new protocol.

R1 is complete (50/50). The global anchor passes 10/10; the unscaled
per-hidden and per-synapse baselines pass 1/10 and 0/10; the dimension-matched
conditions pass 10/10 and 10/10. Their initial weighted arrival gradient per
coordinate matches the global anchor at approximately `.025588`, and no cell
triggers gradient clipping. R2 is therefore not required. The only authorized
next rescue action is the registered 30-cell R3 sealed confirmation. Until R3
passes, call both higher-dimensional recipes provisional. Even after
confirmation, the claim remains explicit-oracle homogeneous delay recovery,
not autonomous routing.

Sealed R3 is also complete (30/30). Global, per-hidden and per-synapse each
pass 10/10 with no clipping or missing artifacts. Both higher-dimensional
recipes confirm, so the registered lower-complexity priority selects
per-hidden-neuron (16 independent delay parameters) over per-synapse (64).
Level 1B-R is complete. This result does not unlock micro-burst or K>1 by
itself; a downstream experiment requires a new explicit decision and protocol.
