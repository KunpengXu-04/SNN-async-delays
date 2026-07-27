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

## 13. Dimension-aware micro-burst rescue decision (2026-07-16)

Do not run the original frozen 60 learned Stage-B cells as the main rescue:
they retain the already diagnosed `1/P` teacher dilution. Preserve them as an
unrun historical protocol. The independent replacement is
`xor_delay_granularity_rescue_microburst_v1`.

The new bridge retains the exact hard-spike model, thresholds, LRs, 500-update
budget, d0 hidden-output delay and delay-4 target. It changes only the encoding
to consecutive events at steps 8 and 9 and transfers the confirmed analytic
weights `.01/.16/.64`. The primary candidate is per-hidden, because sealed R3
selected it over per-synapse on parameter count. Per-synapse remains a
secondary complexity control and per-hidden task-only is diagnostic.

Stage B0 contains five fresh-seed fixed-d4 cells and must pass 5/5 before the
40 learned cells unlock. Formal seeds are `{2333,2351,2371,2389,2411}`. Seed
`2309` was used accidentally during implementation smoke, permanently removed
from the formal set and retained only as invalid smoke provenance. All future
smoke runs use dedicated `99xxx` seeds. No outcome in this protocol directly
authorizes K>1; a full pass authorizes only a newly preregistered
scaffold-withdrawal/task-derived timing test.

Stage B0 is complete. Fixed d4 passes all five fresh formal seeds with exact
four-pattern spike trains at step 15, zero silence/collision, one output event
per trial and complete runtime artifacts. This is a feasibility replication,
not learned-delay evidence. Its mechanical decision unlocks the entire
40-cell learned matrix; no condition or seed may be omitted.

The learned matrix is also complete. Global `.01`, dimension-matched
per-hidden `.16`, and dimension-matched per-synapse `.64` pass all ten cells
each. Per-hidden task-only passes zero of ten. This is a clean separation:
the hard-spike interface and high-dimensional delay parameterizations are
trainable when an explicit coordinate-wise timing teacher is present, but the
registered task loss does not reliably discover the schedule globally.

## 14. Scaffold-withdrawal decision (2026-07-16)

Do not call another task-only from-scratch run “scaffold withdrawal”: that
question has already failed 0/10. Withdrawal must start from an independently
recreated teacher-built per-hidden checkpoint and remove the teacher without
changing the task, interface, optimizer, or training budget.

The next protocol therefore separates retention from restoration. Retention
keeps the exact checkpoint, freezes both weight matrices, sets `lambda=0`, and
updates delays only. Restoration overwrites all 16 functional delays with a
prespecified symmetric perturbation, either 3 or 5 steps. No-update controls
must first show that each perturbation breaks the exact spike interface;
otherwise recovery in that direction is non-informative and remains locked.
Oracle-delay-only recovery is the optimizer positive control. Task-delay-only
recovery is the primary causal test. Task-joint and weight-only branches test
whether apparent recovery is instead compensation by synaptic weights.

Even a 10/10 primary pass supports only local task-derived restoration within
a teacher-created basin. It cannot reverse the 0/10 global-discovery result,
and it cannot authorize heterogeneous schedules, K>1 temporal multiplexing,
compression, or Pareto claims.

W0 is now complete. All five fresh foundations reproduce the exact interface
and all-coordinate delay-four solution. The largest maximum coordinate error
is `.002032` step; no cell clips and all artifact bundles are complete. This
does not add task-derived evidence. It only satisfies the provenance gate for
the ten prespecified no-update d3/d5 damage controls. W2 remains locked.

W1 is complete and satisfies the damage gate. The intervention is clean:
non-delay state is bitwise unchanged, functional delays are exactly three or
five, no component is trainable, and no optimizer update occurs. All d3 cells
emit the correct class at step 14; all d5 cells emit it at step 16. Therefore
classification remains perfect but exact timing fails 10/10. This is precisely
why W2 must retain the exact-spike endpoint rather than use accuracy alone.

Before W2, a declared read-only task-gradient audit found mean target-directed
coordinate fractions `.85` at d3 and `.1375` at d5; all coordinates are
nonzero. This predicts asymmetric restoration and exposes the likely weakness
of the task loss on the late side. It does not alter any W2 arm, seed, LR,
budget, endpoint, or decision rule.

W2 is complete and the formal local-restoration claim fails. Retention passes
5/5 and the oracle delay-only positive control passes 10/10, ruling out generic
optimizer incapacity. Task-delay-only passes 0/10: d3 recovers exact function
in 3/5 but the complete delay schedule in 0/5; d5 recovers neither in 0/5 and
polarizes half its coordinates near the delay bounds. Task-joint exact function
in 8/10 with delay recovery 0/10 is compensation/non-identifiability, not delay
restoration.

Do not respond by tuning LR, threshold, training length, or seeds. The
production interpolation uses `floor(d.detach())`, so an exact integer receives
the right-hand derivative. With globally tied delay and the same W0 weights,
the mean task gradient is `+.00408` at 4.999 but `-.00356` at 5.000; descent
therefore moves the registered d5 condition later. Parameter tying alone does
not fix this. The next method decision must compare symmetric/central or smooth
delay credit and continuous output-level timing losses in a small gradient-field
preflight. K>1 remains locked.

That registered preflight is now complete. Read
`docs/RESULTS_XOR_INTEGER_BOUNDARY_CREDIT_PREFLIGHT_V1.md`. Gaussian STE sigma
1.00 fixes aggregate exact-integer direction without changing forward records,
but per-hidden P1 passes exact function only 2/10 and schedule identification
0/10. Never infer coordinate recovery from mean gradient or exact output
behaviour. The next allowed discriminator is globally tied one-delay task-only
recovery under a new frozen protocol. Only a successful global anchor may
motivate target-free consensus or low-rank per-hidden tying; a failed global
anchor instead requires a continuous task-timing credit redesign. K>1 and
Pareto experiments remain unauthorized.

## 15. Separate capacity from autonomous trainability (2026-07-21)

Close the K=6 centroid-supervised surface as an assisted capacity branch. Its
unstable fresh-seed boundary is not a basis for further K=6 expansion, an
L-shaped law, or autonomous delay claims. Preserve all historical artifacts.

Use two sequential evidence chains. Fixed schedules and selected explicit-
centroid checks answer architectural capacity; task-only SLAYER asks whether
the schedule is learnable without a timing teacher. Learned-delay scaling is
forbidden unless SLAYER passes the registered K=1 and K=2 gates. A failed K=2
gate is a negative trainability result and stops that surface.

Use repeated XOR so K changes query load without changing operation difficulty.
Use deterministic four-step binary one-hot packets so input count is exactly
`8K` and cannot grow with T. Retain the shared windowed MLP for the main
capacity comparison; shared opponent spikes are a separately reported,
conditional follow-up.

SLAYER is an optional backend, not a replacement simulator. Put one
`slayer.axon.Delay` per input value channel (`4K`) before a dense CUBA block and
keep Lava-DL in a pinned environment. On this host, record the pure-PyTorch
fallback for Lava-DL's optional JIT CUDA kernels because MSVC is unavailable;
the SLAYER delay autograd and CUBA model remain in use.

S0 passes both invalid smoke seeds, including full-model CPU/CUDA parity and
checkpoint recovery. This satisfies the technical prerequisite but does not
launch S1: calibration stays locked in YAML. After validated S1/S2 decisions,
freeze N_ref from fixed-oracle results before temporal scanning. Analysis must
retain raw non-monotonic rows, never use isotonic repair, never convert censored
K to a number, and never call a single resource scalar “energy.”
