# Delay SNN: temporal routing under matched resources

This repository investigates a restricted question: whether learnable synaptic
delays improve the **resource--reliability frontier** for multi-query temporal
Boolean tasks, after latency, network size, output interface, delay memory, and
evaluation protocol are matched.

It does **not** currently establish a general energy advantage, a capacity
advantage at fixed resources, or biological temporal multiplexing.  Historical
experiments remain in `runs/` as immutable evidence; their scientific status is
tracked by [`docs/EXPERIMENT_LOG_V2.md`](docs/EXPERIMENT_LOG_V2.md) and
[`docs/CLAIMS_LEDGER.md`](docs/CLAIMS_LEDGER.md), not by directory name.

**Current gate (2026-07-15):** the XOR delay-control matrix rejected WAD
superiority over optimized scalar delay, and the completed 42-cell optimization
audit found no rescue variant. Calibration v1 was stopped because temporal d0
could not produce activity in post-input output windows. Version 2 completed
and froze output threshold 0.2. It calibrated firing but did not establish
routing: temporal worst-balanced remained .5 and only NAND exceeded chance.
The 36-cell spatial pilot is complete. Its primary linear endpoint shows no WAD
advantage over d0 or matched fixed delay; MLP is fully saturated. A secondary
opponent-output exact-trial signal is exploratory only and costs more synaptic
events than scalar/d0. A six-cell temporal viability preflight is preregistered
(not launched) to audit d0/scalar/narrow-fixed/full-support-fixed/scaffold/WAD
using event-support and gradient gates rather than accuracy. That preflight is
now complete and formally failed its hidden-emission gate. A subsequent
checkpoint audit showed that this gate was a wrong proxy: WAD has third-window
output current in 70.3% of exhaustive trials and nonzero window-specific
gradients, but balanced accuracy remains at chance. Preflight v2 is
now complete and passes every mechanism-valid gate on both held-out seeds.
However, WAD worst-window balanced accuracy remains `.5` because the fixed
NOR/window-2 target is not learned. Viability is established; reliable routing
is not. The superseding 45-cell counterbalanced primary-opponent protocol is
complete. No condition passes the preregistered routing floor: WAD's worst
operation-position score is `.5` in every seed. WAD is best on the secondary
mean balanced (`.734`) and exact-trial (`.411`) metrics, but fails late XOR and
NOR and uses more events than fixed-full. This separates the earlier failure
from NOR identity: temporal position is the dominant bottleneck, with an
operation interaction. Do not open test or start K/N/T scaling before deciding
whether to test a coverage-aware delay method or pivot to the negative
methodology result. See
`docs/RESULTS_SIMULTANEOUS_TEMPORAL_COUNTERBALANCED_PERFORMANCE_V1.md`.

The next versioned programme is
`spatial_vs_temporal_pareto_phase0`: first calibrate a K=1 repeated-XOR
opponent target-spike interface and single-query width, then compare K=2
block-diagonal spatial modules against shared spatial/temporal networks. It
uses equal input-event counts and separates neuron-area compression from
latency and synaptic compute. Formal Stage A is complete and failed: no width
passes every reliability/output-interface gate across all three seeds. Stage B,
mixed operations and test access remain locked. See
`docs/RESULTS_SPATIAL_VS_TEMPORAL_PARETO_PHASE0_STAGE_A.md`. The completed
read-only checkpoint audit confirms all delays were d0, decomposes silence into
hidden-representation and output-conversion failures, and shows every collision
contains an early wrong spike at t=10 before the correct t=11 spike. See
`docs/RESULTS_SPATIAL_VS_TEMPORAL_PARETO_PHASE0_STAGE_A_CHECKPOINT_AUDIT.md`.

For a supervisor-facing progress diagnostic, the separate exploratory MLP
scaffold is complete (160/160 cells). At the preregistered robust `.90/.90`
worst-query/exact-trial rule, only the independent spatial baseline and the
fixed query-scheduled temporal oracle pass, both at `T=18, h=24`. Relative to
the spatial baseline, the oracle uses one-half the hidden neurons and hidden
neuron updates, but equal dense MACs, measured synaptic events, delay-value
storage and decoder MACs, plus five times the delay-buffer storage. WAD fails
in every cell (`worst-query balanced accuracy=.5`): its two query delay
distributions remain indistinguishable and the second output window receives
no hidden spikes. Thus the experiment supports only a narrow *fixed-schedule
hidden-area/update* result, not learned temporal routing, compute/energy
compression, or a repaired spiking-output result. Width dominates the measured
accuracy surface; the tested `T` range does not identify a credible `T x h`
cost law. See `docs/RESULTS_SPATIAL_VS_TEMPORAL_PARETO_MLP_SCAFFOLD_V2.md` and
`docs/SPATIAL_VS_TEMPORAL_PARETO_MLP_SCAFFOLD_V2.md`. V1 was aborted after a
runtime-budget error (2,400 repeated full-batch updates per cell) and must not
be pooled with V2.

The learned-delay mechanism fork now starts below XOR. Levels 0A-0C reject a
hard scalar/buffer obstruction and identify current centroid as bidirectional
continuous timing credit. Level 0D is complete (135/135) and passes its hard-
output bridge: hard filtered-spike loss alone is 10/15, while hard loss plus
current centroid at `lambda=.1` is 15/15 with correct nonzero initial direction
in all 13 misaligned pairs. Pre-reset centroid is not a valid substitute
(3/15 soft-only; eight zero-gradient pairs). This remains one fixed-weight
synapse and one event, not XOR or routing evidence. Level 1A K=1 XOR
calibration Stage I is now complete (90/90) and passes. The unique selected
interface is `eta=0,lr_w=.01`, which solves both fixed d0 and fixed delay-4 in
all five seeds with exact hard-spike outputs. Stage II is also complete (85/85)
and Level 1A passes. Task-only delay learning is not reliable (0/10 at
`lr_d=.01`, 1/10 at `.05`), whereas the selected explicit timing scaffold
`lambda=.01,lr_d=.01` reaches delay 4 and exact hard-spike XOR in all ten
initialization/seed cells. This is a scaffold-assisted bridge, not autonomous
routing discovery. Level 1B Stage A is now complete (60/60). The global
scaffold replicates in 10/10 new-seed cells, but per-hidden-neuron and
per-synapse candidates pass only 2/10 and 0/10; the frozen higher-dimensional
extension therefore fails. The ten fixed micro-burst controls are complete:
fixed delay 4 passes 5/5 with exact outputs at `t=15`, while fixed d0 passes
0/5 and emits only at `t=11`. The frozen learned micro-burst matrix is now
mechanically authorized but unrun. K scaling, routing, WAD superiority and
Pareto work remain locked. See
`docs/RESULTS_DELAY_TEMPORAL_CREDIT_LEVEL0B_V1.md` and
`docs/RESULTS_DELAY_SOFT_TRACE_CREDIT_LEVEL0C_V1.md` and
`docs/RESULTS_DELAY_HARD_OUTPUT_SOFT_CREDIT_LEVEL0D_V1.md` and
`docs/XOR_TASK_BRIDGE_LEVEL1A_V1.md` and
`docs/RESULTS_XOR_TASK_BRIDGE_LEVEL1A_STAGE_I.md` and
`docs/RESULTS_XOR_TASK_BRIDGE_LEVEL1A_STAGE_II.md` and
`docs/XOR_DELAY_GRANULARITY_LEVEL1B_V1.md` and
`docs/RESULTS_XOR_DELAY_GRANULARITY_LEVEL1B_STAGE_A.md` and
`docs/RESULTS_XOR_DELAY_GRANULARITY_LEVEL1B_STAGE_B_CONTROLS.md`.

A separate dimension-aware rescue, `xor_delay_granularity_rescue_level1br_v1`,
has completed all 50 formal R1 cells. Its R1 matrix uses analytically matched
`lambda_P=.01P` (`.01/.16/.64`) on entirely new seeds, while retaining
unscaled higher-dimensional baselines. LR or update-budget calibration is
mechanically allowed only when all initial coordinate directions are already
correct. R1 global passes 10/10; unscaled per-hidden/per-synapse pass 1/10 and
0/10; dimension-matched versions pass 10/10 and 10/10. R2 is therefore skipped
and the 30-cell sealed R3 confirmation is complete. Global, per-hidden and
per-synapse each confirm 10/10; the preregistered complexity rule selects
per-hidden-neuron. This tests explicit oracle optimization under teacher
strength that grows with delay dimension; it is not autonomous routing. See
`docs/RESULTS_XOR_DELAY_GRANULARITY_RESCUE_LEVEL1BR_R1.md` and
`docs/RESULTS_XOR_DELAY_GRANULARITY_RESCUE_LEVEL1BR_R3.md`.

The completed independent bridge is
`xor_delay_granularity_rescue_microburst_v1`. It transfers the confirmed
dimension-matched oracle weights to the consecutive `{8,9}` micro-burst using
five fresh formal seeds. Its fixed-d4 replication passes 5/5 and all 40 learned
cells are complete. The selected primary candidate is
per-hidden `lambda=.16`; global `.01`, per-synapse `.64` and per-hidden
task-only are prespecified controls. The three oracle-supervised conditions
pass 10/10; task-only passes 0/10. Read
`docs/XOR_DELAY_GRANULARITY_RESCUE_MICROBURST_V1.md` and
`docs/RESULTS_XOR_DELAY_GRANULARITY_RESCUE_MICROBURST_STAGE_B0.md`, then
`docs/RESULTS_XOR_DELAY_GRANULARITY_RESCUE_MICROBURST_STAGE_B1.md`.

The next preregistered causal audit is
`xor_task_derived_timing_withdrawal_v1`. It does not repeat failed task-only
global discovery. It independently rebuilds five per-hidden scaffolded
foundations, verifies that fixed d3/d5 perturbations genuinely break the exact
interface in ten no-update controls, and only then runs retention plus
oracle-delay-only, task-delay-only, task-joint, and weight-only recovery arms.
The primary claim freezes both weight matrices and asks task loss alone to
restore delays locally. Even success is local basin restoration, not K>1
routing. W0 passes 5/5 and W1 passes 10/10. The d3/d5 interventions preserve
classification but move every output to step 14/16, so W2 is now authorized.
The full 45-cell W2 matrix remains frozen. Read
`docs/XOR_TASK_DERIVED_TIMING_WITHDRAWAL_V1.md`,
`docs/RESULTS_XOR_TASK_DERIVED_TIMING_WITHDRAWAL_W0.md`,
`docs/RESULTS_XOR_TASK_DERIVED_TIMING_WITHDRAWAL_W1.md`, and the config.

W2 is now complete. Retention passes 5/5 and oracle-delay-only passes 10/10,
but task-delay-only passes 0/10; the formal local-restoration protocol fails.
d3 regains exact function in 3/5 task-delay cells without recovering the delay
vector, while d5 is exact in 0/5 and polarizes delays toward the bounds. The
integer-boundary audit shows a gradient sign flip between 4.999 and 5.000 under
the current right-sided interpolation backward. K>1 remains locked. Read
`docs/RESULTS_XOR_TASK_DERIVED_TIMING_WITHDRAWAL_W2.md` before proposing any
credit repair; do not begin with LR, threshold, epoch, or seed tuning.

The registered first credit repair,
`xor_integer_boundary_credit_preflight_v1`, is now complete. A versioned
backward-only Gaussian estimator (sigma 1.00) preserves every tested forward
record bitwise and corrects the aggregate d3/d5 gradient direction, but the
conditional per-hidden recovery still fails: exact interface 2/10, all-delay
schedule recovery 0/10, and d5 exact interface 0/5. This rules out the
one-sided integer derivative as the sole cause. Independent-coordinate credit
and internal-schedule non-identifiability remain. Read
`docs/RESULTS_XOR_INTEGER_BOUNDARY_CREDIT_PREFLIGHT_V1.md`. K>1 remains locked;
the next minimal discriminator is a preregistered one-parameter globally tied
task-only recovery anchor, not an LR/threshold/budget sweep.

For the 2026-07-16 supervisor deadline, a separate non-confirmatory branch is
registered as `mixedop_spatial_temporal_surface_preview_v1`. It uses an MLP
decoder to plot validation accuracy over total hidden neurons and latency for
fixed K=5/8 mixed-operation workloads. Its spatial baseline, hand-scheduled
oracle, and WAD diagnostic must not be interpreted as proof of learned temporal
multiplexing. All six invalid LR=.01 stability cells passed; the unchanged
single-seed grid was initially authorized. See
`docs/MIXEDOP_SPATIAL_TEMPORAL_SURFACE_PREVIEW_V1.md` and the matching YAML.

The researcher then requested a K=5-first gate. All 36 K=5 cells are complete
and K=8 is on hold. Spatial and fixed oracle score 1.0 throughout the K=5
grid, while WAD remains at .5 worst-query balanced accuracy throughout and has
no activity in its last two windows. The constant surfaces cannot identify a
width or latency effect, and the smallest point already saturates both passing
conditions. Read
`docs/RESULTS_MIXEDOP_SPATIAL_TEMPORAL_SURFACE_PREVIEW_V1_K5.md`; do not resume
K=8 without explicit authorization.

The follow-up `mixedop_rate_wad_surface_calibration_v1` pilot is also complete.
A controlled 4K-channel 400/10 Hz rate code gives nonzero WAD delay gradients
for every query but leaves worst-query balanced accuracy at `.50` at all five
corner/center points, with its final two windows silent. All fixed-oracle
controls also miss the registered feasibility floor. At the matched center,
rate input approximately doubles measured synaptic events without improving
the primary endpoint. The full 30-cell surface is locked. Read
`docs/RESULTS_MIXEDOP_RATE_WAD_SURFACE_CALIBRATION_V1.md`.

The subsequent `mixedop_rate_alignment_repair_v1` identifies and fixes the
packet/window mismatch. Event8 fixed oracle is perfect across the selected and
new seeds, but five query-tied task-only and routing-assisted delay learning
both fail 0/3 at `.50` worst-query BAcc. Assistance moves later delays and
reduces routing loss but never reaches the final window or full schedule.
Stage C and larger WAD surfaces remain locked. Read
`docs/RESULTS_MIXEDOP_RATE_ALIGNMENT_REPAIR_STAGE_A.md` and
`docs/RESULTS_MIXEDOP_RATE_ALIGNMENT_REPAIR_STAGE_B.md`.

The supervised temporal-interface repair is now complete. V2 showed that the
old arrival-mass routing loss can yield perfect classification while learning
the wrong registered schedule. V3 replaces it with a window-centroid Huber
objective: oracle and centroid supervision pass 3/3 new seeds, while the
matched old loss fails 0/3. This repairs a five-parameter, explicitly supervised
interface only; autonomous WAD and Pareto claims remain locked. Read
`docs/RESULTS_MIXEDOP_TEMPORAL_WAD_REPAIR_V2.md` and
`docs/RESULTS_MIXEDOP_TEMPORAL_WAD_REPAIR_V3.md`.

The next registered gate is
`mixedop_temporal_wad_scaffold_withdrawal_v1`. It asks separately whether task
BCE can retain a correct warm-start schedule and whether it can restore two
prespecified perturbations after explicit timing supervision is removed. It
does not call either result autonomous discovery. The four invalid
implementation smoke cells pass; only the three-cell W0 scaffold replication
then passes all three new seeds. The frozen 12-cell W1 retention matrix is now
complete: abrupt task-only passes schedule only 1/3, while fixed annealing
followed by 100 task-only updates passes 3/3. The 18-cell W2 restoration matrix
is now complete: centroid restores 6/6, but task-only restores 0/6 despite
near-perfect classification. The current Boolean task loss does not identify
the delay schedule. Read
`docs/MIXEDOP_TEMPORAL_WAD_SCAFFOLD_WITHDRAWAL_V1.md`.

## Source of truth

- [`docs/PUBLICATION_ROADMAP.md`](docs/PUBLICATION_ROADMAP.md): authoritative
  phased research programme, decision gates, publication criteria, and the
  immediate next protocol.
- [`docs/PROJECT_SCOPE.md`](docs/PROJECT_SCOPE.md): research question and
  admissible claims.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): protocol for
  future confirmatory experiments.
- [`docs/METRICS_AND_COST.md`](docs/METRICS_AND_COST.md): reliability and cost
  definitions.
- [`docs/CLAIMS_LEDGER.md`](docs/CLAIMS_LEDGER.md): claim-level evidence audit.
- [`docs/EXPERIMENT_LOG_V2.md`](docs/EXPERIMENT_LOG_V2.md): decisions and
  experiments after the cleanup date.
- [`docs/READOUT_PROTOCOL.md`](docs/READOUT_PROTOCOL.md): explicit observation
  modes and decoder constraints.
- [`docs/RESOURCE_LEDGER.md`](docs/RESOURCE_LEDGER.md): static and measured
  resource-count definitions.
- [`docs/RESEARCH_DECISIONS.md`](docs/RESEARCH_DECISIONS.md): seven-section
  guide for encoding, Plan D, readout, metrics, cost, and diagnostics.

`docs/EXPERIMENT_LOG.md`, legacy presentation files, and historical summaries
are archival material.  They may contain superseded numbers and must not be
used as a source for new claims.

## Status

The controlled XOR matrix rejects a positive learned-delay superiority claim:
an optimized shared scalar delay beats WAD on the primary worst-query endpoint,
while WAD delay shuffling reveals strong within-checkpoint co-adaptation. Read
`docs/RESULTS_XOR_DELAY_CONTROL_MATRIX_V1.md`. Do not open the sealed test set
or add a large K/N sweep; the next protocol must separate timing, distribution,
placement, and co-adaptation.

The event8-aligned K=5 resource preview is preregistered and implemented as
`mixedop_event8_aligned_surface_v2`. It freezes four corner landmarks plus one
center and includes the missing shared-d0 control. Its learned conditions are
separated into a task-only diagnostic and an explicitly curriculum-assisted
model; neither may be reported as autonomous temporal multiplexing. Only five
invalid technical smoke cells are authorized. The 25-cell landmark pilot and
the 120-cell full grid remain locked pending their machine-readable gates. See
`docs/MIXEDOP_EVENT8_ALIGNED_SURFACE_V2.md` and the matching YAML.

The five invalid event8-aligned implementation-smoke cells now pass their
technical gate. They verify exact shared-d0/oracle delays, nonzero task-only
delay gradients, curriculum phase switching, diagnostic panels and resource
ledgers. Their accuracy is not evidence. The unchanged 25-cell landmark pilot
is authorized; the full surface remains locked. See
`docs/RESULTS_MIXEDOP_EVENT8_ALIGNED_SURFACE_V2_SMOKE.md`.

The 25-cell landmark is complete but does not authorize the full surface.
Spatial d0, fixed oracle and curriculum-assisted temporal models already
saturate at the smallest N=20 point; task-only WAD passes function at only one
point and has a 14.23-step schedule error there. The post-run mechanism audit
also exposes an endpoint mismatch: centroid supervision targets
`1.5+w/2+qw`, while the gate compares against `3+qw`. The full grid remains
locked, and these failures must not be reported as a clean WAD optimization
result. See `docs/RESULTS_MIXEDOP_EVENT8_ALIGNED_SURFACE_V2_LANDMARK.md`.

The next supervisor-preview is `mixedop_k6_centroid_supervised_surface_v1`:
six basic logic queries, explicit arrival-centroid supervision, and a
descriptive N=1--60, T=34--166 grid. Only four invalid preflight cells are
authorized; the 112-cell surface remains locked. This is explicitly supervised
temporal routing, not autonomous WAD or a Pareto result. See
`docs/MIXEDOP_K6_CENTROID_SUPERVISED_SURFACE_V1.md`.

Its four extreme preflight cells now pass all technical and final-centroid
gates. The exact 112-cell seed-3907 surface is authorized, while test, extra
seeds and spatial/Pareto claims remain locked. See
`docs/RESULTS_MIXEDOP_K6_CENTROID_SUPERVISED_SURFACE_V1_PREFLIGHT.md`.

The complete 112-cell surface does not show an L-shaped tradeoff. All delay
mechanisms are valid, but N90 is non-monotonic and the hyperbolic fit has the
wrong sign. Width dominates the single-seed descriptive grid variance; T has a
small marginal contribution and is confounded with window/support changes.
Read `docs/RESULTS_MIXEDOP_K6_CENTROID_SUPERVISED_SURFACE_V1.md`. Do not call
this autonomous WAD or spatial-temporal Pareto evidence.

The versioned follow-up `mixedop_k6_boundary_multiseed_confirmation_v1` is
complete (200/200). Neither exact parent reversal reproduces (0/5 each), but
the broader N90 non-monotonicity meets its registered 3/5-seed threshold. The
robust boundary `[4,3,3,3,2,3,3,3]` is not stable at T=34 and does not match
the parent. All mechanisms are valid; low-width failures remain dominated by
XOR/XNOR. This closes the fixed-position surface without a time--width law or
autonomous-WAD/Pareto claim. See
`docs/RESULTS_MIXEDOP_K6_BOUNDARY_MULTISEED_CONFIRMATION_V1.md`.
