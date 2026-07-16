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
