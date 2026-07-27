# Agent working agreement

## Objective and claim discipline

The project asks whether learnable synaptic delays improve the matched
resource--reliability frontier of temporal multi-query computation.  Do not
assume an affirmative answer.  The authoritative research documents are in
`snn_async_delays/docs/`:

1. `PUBLICATION_ROADMAP.md` (read first; governs new work and decision gates)
2. `PROJECT_SCOPE.md`
3. `EXPERIMENT_PROTOCOL.md`
4. `METRICS_AND_COST.md`
5. `CLAIMS_LEDGER.md`
6. `EXPERIMENT_LOG_V2.md`
7. `READOUT_PROTOCOL.md`
8. `RESOURCE_LEDGER.md`

Legacy reports, the old experiment log, presentation slides, and directory
names are not sources of truth.  Never promote an exploratory result to a
claim without updating the ledger and log.

## Repository layout

- `snn_async_delays/`: code, data generation, configs, runs, paper, docs.
- `snn_async_delays/runs/`: immutable experiment artifacts.  Do not delete,
  rename, or overwrite historical runs.  Use `docs/generated/` registries to
  classify them.
- `snn_async_delays/scripts/legacy_tmp/`: archived one-off scripts; do not use
  as canonical pipeline code.
- `presentation/current/`: manually edited deck; never overwrite it
  automatically.
- `presentation/generated/`: generated drafts only.
- `presentation/archive/`: historical decks and backups.

## Required workflow for an experiment change

1. Read the authoritative docs above, especially `PUBLICATION_ROADMAP.md`, and
   inspect the working tree.
2. Write the question, hypothesis, resource vector, readout interface,
   encoding, seeds, primary metric, validation rule, and exclusion rule in
   `EXPERIMENT_LOG_V2.md` before a new sweep.
3. Use `worst_query_accuracy` and `exact_trial_accuracy` as primary
   reliability metrics.  Pooled accuracy is descriptive only.
4. Report actual latency, dimensions, parameters, delay-buffer/state memory,
   event counts, and decoder cost.  Do not call spike counts “energy” without
   a declared hardware model.
5. Select hyperparameters on validation data; evaluate sealed test data once.
6. Record all seed values and update `CLAIMS_LEDGER.md`, including failures and
   alternative explanations.

For long-running sweeps, report only launch, a material error/blocker, and
completion (or a user-requested status). Do not emit routine per-cell updates.

## Readout and encoding constraints

`late-window` count readout is an alignment diagnostic, not a neutral default.
Positive delay claims require matched all-time/time-binned, late-window, and
where relevant spiking-output controls.  Burst and rate encoding test distinct
input regimes.  Jitter must preserve the declared number of input events.

## Code and verification

- Do not modify historical `eval_results.json`, configs, checkpoints, or logs
  in place.  Write re-evaluations to a versioned sibling artifact.
- Keep JSON strict: `NaN` is not a valid output value.
- Run `D:\anaconda3\envs\snn_async\python.exe -m unittest discover -s tests -v`
  after touching core code.
- Build/refresh inventories with:
  `python -m scripts.build_experiment_registry` and
  `python -m scripts.audit_existing_results`.
- Re-evaluate Plan-D checkpoints without touching historical results with
  `python -m scripts.reevaluate_checkpoints --runs-dir <path> --device cpu`,
  then classify with `python -m scripts.classify_reevaluation --input <json>
  --output-dir <dir>`.
- Generate static readout ledgers with
  `python -m scripts.report_readout_resource_ledger`; flatten measured ledgers
  with `python -m scripts.flatten_resource_ledgers --input <json> --output <csv>`.
- Preserve user changes that are unrelated to the task, especially
  `presentation/current/presentation.pptx`.

## Immediate project state (2026-07-13)

Historical runs are exploratory until re-evaluated.  Do not launch a large
sweep before the checkpoint re-evaluation and readout/resource protocol gates
are complete.

`xor_delay_control_matrix_v1` is complete and rejects a positive WAD
superiority claim: optimized scalar beats WAD on the primary endpoint, although
WAD is sensitive to delay shuffling. Read
`docs/RESULTS_XOR_DELAY_CONTROL_MATRIX_V1.md` before new work. Do not open its
test split or expand K/N.

`wad_optimization_audit_v1` is complete: no Stage-B variant passed, so freeze
threshold .3, dmax 30, delay LR .001, constant initialization and joint
optimization. Read `docs/RESULTS_WAD_OPTIMIZATION_AUDIT_STAGE_B.md`.

The d0-only `simultaneous_output_interface_calibration_v1` was stopped as
structurally invalid: spatial completed, but temporal d0 has no activity in the
post-input output windows and all opponent outputs remain silent. It selected
no threshold. `simultaneous_output_interface_calibration_v2` is complete and
freezes output threshold 0.2. It reused nine spatial cells and added nine
temporal cells with a fixed non-learned timing scaffold plus a peak-voltage to
spike-count curriculum. The scaffold is calibration-only and cannot support a
delay/routing claim. Temporal worst balanced accuracy remained .5 in every
seed/threshold; at .2, XOR and NOR stayed at chance and only NAND exceeded
chance. Spatial control is ready, but do not launch the 36-cell temporal matrix
until a versioned matched-training viability preflight is complete. Simultaneous runs
must save exhaustive 64-pattern results as well as validation, checkpoint, NPZ,
panel and resource ledger. Primary reliability is worst balanced operation/window
accuracy; pooled accuracy is descriptive only. After completion use
`scripts.summarize_simultaneous_pilot` for aggregate plots.
Read `docs/WAD_OPTIMIZATION_AUDIT_V1.md` and
`docs/SIMULTANEOUS_PILOTS_V1.md` before changing or launching these protocols.

`simultaneous_spatial_control_pilot_v1` is complete (36/36 artifacts). The
primary linear endpoint does not support WAD superiority; MLP is saturated.
WAD improves opponent exact-trial over scalar in 3/3 seeds, but worst-balanced
is unstable and event cost is higher, so this is exploratory only. Read
`docs/RESULTS_SIMULTANEOUS_SPATIAL_CONTROL_PILOT_V1.md` before interpreting it.
`simultaneous_temporal_viability_preflight_v1` formally failed hidden emission
`[1,.606,0]`, but the read-only mechanism audit found third-window WAD output-
current support `.703`, output activity `.344`, realized arrival mass `.117`
and nonzero per-window gradients. Hidden emission was a wrong arrival proxy.
Preflight v2 is complete and passes every mechanism-valid gate separately in
held-out seeds `{1,42}`. This establishes executable delayed output support and
gradients, not routing performance: WAD worst-window balanced remains `.5` in
both seeds, with effectively zero class-current gap for fixed NOR/window 2.
Operation and temporal position are confounded. Read
`docs/RESULTS_SIMULTANEOUS_TEMPORAL_CHECKPOINT_MECHANISM_AUDIT_V1.md` and
`docs/RESULTS_SIMULTANEOUS_TEMPORAL_VIABILITY_PREFLIGHT_V2.md`. Do not launch
the old fixed-order temporal matrix. Its replacement,
`simultaneous_temporal_counterbalanced_performance_v1`, completed all 45 cells.
No condition passes the locked routing floor; WAD's primary minimum is `.5` in
all seeds and WAD superiority is false. WAD is best only on secondary mean
balanced (`.734`) and exact-trial (`.411`) metrics, while late XOR and NOR stay
at chance. Fixed-full favours late positions but fails early, so the result is
a timing-allocation limitation, not a NOR-only failure. Test remains sealed.
Do not run K/N/T scaling or the eight-operation task. Read
`docs/RESULTS_SIMULTANEOUS_TEMPORAL_COUNTERBALANCED_PERFORMANCE_V1.md`. The next
gate is a researcher decision between one small coverage-aware method test and
a negative methodology framing.

The researcher selected a new versioned direction:
`spatial_vs_temporal_pareto_phase0`. Read
`docs/SPATIAL_VS_TEMPORAL_PARETO_PHASE0.md` before work. Stage A is a 15-cell
K=1 XOR baseline/output-interface calibration using an equal-event binary
one-hot micro-burst input (two events on each selected A/B value channel) and
one class-opponent target spike; it cannot support a multiplexing claim. Stage
B remains locked until Stage A selects the smallest width passing every seed.
Formal Stage A completed and failed on 2026-07-14: no width passes every locked
gate in all three seeds, no baseline width was selected, and Stage B remains
locked. Read `docs/RESULTS_SPATIAL_VS_TEMPORAL_PARETO_PHASE0_STAGE_A.md`.
Do not select h=12/h=24 from mean accuracy, retrospectively choose best epochs,
alter the formal runs, or launch Stage B. The read-only checkpoint audit is
complete; read
`docs/RESULTS_SPATIAL_VS_TEMPORAL_PARETO_PHASE0_STAGE_A_CHECKPOINT_AUDIT.md`.
It confirms all Stage-A delays were d0, finds hidden silence plus output
conversion failures, and localizes every collision to a wrong t=10 spike before
the correct t=11 target. Accuracy-best checkpoints and post-hoc threshold
changes do not rescue the interface. The next gate is a new versioned,
held-out-seed interface-stability calibration.
Do not copy Habashy et al.'s variable-spike input code into the primary task.
Cost must separate hidden area `h'/(Kh)`, hidden updates, latency, dense
synaptic compute, events and delay memory. Do not launch K>2, mixed operations
or test evaluation.

The explicitly separate exploratory branch
`spatial_vs_temporal_pareto_mlp_scaffold_v2` is complete (160/160 cells). Read
`snn_async_delays/docs/RESULTS_SPATIAL_VS_TEMPORAL_PARETO_MLP_SCAFFOLD_V2.md`
before further Pareto work. At the robust `.90` worst-balanced plus `.90`
exact-trial gate in both seeds, only independent spatial d0 and the fixed
query-scheduled oracle pass, at `T=18, h=24`. The oracle halves total hidden
neurons and hidden updates relative to independent spatial, but dense MACs,
measured events, delay-value storage and decoder MACs are unchanged, while
delay-buffer storage is five times larger. WAD is at `.5` worst-balanced in all
32 cells: its query delay distributions overlap, no query-1 synapse reaches the
second window, and second-window hidden activity is zero. Do not describe this
as learned routing, compute/energy compression, or a `T x h` law; width explains
nearly all observed variation and the tested T effect is negligible. The
branch remains non-spiking, validation-only, and cannot unlock formal Stage B
or reinterpret failed Stage A. V1 was aborted after one completed cell because
it scheduled 2,400 redundant updates per cell; never pool its artifacts with
V2. The next research fork must be explicit: either a small query-coverage/
symmetry-breaking WAD bridge at `T=18,h=24`, or a return to the spiking-output
interface at that single feasible fixed-oracle point. Do not rerun the full
surface before this choice.

The learned-delay fork is selected and
`delay_parameter_recovery_level0a_v1` is complete (75/75). Read
`snn_async_delays/docs/RESULTS_DELAY_PARAMETER_RECOVERY_LEVEL0A_V1.md` before
changing delay optimization. The current `.001/200-step` scalar analogue fails
to move `init_raw=-2` from delay `.9536` to target 5 (final `1.1389`), whereas
the preregistered `.05` direct-supervision arm recovers all 15 interior
target/initialization pairs. This rules out a hard scalar `get_delays()` bug but
does not validate task-loss credit or authorize `.05` in XOR. The next allowed
experiment is a versioned Level 0B test through the production spike buffer,
followed separately by one LIF neuron. Do not introduce XOR, K scaling,
pairwise WAD, accuracy claims or a new Pareto surface before that gate.

`delay_temporal_credit_level0b_v1` is now complete (180/180) and formally
fails. Read
`snn_async_delays/docs/RESULTS_DELAY_TEMPORAL_CREDIT_LEVEL0B_V1.md`. The
buffer-current centroid arm passes 15/15 and rules out a hard circular-buffer
gradient obstruction. Buffer filtered trace passes 5/15, hard-LIF centroid
3/15 and hard-LIF filtered trace 10/15. The mechanism audit finds wrong initial
gradient signs in 5/13 filtered-loss pairs and zero gradients in 8/13 hard-LIF
centroid pairs. Do not transfer `.05` to XOR, start K=1 XOR, add pairwise WAD,
or rerun a Pareto surface. The next allowed protocol is a versioned Level 0C
one-event audit of symmetric/global timing credit on soft current or membrane
traces, with bidirectional gradient-sign and coverage gates. Per-neuron delay
tying may be a later controlled factor but cannot by itself repair a wrong
single-delay gradient sign.

`delay_soft_trace_credit_level0c_v1` is complete (360/360) and passes its
preregistered gate. Read
`snn_async_delays/docs/RESULTS_DELAY_SOFT_TRACE_CREDIT_LEVEL0C_V1.md` before
new delay-learning work. Production sigmoid + soft centroid + Adam `.05`
recovers 30/30 current/membrane cells at one common LR with 13/13 correct
nonzero directions per path. All membrane traces remain subthreshold. Causal
filtered loss is path dependent (buffer 5/15, membrane 15/15); symmetric
kernel alignment is biased by the causal membrane tail (current 15/15,
membrane 0/15). This is not XOR or routing evidence. The only authorized next
experiment is a versioned Level 0D hard-spike-forward/soft-credit bridge that
compares current versus pre-reset membrane centroid auxiliaries and evaluates
hard-spike arrival in both directions. Do not start K=1 XOR, pairwise WAD,
per-neuron tying or a Pareto surface before Level 0D passes.

`delay_hard_output_soft_credit_level0d_v1` is now complete (135/135) and
passes. Read
`snn_async_delays/docs/RESULTS_DELAY_HARD_OUTPUT_SOFT_CREDIT_LEVEL0D_V1.md`
before new delay-learning work. The selected bridge is production hard
filtered-spike loss plus synaptic-current centroid at `lambda=.1`, Adam `.05`,
200 updates: 15/15 recovery and 13/13 correct nonzero initial directions.
Hard-only is 10/15. Pre-reset centroid is rejected for this role (3/15
soft-only with eight zero-gradient pairs; no combined arm exceeds 10/15).
Do not transfer `.1` or `.05` into a task without a new protocol. The only
authorized next experiment is a versioned Level 1A K=1 XOR calibration with a
hard-spiking endpoint, explicit task/auxiliary scaling, d0/fixed controls and
separate weight/delay gradient diagnostics. Do not start K>1, pairwise WAD,
per-neuron tying or another Pareto surface before that task-level gate passes.

`xor_task_bridge_level1a_v1` Stage I is now complete (90/90) and passes;
read `snn_async_delays/docs/XOR_TASK_BRIDGE_LEVEL1A_V1.md` and
`snn_async_delays/configs/xor_task_bridge_level1a_v1.yaml` before acting. No
read `snn_async_delays/docs/RESULTS_XOR_TASK_BRIDGE_LEVEL1A_STAGE_I.md`. The
unique locked interface is `eta=0,lr_w=.01`, passing all five d0 and all five
fixed-delay-4 cells. Stage II is now complete (85/85) and Level 1A passes; read
`snn_async_delays/docs/RESULTS_XOR_TASK_BRIDGE_LEVEL1A_STAGE_II.md`. Task-only
delay learning passes 0/10 at `.01` and 1/10 at `.05`. The selected explicit
arrival scaffold is `lambda=.01,lr_d=.01` and passes 10/10 with maximum delay
error `.002628` step; task and auxiliary gradients conflict in 5/10 selected
cells. The fixed-d0/wrong-target control passes 0/5. Describe this only as a
scaffold-assisted joint task/delay bridge, never autonomous routing discovery.
Level 1B K=1 granularity and consecutive-micro-burst preregistration is
complete and implementation-validated as
`xor_delay_granularity_level1b_v1`; read
`snn_async_delays/docs/XOR_DELAY_GRANULARITY_LEVEL1B_V1.md` and its config
before acting. Formal Stage A is complete (60/60); read
`snn_async_delays/docs/RESULTS_XOR_DELAY_GRANULARITY_LEVEL1B_STAGE_A.md`.
Global scaffold passes 10/10, while per-hidden and per-synapse pass only 2/10
and 0/10, so the higher-dimensional extension fails. The fixed micro-burst
controls are now complete; read
`snn_async_delays/docs/RESULTS_XOR_DELAY_GRANULARITY_LEVEL1B_STAGE_B_CONTROLS.md`.
Fixed delay 4 passes 5/5 with exact `t=15` outputs, while d0 passes 0/5 and
emits only at `t=11`. The original 60 learned micro-burst cells are
mechanically authorized but unrun. Never use
mean delay alone: every independent coordinate must pass the `.1`-step and
initial-gradient gates. The scaffold's registered mean over coordinates also
scales each teacher gradient as `1/P`; the result rejects the frozen recipe,
not intrinsic trainability of higher-dimensional delays. Preserve the hard
`4 -> 16 -> 2` interface, d0 hidden-to-output delays, runtime NPZ/panel and
resource separation of delay parameters from physical synapses. Six smoke
cells are invalid for claims. K>1, WAD, routing, scaling and Pareto work remain
locked.

The dimension-aware repair is separately preregistered as
`xor_delay_granularity_rescue_level1br_v1`; read
`snn_async_delays/docs/XOR_DELAY_GRANULARITY_RESCUE_LEVEL1BR_V1.md` and its
config before acting. Formal R1 is complete (50/50). It sets `lambda_P=.01P` to match
teacher gradient per coordinate and retains unscaled new-seed baselines. R2 LR
or budget calibration is allowed only if R1 has correct nonzero direction on
every coordinate but incomplete recovery. R1 global passes 10/10; unscaled
per-hidden/per-synapse pass 1/10 and 0/10; dimension-matched versions pass
10/10 and 10/10 with full coordinate recovery. R2 is not required. The
complete sealed R3 matrix has run: global, per-hidden and per-synapse each pass
10/10, and the registered lower-complexity rule selects per-hidden-neuron.
Never modify Level 1B artifacts or
describe stronger dimension-scaled oracle supervision as autonomous learning.
Six Level-1B-R smoke cells are invalid for claims. Read
`snn_async_delays/docs/RESULTS_XOR_DELAY_GRANULARITY_RESCUE_LEVEL1BR_R1.md`.
Also read
`snn_async_delays/docs/RESULTS_XOR_DELAY_GRANULARITY_RESCUE_LEVEL1BR_R3.md`.

The independent dimension-aware micro-burst bridge is preregistered as
`xor_delay_granularity_rescue_microburst_v1`; read its config and
`snn_async_delays/docs/XOR_DELAY_GRANULARITY_RESCUE_MICROBURST_V1.md` before
acting. It leaves the original 60 learned Stage-B cells frozen and unrun. The
new matrix is five fixed-d4 controls followed conditionally by 40 learned
cells: global `.01`, per-hidden `.16`, per-synapse `.64`, and per-hidden
task-only, each learned candidate spanning two initial directions and five
fresh seeds. The five fixed-d4 controls pass 5/5 and the complete 40-cell
learned matrix is finished. Global `.01`, per-hidden `.16`, and per-synapse
`.64` each pass 10/10; per-hidden task-only passes 0/10. This is
oracle-supervised compatibility, not autonomous routing. Never treat the ten
smoke cells as evidence. Proposed seed `2309` was consumed by smoke and removed
from the formal set; formal seeds are `{2333,2351,2371,2389,2411}`. Success
does not authorize K>1 or Pareto work. The only authorized next protocol is
`xor_task_derived_timing_withdrawal_v1`: five independent teacher-built
foundations, ten locked no-update d3/d5 damage controls, then 45
retention/recovery cells. Read its config and protocol document before acting;
W1 and W2 may not run before their preceding formal decisions pass.
W0 is complete 5/5 and W1 is complete 10/10. d3 and d5 preserve classification
but shift every output to step 14 and 16, respectively. The full 45-cell W2
matrix is authorized and must run unchanged; no subset or directional retuning
is allowed. A disclosed read-only audit predicts asymmetric task credit, but it
does not alter any registered decision rule.
W2 is complete 45/45 and the primary protocol fails. Retention passes 5/5 and
oracle-delay-only passes 10/10, but task-delay-only passes 0/10. d3 has partial
functional recovery without schedule recovery; d5 has wrong-directed integer-
boundary credit and delay polarization. K>1 remains locked. Read
`snn_async_delays/docs/RESULTS_XOR_TASK_DERIVED_TIMING_WITHDRAWAL_W2.md`.
Do not tune LR, thresholds, budget, or seeds as the next action. The next
admissible protocol must first repair and validate symmetric task-derived
temporal credit at exact integer delays and separate functional recovery from
internal schedule identification.
Reproduce the post-W2 evidence with
`python -m scripts.audit_xor_withdrawal_w2_mechanism --device cpu`; its outputs
are analysis artifacts, not additional formal cells.
The first registered backward repair is also complete; read
`snn_async_delays/docs/RESULTS_XOR_INTEGER_BOUNDARY_CREDIT_PREFLIGHT_V1.md` and
`snn_async_delays/configs/xor_integer_boundary_credit_preflight_v1.yaml`.
`right_linear` remains the historical default. Gaussian STE sigma 1.00
preserves tested forward records and repairs aggregate d3/d5 direction, but
per-hidden P1 passes exact function only 2/10 and schedule identification 0/10.
Do not rerun P0/P1, tune sigma/LR/threshold/budget, or claim that the delay
gradient is repaired generally. Do not use exact interface or mean delay as a
proxy for recovery of all 16 coordinates. K>1 remains locked. The next
admissible protocol is a preregistered globally tied one-delay task-only
recovery anchor on the same W0 provenance; its outcome decides between a
target-free parameter-tying rescue and a continuous task-credit redesign.
Read
`snn_async_delays/docs/RESULTS_XOR_DELAY_GRANULARITY_RESCUE_MICROBURST_STAGE_B0.md`.
Also read
`snn_async_delays/docs/RESULTS_XOR_DELAY_GRANULARITY_RESCUE_MICROBURST_STAGE_B1.md`
and `snn_async_delays/docs/XOR_TASK_DERIVED_TIMING_WITHDRAWAL_V1.md`.

An explicitly deadline-driven and scientifically separate supervisor-preview
branch is preregistered as `mixedop_spatial_temporal_surface_preview_v1`.
Read its config and
`snn_async_delays/docs/MIXEDOP_SPATIAL_TEMPORAL_SURFACE_PREVIEW_V1.md` before
touching it. It fixes K={5,8}, seed 307, total hidden {40,80,120,160}, three
latencies per K, and independent-spatial d0 / hand-scheduled oracle / WAD
conditions with an MLP decoder and all learning rates .01. The six invalid
stability cells pass every technical gate. All 36 K=5 cells are complete, but
K=8 is now on hold at the researcher's K=5-first gate. This is a validation-only progress visualization that deliberately
overrides the normal K>1 sequencing at the researcher's request; it does not
unlock or supersede the failed causal mechanism programme. Never call the
oracle learned, the WAD condition temporal multiplexing evidence, or N*T
energy. Do not add seeds, tune after seeing accuracy, open test data, or omit
failed cells. Every cell must write runtime NPZ/panel and the full resource
ledger. The K=5 result is degenerate on the primary metric: spatial/oracle are
1.0 everywhere and WAD is .5 everywhere, so no width/T factor statement is
allowed. WAD covers essentially only the first two windows. Five completed and
two interrupted K=8 cells are excluded; interrupted directories contain audit
records and require versioned sibling reruns if K=8 is later authorized. Read
`snn_async_delays/docs/RESULTS_MIXEDOP_SPATIAL_TEMPORAL_SURFACE_PREVIEW_V1_K5.md`.

A versioned encoding-rescue branch is preregistered as
`mixedop_rate_wad_surface_calibration_v1`. It uses a controlled 4K-channel
one-hot Poisson rate code, an expanded possible N/T domain, and a gated
four-corners-plus-center pilot. It is not event matched to the burst parent
(8.2 expected versus four input events/query), so a positive result could not
be attributed to temporal distribution alone. The two invalid stability cells
and ten pilot cells are complete; the result below governs further work. Do
not tune rates, LR, threshold, seed, updates or grid after seeing the pilot. Read
`snn_async_delays/docs/MIXEDOP_RATE_WAD_SURFACE_CALIBRATION_V1.md` and its
config before acting.

The rate-code pilot is complete and fails its oracle-feasibility, WAD-
performance and routing gates. All five WAD points remain at `.50` worst-query
balanced accuracy; the center has nonzero gradients for every query but no
activity in its final two windows and strongly overlapping delays. The fixed
oracle also stays below `.85` at every point. Rate input approximately doubles
measured center synaptic events without a primary gain. The 30-cell full
surface is locked and must not run. Read
`snn_async_delays/docs/RESULTS_MIXEDOP_RATE_WAD_SURFACE_CALIBRATION_V1.md`.

The versioned repair is `mixedop_rate_alignment_repair_v1`. The previous rate
packet was not aligned: only its last timestep directly entered each declared
window. The repair uses rate steps 6--9 and fixed oracle
`d_q=3+q*w`, then tests two event budgets over N={120,240}, w={4,6,8,12} and
three new seeds. Only two invalid smoke cells are initially authorized. Formal
Stage A requires an explicit config unlock after smoke. Stage B is conditional
on a candidate passing all three seeds and uses exactly five per-query delays;
its task-only and explicitly routing-assisted arms must never be conflated.
Per-synapse WAD, Stage C, full surfaces and sealed test remain locked. Read
`snn_async_delays/docs/MIXEDOP_RATE_ALIGNMENT_REPAIR_V1.md` and its YAML.

Stage A is complete: event4 fails all eight candidates on exact-trial, while
event8 passes all eight across three seeds. The resource rule selects
`event8_N120_w4` (T=30, schedule `[3,7,11,15,19]`). Only the registered nine-
cell Stage B is authorized: new-seed oracle, five-parameter per-query task-only,
and five-parameter routing-assisted. Do not substitute another candidate,
change `.10` routing weight, or call the assisted arm autonomous discovery.
Read `snn_async_delays/docs/RESULTS_MIXEDOP_RATE_ALIGNMENT_REPAIR_STAGE_A.md`.

Stage B is complete and fails: new-seed oracle passes 3/3, task-only passes
0/3, and `.10` routing-assisted passes 0/3. Both learned arms remain at `.50`
worst-query BAcc and fail the five-coordinate schedule. Assisted gradients are
large and nonzero and route loss decreases, so do not call this gradient
vanishing; task/route credit conflict and the q4 sigmoid-boundary traverse are
the main diagnostics. Stage C, per-synapse WAD, full surfaces and test remain
locked. Do not tune weight/LR/updates inside v1. Read
`snn_async_delays/docs/RESULTS_MIXEDOP_RATE_ALIGNMENT_REPAIR_STAGE_B.md`.

The next versioned method test is `mixedop_temporal_wad_repair_v2`. It retains
event8/N120/w4 but expands support to 23, initializes five delays at 3, selects
joint routing scale only from a three-seed zero-update gradient rule, and adds
a decoupled arm whose delays receive routing gradients only. Every trained
cell must save accuracy-best and schedule-best artifacts. Only the gradient
preflight is initially authorized; smoke/formal/withdrawal/test remain locked.
Assisted success would be supervised routing, never autonomous discovery. Read
`snn_async_delays/docs/MIXEDOP_TEMPORAL_WAD_REPAIR_V2.md` and its YAML.

Temporal WAD repairs v2/v3 are complete. V2's decoupled arrival-mass arm is
classification-perfect but schedule-invalid because its objective drives q0
near delay 1; do not describe it as insufficient training. V3 changes only the
delay-only objective to output-window-centroid Huber and passes the joint
function/activity/schedule gate on 3/3 new seeds; the matched mass-CE comparator
fails 0/3. This is explicitly supervised per-query routing, not autonomous WAD
or time multiplexing. Read
`snn_async_delays/docs/RESULTS_MIXEDOP_TEMPORAL_WAD_REPAIR_V2.md`,
`snn_async_delays/docs/RESULTS_MIXEDOP_TEMPORAL_WAD_REPAIR_V3.md`, and both
configs before continuing. Do not launch K=8, per-synapse WAD, sealed test or a
Pareto surface. The next admissible step is a separately preregistered scaffold-
withdrawal protocol; `withdrawal_launch` remains false.

The withdrawal protocol is now preregistered as
`mixedop_temporal_wad_scaffold_withdrawal_v1`. Read its YAML and
`snn_async_delays/docs/MIXEDOP_TEMPORAL_WAD_SCAFFOLD_WITHDRAWAL_V1.md` before
implementation. W0 replicates the centroid scaffold on seeds 3657/3669/3691;
W1 tests frozen, continued-centroid, abrupt-task-only and fixed annealed credit;
W2 tests frozen/centroid/task-only restoration after uniform -2 and
`[0,-1,-2,-3,-4]` perturbations. Final checkpoints are primary. Only invalid
implementation smoke is authorized; do not run W0/W1/W2 until the preceding
gate is audited and the config authorization is explicitly changed.
The implemented entry point is
`python -m scripts.run_mixedop_temporal_wad_scaffold_withdrawal --stage <smoke|w0|w1|w2> --device cuda`.
Dry-run bypasses launch authorization only to enumerate paths. Never reuse an
incomplete directory; audit it and create a versioned sibling if required.
The four implementation smoke cells are complete and pass all technical gates.
Do not cite their accuracy. W0 is now authorized for exactly seeds
3657/3669/3691 and 400 updates with centroid-only delay credit; W1/W2 remain
locked until the generated W0 decision passes all three seeds.
W0 is now complete and passes 3/3. The frozen W1 matrix is authorized for the
same three source checkpoints and exactly four arms: delay-frozen,
centroid-continue, abrupt-task-only and annealed-joint, 200 updates each. Source
updates are 380/320/400 with recorded hashes. W2 remains locked until either
abrupt or annealed retention passes all seeds at the final checkpoint.
W1 is complete: abrupt task-only passes only 1/3 schedule gates despite
near-perfect function; annealed-joint passes 3/3 after its final 100 task-only
updates. The exact W2 18-cell matrix is authorized. It must start from untouched
W0 sources, not W1 checkpoints, and apply only the two registered perturbations.
Do not interpret W1 as restoration or autonomous discovery.
W2 is complete and fails task-only restoration 0/6; centroid positive controls
pass 6/6. Classification remains near-perfect while task-only schedule errors
remain 1.35--2.20 steps. Treat the current Boolean BCE schedule as locally
non-identifiable. Do not open test, tune LR/threshold/updates, or label annealed
surface models autonomous WAD. Read
`snn_async_delays/docs/RESULTS_MIXEDOP_TEMPORAL_WAD_SCAFFOLD_WITHDRAWAL_W2.md`.

The next supervisor-preview branch is preregistered and implemented as
`mixedop_event8_aligned_surface_v2`. Its five landmarks are the four corners
`(N,w)={(20,4),(20,12),(240,4),(240,12)}` plus `(80,8)`. Each compares
independent spatial d0, shared d0, fixed event8 oracle, task-only five-delay
WAD, and a curriculum-assisted five-delay temporal model. The last condition
uses explicit centroid timing supervision before a fixed withdrawal and must
never be called autonomous WAD. Only the five invalid smoke cells are
authorized; the 25-cell landmark and 120-cell full grid are locked. Use
`python -m scripts.run_mixedop_event8_aligned_surface_v2`, not direct script
execution. Read the matching YAML and
`snn_async_delays/docs/MIXEDOP_EVENT8_ALIGNED_SURFACE_V2.md` before acting.

The five event8-aligned v2 implementation-smoke cells are complete and pass
all technical gates. Oracle schedule `[3,11,19,27,35]`, shared-d0 zeros,
task-only delay gradients, curriculum phase transition, runtime panels and
resource ledgers were independently verified. Smoke accuracy is invalid and
must not be cited. Exactly the frozen 25-cell landmark pilot is now authorized;
the full surface, K=8 and sealed test remain locked. Read
`snn_async_delays/docs/RESULTS_MIXEDOP_EVENT8_ALIGNED_SURFACE_V2_SMOKE.md`.

The 25-cell event8-aligned v2 landmark is complete and the full-surface gate is
negative. Oracle passes all required points; curriculum passes only the two
`w=4` points, not the center or either `w=12` point. A post-run audit finds the
centroid loss target `d_q=1.5+w/2+qw` is inconsistent with the registered
oracle-vector gate `d_q=3+qw`, creating deterministic offsets of .5/2.5/4.5
steps for w=4/8/12. Do not describe the three curriculum gate failures simply
as training failure and do not amend the v2 gate post hoc. Spatial, oracle and
curriculum already saturate at N=20, so the remaining larger-N grid cannot
identify compression or an N/T factor law. Full v2, K=8 and test remain locked.
Read `snn_async_delays/docs/RESULTS_MIXEDOP_EVENT8_ALIGNED_SURFACE_V2_LANDMARK.md`.

The separate supervisor-preview `mixedop_k6_centroid_supervised_surface_v1` is
preregistered and implemented. Read its YAML and
`snn_async_delays/docs/MIXEDOP_K6_CENTROID_SUPERVISED_SURFACE_V1.md`. It uses
six fixed basic operations, N=1--60, T=34--166 and six query-tied delays trained
only by explicit arrival-centroid Huber credit; weights/readout receive task
BCE only. This is explicitly supervised temporal routing, never autonomous
WAD. Only four invalid extreme preflight cells are authorized; the 112-cell
surface stays YAML-locked until the generated preflight decision passes and
authorization is explicitly changed. Common initialization equals q0's target,
so q0 is not required to have nonzero initial routing gradient; q1--q5 must be
nonzero and target-directed. Do not change seed, LR, 600-update budget,
threshold, support, grid or gates, and do not open test. Use
`python -m scripts.run_mixedop_k6_centroid_supervised_surface`; retain runtime
NPZ, six-query diagnostic panels and full resource ledgers.

The four K=6 centroid-supervised preflight cells are complete and pass. Read
`snn_async_delays/docs/RESULTS_MIXEDOP_K6_CENTROID_SUPERVISED_SURFACE_V1_PREFLIGHT.md`.
The exact 112-cell seed-3907 surface is authorized unchanged. Do not use the
preflight accuracy as evidence: N=1 is near chance and N=60 perfect at both T
extremes, which only suggests the grid crosses a width transition. Monitor
clipping, especially N=60 at long T, but do not change LR, clipping rule,
updates or grid. Test, extra seeds, spatial controls and publication claims
remain locked.

The full K=6 centroid-supervised surface is complete (112/112). Read
`snn_async_delays/docs/RESULTS_MIXEDOP_K6_CENTROID_SUPERVISED_SURFACE_V1.md`.
All cells are mechanism-valid, but raw N90 `[3,3,2,2,3,4,3,3]` is not L-shaped;
the hyperbolic fit has negative C and `R2=.129`. Width accounts descriptively
for .811 of worst-BAcc grid SS, T .013 and interaction .176, with only one seed.
N>=5 is mostly saturated and two higher-width reversals invalidate a stable
boundary claim. Do not smooth the curve, enlarge the grid, call T causal, or
claim autonomous WAD/Pareto benefit. The next admissible protocol is a new
multi-seed confirmation near N=1--5; operation-window counterbalancing and
spatial/shared controls remain separate later gates.

`mixedop_k6_boundary_multiseed_confirmation_v1` is complete (200/200); read
its result document before new surface work. Both named seed-3907 reversals
fail to reproduce (0/5 each), while broader raw-N90 non-monotonicity confirms
at the registered 3/5 threshold. Robust 4/5 N90 is
`[4,3,3,3,2,3,3,3]`; strict 5/5 is `[5,3,3,3,3,3,3,3]`. The boundary is not
stable because T=34 spans N90=2--4 and it does not reproduce the parent curve.
All cells are centroid-mechanism-valid; N=2 instability is primarily XOR/XNOR
classification. Do not add seeds, smooth the curve, claim a causal T effect,
open test, or infer autonomous WAD/Pareto. The next admissible scientific gate
is a new operation-window counterbalancing protocol; selected matched
spatial/shared-d0/oracle controls come only after that separate gate.
