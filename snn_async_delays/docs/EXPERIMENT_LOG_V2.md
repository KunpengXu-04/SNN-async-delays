# Experiment log v2

This log begins during the July 2026 cleanup.  The historical narrative is
preserved in `EXPERIMENT_LOG.md`; it is not retroactively edited and contains
superseded claims and numbers.

## 2026-07-11 — Protocol reset

- **Decision:** The canonical objective is the resource--reliability frontier
  for temporal multi-query computation, not an assumed delay advantage.
- **Reason:** Existing pooled-accuracy, late-window, and cost analyses do not
  identify temporal multiplexing under matched resources.
- **Status:** active.
- **Artifacts:** `PROJECT_SCOPE.md`, `METRICS_AND_COST.md`,
  `EXPERIMENT_PROTOCOL.md`, `CLAIMS_LEDGER.md`, `RESEARCH_DECISIONS.md`.
- **Next gate:** Create an immutable run registry and re-evaluate existing
  checkpoints with position-sensitive metrics before launching new sweeps.

## 2026-07-11 — Evidence inventory and first re-evaluation

- **Decision:** Historical runs remain in place and are classified through a
  registry; presentation assets and temporary utilities are physically
  separated from canonical code.
- **Evidence:** 883 run records: 767 exploratory, 113 archived, 3 incomplete.
  Of 879 historical evaluation files, 60 pass pooled 90% while failing the
  recovered worst-query 90% test.
- **Artifacts:** `docs/generated/experiment_registry.csv`,
  `docs/generated/historical_result_audit.csv`,
  `scripts/reevaluate_checkpoints.py`, and
  `scripts/summarize_worst_query.py`.
- **Caveat:** Historical `per_query_acc` permits triage but cannot reconstruct
  exact-trial correctness.  Checkpoint re-evaluation writes separate versioned
  outputs and never overwrites historical `eval_results.json`.

## 2026-07-11 — Full NAND burst Plan-D checkpoint re-evaluation (v1)

- **Scope:** all 177 checkpoints under `runs/NAND_compress_burst_(planD)`;
  deterministic burst encoding, original late-window interface, fresh test
  reconstruction, evaluation seed `20260711`.
- **Integrity:** 177 loaded and evaluated successfully; no skip or error.
- **Result:** requiring every available seed to reach worst-query accuracy
  `>= 0.90`, learned-delay WAD passes only K=1 (minimum tested h=10) and K=2
  (minimum tested h=60).  No tested WAD h through 150 passes K>=3.  The d0
  condition passes no K in its tested h range.  This d0 failure remains an
  alignment-interface observation, not a fair general no-delay conclusion.
- **Classification:** 177 runs / 89 grouped cells remain `exploratory`; zero
  are canonical because the series has at most two seeds and lacks matched
  readout/resource controls.  Zero are archived: failed results remain useful
  negative evidence and loaded reproducibly.
- **Artifacts:** `docs/generated/reevaluation_NAND_compress_burst_planD_v1.json`,
  `docs/generated/classification_NAND_compress_burst_planD_v1/`.
- **Decision:** retire any K=3+ @90% statement for this series.  Do not launch
  an enlarged Plan-D sweep until readout-interface and matched-baseline gates
  are specified.

## 2026-07-11 — Explicit readout interface implementation

- **Question:** Is the historical final readout window a neutral measurement
  interface or a treatment that favours delay-based alignment?
- **Implementation:** `SNNSimultaneousModel` now declares
  `observation_mode = late_window | all_time | time_binned`.
  `late_window` is byte/numerically compatible with the historical default;
  `all_time` observes the full trial with the same decoder feature dimension;
  Plan-D `time_binned` exposes K input bins plus the final read bin and reports
  its larger decoder dimension and parameter count.
- **Compatibility check:** four representative historical checkpoints produce
  zero difference in pooled, worst-query, exact-trial, and spike-count metrics
  under the explicit default.
- **Spiking-output finding:** the current direct spike count is non-negative but
  is used as a BCE logit, so it cannot express negative evidence for class 0.
  It remains exploratory pending an opponent-pair or compatible count-loss
  design.  Time-binned direct spike output is rejected explicitly.
- **Artifacts:** `docs/READOUT_PROTOCOL.md`,
  `tests/test_readout_observation.py`, and
  `docs/generated/readout_default_compatibility_v1.json`.
- **Training decision:** no new model was trained in this step.

## 2026-07-11 — Resource ledger v1

- **Objective:** replace hidden-spike-only “energy/throughput” claims with an
  auditable resource vector.
- **Static fields:** latency; layer dimensions; delayed synapses; trainable and
  stored parameters; decoder parameters/MACs; neuron updates; dense simulator
  synapse MACs; delay floor/ceil reads and interpolation operations; delay
  buffer and neuron-state memory; observation accumulation operations.
- **Dynamic fields:** measured input, layer-wise hidden, and output spikes;
  event-driven synaptic events computed as presynaptic spikes times fan-out.
- **Important distinction:** the present PyTorch simulator evaluates delayed
  synapses densely every timestep.  Event-driven synaptic-event counts are a
  separate prospective hardware model and are not measured runtime or energy.
- **Checkpoint sanity check:** for historical NAND K=3 seed42, d0 h20 has 800
  neuron updates, 1,600 dense synapse MACs, 552 MLP decoder MACs, and about
  179 input-fan-out events per trial.  WAD h50 has 2,000 neuron updates, 4,000
  dense synapse MACs, 2,650 decoder MACs, and about 448 input-fan-out events.
  Thus the historical lower hidden-spike count does not imply lower total
  compute or event cost.
- **Readout sanity check:** at K=3/h50, time-binned linear storage rises from
  353 to 803 scalars; under the current width rule time-binned MLP rises from
  2,903 to 41,003.  Spiking output increases delay-buffer storage per sample
  from 62 to 1,612 FP32 elements because it adds hidden-history buffering.
- **Artifacts:** `docs/RESOURCE_LEDGER.md`,
  `utils/resource_ledger.py`, `scripts/report_readout_resource_ledger.py`,
  `scripts/flatten_resource_ledgers.py`,
  `docs/generated/readout_resource_ledger_v1.csv`, and
  `docs/generated/resource_ledger_checkpoint_sample_v1.csv`.
- **Training decision:** no pilot was launched; the ledger gate is complete.

## 2026-07-11 — Pre-registration: XOR readout-interface pilot v1

- **Question:** under balanced deterministic-burst XOR at K=2, do WAD and d0
  change relative behaviour when observation moves from late-window censoring
  to all-time or time-binned access?
- **Fixed factors:** h=50; seeds 0/1/42; threshold 0.3; no homeostasis;
  2,000/500/1,000 train/validation/test samples; 100 epochs; no jitter.
  Threshold 0.3 is locked because historical XOR at 1.0 was all-silent.
- **Factorial design:** WAD/d0 × late_window/all_time/time_binned × linear/MLP
  × 3 seeds = 36 cells.  Time-binned is an explicitly resource-expanded
  control.  All cells remain exploratory.
- **Primary outputs:** worst-query, exact-trial, balanced accuracy, and
  `resource_ledger_v1`; pooled accuracy is descriptive only.
- **Artifacts:** `configs/pilot_xor_readout_v1.yaml`,
  `docs/PILOT_XOR_READOUT_V1.md`, and `scripts/run_xor_readout_pilot.py`.
- **Stop rule:** execute every registered cell once; do not alter hyperparameters
  or extend the grid based on intermediate results.

## 2026-07-12 — XOR readout-interface pilot v1 completed

- **Completion:** all 36 preregistered cells completed; all 12 groups have all
  three seeds. No factor was changed during execution.
- **Key result:** WAD's apparent advantage is interface-dependent. Under
  late-window MLP, d0/WAD worst-query are `0.490±0.017` / `0.727±0.132`; under
  all-time MLP, both are `1.000±0.000`. Under all-time linear, d0/WAD are
  `0.850±0.090` / `0.877±0.054`. Time-binned conditions are all 1.0 but have
  much larger decoder resources.
- **Interpretation:** late-window readout censors d0's useful early activity;
  this pilot rejects using its WAD–d0 gap as evidence for a general capacity
  advantage. WAD may still be an alignment mechanism; its general benefit is
  untested.
- **Resources:** time-binned MLP raises scalar storage from 2,852 to 23,152
  and decoder MACs from 2,600 to 22,800. It is an explicit temporal-feature
  control, not a matched-cost comparator.
- **Diagnostics:** six non-cherry-picked panels generated after completion:
  seed 0, MLP, every delay condition and observation mode. Panel annotations
  now report the actual observation mode.
- **Artifacts:** `RESULTS_XOR_READOUT_PILOT_V1.md`, pilot generated summary,
  resource figure, and diagnostic selection manifest.

## 2026-07-12 — Publication roadmap v1 locked

- **Objective:** convert the diagnostic findings into a falsifiable, gated
  programme that can support either a positive learned-delay claim or an
  academically useful conditional/negative result.
- **Primary question:** whether learnable heterogeneous delays improve the
  reliability--resource Pareto frontier over d0, optimized scalar delay, and
  fixed heterogeneous-delay controls under neutral all-time observation.
- **Binding decisions:** all-time is primary; late-window is an alignment
  diagnostic; time-binned is an expanded-resource upper bound; `K`, hidden
  size `N`, and duration `T` must be separated; no energy scalar without a
  hardware model; mechanism claims require causal interventions.
- **Next gate:** run validation-only `xor_difficulty_calibration_v1` to select a
  non-saturated regime, then preregister the full causal delay-control matrix.
- **Artifact:** `docs/PUBLICATION_ROADMAP.md`.
- **Status:** canonical planning decision; no new empirical claim and no
  training run in this entry.

## 2026-07-12 — Pre-registration: XOR difficulty calibration v1

- **Question:** which compact `(K,N,T)` settings place deterministic-burst XOR
  in a stable non-floor, non-ceiling regime suitable for a later causal
  delay-control comparison?
- **Status:** validation-only exploratory calibration; it cannot support a WAD
  superiority, capacity, multiplexing, or test-set claim.
- **Factors:** fixed-subwindow/growing-`T` with `K={2,3,4}`, `N={20,35,50}`,
  WAD/d0, paired seeds `{0,1,42}`: 54 cells. All-time linear readout is fixed.
- **Data:** 2,000 train and 500 validation samples. Best-checkpoint selection
  and calibration evaluation both use validation, so reported values are
  optimistic. The test split must not be opened.
- **Primary metrics:** validation worst-query and exact-trial accuracy;
  balanced accuracy and `resource_ledger_v1` are required.
- **Selection rule:** prefer adjacent settings where both condition means lie
  in worst-query `[0.65,0.95]`, neither lies outside `[0.55,0.98]`, and seed SD
  is at most `0.12`. Failure to qualify requires a new protocol, not post-hoc
  test inspection.
- **Diagnostics:** after completion, generate exactly six outcome-independent
  panels: seed 0, `N=35`, every K and both delay conditions, trace seed 999.
- **Artifacts:** `configs/xor_difficulty_calibration_v1.yaml`,
  `docs/XOR_DIFFICULTY_CALIBRATION_V1.md`,
  `scripts/run_xor_difficulty_calibration.py`, and
  `scripts/generate_xor_difficulty_calibration_diagnostics.py`.
- **Important limitation:** this version varies `K` while keeping subwindow
  duration fixed, so total `T` grows. It does not yet identify fixed-`T`
  temporal packing effects.

## 2026-07-12 — XOR calibration diagnostic-retention amendment

- **Researcher requirement:** retain `diagnostic_data.npz` and generate
  `diagnostic_panel.png` immediately for every completed calibration cell,
  rather than generate a six-cell subset after sweep completion.
- **Scope:** artifact-generation timing and coverage only. Task factors,
  metrics, seeds, validation-only boundary, checkpoints, and selection rule are
  unchanged.
- **Execution:** the active process was stopped after 25 completed cells; those
  cells are backfilled with trace seed 999. The runner resumes immutably,
  skipping completed result files and generating diagnostics for each new
  completed cell. The interrupted 26th cell is rerun from its registered seed.
- **Evidence boundary:** all panels remain illustrative and are prohibited as
  a difficulty-selection criterion.

## 2026-07-12 — XOR difficulty calibration v1 completed

- **Completion:** 54/54 validation-only cells, NPZ files, and diagnostic
  panels. Final diagnostics were regenerated uniformly after one intermediate
  race artifact was detected.
- **Primary pattern:** WAD−d0 worst-query is approximately
  `+.078/+.077/+.013` at K=2 for N=20/35/50,
  `+.037/+.013/−.031` at K=3, and `−.003/+.018/−.011` at K=4.
  There is no stable learned-delay scaling advantage.
- **Selected settings:** K=3,N=35,T=40 is primary; K=4,N=50,T=50 is a stress
  point. These are calibration choices, not test claims.
- **Resources at primary point:** d0/WAD trainable parameters 178/248; both
  store 248 scalars and perform 1,400 neuron updates, 2,800 dense synapse MACs,
  and 105 decoder MACs. Hidden spikes 10.68/9.89 do not establish energy gain.
- **Diagnostics:** useful as trace illustrations but insufficient for a
  mechanism claim; membrane/readout traces, clear non-spiking output semantics,
  aggregates, and causal delay interventions remain missing.
- **Status:** exploratory calibration complete; test split remained sealed.
- **Artifacts:** `docs/RESULTS_XOR_DIFFICULTY_CALIBRATION_V1.md`, generated
  group/paired tables, selection JSON, and accuracy/delta figures.
- **Next gate:** implement full delay controls and preregister
  `xor_delay_control_matrix_v1`. WAD must beat the strongest non-learned delay
  control under all-time observation to support a positive programme.

## 2026-07-12 — Pre-registration: XOR delay control matrix v1

- **Question:** under all-time observation at calibrated non-saturated points,
  does learned heterogeneous delay placement outperform d0, an optimized
  shared scalar delay, and fixed heterogeneous delays?
- **Settings:** primary K=3,N=35,T=40 with linear primary/MLP secondary;
  stress K=4,N=50,T=50 with linear only. Five paired seeds
  `{0,1,2,3,42}` and four delay conditions produce 60 training cells.
- **Primary endpoint:** validation worst-query accuracy for WAD versus the
  strongest non-learned control at primary/linear. Test remains sealed.
- **Causal interventions:** deterministic post-training WAD delay shuffle
  preserving the exact delay multiset, weights, and decoder; and a same-
  checkpoint late-window probe for every condition.
- **Gate:** WAD requires mean paired advantage >=.03, positive difference in
  at least 4/5 seeds, no resource domination, and shuffle degradation in at
  least 4/5 seeds. Beating d0 alone is insufficient.
- **Controls implemented:** one trainable shared delay scalar; frozen local-
  seed uniform heterogeneous bank; deterministic learned-delay permutation.
- **Diagnostics corrected:** hidden membrane heatmap, query-wise labels,
  predictions and logits, and explicit separation of decoder decisions from
  true output spikes. Every training/intervention cell retains NPZ and panel.
- **Verification:** 19 unit tests pass; dry-run expands exactly 60 uniquely
  named validation-only cells. No matrix cell has been launched.
- **Artifacts:** `configs/xor_delay_control_matrix_v1.yaml`,
  `docs/XOR_DELAY_CONTROL_MATRIX_V1.md`,
  `scripts/run_xor_delay_control_matrix.py`,
  `scripts/run_xor_delay_control_interventions.py`,
  `utils/delay_controls.py`, and `tests/test_delay_controls.py`.
- **Status:** preregistered, executable, not launched.

## 2026-07-12 — XOR delay control matrix v1 completed

- **Completeness:** 60/60 training cells, 60/60 late-window probes, 15/15 WAD
  shuffle probes, and 135/135 NPZ/panels. Test remained sealed.
- **Primary result:** optimized scalar is the strongest non-learned linear
  control (`.699±.019` worst-query); WAD is `.678±.036`. Paired WAD−scalar is
  `−.028,−.060,−.030,−.036,+.050`, mean `−.0208`, only 1/5 positive. The
  preregistered positive gate fails decisively.
- **Secondary result:** MLP ranking is fixed heterogeneous `.809`, scalar
  `.780`, d0 `.775`, WAD `.770`, with high fixed-bank variance. Stress-linear
  scalar/WAD are `.652/.644`; no WAD scaling advantage.
- **Shuffle:** WAD shuffle decreases worst-query in 15/15 cells; primary-linear
  mean drop `.116`, primary-MLP `.142`, stress-linear `.098`. This supports
  co-adapted placement sensitivity, not WAD superiority.
- **Window probe:** every all-time-trained checkpoint degrades under late-only
  evaluation. WAD is not uniquely aligned to the final window.
- **Resources:** primary scalar/WAD use 1/70 trainable delays and 179/248
  stored scalars with identical dense compute; WAD has fewer hidden spikes but
  no supported energy/frontier advantage.
- **Critical limitation:** fixed heterogeneous is uniform 0–30 (mean ~15),
  while WAD is concentrated around 1–9 (mean ~3.5). It is not a matched
  distributional control.
- **Decision:** reject the positive learned-delay superiority programme. Do
  not open test or expand K/N. Next gate is a compact causal decomposition of
  scalar timing, matched delay distribution, placement, and co-adaptation.
- **Artifacts:** `docs/RESULTS_XOR_DELAY_CONTROL_MATRIX_V1.md`, generated raw/
  grouped/paired/intervention/delay tables, decision JSON, and two figures.

## 2026-07-13 — Pre-registration: WAD optimization audit v1

- **Purpose:** determine whether the failed WAD result is plausibly caused by
  an unfair/ineffective delay-optimization regime, without reopening the test
  split or turning a broad search into a post-hoc rescue.
- **Stage A (locked):** XOR, K=3, N=35, burst, sequential, all-time linear;
  thresholds `{0.2,0.3,0.5}` crossed with scalar/WAD and paired seeds
  `{0,1,42}` (18 validation cells). Threshold is selected by predeclared
  activity and gradient viability, not by the best accuracy.
- **Why threshold is included:** threshold controls spike availability,
  surrogate-gradient support, event count, and saturation. Treating 0.3 as an
  invisible constant would confound any conclusion about delay learning.
- **Stage B (locked pending Stage A):** matched scalar/WAD audit of dmax,
  delay learning rate, noisy initialization, warm-up and alternating schedules
  (42 cells). `selected_threshold` remains null until a dated Stage-A decision.
- **Positive gate:** WAD must improve the original WAD baseline by at least
  .03 in 2/3 seeds and remain competitive with the matched scalar control;
  delay motion and non-degenerate activity are required.
- **Artifacts:** `configs/wad_optimization_audit_v1.yaml`,
  `docs/WAD_OPTIMIZATION_AUDIT_V1.md`,
  `scripts/run_wad_optimization_audit.py`.
- **Status:** preregistered; Stage A implementation verified by dry-run (18
  unique cells) and launched 2026-07-13 with the locked CUDA command; test
  split sealed. Stage B remains locked pending the Stage-A viability decision.

## 2026-07-13 — WAD optimization audit Stage A completed

- **Completeness:** 18/18 validation cells, checkpoints, train logs, NPZ files
  and diagnostic panels; test remained sealed.
- **Viability:** thresholds 0.2 and 0.3 pass every per-cell gate. Threshold 0.5
  fails the worst-query lower bound and exhibits low activity (2.677 spikes
  across conditions).
- **Locked selection:** threshold 0.3, because across-condition mean activity
  is 10.066 spikes, closest to the preregistered target 10. Accuracy was not
  used to break the tie.
- **Optimization evidence at 0.3:** every WAD epoch has finite nonzero delay
  gradient; mean gradient norm is .0449, final delay movement averages .989
  steps (minimum .883), and saturation is zero.
- **Reliability:** paired WAD−scalar worst-query differences at threshold 0.3
  are `-.028,-.060,+.050` (mean `-.0127`, 1/3 positive). Stage A supplies no
  WAD superiority evidence.
- **Decision:** unlock Stage B at threshold 0.3. This remains an exploratory
  optimization audit and cannot retroactively rescue the failed confirmatory
  claim.
- **Stage-B launch:** launched 2026-07-13 on CUDA after confirming 0/42 prior
  results and no duplicate Stage-B process. The locked matrix contains 42
  validation-only cells; no test data are opened.
- **Stage-B completion:** 42/42 cells and required artifacts. No variant passes
  the rescue gate. Mean WAD worst-query is baseline `.676`, d10 `.638`, low-LR
  `.654`, high-LR `.607`, noisy-init `.654`, warm-up `.624`, alternating `.641`.
  Every intervention is below baseline; baseline settings are frozen for the
  simultaneous pilots. See `RESULTS_WAD_OPTIMIZATION_AUDIT_STAGE_B.md`.
- **Artifacts:** `docs/RESULTS_WAD_OPTIMIZATION_AUDIT_STAGE_A.md` and
  `docs/generated/wad_optimization_audit_v1/`.

## 2026-07-13 — Pre-registration: simultaneous pilots v1

- **Spatial control:** three simultaneous fixed operation channels
  (XOR/NAND/NOR) with separate output identities. This tests spatial parallel
  multi-task computation and is explicitly not evidence for temporal routing.
- **True temporal routing:** the same simultaneous inputs must be decoded by
  one shared decoder or one shared opponent-spike pair in three ordered output
  windows. Separate per-query heads are forbidden in the primary interface.
- **Endpoints:** parameter-shared linear and MLP window decoders are
  representation diagnostics; paired opponent output neurons provide a
  task-native signed spike decision. A single non-negative spike count is not
  used as a binary logit.
- **Controls:** d0, optimized scalar, frozen matched heterogeneous delay bank,
  and WAD, with seeds `{0,1,42}`; 36 cells per pilot. Every completed cell must
  retain NPZ, diagnostic panel, validation metrics and resource ledger.
- **Lock:** both pilots are executable only after the WAD audit freezes
  threshold, dmax, delay LR, initialization and optimization schedule. This
  prevents pilot outcomes from choosing the WAD recipe.
- **Artifacts:** `configs/simultaneous_spatial_control_pilot_v1.yaml`,
  `configs/simultaneous_temporal_routing_pilot_v1.yaml`,
  `docs/SIMULTANEOUS_PILOTS_V1.md`, `scripts/run_simultaneous_pilot.py`.
- **Status:** preregistered and dry-run verified (36+36 cells), deliberately
  not launched.

## 2026-07-13 — Simultaneous prelaunch evaluation amendment

- **Reason:** before any simultaneous result existed, code review found that
  raw accuracy is confounded by XOR/NAND/NOR prevalence, temporal decision
  markers were not window-specific, and output threshold .3 was uncalibrated.
- **Metrics:** primary becomes worst operation/window balanced accuracy;
  pooled accuracy is descriptive. Add exhaustive 64-pattern evaluation,
  exact-trial completion, cross-target routing matrix/selectivity, opponent
  silent/tie/collision, per-window spikes and signed margins.
- **Diagnostics:** seed/endpoint provenance, six-channel and opponent labels,
  actual output-window boundaries and decision centres, per-window opponent
  counts, and shared-window mechanism annotations are implemented. Aggregate
  paired reliability, resource, interface-failure and truth-table plots are
  generated separately from the single-sample panel.
- **Output interface gate:** preregistered 18-cell d0-only threshold calibration
  over `{.2,.3,.5}` and spatial/temporal opponent interfaces. Accuracy is only
  a viability gate; distance to one output spike per operation/window selects.
  Both 36-cell pilots remain locked until this threshold is frozen.
- **Status:** code implemented and dry-run verified; no calibration or pilot
  cell had been launched at preregistration; test sealed.
- **Calibration launch:** launched 2026-07-13 on CUDA after confirming 0/18
  existing results and no duplicate process. The two 36-cell pilots remain
  locked pending the calibration decision.
- **Interrupted-run audit:** apparent completion was checked against required
  artifacts: spatial was 9/9 complete but temporal was 0/9; its first cell had
  only config/checkpoint and no train log or evaluation artifacts. The same
  immutable command was resumed on 2026-07-13; completed spatial cells are
  skipped and temporal cells continue. No threshold decision is permitted
  until 18/18 required artifacts exist.
- **Calibration structural failure:** after resume, spatial completed 9/9 and
  temporal completed thresholds .2/.3 for all three seeds (6/9). Every temporal
  run has train loss exactly `.693147` from epoch 1–100, 100% silent/tie rate,
  zero output spikes and worst-balanced .5. Weight gradients are nonzero and
  hidden spike counts grow, but fixed diagnostic samples show all hidden spikes
  in 0–10 ms and none in any post-input output window. The d0 feedforward model
  has no delayed/recurrent route into those windows, so d0-only temporal output
  calibration is structurally invalid. The remaining .5 cells were stopped;
  no output threshold was selected and both pilots remain locked. This is a
  protocol-design failure, not negative evidence about WAD.

## 2026-07-13 — Preregistration amendment: output-interface calibration v2

- **Reason:** v1's temporal d0 arm was incapable of placing presynaptic events
  in the three post-input output windows. Continuing it or merely lowering the
  output threshold would confound structural silence with excitability.
- **Preserved evidence:** all v1 artifacts remain immutable. The completed nine
  spatial-d0 cells are reused; the six silent temporal cells remain invalid
  protocol evidence and are not pooled into v2.
- **New temporal calibration device:** fixed, non-learned per-input-channel
  delays `[8,8,18,18,28,28]` route the three simultaneous operation channels
  into the three 10-step output windows. Hidden-to-output delay remains zero.
  This scaffold calibrates the shared opponent output interface only; it is
  prohibited as evidence that fixed delays or WAD solve temporal routing.
- **Optimization interface:** epochs 1–20 optimize class-wise peak pre-reset
  output voltage relative to the candidate threshold. Epochs 21–100 optimize
  signed opponent spike-count BCE plus a fixed 0.2 voltage auxiliary. Reported
  predictions always use spike counts; membrane logits never enter evaluation.
- **Checkpoint rule:** v2 evaluates the final checkpoint. A pooled-accuracy
  best checkpoint is not valid during a discrete membrane-to-spike curriculum,
  because it can remain the epoch-1 all-silent state.
- **Factors / cells:** output thresholds `{.2,.3,.5}`, paired seeds `{0,1,42}`,
  one temporal shared-opponent scaffold arm: nine new validation-only cells.
  Selection combines these with the nine completed spatial-v1 cells using the
  unchanged dual-interface viability gates and one-spike target.
- **Diagnostics:** epoch logs separately record spike-count and membrane losses.
  NPZ/panel are emitted in-run and include hidden membrane, true output spikes,
  output membrane against the firing threshold, window boundaries and opponent
  counts. Accuracy is expected to remain flat before a threshold crossing and
  is not, by itself, evidence that the optimizer is frozen.
- **Verification:** 25 unit tests pass, including backpropagation through the
  peak-voltage shared-window path. A full-data-budget 40-epoch smoke at threshold
  .3 reduced validation loss from `.8576` to `.7179`, increased hidden spikes
  from `.68` to `10.43`, and produced mean output spikes `[0,.104,.75]`; silent
  and tie rates fell from 1.0 to `.715`. Accuracy remained at the all-zero
  baseline and the smoke fails the formal viability gate. It proves executable
  gradient flow only and is not calibration evidence.
- **Status at preregistration:** smoke-verified with 0/9 formal v2 cells.
  The locked 9-cell CUDA run was launched on 2026-07-13 after confirming 0
  completed formal results and a unique 9-path dry expansion. Both 36-cell
  simultaneous pilots remain locked and the test split is sealed.
- **Artifacts:** `configs/simultaneous_output_interface_calibration_v2.yaml`,
  `scripts/run_simultaneous_output_calibration_v2.py`,
  `scripts/summarize_simultaneous_output_calibration_v2.py`, and
  `docs/SIMULTANEOUS_OUTPUT_INTERFACE_CALIBRATION_V2.md`.

## 2026-07-13 — Output-interface calibration v2 completed

- **Completeness:** 9/9 new temporal cells plus 9/9 reused spatial-v1 cells;
  every new cell has config, best/final checkpoint, 100-epoch log, validation,
  exhaustive 64-pattern result, resource ledger, NPZ and diagnostic panel.
  Test remained sealed.
- **Locked selection:** thresholds .2 and .3 pass the arm-level gates; .5 fails
  temporal silent/tie. Pooled spikes per operation/window are `.9668` at .2 and
  `.9381` at .3, so the one-spike tie-breaker selects **0.2**.
- **Spatial evidence at .2:** mean worst balanced `.742`, silent `.148`, tie
  `.204`, collision `.056`, spikes/query `1.076`.
- **Temporal evidence at .2:** mean worst balanced `.500`, silent/tie `.291`,
  collision `0`, spikes/window `.858`; exact-trial `.248/.216/.182`.
- **Failure pattern:** XOR and NOR balanced accuracy are `.500` in every seed;
  only NAND rises above chance (`.807/.679/.579`). Exhaustive truth tables agree.
  V2 calibrated firing but did not validate three-window logical routing.
- **Decision boundary:** freeze output threshold .2 without claiming routing.
  Spatial control is ready. Temporal routing remains locked pending a matched
  output-training viability preflight; d0 is a structural lower bound and WAD
  must be judged against scalar/fixed delay controls.
- **Artifacts:** `docs/RESULTS_SIMULTANEOUS_OUTPUT_INTERFACE_CALIBRATION_V2.md`
  and `docs/generated/simultaneous_output_interface_calibration_v2/`.

## 2026-07-13 — Spatial-control pilot launched

- **Protocol:** `simultaneous_spatial_control_pilot_v1`, 36 validation-only
  cells: four delay conditions × three endpoints × seeds `{0,1,42}`.
- **Frozen interface:** output threshold 0.2 from calibration v2; frozen WAD
  recipe unchanged. Dry expansion produced 36 unique paths and the formal
  directory contained 0 completed results before launch.
- **Execution:** locked CUDA run launched in unified session `47364`. Test is
  sealed. Routine per-cell progress is not used for decisions or reporting.
- **Status:** running; no result interpretation permitted until completeness
  and artifact audit.

## 2026-07-13 — Preregistration: temporal viability preflight v1

- **Reason:** calibration v2 made the shared opponent pair excitable but left
  XOR/NOR at chance. In addition, the current frozen heterogeneous range
  `[1,9]` cannot directly support the complete 30-step output horizon.
- **Design:** six seed-0 opponent cells: d0, scalar, narrow fixed `[1,9]`, full-
  support fixed `[0,30]`, scaffold positive control and WAD. Threshold .2,
  20-epoch membrane curriculum, .2 auxiliary, 100 epochs and final checkpoint
  are identical across cells; scaffold applies only to its named control.
- **Decision variables:** artifact/finite status, per-window hidden activity,
  output-weight gradient, output silence, and WAD delay gradient/movement/
  saturation. Accuracy and relative condition ranking are descriptive only.
- **Gate:** scaffold must demonstrate execution support; fixed-full and WAD
  must cover every window and retain output gradients; WAD must retain delay
  gradients and movement without saturation. Failure keeps temporal locked.
- **Artifacts:** `configs/simultaneous_temporal_viability_preflight_v1.yaml`,
  `docs/SIMULTANEOUS_TEMPORAL_VIABILITY_PREFLIGHT_V1.md`, runner and summarizer.
- **Status:** preregistered, not launched; test sealed.

## 2026-07-13 — Simultaneous spatial-control pilot v1 completed

- **Completeness:** 36/36 cells complete. Every cell has config, best/last
  checkpoint, training log, validation, exhaustive 64-pattern truth table,
  diagnostic NPZ and run-time diagnostic panel. Test remains sealed.
- **Primary endpoint:** linear worst-operation balanced accuracy. Means for
  d0/scalar/fixed/WAD are `.961/.944/.960/.956`. Paired WAD-minus-d0 is
  `[-.030,+.038,-.021]` (mean `-.0043`); WAD-minus-fixed mean is `-.0034`.
  There is no WAD superiority on the primary spatial endpoint.
- **MLP diagnosis:** all 12 cells are exactly 1.0 on worst-balanced and exact-
  trial accuracy, including exhaustive truth tables. This endpoint is at
  ceiling and cannot discriminate delay mechanisms.
- **Opponent diagnosis:** WAD and scalar worst-balanced are `.804/.802`, but
  WAD seed differences are inconsistent and WAD SD is `.138`. WAD-minus-scalar
  exact-trial is positive in 3/3 seeds (mean `+.110`) and accompanies lower
  silence/tie/collision. Retain this only as an exploratory interface signal.
- **Resources:** opponent WAD uses about 548 measured synaptic events and 16.45
  hidden spikes per trial versus scalar 534 and 14.12. WAD stores 600 trainable
  delay values versus two scalar delays. No resource-frontier claim is supported.
- **Scope:** dedicated spatial inputs/outputs cannot establish shared temporal
  routing. Do not expand this pilot or open test data.
- **Decision:** next run is the already-preregistered six-cell temporal
  viability preflight; its decision gates are activity/gradient support, not
  accuracy or WAD ranking. Full temporal matrix stays locked.
- **Artifacts:** `docs/RESULTS_SIMULTANEOUS_SPATIAL_CONTROL_PILOT_V1.md` and
  `docs/generated/simultaneous_spatial_control_pilot_v1/`.

## 2026-07-13 — Temporal viability preflight v1 launched

- **Pre-launch audit:** dry-run expanded exactly six unique locked cells and
  the formal output directory contained 0 completed validation results.
- **Execution:** CUDA unified session `5312`; no condition, seed, gate,
  curriculum or checkpoint-selection rule changed after preregistration.
- **Interpretation lock:** accuracy remains descriptive and cannot pass/fail
  this preflight. No full temporal matrix or N/T/K resource sweep may be
  launched until the support/gradient gates are evaluated.
- **Status:** running; test sealed.

## 2026-07-13 — Temporal viability preflight v1 completed: failed

- **Completeness:** 6/6 cells and all required artifacts are complete and
  finite; test remains sealed.
- **Passed controls:** scaffold activity fractions `[1,1,.75]`, silence `.216`;
  fixed-full activity `[1,1,1]` and output-gradient fraction `1.0`.
- **Failed WAD gate:** activity fractions `[1,.606,0]`; the third-window minimum
  `.05` is not met. Output-gradient fraction `1.0`, delay-gradient fraction
  `1.0`, movement `2.572` and saturation `0` all pass.
- **Localization:** d0 is silent; scalar and narrow fixed reach only window 1.
  Full-support fixed and scaffold demonstrate that the simulator, shared output
  and window metrics can support all windows. WAD learned delays remain mostly
  short (input-hidden median `3.43`, hidden-output median `3.90`).
- **Accuracy is descriptive:** WAD per-window balanced accuracy is
  `[.948,.878,.500]`; exact trial `.528` is partly compatible with a default
  majority decision for the imbalanced NOR window and cannot override zero
  third-window support.
- **Decision:** preflight failed; full temporal matrix remains locked. Any
  change to initialization, objective, curriculum or support regularization
  requires a new version and preregistration.
- **Artifacts:** `docs/RESULTS_SIMULTANEOUS_TEMPORAL_VIABILITY_PREFLIGHT_V1.md`
  and `docs/generated/simultaneous_temporal_viability_preflight_v1/decision.json`.

## 2026-07-13 — Temporal checkpoint mechanism audit v1 completed

- **Scope:** read-only audit of the six v1 final checkpoints on all 64 input
  patterns; no training, checkpoint mutation or model selection.
- **Gate correction:** WAD has zero hidden emission in window 3 but output-
  current support `.703`, output activity `.344`, and normalized realized
  arrival mass `.117`. The v1 formal fail remains, but hidden emission is
  invalidated as a proxy for delayed output arrival.
- **Window gradients:** WAD combined-loss input-hidden delay-gradient norms are
  `.379/.599/1.195`; hidden-output norms `.459/.312/.293`; output-weight norms
  `.135/.175/.187`. Window 3 has substantial credit assignment.
- **Local direction:** the window-3 first-order descent proxy shifts mean
  effective input-hidden/hidden-output delays by `-.061/-.019`, so the current
  objective does not simply push delays later.
- **Discriminability:** WAD signed-current class gaps are `11.20/20.14/2.31`;
  window 3 receives events but carries weak class separation and balanced
  accuracy remains `.5`.
- **Objective mismatch:** pooled BCE weights samples from XOR/NAND/NOR despite
  positive prevalences `.50/.75/.25`, while evaluation is worst-window
  balanced. A class/window-balanced objective is required for the next
  interface audit and must apply to every condition.
- **Artifacts:**
  `docs/RESULTS_SIMULTANEOUS_TEMPORAL_CHECKPOINT_MECHANISM_AUDIT_V1.md` and
  `docs/generated/simultaneous_temporal_checkpoint_mechanism_audit_v1/`.

## 2026-07-13 — Preregistration: temporal viability preflight v2

- **Separation:** seed 0 informed the mechanism audit and gate thresholds; v2
  excludes it and uses held-out seeds `{1,42}`.
- **Six cells:** fixed-full/scaffold/WAD × two seeds, shared opponent output,
  unchanged WAD recipe, threshold `.2`, final checkpoint and training budget.
- **Common correction:** every condition uses window/class-balanced BCE. No
  WAD-only initialization, delay range, optimizer or threshold change.
- **Locked gates:** delayed output-current support, output spike activity,
  realized hidden-to-output arrival mass, per-window output/delay gradients,
  WAD movement and saturation. Every held-out seed must pass separately.
- **Non-decision variables:** accuracy, exact trial, class-current gap and
  relative condition ranking are descriptive only; test remains sealed.
- **Status:** preregistered and dry-run verified as six unique cells; not
  launched.
- **Artifacts:** `configs/simultaneous_temporal_viability_preflight_v2.yaml`,
  `docs/SIMULTANEOUS_TEMPORAL_VIABILITY_PREFLIGHT_V2.md`, runner and summarizer.

## 2026-07-13 — Temporal viability preflight v2 launched

- **Pre-launch audit:** dry-run expanded exactly six unique held-out cells and
  the formal output directory contained 0 completed validation results.
- **Execution:** unified CUDA session `59956`; fixed-full/scaffold/WAD × seeds
  `{1,42}`. No preregistered field or gate changed.
- **Lock:** test remains sealed; accuracy is descriptive; full temporal matrix
  remains locked until the mechanism-valid v2 decision is complete.
- **Status:** running; routine per-cell progress is not reported.

## 2026-07-13 — Temporal viability preflight v2 completed: passed

- **Completeness:** 6/6 held-out cells have finite logs and every required
  checkpoint/evaluation/truth-table/NPZ/panel artifact. Test remains sealed.
- **Formal decision:** all mechanism-valid gates pass separately in seeds 1
  and 42; no averaging and no accuracy decision. Limiting WAD third-window
  current support is `.500/.766`, output activity `.234/.328`, and realized
  arrival mass `.051/.084` against gates `.50/.10/.05`.
- **Trainability:** minimum WAD per-window output-weight gradients are
  `.150/.110`, total delay gradients `.605/.548`; delay movement is
  `2.303/2.302` with zero saturation.
- **Descriptive performance:** WAD per-window balanced accuracy is
  `[.983,.891,.500]` and `[.931,.819,.500]`; exact trial `.604/.474`. Third-
  window signed-current class gaps are approximately `-.038/-.069`, so event
  support does not produce NOR discrimination.
- **Control evidence:** fixed-full learns NOR (`.743/.746`) but is seed-unstable
  on XOR, yielding worst-balanced `.702/.487`; scaffold remains an execution
  control at worst-balanced `.5`.
- **Remaining confound:** fixed order maps XOR/NAND/NOR to windows 0/1/2, so
  operation identity, temporal position and their interaction are inseparable.
- **Decision:** viability is established, routing performance is not. Do not
  launch the old fixed-order full matrix. Preregister a counterbalanced primary
  opponent-output matrix with each operation in each temporal position.
- **Artifacts:** `docs/RESULTS_SIMULTANEOUS_TEMPORAL_VIABILITY_PREFLIGHT_V2.md`
  and `docs/generated/simultaneous_temporal_viability_preflight_v2/`.

## 2026-07-13 — Preregistration: counterbalanced temporal performance v1

- **Question:** after mechanism viability, can any delay condition reliably
  route XOR/NAND/NOR through one shared opponent pair when operation identity
  is separated from temporal position?
- **Counterbalance:** cyclic orders `XOR/NAND/NOR`, `NAND/NOR/XOR`, and
  `NOR/XOR/NAND`; every operation occupies every window once.
- **Matrix:** d0/scalar/narrow-fixed/full-fixed/WAD × new paired seeds
  `{7,19,73}` × three orders = 45 validation-only cells. Seeds `{0,1,42}` are
  excluded because they informed interface design.
- **Frozen interface:** opponent shared output only, window/class-balanced loss,
  threshold `.2`, 20-epoch membrane curriculum, final checkpoint, burst input,
  unchanged WAD recipe and full diagnostics/resources.
- **Primary estimand:** for each condition/seed, minimum balanced accuracy over
  the nine operation × position combinations.
- **Routing label:** every seed primary >=`.55` and mean primary >=`.60`.
- **WAD superiority label:** mean paired primary advantage >=`.03` over the
  strongest observed non-learned control, positive in 3/3 seeds, and mean
  exact-trial difference >=`-.02`.
- **Scope:** exploratory validation decision only; test remains sealed and a
  positive result still requires confirmatory new seeds/test.
- **Verification:** dry-run expands 45 unique paths; counterbalance test and all
  30 unit tests pass. Formal directory contains 0 results before launch.
- **Artifacts:**
  `configs/simultaneous_temporal_counterbalanced_performance_v1.yaml`,
  `docs/SIMULTANEOUS_TEMPORAL_COUNTERBALANCED_PERFORMANCE_V1.md`, runner and
  summarizer.

## 2026-07-13 — Counterbalanced temporal performance v1 launched

- **Pre-launch audit:** 0/45 completed results, 45 unique dry-run paths, CUDA
  available and 30/30 unit tests passing.
- **Execution:** unified CUDA session `31606`; no preregistered order, seed,
  condition, objective, interface or decision rule changed.
- **Status:** running; test sealed; routine per-cell progress is not reported.

## 2026-07-14 — Counterbalanced temporal performance v1 completed: primary negative

- **Completeness:** 45/45 cells and all required checkpoint/log/validation/
  exhaustive/ledger/NPZ/panel artifacts are present; no exclusions; test sealed.
- **Primary:** condition/seed minima over nine operation-position cells are
  exactly `.5` for d0, scalar, fixed-matched and WAD. Fixed-full is
  `.500/.486/.482`. No condition passes the `.55` per-seed and `.60` mean
  routing gate; WAD superiority is false.
- **Tie correction:** d0, scalar and fixed-matched tie as strongest controls on
  the primary score. Selecting d0 alone is a list-order artefact, not evidence
  that d0 is scientifically strongest.
- **Secondary:** WAD has the highest mean operation-position balanced accuracy
  (`.734`) and exact-trial accuracy (`.411`). Its paired gains over fixed-full
  are `+.0777` and `+.0847`, respectively, but it uses more measured synaptic
  events and still has chance-level cells.
- **Counterbalanced diagnosis:** WAD learns NOR in positions 0/1 but fails NOR
  and XOR at position 2; NAND remains partly learnable there. Fixed-full has
  the opposite position trend and performs best late. The prior NOR/window-2
  failure is therefore primarily a temporal-position limitation with an
  operation interaction, not an operation-only explanation.
- **Decision:** no test opening and no K/N/T or eight-operation expansion. The
  next gate is an explicit choice between a small preregistered temporal-
  coverage method intervention and a negative methodology framing.
- **Artifacts:**
  `docs/RESULTS_SIMULTANEOUS_TEMPORAL_COUNTERBALANCED_PERFORMANCE_V1.md` and
  `docs/generated/simultaneous_temporal_counterbalanced_performance_v1/`.

## 2026-07-14 — Preregistration and smoke: MLP Pareto scaffold v1

- **Question:** with output-spike conversion removed, do independent spatial
  modules, ordinary shared representation, fixed temporal routing and WAD form
  an interpretable XOR latency-width-resource surface?
- **Scope:** exploratory supervisor-facing diagnostic, not a formal Stage-B
  continuation. The failed spiking-output Stage A and its conclusions remain
  unchanged.
- **Matrix:** K=2 repeated XOR; conditions `spatial_independent_d0`,
  `shared_spatial_d0`, `shared_temporal_d0`, `shared_temporal_oracle`, and
  `shared_temporal_wad`; `T={18,22,26,30}`, surface
  `h={4,8,16,24}`, seeds `{307,613}`; 160 cells.
- **Width semantics:** spatial `h` is per independent module and total hidden is
  `2h`; shared conditions use total hidden `h`. Temporal conditions use one
  shared MLP applied separately to two output windows. The all-time K-output
  MLP appears only in the shared-spatial control.
- **Data/objective:** every optimizer batch contains all 16 joint XOR patterns
  with balanced labels per query. Formal exploratory training is 300 epochs;
  the checkpoint rule is preregistered `best_pooled_accuracy` before any formal
  cell because this scaffold measures representational capacity rather than
  output-spike event stability. Primary reporting remains worst-query balanced
  plus exact-trial accuracy; pooled accuracy is descriptive.
- **Resource vector:** total hidden, latency, neuron updates, dense synapse MACs,
  measured synaptic events, trainable/decoder parameters, decoder operations
  and delay storage. Area compression alone is not a cost conclusion.
- **Artifacts:** each run writes both checkpoints, exact predictions, ledger,
  diagnostic NPZ and panel during execution. Summary outputs include mean and
  worst-seed `T x h` planes, exact-trial plane and six resource frontiers.
- **Smoke:** all five conditions produced complete artifacts with finite
  metrics and nonzero hidden activity. An additional locked-grid viability
  point at h=24/T=30/seed307 reached spatial worst-balanced/exact `1.0/1.0`
  and temporal-oracle `.9375/.9375` using the preregistered best checkpoint.
  This is smoke evidence only and is not a result claim.
- **Verification:** 42/42 repository tests pass; the full 160 formal-
  exploratory paths are empty and unique. Status is `preregistered_ready`.
- **Limits:** no sealed test, no mixed operations, no K>2, no direct output
  spikes, no formal Stage-B unlock, and no publication claim from two seeds.
- **Artifacts:** `configs/spatial_vs_temporal_pareto_mlp_scaffold_v1.yaml`,
  `docs/SPATIAL_VS_TEMPORAL_PARETO_MLP_SCAFFOLD_V1.md`, runner, summarizer,
  `SNNSpatialParallelModel`, resource-ledger support and diagnostic adaptation.

## 2026-07-14 — Runtime amendment: MLP Pareto scaffold v2

- **Why V1 stopped:** V1 scheduled eight identical exhaustive batches per
  epoch for 300 epochs (2,400 optimizer updates per cell). CUDA execution is
  dominated by Python timestep dispatch, and the first buffered result took
  minutes. The process was stopped for runtime before its output was read.
- **Observed provenance after stop:** artifact inspection found one completed
  V1 cell (`spatial_independent_d0/T18/h4/seed307`) and one interrupted cell.
  They are retained but V1 is `aborted_after_one_cell_runtime_budget_error` and
  must not be pooled with any result surface.
- **V2 amendment:** one exact 16-pattern batch per epoch, 200 epochs, exactly
  200 optimizer updates per cell. All scientific factors, models, seeds,
  endpoints, grids, metrics, resource fields and checkpoint selection remain
  unchanged. The amendment decision preceded inspection of V1 accuracy.
- **Verification:** V2 repeated all five smoke conditions in its own directory;
  complete checkpoints, truth tables, ledgers, NPZ files and panels are present.
  Targeted runner/model/resource tests pass. V2 status is
  `preregistered_ready`.
- **Boundary:** V1 is aborted provenance; V2 remains an exploratory MLP
  scaffold and cannot unlock formal spiking-output Stage B.
- **Artifacts:** `configs/spatial_vs_temporal_pareto_mlp_scaffold_v2.yaml` and
  `docs/SPATIAL_VS_TEMPORAL_PARETO_MLP_SCAFFOLD_V2.md`.

## 2026-07-14 — MLP Pareto scaffold v2 launched

- **Pre-launch:** five V2 smoke cells complete; targeted protocol/model/ledger
  tests pass; 160 unique formal-exploratory paths were empty.
- **Execution:** full 160-cell V2 grid launched on the NVIDIA RTX 4070 Laptop
  GPU. Per-cell output is intentionally suppressed from progress reporting;
  only material failure or completion is reported.
- **Locks:** no condition, grid, seed, metric, checkpoint rule, artifact rule or
  resource definition changed. Direct-spiking-output Stage B remains locked.
- **Status:** `formal_exploratory_running`; validation/exhaustive XOR only, no
  sealed test access.

## 2026-07-14 — MLP Pareto scaffold v2 completed

- **Completeness:** 160/160 cells; 160 each of validation, exhaustive truth
  table, best/final checkpoint, log, ledger, NPZ and panel; no exclusions and no
  sealed test access.
- **Robust `.90` result:** only spatial independent d0 and the fixed temporal
  oracle pass both worst-balanced and exact-trial `.90` in both seeds. At their
  minimum-resource feasible point `T=18,h=24`, spatial has 48 total hidden and
  the oracle 24; both have min-seed worst-balanced/exact `1.0/1.0`.
- **Cost interpretation:** oracle/spatial ratios are `.5` for hidden area and
  neuron updates, `1.0` for dense MACs, measured events, stored delay values
  and decoder MACs, and `5.0` for delay-buffer elements. No compute, event,
  energy or total-memory saving is established.
- **WAD negative:** worst-balanced is `.5` in all 32 WAD cells. Per-query mean
  BAcc is `[.865,.500]`; window-1 hidden spikes and activity are exactly zero.
  Selected WAD delays remain at about `.117/.119` window lengths for query
  0/1, with no query-1 synapse reaching window 1. Final checkpoints agree.
- **Mechanism:** the oracle `[0,1 window]` schedule produces activity in both
  output windows and passes at h=24. Therefore the simulator/shared hidden/MLP
  interface can express routing; the frozen WAD method fails to learn the
  required query-conditioned separation despite nonzero delay gradients.
- **Factor result:** hidden width accounts for 93.7%, 97.7% and 99.6% of
  seed-mean grid variation in spatial, shared-spatial and oracle conditions;
  T accounts for only 1.6%, .9% and .05%. WAD and temporal d0 are flat chance.
  This phase does not identify an accuracy law in T.
- **Decision:** classify as positive fixed-oracle feasibility and negative WAD
  learning evidence. Direct-spiking-output Stage B remains locked. Next choose
  between one small oracle-to-WAD bridge and a spatial-vs-oracle spiking-output
  interface study at `T=18,h=24`; do not repeat the full surface.
- **Artifacts:**
  `docs/RESULTS_SPATIAL_VS_TEMPORAL_PARETO_MLP_SCAFFOLD_V2.md` and
  `docs/generated/spatial_vs_temporal_pareto_mlp_scaffold_v2/exploratory/`.

## Entry template

## 2026-07-14 — Preregistration: spatial versus temporal Pareto Phase 0

- **Pivot:** compare K independent block-diagonal spatial modules with a shared
  hidden temporal candidate at matched reliability. Neuron area, latency,
  area-time, dense synaptic compute and causal reuse are separate endpoints.
- **Encoding decision:** do not copy Habashy et al.'s variable 0–5 input-spike
  code. Primary input uses binary one-hot value channels and exactly four
  micro-burst events per query, invariant to bit values. Output uses one target
  spike in the correct
  neuron of a class-opponent pair at the earliest d0-causally-reachable output
  offset, a
  filtered spike-train loss with `tau=5`, signed count BCE and membrane warmup.
- **Stage A:** 15 d0 K=1 XOR cells: hidden `{4,8,12,16,24}` × new seeds
  `{107,211,509}`. Select the smallest width passing every preregistered
  reliability, truth-table, silence/collision, spike-count and timing gate in
  every seed. Test remains sealed.
- **Stage B lock:** K=2 repeated XOR; block-diagonal spatial d0, shared-spatial
  d0 and shared-temporal d0/fixed/oracle/WAD. Input-to-hidden is the only delay
  locus and hidden-to-output delays remain d0. Stage B cannot be materialized
  before Stage A selects a width.
- **Cost:** area `h'/(Kh)`, hidden-update `h'T_B/(KhT_A)`, and dense-synapse
  compute `h'T_B/(hT_A)` are reported separately with the measured ledger.
- **Artifacts:** `docs/SPATIAL_VS_TEMPORAL_PARETO_PHASE0.md`,
  `configs/spatial_vs_temporal_pareto_phase0.yaml`, runner, summarizer and
  `utils/pareto_cost.py`.
- **Status:** Stage A preregistered; dry-run/smoke verification required before
  formal launch.
- **Pre-formal amendment:** binary one-hot smoke at inherited hidden threshold
  `.3` was all-silent after 100 epochs; maximum hidden membrane was `.242`
  despite nonzero gradients. Because `.3` came from a different fan-in and
  unequal-event burst code, Phase 0 freezes `.2` before any formal cell. This is
  a structural excitability correction, not accuracy-based model selection.
  At hidden threshold `.2`, sparse hidden activity appeared but output remained
  silent: maximum output pre-reset voltage was `.046` against inherited `.2`.
  Output threshold is therefore frozen at `.03` for the final pre-formal smoke;
  each selected one-hot value channel uses a two-spike micro-burst to avoid an
  all-but-silent hidden representation while preserving equal event counts.
  Existing collision/one-spike/timing gates reject over-excitability.
- **Final smoke verification:** all 36 repository tests passed; the 15 formal
  paths were unique and empty. The final h=24/seed=107 smoke achieved validation
  balanced accuracy 1.0 and exhaustive 4/4 classification, but timing-hit was
  only `.773` (`.75` exhaustive) and silent rate `.227` (`.25` exhaustive).
  It therefore proves trainability but deliberately does not pass the formal
  interface gates. The in-run diagnostic NPZ/panel now marks target `t=11`,
  timing loss, output membrane/threshold, true output spikes and routing.
- **Formal launch:** the unchanged 15-cell Stage A matrix was launched on CUDA
  on 2026-07-14. Status is `formal_running`; no test split is open and Stage B
  remains locked.

## 2026-07-14 — Formal result: spatial versus temporal Pareto Phase 0 Stage A

- **Completeness:** 15/15 validation results, truth tables, final checkpoints,
  diagnostic NPZ files and diagnostic panels are present.
- **Decision:** formal failure. No hidden size passes every gate in all three
  seeds; `selected_baseline_hidden_per_query=null` and Stage B remains locked.
  Six individual cells pass, but cells may not be cherry-picked across widths.
- **Pattern:** width is not monotonic. h=12 and h=24 each have two perfect seeds
  but seed 211 fails through opponent collisions (`.535` and `.234`). h=8 and
  h=16 failures are primarily silence. Every failed truth-table classification
  is one asymmetric positive XOR pattern with zero signed count.
- **Metric warning:** zero ties default to class 0. Accuracy alone can therefore
  mark a silent class-0 output correct; h=4/seed=211 demonstrates this with
  balanced accuracy 1.0 but silent rate `.287`.
- **Timing:** conditional absolute timing error is zero throughout. The problem
  is spike existence/exclusivity, not target-time placement.
- **Stability:** 11/15 cells reach validation accuracy 1.0 at some epoch, but
  several regress before the frozen final checkpoint. This is evidence for a
  new stability protocol, not permission for retrospective checkpoint picking.
- **Artifacts:**
  `docs/RESULTS_SPATIAL_VS_TEMPORAL_PARETO_PHASE0_STAGE_A.md` and
  `docs/generated/spatial_vs_temporal_pareto_phase0/stage_a/`.
- **Next gate:** read-only successful/silent/collision checkpoint audit, then a
  versioned interface-stability calibration. Do not launch Stage B, mixed
  operations, K/N/T scaling or test evaluation.

## 2026-07-14 — Phase-0 Stage-A checkpoint mechanism audit

- **Scope:** read-only replay of best/final checkpoints from all 15 formal
  cells on all four XOR patterns: 30 checkpoints and 120 forward passes. No
  training or test access.
- **Delay verification:** both input-to-hidden and hidden-to-output delays are
  exactly zero in every checkpoint. Stored `input_to_hidden_only` metadata was
  misleading; Stage A was operationally all-d0.
- **Failure decomposition:** 6 final cells are successful, 7 have silence and
  2 have collision. Of 10 silent patterns, 7 have zero hidden spikes and 3
  have hidden activity but fail at output conversion.
- **Collision timing:** all three collided patterns contain an incorrect spike
  at `t=10` followed by the correct target spike at `t=11`. This is early-event
  leakage into the all-time opponent count, not two neurons firing together at
  the target.
- **Checkpoint selection:** only 4/15 accuracy-selected best checkpoints have
  four valid interface patterns versus 6/15 final checkpoints; no cell loses
  valid-pattern count from best to final. Retrospective best selection is not a
  repair and accuracy is misaligned with output semantics.
- **Threshold diagnostic:** counterfactual `.015-.06` output thresholds do not
  make any width pass all three seeds. Lower thresholds improve only one cell;
  higher thresholds reduce collision marginally while sharply increasing
  silence.
- **Artifacts:**
  `docs/RESULTS_SPATIAL_VS_TEMPORAL_PARETO_PHASE0_STAGE_A_CHECKPOINT_AUDIT.md`
  and `docs/generated/spatial_vs_temporal_pareto_phase0/stage_a_checkpoint_audit/`.
- **Next gate:** preregister a versioned interface-stability calibration that
  separates truth-table batching, global one-target/no-early-spike supervision,
  and micro-burst structure using calibration versus held-out seeds. Stage B
  remains locked.

```text
Date / experiment id:
Question and predeclared hypothesis:
Protocol version and code commit:
Independent and controlled variables:
Primary metric / selection rule / seeds:
Result (all seed values):
Alternative explanations checked:
Status: canonical | exploratory | invalid | archived
Artifacts:
Decision and next gate:
```

## 2026-07-15 — Preregistration: delay parameter recovery Level 0A v1

- **Question:** can the production sigmoid delay parameterization recover
  declared scalar targets before any spike-buffer, LIF, XOR or multi-query
  optimization is introduced?
- **Scope:** deterministic one-parameter unit experiment using the actual
  `DelayedSynapticLayer.get_delays()` implementation. This is not oracle
  imitation and cannot support a routing or performance claim.
- **Matrix:** `dmax=8`; raw initializations `{-4,-2,0,2,4}`; interior targets
  `{1,5,7}`; descriptive boundary stresses `{0,8}`; Adam learning rates
  `{.001,.01,.05}`; 200 optimizer steps; 75 cells; no stochastic seeds.
- **Loss:** `0.5*(delay-target)^2` in nominal simulator-step units.
- **Primary metric:** final absolute delay error; recovery tolerance `.1` step.
- **Locked gates:** the current-recipe analogue is
  `init_raw=-2, lr=.001, target=5`. Interior numerical recoverability requires
  every interior target/initialization pair to be rescued by at least one
  preregistered learning rate. Boundary targets do not gate this claim.
- **Artifacts:** per-cell config/metrics/state/NPZ/panel plus aggregate CSV,
  decision JSON and error/convergence heatmaps.
- **Next gate:** Level 0B may be designed only after Level 0A distinguishes
  parameter/optimizer limitations from downstream temporal-credit limitations.
- **Protocol:** `configs/delay_parameter_recovery_level0a_v1.yaml` and
  `docs/DELAY_PARAMETER_RECOVERY_LEVEL0A_V1.md`.
- **Status:** preregistered, not yet run.

## 2026-07-15 — Result: delay parameter recovery Level 0A v1

- **Completeness:** 75/75 cells have config, metrics, scalar state, NPZ and
  diagnostic panel; aggregate CSV, decision and both heatmaps are present.
- **Current-recipe decision:** failed. For `init_raw=-2,target=5`, `.001` moves
  delay only `.9536 -> 1.1389` in 200 steps; final error `3.8611` and no
  convergence. Across all interior cells `.001` recovers only 2/15, both
  initialized near their target.
- **Numerical-recoverability decision:** passed narrowly. The preregistered
  `.05` arm recovers 15/15 interior target/initialization combinations;
  maximum final error is `.0781`. For the declared target-5 cell it first
  reaches `.1` error at step 56 and ends at `4.9997`.
- **Boundary stress:** descriptive only. With `.05`, 4/10 boundary cells pass;
  finite sigmoid parameters cannot attain exact 0 or `dmax`, confirming that
  an oracle target placed at the boundary is unnecessarily hostile.
- **Interpretation:** no hard production sigmoid implementation bug is found,
  but the current raw-parameter learning-rate/step budget is inadequate even
  under direct supervision. `.05` is not authorized for XOR transfer.
- **Artifacts:** `docs/RESULTS_DELAY_PARAMETER_RECOVERY_LEVEL0A_V1.md`,
  `docs/generated/delay_parameter_recovery_level0a_v1/`, and
  `runs/exploratory/delay_parameter_recovery_level0a_v1/`.
- **Next gate:** preregister Level 0B to test arrival-time loss through the
  production delay buffer first, then add one LIF neuron. Preserve
  `.001/.01/.05` as diagnostic arms and do not introduce XOR yet.
- **Status:** exploratory mechanism diagnostic complete.

## 2026-07-15 — Preregistration: delay temporal credit Level 0B v1

- **Question:** can output-timing loss move one production sigmoid delay
  through the actual circular buffer, and does that credit survive one fixed
  production LIF neuron?
- **Paths:** buffer-current trace and one-LIF spike trace. Input is one event at
  `t=2`; `T=16`, `dmax=8`; LIF fixed weight 4, `tau=10`, threshold `.2 a.u.`,
  reset 0, refractory 2, surrogate beta 4.
- **Objectives:** global arrival-centroid loss and causal filtered-trace MSE
  (`tau_filter=3`). Neither directly reads the delay value.
- **Matrix:** targets `{1,5,7}` nominal steps (arrivals `{4,8,10}`), raw
  initializations `{-4,-2,0,2,4}`, Adam LRs `{.001,.01,.05}`, 200 steps,
  `2 x 2 x 3 x 5 x 3 = 180` deterministic cells.
- **Primary metric:** final output-arrival error; pass requires error `<=.1`
  step and output trace mass `>=.5`.
- **Gate:** each path/loss must recover every target/initialization pair under
  at least one preregistered LR. Strict Level 0B requires all four to pass.
- **Exclusions:** no XOR, trainable weights, readout, direct delay loss,
  oracle imitation, accuracy, resource or publication claim.
- **Artifacts:** per-cell config/metrics/state/NPZ/six-panel diagnostic plus
  aggregate CSV, decision, four heatmaps and recovery summary.
- **Protocol:** `configs/delay_temporal_credit_level0b_v1.yaml` and
  `docs/DELAY_TEMPORAL_CREDIT_LEVEL0B_V1.md`.
- **Status:** preregistered, not yet run.

## 2026-07-15 — Result: delay temporal credit Level 0B v1

- **Completeness:** 180/180 deterministic cells completed. Every cell contains
  config, metrics, final scalar parameter, runtime NPZ and six-panel diagnostic;
  aggregate CSV, decision JSON, four error heatmaps, recovery summary and
  initial-gradient summary are present.
- **Formal decision:** strict Level-0B failed. `buffer_current +
  arrival_centroid` recovers 15/15 target/initialization pairs, but
  `buffer_current + filtered_trace`, `lif_spike + arrival_centroid` and
  `lif_spike + filtered_trace` recover 5/15, 3/15 and 10/15 respectively.
- **Current recipe:** `init_raw=-2, lr=.001, target=5` fails all four arms;
  final arrival errors are `3.8611`, `4.1823`, `4.0` and `4.0` steps.
- **Buffer conclusion:** buffer centroid has the correct initial gradient sign
  in all 13 initially misaligned pairs and reaches 15/15 at `.05`. The actual
  circular buffer therefore does not impose a hard scalar-gradient obstruction
  in this one-event setup.
- **Objective conclusion:** causal filtered trace starts with a wrong nonzero
  gradient in 5/13 misaligned pairs both before and after LIF. Buffer recovery
  stays at 5/15 when LR rises from `.01` to `.05`; this is not a step-size-only
  failure.
- **Hard-spike conclusion:** LIF centroid starts with zero raw-delay gradient in
  8/13 misaligned pairs. A normalized centroid of one hard spike is piecewise
  constant and is not a usable general delay-learning objective.
- **Directional result:** LIF filtered trace often moves early spikes later but
  fails late-to-early movement. This 10/15 partial result is mechanism evidence,
  not permission to relax the bidirectional gate.
- **Scope:** no XOR labels, trainable weights, query identities, readout,
  accuracy, resource comparison or test split. No XOR LR, routing or Pareto
  claim is authorized.
- **Artifacts:** `docs/RESULTS_DELAY_TEMPORAL_CREDIT_LEVEL0B_V1.md`,
  `docs/generated/delay_temporal_credit_level0b_v1/`, and
  `runs/exploratory/delay_temporal_credit_level0b_v1/`.
- **Next gate:** preregister Level 0C at the same one-event scale to compare
  symmetric/global timing credit on soft current or membrane traces. Require
  correct early-to-late and late-to-early gradient signs plus full target/init
  recovery before K=1 XOR. Per-neuron tying may be tested only as a separate
  factor; it cannot repair a wrong single-delay gradient direction.
- **Status:** exploratory mechanism diagnostic complete; formal gate negative.

## 2026-07-15 — Preregistration: delay soft-trace credit Level 0C v1

- **Question:** can a global/symmetric loss computed only from a soft current or
  subthreshold membrane trace provide correct bidirectional delay credit, and
  is any residual failure caused by sigmoid parameterization?
- **Paths:** production circular-buffer current and one production LIF membrane
  with fixed weight 1, `tau=10`, threshold `.2 a.u.`, reset 0, refractory 2.
  The membrane path must emit zero hard spikes.
- **Targets:** one input at `t=2`, `T=16`, `dmax=8`; nominal targets `{1,5,7}`
  give current arrivals `{4,8,10}`. Membrane targets are detached production-LIF
  responses to the corresponding target currents. Loss never reads delay.
- **Objectives:** raw causal filtered MSE (`tau=3`) as a selection-ineligible
  negative control; normalized global soft-centroid loss; normalized global
  symmetric Laplace-kernel alignment (`K_ij=exp(-|i-j|/4)`).
- **Parameterizations:** production sigmoid and production direct. Functional
  initial delays are matched as `8*sigmoid(r)` for labels
  `r∈{-4,-2,0,2,4}`; direct parameters start at those functional values.
- **Matrix:** two parameterizations, two paths, three objectives, three targets,
  five initializations, Adam `{.01,.05}`, 200 updates: 360 deterministic cells.
- **Primary endpoint:** normalized trace `W1=sum|CDF_out-CDF_target|` in steps;
  recovery requires `W1<=.1`, mass `>=.05` and path validity.
- **Gradient gate:** every initially misaligned pair must have a nonzero
  objective gradient whose sign moves output centroid toward target centroid.
- **Selection gate:** one candidate objective at one common LR must recover all
  30 production-sigmoid cells across both paths and pass every gradient gate.
  Pair-specific LR mixing is descriptive only. Direct-only success localizes a
  parameterization issue but does not pass. If multiple candidates pass,
  choose `symmetric_kernel_alignment` before `soft_centroid`, then the lowest
  common LR.
- **Exclusions:** no hard-spike training trace, XOR, labels, trainable weights,
  readout, delay supervision, tying comparison, accuracy, resources or test.
- **Next authorization:** Level-0C success permits only a versioned Level-0D
  hard-spike forward/soft-auxiliary bridge; XOR remains locked.
- **Protocol:** `configs/delay_soft_trace_credit_level0c_v1.yaml` and
  `docs/DELAY_SOFT_TRACE_CREDIT_LEVEL0C_V1.md`.
- **Status:** preregistered, not yet run.

## 2026-07-15 — Pre-run amendment: Level 0C symmetric objective

- **Trigger:** before smoke or formal data generation, a unit test found that
  the originally proposed squared-CDF loss has exact zero delay gradient when
  the current delay is an integer and the target lies earlier. Production
  floor/ceil interpolation exposes a right-hand mixture; squaring its initially
  zero boundary CDF difference removes the needed first-order signal.
- **Change:** replace only that candidate with normalized symmetric Laplace-
  kernel alignment, `-log(sum_ij p_i exp(-|i-j|/4) q_j + eps)`.
- **Unchanged:** W1 primary endpoint, paths, parameterizations, targets,
  initializations, LRs, update budget, artifacts and all decision gates.
- **Integrity:** no smoke/formal Level-0C cell existed when this amendment was
  recorded. The failed unit test is design verification, not experiment data.

## 2026-07-15 — Result: delay soft-trace credit Level 0C v1

- **Completeness:** 360/360 deterministic cells contain config, metrics, final
  scalar state, runtime NPZ and runtime six-panel diagnostic. Aggregate CSV,
  decision JSON, twelve W1 heatmaps and both summary figures are present.
- **Formal decision:** Level 0C passes. The selected production candidate is
  sigmoid + soft centroid + Adam `.05`. At this one common LR it recovers all
  30 buffer-current and LIF-membrane cells, passes 13/13 correct nonzero initial
  directions per path and keeps every membrane cell subthreshold.
- **Selected-arm precision:** maximum final W1 is `.0781` on current and `.0583`
  on membrane; maximum delay error is `.0781/.0870`; the latest first gate
  crossing occurs at step `185/168`. Minimum membrane mass is `.451`.
- **Learning-rate result:** `.01` recovers only 6/15 per selected path; `.05`
  is selected under the fixed common-LR rule. This remains an auxiliary-unit
  result, not permission to change XOR delay LR.
- **Filtered-control interaction:** raw causal filtered loss stays at 5/15 with
  5/13 wrong directions on buffer current, but becomes 15/15 with 13/13 correct
  directions after passive membrane smoothing. Failure belongs to the
  loss-by-observed-state interaction, not to the loss name alone.
- **Kernel failure:** symmetric Laplace-kernel alignment is 15/15 on current
  but 0/15 on membrane under both parameterizations. It usually converges one
  delay step late because one-sided membrane decay and finite truncation bias
  the cross-similarity optimum; one late pair has a wrong initial direction.
- **Direct control:** matched direct centroid has correct directions but only
  11/15 recovery per path at `.05`; the four extreme functional moves remain
  `.16-.61` W1 away. Equal parameter-coordinate LR does not match delay-space
  travel, so this is not intrinsic sigmoid superiority.
- **Scope:** one event, one fixed-weight synapse, no hard spikes, labels,
  trainable weights, readout, generalization, resources or test split.
- **Authorization:** Level 0D only. Hard-spike arrival must be evaluated while a
  prespecified current or pre-reset-membrane centroid supplies auxiliary
  credit. XOR, pairwise WAD, tying and Pareto work remain locked.
- **Artifacts:** `docs/RESULTS_DELAY_SOFT_TRACE_CREDIT_LEVEL0C_V1.md`,
  `docs/generated/delay_soft_trace_credit_level0c_v1/`, and
  `runs/exploratory/delay_soft_trace_credit_level0c_v1/`.
- **Status:** exploratory mechanism diagnostic complete; formal gate positive.

## 2026-07-15 — Preregistration: delay hard-output / soft-credit Level 0D v1

- **Question:** can current or pre-reset-membrane centroid credit place one
  suprathreshold hard LIF spike at the declared time in both directions, and
  remain effective when combined with hard filtered-spike loss?
- **Forward path:** one input event at `t=2`, `T=16`, `dmax=8`; production
  circular delay layer with fixed weight 4 and one sigmoid delay; production
  LIF `tau=10`, threshold `.2 a.u.`, reset 0, refractory 2, surrogate beta 4.
- **Endpoint:** target delays `{1,5,7}` give one-hot hard target spikes at
  `{4,8,10}`. Recovery requires final output spike count exactly 1 and final
  hard-train W1 `<=.1` step. Transient crossings do not count.
- **Losses:** Level-0B causal filtered hard-spike MSE (`tau=3`); Level-0C
  centroid auxiliary from synaptic current or production pre-reset voltage;
  combined `L_hard + lambda*L_aux` for `lambda={.01,.1,1}`.
- **Conditions:** hard-only, two soft-only controls, and six hard-plus-auxiliary
  candidates: nine conditions total.
- **Matrix:** nine conditions, three targets, five raw initializations, fixed
  Adam `.05`, 200 updates: 135 deterministic cells.
- **Gradient audit:** store initial hard, auxiliary and total raw-delay
  gradients and their conflict. Every initially misaligned pair in a passing
  condition requires a correct nonzero total direction.
- **Gate:** Level 0D requires one combined condition to recover 15/15 and pass
  all directional checks. Prefer pre-reset over current, then lowest passing
  lambda. Soft-only success is not sufficient.
- **Exclusions:** no post-reset auxiliary, trainable weights, XOR labels,
  readout, multiple neurons/synapses, accuracy, resources or test split.
- **Authorization:** a pass permits only preregistration of K=1 XOR calibration;
  pairwise WAD, tying, K scaling and Pareto work remain locked.
- **Protocol:** `configs/delay_hard_output_soft_credit_level0d_v1.yaml` and
  `docs/DELAY_HARD_OUTPUT_SOFT_CREDIT_LEVEL0D_V1.md`.
- **Status:** preregistered, not yet run.

## 2026-07-15 — Pre-formal implementation validation: Level 0D

- A unit test reproduced an exact zero pre-reset-centroid gradient at the
  declared integer-delay boundary (`target=1, init_raw=0`). The pre-reset arm
  and every gate were retained unchanged so this remained a tested hypothesis,
  not a screened-out failure.
- The first smoke invocation stopped while drawing the synaptic-current panel
  because the display key used `synaptic_current` rather than the stored
  `current` array. Only the diagnostic key mapping was corrected; the smoke
  cells were rerun. No formal cell existed, and no loss, target, lambda,
  optimizer, metric or decision rule changed.
- The completed ten-cell smoke has all required runtime artifacts and is marked
  invalid-for-claims by the registry policy.

## 2026-07-15 — Result: delay hard-output / soft-credit Level 0D v1

- **Completion:** 135/135 deterministic cells. Every cell has config, metrics,
  final scalar state, runtime NPZ and runtime eight-panel diagnostic. Aggregate
  CSV, decision JSON, nine hard-W1 heatmaps and three mechanism summaries are
  complete.
- **Formal decision:** pass. `hard + current centroid, lambda=.1` is selected:
  15/15 recovery and 13/13 correct nonzero initial total-gradient directions
  among hard-misaligned pairs. `lambda=1` also passes; `.01` remains 10/15.
- **Negative control:** hard filtered-spike loss alone recovers 10/15 and has
  five wrong initial directions. Its five failed late-to-early cells move
  toward the upper delay bound, so this is directional failure, not merely slow
  convergence.
- **Auxiliary mechanism:** current centroid alone is 15/15. Hard and current
  gradients conflict in five cells; `lambda=.1` is sufficient to correct the
  total direction in all of them.
- **Rejected hypothesis:** pre-reset centroid alone is 3/15 and exactly flat in
  eight of 13 initially misaligned pairs. None of the combined pre-reset arms
  exceeds 10/15. Output proximity does not preserve continuous delay credit
  through hard threshold/reset segmentation.
- **Claim boundary:** one fixed-weight synapse, one event and directly timed
  current moment only. No XOR, generalization, WAD, tying, routing, accuracy or
  resource conclusion follows.
- **Authorization:** preregister Level 1A K=1 XOR calibration only. Keep hard
  output, name the current-centroid auxiliary explicitly, lock its scale versus
  task loss, include d0/fixed controls, and audit delay/weight gradients.
- **Artifacts:** `docs/RESULTS_DELAY_HARD_OUTPUT_SOFT_CREDIT_LEVEL0D_V1.md`,
  `docs/generated/delay_hard_output_soft_credit_level0d_v1/`, and
  `runs/exploratory/delay_hard_output_soft_credit_level0d_v1/`.
- **Status:** exploratory mechanism diagnostic complete; formal gate positive.

## 2026-07-16 — Preregistration: XOR task bridge Level 1A v1

- **Question:** can the Level-0D hard-output/continuous-current credit bridge
  survive the smallest complete task with trainable weights, two classes and
  all four XOR patterns? This is not a test that delay improves XOR accuracy.
- **Task and encoding:** exhaustive K=1 XOR truth table on every update;
  one-hot channels `[A0,A1,B0,B1]`; one event at `t=9` on each selected channel,
  exactly two simultaneous input events for every pattern. Consecutive
  two-event micro-bursts are deferred to Level 1B.
- **Model:** `4 -> 16 -> 2` production LIF network; thresholds `.2/.03 a.u.`,
  `tau=10`, reset 0, refractory 2 and surrogate beta 4. Hidden-to-output delays
  are d0. The learned input-to-hidden delay is one global sigmoid scalar,
  broadcast over all 64 pairs, with range `[0,8]`.
- **Timing:** the production buffer contributes `d+1` steps per delayed layer.
  With input at 9 and d0 hidden-to-output, the target output is at `11+d_ih`:
  t=11 for d0 and t=15 for the fixed/learned delay-4 schedule.
- **Objective:** causal filtered hard-spike MSE plus a globally balanced
  pre-reset-voltage envelope (`eta={0,.1,1}`), and in Stage II an explicitly
  labelled unweighted input-arrival centroid teacher
  (`lambda={0,.01,.1,1}`). The arrival term is an oracle timing scaffold, not
  learned routing evidence.
- **Stage I:** 90 indivisible cells: two fixed schedules, three eta values,
  three weight LRs and five seeds. A `(eta,lr_w)` candidate must pass all ten
  schedule/seed cells; choose lowest eta, then lowest LR. Stage II is locked
  unless the complete decision passes.
- **Stage II:** five fixed-d0/wrong-target negative cells plus 80 learned-delay
  cells spanning four lambdas, two delay LRs, both early/late initializations
  and five seeds. A `(lambda,lr_d)` candidate must pass all ten
  initialization/seed cells; choose lowest lambda, then lowest delay LR.
- **Exact gate:** all four labels and balanced accuracy 1; exact equality of
  actual and one-target opponent spike trains; no silence/collision; one output
  spike per pattern at the target time; hidden activity in every pattern.
  Learned cells also require `|d_final-4|<=.1` and correct nonzero initial total
  delay-gradient direction.
- **Artifacts:** strict config/metrics, checkpoint, truth-table output, resource
  ledger, runtime NPZ and runtime 12-panel diagnostic for every cell; aggregate
  tables, heatmaps and decision files per stage.
- **Scope lock:** K>1, WAD, per-neuron/per-synapse granularity, micro-burst
  robustness, Pareto and energy conclusions remain unauthorized.
- **Protocol:** `configs/xor_task_bridge_level1a_v1.yaml` and
  `docs/XOR_TASK_BRIDGE_LEVEL1A_V1.md`.
- **Status:** preregistered and implementation-validated; 0 formal cells.

## 2026-07-16 — Pre-formal implementation validation: XOR Level 1A

- Eleven protocol tests pass, including exact truth-table encoding, target
  timing, shared-delay tying, separate optimizer rates, production `d+1`
  arrival semantics, bidirectional centroid gradients, envelope coverage and
  collision rejection. Dry expansion is exactly 90 Stage-I and 85 Stage-II
  paths; Stage II refuses formal materialization without a passing Stage-I
  decision.
- Two smoke cells per stage generated config, metrics, checkpoint,
  truth-table output, resource ledger, NPZ and diagnostic panel during the run.
  Stage-II smoke used a synthetic Stage-I selection only to exercise code.
- The four smoke cells are invalid-for-claims. Their failures are not formal
  evidence, were not used to select settings, and caused no amendment to the
  preregistered task, losses, parameter grid or gates.
- **Status:** implementation validation complete; formal Stage I not started.

## 2026-07-16 — Formal result: XOR task bridge Level 1A Stage I

- **Completion:** 90/90 formal cells. All cells contain config, metrics, final
  checkpoint, training log, exhaustive truth table, resource ledger, runtime
  NPZ and runtime diagnostic panel. Aggregate cells CSV, heatmap and decision
  JSON are complete.
- **Interruption provenance:** the first terminal-bound process was terminated
  by a new chat message after 47 complete cells. The runner skipped those 47
  cells by their complete `metrics.json`; the interrupted 48th cell had only a
  config and was rerun from initialization. The resumed process completed with
  exit code 0. No experimental setting or decision rule changed.
- **Decision:** Stage I passes. The unique selected candidate is `eta=0` and
  `lr_w=.01`, with 10/10 exact-interface passes across fixed d0, fixed delay 4
  and five seeds. Every selected cell has balanced accuracy 1, exact hard-spike
  truth-table completion, zero silence/collision, correct target timing, one
  output event per pattern and nonzero hidden activity.
- **Non-selected candidates:** pass counts are 0/10, 8/10 and 10/10 for
  `eta=0` as LR rises; 0/10, 4/10 and 7/10 for `eta=.1`; and 0/10, 0/10 and
  8/10 for `eta=1`. The voltage-envelope auxiliary is therefore not required
  by the locked selection and positive weights are not seed-robust here.
- **Claim boundary:** this validates the fixed-schedule K=1 hard-spike
  interface only. No delay was learned, and no routing, compression, WAD,
  generalization or Pareto conclusion follows.
- **Authorization:** run the complete 85-cell Stage II with `eta=0,lr_w=.01`
  read from the Stage-I decision. K>1 and Level 1B remain locked.
- **Artifacts:** `docs/RESULTS_XOR_TASK_BRIDGE_LEVEL1A_STAGE_I.md`,
  `docs/generated/xor_task_bridge_level1a_v1/stage_i/`, and
  `runs/exploratory/xor_task_bridge_level1a_v1/stage_i/`.
- **Status:** formal Stage I complete and positive; Stage II authorized but not
  started.

## 2026-07-16 — Formal result: XOR task bridge Level 1A Stage II

- **Completion:** 85/85 cells. All cells have config, metrics, final checkpoint,
  full training log, exhaustive truth table, resource ledger, runtime NPZ and
  diagnostic panel. Aggregate CSV, decision JSON and heatmap are complete.
- **Frozen interface:** Stage II read `eta=0,lr_w=.01` from the complete Stage-I
  decision; no manual substitution was used.
- **Decision:** Level 1A passes. Positive arrival-scaffold arms pass 10/10 for
  every tested lambda/LR pair. The locked priority rule selects the smallest
  values, `lambda=.01,lr_d=.01`.
- **Selected arm:** 10/10 pass across raw initializations `-2/+2` and five
  seeds. Final delay range is `[3.997372,4.001881]`, maximum error `.002628`
  step; balanced accuracy, exact hard-spike truth-table completion, target
  timing and initial total-gradient direction all pass 10/10. Task and arrival
  gradients conflict in 5/10 selected cells.
- **Task-only result:** `lambda=0` passes 0/10 at `lr_d=.01` and 1/10 at `.05`.
  Only 5/10 initial task-gradient directions are correct at either LR. At `.05`,
  8/10 cells reach exact spike trains but only one also meets delay and gradient
  gates, so output accuracy does not establish timing-parameter recovery.
- **Timing-specificity control:** fixed d0 with target-at-15 passes 0/5 and has
  correct-target-time rate zero in every seed. One seed has balanced accuracy 1
  but remains temporally invalid.
- **Claim boundary:** this is a scaffold-assisted K=1 joint task/delay bridge.
  The arrival loss explicitly encodes the oracle delay-4 schedule. It is not
  evidence for task-derived routing discovery, WAD, multiplexing,
  generalization, compression or a Pareto law.
- **Authorization:** preregister Level 1B K=1 delay-granularity and consecutive
  micro-burst robustness. Keep K>1 and test access locked.
- **Artifacts:** `docs/RESULTS_XOR_TASK_BRIDGE_LEVEL1A_STAGE_II.md`,
  `docs/generated/xor_task_bridge_level1a_v1/stage_ii/`, and
  `runs/exploratory/xor_task_bridge_level1a_v1/stage_ii/`.
- **Status:** Level 1A complete and positive within its narrow scaffold-assisted
  claim; Level 1B preregistration authorized.

## 2026-07-16 — Preregistration: XOR delay granularity Level 1B v1

- **Question:** at K=1, does the explicit Level-1A delay-4 scaffold remain
  robust as independent input-delay dimensionality grows from global (1) to
  per-hidden-neuron (16) or per-synapse (64), and does any named
  parameterization survive a consecutive two-event micro-burst?
- **Frozen model:** exhaustive XOR, `4 -> 16 -> 2` hard LIF, hidden/output
  thresholds `.2/.03` arbitrary simulator units, `dmax=8`, sigmoid delays,
  d0 hidden-to-output, Adam weight/delay LR `.01/.01`, 500 full-batch updates,
  final checkpoint only.
- **Stage A:** `3 granularities x 2 lambda conditions x 2 initial raw values x
  5 new seeds = 60` cells. Seeds `{607,709,811,919,1021}` are disjoint from
  Level 1A. Single-event inputs retain the exact Level-1A event load and timing.
- **Loss:** hard filtered opponent-spike loss plus `lambda in {0,.01}` times a
  per-independent-coordinate arrival-centroid loss targeted to delay 4. The
  positive condition is an explicit oracle teacher, not task-derived routing.
- **Cell gate:** exact four-pattern output spike train, zero silence/collision,
  target timing one, one output event and four hidden-active patterns; every
  independent delay must finish within `.1` step and every initial total
  raw-delay gradient must be nonzero and point toward four.
- **Stage-A decisions:** each `(granularity,lambda)` spans ten cells and requires
  10/10. Only a 10/10 global-scaffold held-out replication unlocks Stage-B
  controls; higher-dimensional extension is reported separately.
- **Stage B:** micro-burst events occur at steps 8 and 9 on both selected input
  channels, four input events per trial, but the target remains exactly one
  correct opponent spike at step 15. Five fixed-delay-4 cells form the
  feasibility gate; five fixed-d0 cells are timing-specificity negatives. A
  fixed-oracle 5/5 pass is required before the fixed 60 learned cells run.
- **Artifacts:** checkpoint, strict metrics, full training log, exhaustive
  truth table, resource ledger, runtime NPZ and granularity-aware 12-panel
  diagnostic are mandatory per cell. Mean delay cannot replace maximum error
  and coordinate coverage.
- **Claim boundary:** no heterogeneous routing, K>1, WAD superiority,
  generalization, compression, energy or Pareto conclusion is admissible.
- **Protocol:** `configs/xor_delay_granularity_level1b_v1.yaml` and
  `docs/XOR_DELAY_GRANULARITY_LEVEL1B_V1.md`.

## 2026-07-16 — Pre-formal implementation validation: XOR Level 1B

- Dry expansion is exactly 60 Stage-A, 10 fixed-control and 60 learned
  micro-burst cells. Downstream formal stages refuse to materialize without
  their complete decision files.
- Ten Level-1B structural tests and the full 88-test project suite pass. Tests
  cover event counts/timing, 1/16/64 tying, d0 output delays, resource charging,
  optimizer rates, production `d+1` arrival semantics, per-coordinate
  bidirectional scaffold gradients and stage locks.
- Two smoke cells per stage generated checkpoint, strict JSON, full log,
  exhaustive truth-table output, resource ledger, NPZ and diagnostic panel
  during execution. Synthetic decisions were used only to exercise the two
  downstream smoke paths.
- Smoke illustrates why the gate is coordinate-wise: a per-synapse scaffold
  cell reached exact XOR and mean delay `3.99996` yet failed because maximum
  coordinate error was `.378` and coverage was `.859`. These values are not
  formal evidence and did not change any setting or gate.
- **Status:** implementation validation complete; Stage A is launch-ready; no
  formal Level-1B cell has been launched. Stage B remains mechanically locked.

## 2026-07-16 — Formal result: XOR delay granularity Level 1B Stage A

- **Completion:** 60/60 formal cells. Every cell contains config, metrics,
  checkpoint, full training log, exhaustive truth table, resource ledger,
  runtime NPZ and runtime diagnostic panel. Aggregate CSV, heatmap and decision
  JSON are complete. No test split was opened.
- **Global replication:** scaffold `lambda=.01` passes 10/10 across both raw
  initializations and five new seeds. Exact hard-spike XOR passes in all ten;
  maximum final delay error is `.001864` step. The registered global
  replication gate passes.
- **Higher-dimensional result:** per-hidden scaffold passes 2/10 and
  per-synapse scaffold passes 0/10, so the extension gate fails. Per-hidden
  nevertheless preserves the exact interface in 10/10; only 3/10 achieve full
  delay coverage and 4/10 have the correct initial total-gradient direction on
  every coordinate. Per-synapse has interface 7/10, delay coverage 0/10 and
  all-coordinate initial direction 0/10.
- **Task-only comparators:** global passes 1/10; per-hidden and per-synapse pass
  0/10. This remains non-robust and does not support autonomous task-derived
  delay learning.
- **Mechanism caveat:** the frozen per-coordinate scaffold is averaged over
  `P`, so its initial arrival-gradient magnitude per coordinate is
  approximately `2.5588/P`. Increasing dimension therefore also weakens each
  coordinate's teacher signal at fixed lambda/LR/update budget. The formal
  conclusion is failure of the frozen recipe to scale, not intrinsic
  impossibility of per-neuron or per-synapse delays.
- **Resources:** physical synapses, neuron updates, dense MACs and buffer
  elements are matched. Trainable input-delay values are 1/16/64 and total
  delay storage is 33/48/96 elements for global/per-hidden/per-synapse.
- **Decision:** global replication passes; higher-dimensional extension fails;
  the complete ten-cell fixed micro-burst control matrix is authorized. The
  60 learned micro-burst cells remain locked pending fixed-d4 5/5.
- **Artifacts:** `docs/RESULTS_XOR_DELAY_GRANULARITY_LEVEL1B_STAGE_A.md`,
  `docs/generated/xor_delay_granularity_level1b_v1/stage_a/`, and
  `runs/exploratory/xor_delay_granularity_level1b_v1/stage_a/`.

## 2026-07-16 — Formal result: Level 1B fixed micro-burst controls

- **Completion:** 10/10 cells across fixed delay 4 and fixed d0. Every cell has
  config, final checkpoint, strict metrics, full training log, exhaustive
  truth table, resource ledger, runtime NPZ and runtime diagnostic panel.
- **Fixed oracle:** delay 4 passes 5/5. All 20 truth-table patterns have
  balanced accuracy one, exact output spike trains, zero silence/collision,
  hidden activity and exactly one correct output at step 15.
- **Timing-specificity negative:** d0 passes 0/5. Balanced accuracy ranges
  `.50--.75`, exact output matches are 0/20 and target-time rate is zero in
  every seed. All 27 emitted output spikes occur at step 11; every pattern
  nevertheless has hidden activity.
- **Interpretation:** the consecutive four-event input interface is feasible
  under an experimenter-supplied delay-4 schedule, while weights with d0 cannot
  imitate the delayed timing. This is not learned timing or routing evidence.
- **Decision:** the fixed-oracle 5/5 gate passes, mechanically authorizing the
  frozen 60-cell learned micro-burst matrix. It remains unrun. Post-result
  lambda/LR changes are forbidden; a dimension-aware rescue requires a new
  protocol and new seeds.
- **Artifacts:**
  `docs/RESULTS_XOR_DELAY_GRANULARITY_LEVEL1B_STAGE_B_CONTROLS.md`,
  `docs/generated/xor_delay_granularity_level1b_v1/stage_b_controls/`, and
  `runs/exploratory/xor_delay_granularity_level1b_v1/stage_b_controls/`.

## 2026-07-16 — Preregistration: dimension-aware delay rescue Level 1B-R v1

- **Question:** does the Level-1B higher-dimensional failure arise because its
  mean scaffold weakens each coordinate's timing-teacher gradient by `1/P`?
  Can analytically matched per-coordinate supervision recover per-hidden or
  per-synapse delay 4 while preserving exact K=1 hard-spike XOR?
- **Protocol separation:** this is a new identifier,
  `xor_delay_granularity_rescue_level1br_v1`. It does not amend, overwrite or
  reinterpret the completed Level-1B candidates.
- **Frozen interface:** single-event exhaustive K=1 XOR, `4 -> 16 -> 2` hard
  LIF, target step 15, hidden-output d0, sigmoid `dmax=8`, thresholds `.2/.03`,
  weight LR `.01`, initial delay LR `.01`, 500 updates and final checkpoint.
- **R1 intervention:** retain the mean per-coordinate arrival-centroid loss but
  set `lambda_P=.01P`, giving effective lambdas `.01/.16/.64` for global,
  per-hidden and per-synapse delays. This is equivalent to `.01` times the sum
  of coordinate losses and is fixed analytically rather than selected.
- **R1 grid:** global anchor; scaled and unscaled per-hidden; scaled and
  unscaled per-synapse; two initial raw values; five new seeds
  `{1123,1229,1321,1427,1523}`: 50 cells. Every named candidate requires
  10/10 under the exact interface and all-coordinate delay/gradient gates.
- **Conditional R2:** LR/budget calibration is permitted only for a scaled R1
  candidate whose ten cells all have correct nonzero coordinate directions but
  which fails final recovery. It tests exactly `.05/500`, `.01/1000` and
  `.05/1000` on separate seeds `{1601,1693,1789}`, with first 6/6 passing
  intervention selected. Wrong-direction R1 results prohibit R2.
- **R3 confirmation:** sealed seeds `{2003,2011,2027,2039,2053}` remain
  inaccessible until preceding decisions complete. The global anchor and every
  provisional higher-dimensional recipe require independent 10/10 passes.
  Per-hidden has priority if both confirm because it has 16 versus 64 delay
  parameters.
- **Mechanism audit:** record unweighted/weighted per-coordinate gradients,
  task conflict, pre-clip global norm and clip coefficient. Any coefficient
  below `.999` is flagged.
- **Claim boundary:** success would establish only optimization under stronger
  total explicit oracle supervision as `P` grows. It cannot support autonomous
  routing, heterogeneous specialization, micro-burst, K>1, compression or
  Pareto claims.
- **Protocol:** `configs/xor_delay_granularity_rescue_level1br_v1.yaml` and
  `docs/XOR_DELAY_GRANULARITY_RESCUE_LEVEL1BR_V1.md`.

## 2026-07-16 — Pre-formal implementation validation: Level 1B-R

- R1 dry expansion is exactly 50 cells; R2 is conditionally 0--36 and R3 is
  conditionally 20--30. Locked stages require complete decision files.
- Nine rescue-specific structural tests and the full 97-test project suite
  pass. Tests cover analytic `.01P` scaling, matched weighted coordinate
  gradients, disjoint seeds, conditional LR authorization, sealed confirmation,
  explicit optimizer overrides, d0 output delays and exclusion of micro-burst/K.
- Two smoke cells per stage generated every required artifact during the run.
  Synthetic decisions were used only for locked-path validation. All six smoke
  cells are invalid for claims.
- Smoke confirms implementation mathematics: global, per-hidden and
  per-synapse scaled conditions have weighted initial arrival-gradient magnitude
  per coordinate near `.025588`, and sampled cells do not trigger clipping.
  These outcomes did not alter the matrix or gates.
- **Status:** implementation validation complete; R1 launch-ready; no formal
  Level-1B-R cell launched.

## 2026-07-16 — Level 1B-R R1 complete: dimension normalization succeeds

- **Execution:** all 50 preregistered R1 cells completed; every cell contains
  the eight mandatory runtime/checkpoint artifacts. Test data remained sealed.
- **Global anchor:** 10/10 full pass.
- **Unscaled replication:** per-hidden mean baseline 1/10; per-synapse mean
  baseline 0/10. Both retain 10/10 exact hard-spike interfaces but fail
  coordinate-wise delay recovery.
- **Dimension-matched intervention:** per-hidden (`lambda=.16`) 10/10 and
  per-synapse (`lambda=.64`) 10/10. Every coordinate has correct nonzero
  initial total-gradient direction and finishes within `.1` step of delay 4.
- **Mechanism:** scaled conditions match the global initial weighted arrival
  gradient per coordinate at approximately `.025588`; unscaled values are
  `.001599` and `.0003998`, the predicted `1/16` and `1/64`. No cell triggers
  gradient clipping.
- **Decision:** R2 is not required and must not run. R3 direct authorization is
  true. The next allowed execution is the complete 30-cell sealed R3 matrix.
- **Claim limit:** R1 supports normalization sufficiency only for explicit
  homogeneous oracle delay recovery. It does not support autonomous discovery,
  heterogeneous routing, micro-burst, K>1, WAD, compression or Pareto claims.
- **Results:** `docs/RESULTS_XOR_DELAY_GRANULARITY_RESCUE_LEVEL1BR_R1.md`.

## 2026-07-16 — Level 1B-R sealed R3 complete on GPU

- **Execution:** the complete 30-cell sealed matrix ran on an NVIDIA GeForce
  RTX 4070 Laptop GPU; no subset or intermediate selection was used.
- **Confirmation:** global 10/10, per-hidden-neuron 10/10 and per-synapse
  10/10. Every cell passes the exact hard-spike interface, initial
  coordinate-gradient direction and final full-delay-coverage gates.
- **Numerics:** maximum final delay errors are `.017347`, `.000497` and
  `.000217` step for global, per-hidden and per-synapse respectively. No cell
  triggers the gradient-clipping flag.
- **Artifacts:** 30/30 complete diagnostic/checkpoint bundles; zero missing.
- **Decision:** Level 1B-R passes. Both higher-dimensional candidates confirm;
  the preregistered complexity rule selects per-hidden-neuron (16 rather than
  64 independent delay values). R2 remains unrun and unnecessary.
- **Boundary:** micro-burst and K>1 remain unauthorized. The result is
  homogeneous explicit-oracle recovery, not autonomous temporal routing.
- **Results:** `docs/RESULTS_XOR_DELAY_GRANULARITY_RESCUE_LEVEL1BR_R3.md`.
