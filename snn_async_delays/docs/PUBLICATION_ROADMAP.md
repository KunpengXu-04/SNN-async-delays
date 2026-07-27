# Publication roadmap: learned delays under observation and resource controls

**Status:** authoritative forward plan, version 1.1 (2026-07-13)  
**Scope:** this document governs new experiments. Historical artifacts remain
evidence, but they do not override the gates below. Changes to hypotheses,
primary endpoints, or decision gates require a dated amendment in
`EXPERIMENT_LOG_V2.md`; they must not be silently edited after results exist.

**Current branch decision (updated 2026-07-13):** Gate 1 positive superiority failed.
Optimized scalar outperformed WAD on the primary XOR endpoint, while WAD delay
shuffle showed strong within-checkpoint co-adaptation. The active path is now a
causal decomposition / methodology-negative-result programme. Before matched-
distribution confirmation, audit whether WAD was optimization-limited and test
whether its value changes when simultaneous inputs must be routed through time.
Do not open the sealed test set or expand K/N. See
`RESULTS_XOR_DELAY_CONTROL_MATRIX_V1.md`.

Both stages of the optimization audit are complete. Threshold 0.3 was selected by
the preregistered activity/gradient rule (mean activity 10.066 versus target
10), not by accuracy. No Stage-B optimization variant passed; all lowered mean
WAD worst-query relative to baseline. Baseline is frozen. The d0-only output
calibration v1 was stopped as structurally invalid. Version 2 completed and
selected output threshold 0.2, but its temporal scaffold learned only NAND;
XOR/NOR and worst balanced accuracy remained at chance. The spatial pilot is
the active protocol. The temporal matrix remains conditional on a matched
output-training viability preflight.

## 1. Executive decision

The defensible project is not currently “delays enable temporal multiplexing”
or “delays save energy.” The present evidence supports a narrower and more
valuable question:

> Under a neutral observation interface and an explicitly matched resource
> vector, do learnable heterogeneous synaptic delays improve the
> reliability--cost Pareto frontier of sequential temporal computation over
> zero-delay, fixed-delay, and non-learned heterogeneous-delay controls?

The completed XOR pilot shows that a late-window decoder can manufacture an
apparent WAD advantage by censoring useful early d0 activity. With all-time MLP
observation, both d0 and WAD reach 1.0; time-binned decoding also reaches 1.0
but spends far more decoder resources. Therefore:

1. `all_time` is the primary neutral observation interface.
2. `late_window` is an alignment/censoring diagnostic, not the default basis
   for a capacity claim.
3. `time_binned` is an expanded-information upper bound unless its decoder
   storage and operations are explicitly matched.
4. Existing Plan-D experiments should be called **sequential delayed-retention
   / routing tasks**, not proven temporal multiplexing.
5. Burst and rate encodings define different regimes. A conclusion from one
   cannot be transferred to the other without a controlled robustness study.
6. No scalar “energy” claim is admissible without a declared hardware and
   coefficient model. Report the resource vector.

This roadmap is deliberately capable of producing a publishable negative or
conditional result. A positive delay claim is not assumed.

## 2. Claims and falsifiable hypotheses

### 2.1 Primary claim candidate

At matched task, observation interface, latency and declared resources,
learnable heterogeneous delays improve the lower-confidence reliability--cost
frontier relative to all prespecified controls.

This claim is supported only if the advantage survives an optimized scalar
delay, a fixed random heterogeneous delay bank, paired seeds, a sealed test
set, and at least one non-synthetic temporal task. Beating d0 alone is
insufficient.

### 2.2 Hypotheses

- **H0:** WAD has no advantage over the strongest matched non-learned-delay
  control under all-time observation.
- **H1 (resource frontier):** for a target reliability, WAD reduces at least
  one prespecified cost without worsening the others, or raises reliability
  at an otherwise matched resource vector.
- **H2 (interface interaction):** a WAD--d0 gap is larger under late-window
  than all-time observation. Such a result is evidence of alignment, not
  general representational capacity.
- **H3 (delay-specific causality):** learned delay placement matters beyond
  merely adding a delay distribution; shuffling trained delays or replacing
  them with a distribution-matched random bank degrades performance.
- **H4 (scaling):** reliability depends jointly on query count `K`, hidden
  neurons `N`, trial duration `T`, encoding and delay condition; no one-factor
  “maximum K” is meaningful without the other factors.
- **H5 (encoding robustness):** any claimed benefit persists in the
  preregistered burst regime under controlled timing noise. Rate coding is a
  separate secondary regime, not pooled with burst results.

### 2.3 Claims that remain prohibited

- biological temporal multiplexing or biological plausibility;
- hardware energy savings inferred from hidden spike count;
- universal capacity gain from a single late-window comparison;
- “solves K queries” from pooled accuracy alone;
- superiority based on time-binned features without charging decoder cost;
- transfer of NAND/rate conclusions to XOR/burst or vice versa.

## 3. Intended scholarly contribution and positioning

The strongest coherent paper has four linked contributions:

1. **Protocol:** expose observation-window and decoder-resource confounding in
   delay-SNN evaluation.
2. **Controlled evidence:** compare learned delays against delay-specific
   causal controls rather than d0 alone.
3. **Scaling analysis:** estimate reliability as a function of `K`, `N`, `T`
   and a resource vector, with iso-accuracy and Pareto plots.
4. **Mechanism diagnosis:** separate useful temporal alignment from genuine
   retention/routing using interventions, not raster interpretation alone.

The novelty bar must account for recent work showing that temporal Boolean
logic is highly encoding-dependent and can be implemented through delays,
time constants, and bursting; learned-delay methods on temporal speech and
event tasks already exist. Consequently, “we learn delays” is not novel. The
paper must instead earn novelty through controlled causal evaluation,
resource accounting, and a clearly demonstrated failure mode or frontier.

**Likely venue status:** the current evidence is an exploratory/negative-result
study suitable for a workshop or thesis chapter. A serious conference
submission requires Phases 1--5 below. A journal claim requires broader task
coverage, stronger uncertainty analysis, and preferably hardware-calibrated
cost. If delay causality fails, the honest output is a methodology/negative
result paper or engineering report, not a weakened positive claim.

## 4. Canonical task suite

### 4.1 Synthetic mechanism tasks

Use balanced XOR as the primary logic primitive because a same-operation task
avoids conflating routing with operation identity. XNOR may be a symmetry
check. NAND is retained only for historical comparison.

The suite must distinguish:

- simultaneous queries;
- sequential queries with fixed total `T` and shrinking subwindows as `K`
  grows;
- sequential queries with fixed subwindow duration and growing `T`;
- delayed recall/retention with explicitly varied query-to-read delay;
- held-out temporal compositions or schedules, not only i.i.d. examples from
  the same finite support.

These two `K` regimes are essential: if `K` and `T` always grow together, the
study cannot attribute a change to temporal packing rather than additional
compute or latency. Enumerate logical input states where feasible and verify
query/label balance programmatically.

### 4.2 External validity task

Only after the synthetic causal gate passes, select one established temporal
benchmark (preferably SHD or SSC, subject to compute and licensing). The goal
is not leaderboard performance. It is to test whether the controlled finding
survives a non-Boolean temporal input distribution with the same resource and
observation discipline. One well-executed benchmark is preferable to several
under-tuned ones.

## 5. Models and required controls

The minimum delay-condition family is:

1. `d0`: effective zero-delay baseline with semantics unit-tested;
2. optimized shared scalar delay;
3. fixed random heterogeneous delays with the same range/distribution and
   stored-delay footprint as WAD;
4. WAD: learnable heterogeneous delays;
5. post-training shuffled/permuted WAD delays;
6. distribution-matched random replacement of trained WAD delays.

Where useful, add a no-delay recurrent or time-constant baseline to test
whether delays are necessary rather than merely one temporal-memory mechanism.
Do not create a large ANN/RNN zoo; one credible matched temporal baseline is
enough.

Weights, thresholds, delay range, optimizer budget and selection rules must be
matched or explicitly charged. Parameter equality alone is not resource
equality. Failed optimization is not evidence of representational incapacity;
training curves, convergence and a reasonable tuning budget must be reported.

## 6. Readout and encoding policy

### 6.1 Readout interfaces

- **Primary:** all-time observation with a linear probe where the task is not
  saturated. This asks whether temporally aggregated network activity is
  separable without giving the decoder explicit time bins.
- **Secondary:** a small MLP, with its parameters and operations charged. Use
  it to distinguish representation failure from linear-readout failure.
- **Diagnostic:** late-window, with window start/end displayed. It measures
  final-time alignment and censoring.
- **Upper bound:** time-binned decoding. It is not a matched comparator unless
  feature dimension and decoder resources are controlled.
- **Future spike output:** use an opponent pair (class-0/class-1 spike counts)
  or a valid count/probability loss. A single nonnegative spike count passed as
  a BCE logit is invalid because it cannot express negative evidence.

Avoid saturated settings: a readout at 1.0 for both conditions cannot estimate
an advantage. Difficulty calibration uses validation only and precedes the
confirmatory comparison.

### 6.2 Encoding

Deterministic burst encoding is the canonical mechanism-visible regime. Add
timing jitter, event deletion/insertion, and amplitude/event-count controls in
the robustness phase. Jitter must preserve the declared event count unless
event-count corruption is itself the factor. Rate encoding is a secondary
replication regime because it changes both information representation and
activity cost.

## 7. Reliability and cost mathematics

### 7.1 Reliability endpoints

Primary endpoints are worst-query accuracy and exact-trial accuracy. Balanced
accuracy is a guard against class imbalance; pooled accuracy is descriptive.
Report every seed and confidence intervals, not only means.

For method `m`, define the empirical feasible set over resource settings
`r=(K,N,T,...)`. For target reliability `a`, report

```text
C_min^m(a, K, regime) = min C(r)
subject to lower_CI(A_worst^m(r)) >= a
```

There is no universal scalar cost. The canonical vector is:

```text
R = [latency,
     trainable/stored parameters,
     delay-buffer and state memory,
     neuron updates,
     dense simulator synapse MACs,
     delay reads/interpolation operations,
     decoder MACs,
     input/hidden/output spikes,
     event-driven synaptic fan-out events]
```

Dense simulator work and prospective event-driven work are separate models.
If a scalar cost is later used, coefficients and target hardware must be
declared before evaluation. Also report a normalized scaling ratio, for
example `rho_C(K)=C(K)/(K*C(1))`, without interpreting it as energy.

### 7.2 Factor analysis

Fit a prespecified hierarchical/binomial or logistic response model, supported
by nonparametric plots:

```text
logit P(correct) = beta_0 + beta_K*K + beta_N*log(N) + beta_T*log(T)
                 + beta_m*method + beta_o*observation
                 + prespecified two-way interactions + seed/task effects
```

Do not infer smooth scaling from too few grid points. The main visual outputs
are:

- iso-accuracy contours over `(N,T)` for fixed `K` and method;
- minimum `N` and resource vector needed to clear a reliability threshold;
- worst-query and exact-trial curves versus `K` with uncertainty;
- Pareto fronts, with dominated points visibly marked;
- method-by-observation and method-by-encoding interaction plots.

The old `acc=90%` line is not a scientific baseline. It may be one operational
threshold, but conclusions must be stable across several preregistered targets
(for example 0.80, 0.90 and 0.95) and use confidence bounds.

## 8. Phased experimental programme and decision gates

### Phase 0 — trustworthy infrastructure (complete)

Historical reevaluation, strict metrics, resource ledger, readout interfaces,
run classification and the 36-cell XOR pilot are complete. Preserve these as
exploratory evidence.

### Phase 1 — difficulty calibration and causal delay controls

**1A. Validation-only calibration.** Use all-time XOR with `K` varied
independently of `N` and `T`, initially WAD/d0, linear readout, three seeds.
Choose a regime in which neither method is at floor or ceiling and where seed
variance is estimable. No test-set claim may be made from this search.

**1B. Preregistered causal matrix.** At the locked regime compare d0, shared
scalar, fixed heterogeneous, and WAD; use paired seeds and all-time as primary.
Late-window is a prespecified interaction diagnostic. Linear is primary if it
is non-saturated; MLP is a secondary representation check.

**Gate 1:** proceed to a positive learned-delay programme only if WAD improves
over the strongest non-learned control by a practically meaningful,
uncertainty-supported amount under all-time observation. If it beats d0 only,
the delay-learning claim fails. If an advantage exists only in late-window,
pivot to an observation-alignment/failure-mode paper.

**1C. WAD optimization audit (complete; rescue failed).** At K=3,N=35, burst XOR,
all-time linear, use validation only. Stage A screens threshold
`{0.2,0.3,0.5}` using firing/gradient viability with matched scalar controls.
Stage B tests a compact, preregistered set of `d_max`, effective delay learning
rate, initialization, and joint/warm-up/alternating schedules only at viable
thresholds. Record delay and weight gradient norms, effective delay movement,
saturation, spike activity, and convergence. The audit may diagnose an
optimization failure but cannot retroactively turn the completed matrix into a
positive result.

**Optimization gate:** retain a WAD configuration only if it is non-silent,
non-saturated, moves delays materially from initialization, has mean paired
worst-query gain >=.03 over the fresh Stage-B WAD baseline with at least 2/3
individual gains >=.03, and has mean WAD-minus-matched-scalar >=-.01. Scalar
receives the identical tuning budget. Otherwise retain baseline and freeze the
negative optimization-rescue conclusion.

**1D. Simultaneous-input pilots.** Run two explicitly separated tasks:

- **Spatial control:** simultaneous query inputs with separate output pairs;
  this measures spatial parallel multi-task computation and must not be called
  temporal multiplexing.
- **True temporal routing:** simultaneous query inputs, a shared hidden layer,
  and one shared opponent output pair reused across ordered output windows.
  The decision logit in window k is `count(class1)-count(class0)`. This is the
  task-native test of heterogeneous delay routing.

For both pilots, retain a per-window shared linear probe as the primary
representation diagnostic, a small shared MLP as secondary, and the opponent-
pair spiking output as the task-native endpoint. All-time count is an ablation;
time-binned decoding is a charged information upper bound. Dedicated output
heads are prohibited as evidence of temporal multiplexing.

**Simultaneous gate:** a temporal-routing contribution requires WAD to beat
scalar and matched fixed-delay controls on worst-window and exact-trial
reliability with the shared opponent output, while the spatial control shows
whether the gain is specifically temporal rather than generic model capacity.
At least 4/5 paired seeds must agree before escalation.

### Phase 2 — `K,N,T` response surface and resource frontier

Use a staged design rather than an uncontrolled exhaustive sweep. Separate
fixed-`T` packing from fixed-subwindow growing-`T`. Sample enough `N` and `T`
levels to estimate monotonicity, interactions, and threshold crossings.
Exploratory surface fitting uses at least five paired seeds; promising frontier
points are rerun confirmatorily with at least ten paired seeds.

**Gate 2:** a scaling claim requires stable ordering across adjacent settings,
non-overreliance on a single threshold, and a non-dominated resource gain. If
WAD merely exchanges latency for neurons or decoder work, report that tradeoff
rather than “efficiency.”

### Phase 3 — causal mechanism tests

At representative success and failure points perform:

- trained-delay shuffle and distribution-matched replacement;
- observation-window sweep and temporal translation of the same window;
- arrival-time and readout-feature attribution;
- freeze-delay/retrain-weight and freeze-weight/perturb-delay interventions;
- hidden-neuron reuse/participation statistics across query subwindows;
- threshold and delay-range sensitivity.

**Gate 3:** raster plots and learned-delay histograms alone cannot establish a
mechanism. A mechanism claim requires a targeted intervention whose predicted
performance change occurs and is replicated across seeds.

### Phase 4 — encoding and perturbation robustness

Repeat a compact set of frontier points under burst timing jitter, event-count
noise, and the separately defined rate regime. Keep information budget and
activity changes visible. Test held-out schedules or temporal compositions.

**Gate 4:** if the effect disappears under small realistic timing changes,
state it as a brittle deterministic-timing phenomenon. Do not average regimes
to hide the interaction.

### Phase 5 — one external temporal benchmark

Preregister dataset split, preprocessing, tuning budget, model sizes,
baselines, and primary metric. Use the same causal delay controls and resource
ledger where technically meaningful. Test once on the sealed test set.

**Gate 5:** a general method paper requires survival of this phase. Without it,
the work remains a synthetic mechanism/protocol paper, which can still be
publishable if framed accurately.

### Phase 6 — confirmation and manuscript

Freeze code/configs, rerun all headline cells from clean commands, generate
tables from machine-readable results, and have a second pass reproduce at
least one central figure. The abstract must be written from the claims ledger,
not from the most attractive plot.

## 9. Statistical and anti-cherry-picking protocol

- Use paired seeds across delay conditions and publish individual seed points.
- Use at least five seeds for exploratory comparisons and ten for headline
  confirmation; increase if pilot variance implies inadequate precision.
- Select architecture/hyperparameters on validation data only. A test set is
  opened once per frozen protocol, not once per cell revision.
- Declare one primary endpoint and comparison family per phase. Correct or
  transparently qualify multiple comparisons.
- Report paired effect sizes and interval estimates. Do not treat overlapping
  standard deviations as a hypothesis test or tiny seed counts as Gaussian.
- Define handling of crashes, non-convergence and missing cells before runs;
  never silently exclude failures.
- Store all trial-level predictions needed to recompute worst-query,
  exact-trial, balance and uncertainty.
- Distinguish exploratory p-values/intervals from confirmatory inference.

## 10. Diagnostic-panel standard

The diagnostic panel is necessary but illustrative. Every canonical panel
must state protocol ID, seed, sample ID, condition, `K,N,T`, encoding, delay
range, observation mode and window boundaries. It should contain:

1. input and hidden spike raster with subwindow and observation overlays;
2. membrane/readout traces, not spikes alone;
3. learned-delay or effective-arrival distribution with d0/random controls;
4. query-wise predictions and errors;
5. mechanism plot tied to a defined statistic (arrival alignment,
   participation/reuse, or censoring), not an aesthetic schematic;
6. the resource vector for the shown run.

Illustrative samples must be selected by an outcome-independent rule before
inspection (fixed seed/sample IDs or a preregistered median/failure rule).
Headline claims require aggregate versions across all seeds: heatmaps,
distributions, confidence bands and intervention effects. A visually striking
raster is never causal evidence.

## 11. Reproducible workflow for future agents

For every new phase:

1. read `AGENTS.md`, this roadmap, scope, protocol, metrics/cost, readout,
   resource and claims ledgers;
2. inspect git status and preserve unrelated user work;
3. assign a protocol ID and preregister question, hypotheses, factors, seeds,
   endpoints, resource vector, stopping/exclusion rules and gates;
4. add schema/tests before a large sweep; run a smoke cell and verify strict
   JSON, trial predictions, ledger fields and deterministic reload;
5. run immutable cells into a new versioned directory; never edit historical
   results or checkpoints in place;
6. report only launch, material failure and completion for long sweeps;
7. validate cell completeness and aggregate from raw machine-readable files;
8. generate diagnostics using a frozen selection manifest;
9. update `EXPERIMENT_LOG_V2.md`, `CLAIMS_LEDGER.md` and result documentation,
   including null results and alternative explanations;
10. classify artifacts as `canonical`, `exploratory`, `invalid`, or
    `archived` only after the declared gate is evaluated.

Each result manifest should include code commit/diff status, environment,
device, command, seed, config hash, dataset/split hash, checkpoint hash,
start/end time and completion status.

## 12. Timeline and deliverables

### First week — causal foundation

- implement and unit-test shared-scalar, fixed-heterogeneous and delay-shuffle
  controls plus effective-delay semantics;
- create a sealed balanced XOR dataset/split generator and prediction schema;
- run validation-only difficulty calibration, explicitly avoiding saturation;
- preregister (but do not yet expand) the Phase-1B causal matrix;
- upgrade the diagnostic panel with observation masks, membrane/readout traces
  and a frozen sample-selection manifest.

**Deliverable:** one locked, executable Phase-1B protocol and a calibration
report; not a paper claim.

### Weeks 2--4 — decisive controlled evidence

- run Phase 1B with complete paired seeds;
- compute paired uncertainty, delay-condition × observation interaction and
  full resource vectors;
- execute shuffle/replacement intervention at the strongest candidate point;
- make the Gate-1 decision and update the paper framing immediately.

**Deliverable:** either credible evidence that learned delay placement matters,
or a documented pivot to the observation-confound/negative-result story.

### Months 1--2 — scaling and mechanism

- run the staged `K,N,T` response-surface study;
- estimate iso-accuracy contours and Pareto fronts with confirmatory reruns;
- complete mechanism interventions and aggregate diagnostic figures;
- run compact burst-noise/rate robustness checks;
- draft Methods, protocol contribution and synthetic results while results are
  generated from scripts.

**Deliverable:** a complete synthetic mechanism/protocol manuscript core.

### Months 2--3, conditional — external validity and submission

- run one external temporal benchmark only if Gate 1 supports it or the chosen
  failure-mode paper specifically requires it;
- freeze headline configurations, rerun from clean commands, complete
  reproducibility package and manuscript;
- select venue based on the evidence actually obtained.

## 13. Publication decision tree

- **WAD beats all matched delay controls under all-time, survives causal
  interventions, scaling and an external task:** conference method/empirical
  paper; journal extension possible with broader cost calibration.
- **WAD advantage exists only under late-window or disappears when decoder
  information is matched:** methodology paper on observation-induced delay
  advantages, with the XOR pilot as the motivating result.
- **WAD matches fixed heterogeneous delays but both beat d0:** paper about
  temporal delay structure, not delay learning; remove learning novelty.
- **No robust delay benefit after controls:** negative result/engineering
  report or thesis chapter; publish protocol and failure analysis if breadth
  and rigor are sufficient.
- **Results remain unstable or optimization-sensitive:** project is not ready
  for a claim; prioritize implementation/optimization audit over more sweeps.

## 14. Definition of “publication-ready”

The work is not publication-ready until all applicable items hold:

- a single falsifiable primary question and locked claim vocabulary;
- neutral observation plus delay-specific controls;
- non-saturated tasks and independently varied `K,N,T`;
- worst-query and exact-trial results with paired uncertainty;
- complete resource vector and no unsupported energy language;
- causal mechanism intervention, not just diagnostic visualization;
- robustness to seeds and timing perturbations;
- one external task for a general-method claim;
- immutable configs/results, full manifests and clean rerun commands;
- claims ledger consistent with every abstract/conclusion sentence.

## 15. Immediate next protocols

Execute in this order:

1. `simultaneous_spatial_control_pilot_v1` is complete: primary linear results
   are negative for WAD superiority; MLP is saturated; the opponent exact-trial
   signal is exploratory and not resource-superior;
2. temporal preflight v1 formally failed its hidden-emission gate, but the
   checkpoint audit invalidated that quantity as an arrival proxy: WAD has
   third-window output current/arrivals and gradients, while the pooled BCE is
   misaligned with balanced evaluation;
3. preflight v2 is complete and passes every mechanism-valid gate in both held-
   out seeds, but WAD worst-window balanced remains `.5` because fixed
   NOR/window 2 is not learned;
4. the counterbalanced opponent-output protocol is complete and primary-
   negative: no condition passes the routing floor. WAD has the best average
   balanced/exact performance, but late XOR and NOR remain at chance; fixed-
   full instead favours late over early positions;
5. the researcher selected a versioned spatial-versus-temporal Pareto pivot.
   `spatial_vs_temporal_pareto_phase0` first calibrates a K=1 repeated-XOR
   baseline with equal-event binary one-hot micro-bursts and class-opponent
   target spikes. Its 15-cell Stage A formally failed: no width passes every
   gate in all three seeds, so no baseline width was selected and the K=2
   block-diagonal spatial versus shared temporal matrix remains locked;
6. keep mixed operations, K>2, full K/N/T response surfaces and test access
   locked until Stage B distinguishes neuron-area compression from latency,
   dense-synapse compute and ordinary shared-representation alternatives.
7. the successful/silent/collision checkpoint audit is complete. It verifies
   all-d0 delays, separates hidden-silent from output-conversion failures, and
   finds early wrong spikes at t=10 in every collision; neither accuracy-best
   checkpoints nor a universal threshold sweep rescues the interface;
8. before any Stage-B attempt, preregister a versioned seed-robust interface-
   stability calibration separating exact truth-table batching, global one-
   target/no-early-spike supervision, and micro-burst structure. Do not
   substitute mean accuracy or retrospective best epochs for the failed rule.
9. the separate supervisor-facing MLP scaffold is complete (160/160 cells).
   Only independent spatial d0 and the fixed query-scheduled oracle pass the
   robust `.90` worst-balanced and `.90` exact-trial gate in both seeds, at
   `T=18,h=24`. The fixed oracle halves total hidden neurons and hidden updates,
   but leaves dense MACs, measured synaptic events, delay-value storage and
   decoder MACs unchanged and requires five times the delay-buffer storage.
   WAD remains at `.5` worst-balanced in all 32 cells; its learned delays do not
   separate by query and produce zero second-window hidden activity. This is a
   narrow positive fixed-schedule hidden-area/update result and a negative WAD
   result, not evidence for learned routing, energy/compute compression, a
   general `T x h` law, or a repaired spike-output interface. See
   `RESULTS_SPATIAL_VS_TEMPORAL_PARETO_MLP_SCAFFOLD_V2.md`. V1 remains excluded.
10. before another surface experiment, choose one scientific fork and
    preregister it: (a) if learnable delays are the contribution, test one small
    query-coverage/symmetry-breaking bridge at `T=18,h=24` against d0 and the
    fixed oracle, with delay coverage and second-window activity as mechanism
    gates; (b) if resource architecture is the contribution, return to the
    task-native spiking-output interface only at the feasible spatial/oracle
    point. Do not claim the exploratory MLP oracle result as publication-ready
    until the selected endpoint survives more seeds and a held-out test.
11. the deterministic `delay_parameter_recovery_level0a_v1` unit experiment is
    complete (75/75). The current `.001/200-step` recipe fails the declared
    low-init to delay-5 recovery (`.9536 -> 1.1389`, error `3.8611`), while the
    preregistered `.05` arm recovers all 15 interior target/initialization
    combinations. This rejects a hard scalar implementation bug but shows the
    current update budget is inadequate even under direct supervision. It does
    not authorize `.05` in XOR. Next preregister Level 0B: arrival-time credit
    through the production buffer, followed separately by one LIF neuron.
12. `delay_temporal_credit_level0b_v1` is complete (180/180) and fails its
    preregistered strict gate. Buffer-current centroid recovers 15/15
    target/initialization pairs, proving that the production circular buffer can
    transport a global timing gradient. Buffer filtered trace recovers 5/15,
    hard-LIF centroid 3/15 and hard-LIF filtered trace 10/15. Initial-gradient
    auditing localizes the failure: causal filtered loss has the wrong direction
    in 5/13 misaligned pairs, while hard-spike centroid is exactly flat in 8/13.
    XOR, per-query/full WAD and any LR transfer remain locked. The next allowed
    gate is a versioned Level 0C symmetric/global timing-objective audit at the
    same one-event scale; do not expand task complexity first.
13. `delay_soft_trace_credit_level0c_v1` is complete (360/360) and passes. The
    selected production arm is sigmoid + soft centroid + Adam `.05`: one common
    LR recovers 30/30 current/membrane cells, with 13/13 correct nonzero initial
    directions per path and no hard spikes. Causal filtered loss remains 5/15
    on buffer current but becomes 15/15 after passive membrane smoothing,
    proving a loss-by-state interaction. Symmetric kernel alignment is 15/15 on
    current and 0/15 on membrane because the causal tail/truncation shifts its
    optimum. Direct centroid is 11/15 within the unmatched parameter-coordinate
    budget and does not implicate sigmoid. Level 0D is authorized; it must keep
    a hard-spike endpoint while using a prespecified continuous-state centroid
    auxiliary. XOR and Pareto work remain locked.
14. `delay_hard_output_soft_credit_level0d_v1` is complete (135/135) and
    passes. Hard filtered-spike loss alone remains 10/15 with five wrong
    initial directions. Current centroid alone is 15/15, and the combined
    hard-plus-current conditions at `lambda=.1` and `1` are both 15/15 with
    13/13 correct nonzero directions; the locked rule selects `.1`. The
    pre-reset hypothesis fails: soft-only is 3/15 with eight zero-gradient
    pairs, and every combined pre-reset arm remains 10/15. This establishes a
    narrow production hard-spike/continuous-current credit bridge. It permits
    preregistration of Level 1A K=1 XOR calibration only; pairwise WAD,
    per-neuron tying, K scaling and Pareto work remain locked.
15. `xor_task_bridge_level1a_v1` Stage I is complete (90/90) and passes. It uses
    the exhaustive K=1 XOR truth table, a
    `4 -> 16 -> 2` hard-spiking network, equal two-event one-hot inputs, exact
    opponent-spike targets and one global shared input-to-hidden delay. Stage I
    required both fixed d0 and fixed delay-4 schedules to pass the exact output
    interface in all five seeds while selecting the voltage-envelope scale and
    weight learning rate. The unique passing candidate is `eta=0,lr_w=.01`
    with 10/10 exact-interface cells. Stage II is also complete (85/85) and
    Level 1A passes. Task-only delay learning is 0/10 at `lr_d=.01` and 1/10 at
    `.05`; initial task-gradient direction is correct in only 5/10 cells. Every
    tested positive arrival scaffold passes 10/10, and the locked rule selects
    `lambda=.01,lr_d=.01`. Its final delay error is at most `.002628` step and
    it retains exact hard-spike XOR output in all ten cells. The d0/wrong-time
    control fails 0/5 with zero correct-target-time rate. This supports a narrow
    scaffold-assisted joint task/delay bridge, not task-derived timing
    discovery. Level 1B K=1 granularity and micro-burst preregistration is now
    authorized; K>1 and Pareto surfaces remain locked.
16. `xor_delay_granularity_level1b_v1` Stage A is complete (60/60). Stage A is
    a fixed K=1 matrix:
    global, per-hidden-neuron and per-synapse input-delay coordinates crossed
    with task-only/scaffold loss, two matched homogeneous initial directions
    and five new seeds. Every coordinate—not merely the mean—must have the
    correct nonzero initial total-gradient direction and finish within `.1`
    step of delay 4 while retaining the exact hard-spike XOR interface. A
    10/10 held-out global-scaffold replication is required before ten fixed
    micro-burst controls may run; the 60 learned micro-burst cells remain
    locked unless fixed delay 4 passes 5/5. The micro-burst target remains one
    exact opponent spike at `t=15`, so early or duplicate spikes fail. Runtime
    NPZ, a granularity-aware 12-panel diagnostic and a resource ledger that
    separates 1/16/64 trainable delay values from 64 physical synapses are
    mandatory. The global scaffold passes 10/10 new-seed cells with maximum
    delay error `.001864` step, so fixed micro-burst controls are authorized.
    Per-hidden and per-synapse scaffold candidates pass only 2/10 and 0/10;
    the higher-dimensional extension fails. Per-hidden retains the exact
    output interface in 10/10, localizing its failure to coordinate-wise delay
    recovery. Because the registered mean scaffold divides per-coordinate
    arrival gradients by `P`, this result rejects scaling of the frozen recipe,
    not the intrinsic trainability of higher-dimensional delays. The ten fixed
    micro-burst controls are also complete: delay 4 passes 5/5 with all 20
    outputs exactly at `t=15`; d0 passes 0/5, with all 27 emitted spikes at
    `t=11` and target-time rate zero. Thus the harder interface is feasible and
    timing-specific. The frozen 60 learned micro-burst cells are mechanically
    authorized but unrun. Changing their lambda/LR would violate the protocol;
    a dimension-aware rescue requires a new identifier and new seeds. This
    protocol cannot support heterogeneous routing, K>1, WAD superiority,
    compression or a Pareto law.
17. `xor_delay_granularity_rescue_level1br_v1` R1 is complete (50/50). It preserves the
    single-event K=1 hard-spike interface and treats the Level-1B high-
    dimensional failure as a normalization hypothesis. R1 fixes
    `lambda_P=.01P`, giving effective lambdas `.01/.16/.64` for 1/16/64 delay
    coordinates, and crosses a global anchor plus scaled/unscaled per-hidden
    and per-synapse conditions with two initial directions and five new seeds
    (50 cells). This matches the analytic weighted arrival gradient per
    coordinate; it does not constitute a tuned sweep. R2 is allowed only when
    all R1 coordinate directions are correct but recovery remains incomplete;
    it compares LR-only, budget-only and combined interventions on separate
    calibration seeds. Any provisional recipe must pass R3 on five sealed
    seeds, with a 10/10 global anchor and 10/10 higher-dimensional candidate.
    Per-hidden has predeclared priority if both pass because it uses fewer
    delay parameters. Stronger total oracle supervision as `P` grows must be
    disclosed; success is not autonomous routing. Six smoke cells validate
    implementation only. In formal R1, global passes 10/10; unscaled
    per-hidden/per-synapse pass 1/10 and 0/10; dimension-matched versions pass
    10/10 and 10/10 with every coordinate correctly directed and recovered,
    and without clipping. This supports the normalization explanation
    narrowly. R2 was skipped by rule. Sealed R3 is complete (30/30): global,
    per-hidden and per-synapse each pass 10/10 with no clipping. Both
    higher-dimensional recipes confirm and the preregistered lower-parameter
    priority selects per-hidden-neuron. Level 1B-R is complete, while
    micro-burst, K>1 and routing claims remain outside its authorization.
18. `xor_delay_granularity_rescue_microburst_v1` is preregistered and
    implementation-validated as the independent dimension-aware Stage-B
    rescue. It does not modify the frozen original 60-cell learned matrix. The
    new protocol uses consecutive events at steps 8 and 9, exact output at
    step 15, fresh formal seeds `{2333,2351,2371,2389,2411}`, and unchanged
    `.01` weight/delay LRs with 500 updates. Five fixed-d4 cells must pass
    before 40 learned cells materialize. Learned candidates are global
    `lambda=.01`, selected per-hidden `lambda=.16`, secondary per-synapse
    `lambda=.64`, and diagnostic per-hidden task-only. Primary success requires
    fixed d4 5/5 plus global and per-hidden 10/10 under the full exact-interface
    and coordinate-delay gates. Ten implementation smoke cells are invalid;
    proposed seed `2309` was consumed by smoke and replaced before formal
    launch. The five formal fixed-d4 controls are complete and pass 5/5 with
    exact step-15 outputs, so the complete 40-cell learned matrix is now
    launch-ready. The complete learned matrix has now run. Global `.01`,
    per-hidden `.16`, and per-synapse `.64` each pass 10/10; per-hidden
    task-only passes 0/10. The result establishes an oracle-supervised
    micro-burst bridge but simultaneously rejects reliable task-only global
    discovery under this loss and optimizer. The next authorized experiment is
    a fresh scaffold-withdrawal protocol that separates (i) retention after
    teacher removal from (ii) local restoration after symmetric delay
    perturbations. It does not authorize K>1 or Pareto work.
19. `xor_task_derived_timing_withdrawal_v1` is the next preregistered causal
    audit. Five fresh-seed per-hidden `.16` foundation cells must first recreate
    exact teacher-built checkpoints. Ten no-update controls then set every
    independent delay to either 3 or 5 steps; both perturbation directions must
    destroy the exact interface before recovery arms unlock. The primary arm
    freezes both weight matrices, removes the arrival teacher (`lambda=0`), and
    asks whether task loss alone restores all 16 delays to 4 and the exact
    spike train. Oracle-delay-only is the positive optimizer control;
    task-joint and weight-only are compensation controls; unperturbed
    task-delay-only retention is reported separately. A positive result means
    local task-derived restoration inside a teacher-created basin, never
    from-scratch discovery, heterogeneous routing, K>1, or a Pareto advantage.
    W0 is complete: all five fresh foundations pass the exact interface and
    coordinate-delay gates, with maximum error `.002032` step and complete
    artifacts. This opens only the ten no-update d3/d5 damage controls in W1;
    all 45 W2 cells remain mechanically locked.
    W1 is now complete and passes 10/10: d3 shifts every output to step 14 and
    d5 shifts every output to step 16 while preserving classification accuracy
    one. W2 is authorized. A disclosed no-training gradient audit predicts a
    major directional asymmetry (`.85` target-directed coordinates at d3 versus
    `.1375` at d5); the frozen 45-cell matrix must nevertheless run unchanged.
    W2 is complete and the primary protocol fails. Retention passes 5/5 and
    oracle-delay-only passes 10/10, but task-delay-only passes 0/10. Functional
    recovery is asymmetric (d3 3/5 exact, d5 0/5) and no task-only cell
    recovers all 16 delays. Joint training reaches the exact interface in 8/10
    but never the oracle schedule, demonstrating compensation/non-identifiability.
    A post-result audit localizes the late-side failure to the integer boundary:
    the globally tied mean gradient flips from `+.00408` at `d=4.999` to
    `-.00356` at `d=5.000` because the current interpolation backward selects
    the right-hand interval. K>1 remains locked. The next admissible work is a
    small preregistered integer-boundary temporal-credit preflight, not an LR,
    threshold, budget, or seed sweep.
20. `xor_integer_boundary_credit_preflight_v1` is complete. P0 evaluates 100
    zero-update probes over five W0 foundations, d3/d5, global/per-hidden
    tying, and five backward estimators. All forward records are bitwise
    identical. Gaussian STE sigma 1.00 alone passes the registered aggregate
    direction rule (global and per-hidden mean 10/10); it raises d5
    coordinate-direction fraction from `.1375` to `.4875`, still below a
    coordinate-majority guarantee. Conditional P1 then freezes weights and
    runs ten per-hidden task-only recovery cells. It fails: exact interface
    2/10 (both d3), full 16-delay recovery 0/10, and d5 exact interface 0/5.
    Exact cells reach zero task loss at non-oracle schedules, while d5 delays
    split toward the bounds. The integer derivative was a real but incomplete
    cause. K>1 remains locked. The next admissible causal discriminator is a
    new preregistered globally tied one-delay recovery anchor using the selected
    backward. If that succeeds, compare target-free consensus/low-rank delay
    parameterizations; if it fails, redesign continuous task timing credit.
    Do not tune LR, threshold, update budget, sigma, or seeds using P1.

All are validation-only pilots. No test split is opened. The optimization audit
must not use the simultaneous pilots as an additional hyperparameter search;
the WAD configuration is frozen before those tasks run.

## 15A. Deadline-driven mixed-operation surface preview (non-confirmatory)

At the researcher's explicit request, `mixedop_spatial_temporal_surface_preview_v1`
is a separate progress-visualization branch and an exception to the causal
programme's normal K>1 sequencing. It does not alter the publication gates.
The frozen single-seed grid compares an independent spatial d0 architecture, a
hand-scheduled temporal oracle and the current WAD diagnostic at K=5/8 over
four total hidden widths and three latencies. All learning rates are `.01`.
Six invalid technical cells passed the preregistered stability gate without
using accuracy, authorizing the unchanged 72-cell matrix.

The researcher subsequently introduced a K=5-first decision gate. All 36 K=5
cells are complete; K=8 is on hold. K=5 does not produce an informative factor
surface: spatial and fixed oracle are 1.0 everywhere, while WAD is .5
everywhere on worst-query balanced accuracy. Width/T marginal ranges are zero
and SS fractions are undefined. WAD has activity in essentially only its first
two output windows and does not learn the query-spaced delay schedule. At the
smallest tested N/T, spatial and oracle are both perfect, while oracle carries
5x dense MAC/event cost and 17x delay-buffer storage. This branch therefore
does not currently supply a cost law or learned routing result. Read
`RESULTS_MIXEDOP_SPATIAL_TEMPORAL_SURFACE_PREVIEW_V1_K5.md`; do not resume K=8
without an explicit research decision.

The only potentially admissible conclusion is a descriptive statement about
width-versus-latency sensitivity in this exact workload, and only under the
registered marginal-range and balanced-grid-SS thresholds. The surface cannot
establish temporal credit, internal-schedule identification, WAD superiority,
output-spike timing, general mixed-operation routing, or hardware energy. A
positive oracle surface is architectural feasibility under experimenter timing;
a positive WAD surface remains compatible with shared representation and
alignment. These results must be displayed alongside the failed causal timing
programme, not used to erase it.

## 16. Decisions that still require the researcher

Before Phase 2 is locked, record:

- intended submission window and acceptable venue class;
- compute budget and maximum confirmatory seed count;
- whether latency `T` is a constrained resource or merely an axis of study;
- the target deployment/hardware model, if any scalar cost is desired;
- which external dataset is scientifically relevant and legally available;
- whether the thesis contribution prioritizes a learned-delay method or an
  evaluation/methodology result when evidence favours the latter.

These decisions affect scope and cost modelling, but they do not block the
immediate calibration and causal-control work.

## 17. Gated rate-code rescue calibration

`mixedop_rate_wad_surface_calibration_v1` is a versioned exploratory response
to the K=5 WAD floor. It keeps the 4K value-channel interface and changes the
micro-burst into stochastic one-hot rate trains. This tests whether distributed
input events improve optimization and output-window coverage, but it is not
event matched: 8.2 expected events/query replace four. Resource reporting must
therefore accompany every accuracy comparison.

Only four expanded-surface corners and the center are evaluated first, each
with a fixed-oracle feasibility control and WAD. The 30-cell WAD surface stays
locked unless the preregistered gradient, oracle and WAD performance gates all
pass. Even then, a positive surface is descriptive unless the separate routing
gate passes. Failure stops this branch; it does not authorize rate, threshold,
LR or grid tuning on the same validation workload. Read
`MIXEDOP_RATE_WAD_SURFACE_CALIBRATION_V1.md` before execution.

## 18. Rate-alignment repair outcome

The versioned late-rate repair confirms that analytical packet alignment and
event redundancy can make the fixed K=5 oracle perfect. Stage A selects the
near-deterministic event8/N120/w4 interface; event4 never reaches the exact-
trial gate. Stage B then rejects both autonomous and `.10` routing-assisted
five-parameter delay learning: oracle passes 3/3, while both learned arms have
worst-query `.50` in all seeds and fail the internal schedule.

Assisted gradients are large, routing loss falls and later-window coverage
partially grows, so this is not gradient disappearance. The remaining design
risks are conflict between task and routing credit, a late target at the
sigmoid support boundary, and insufficient schedule identifiability. Scaffold
withdrawal, per-synapse WAD and full surfaces remain closed. A further repair,
if pursued, must be a separately preregistered method test with interior target
support and explicit gradient-scale calibration; it cannot be an undisclosed
increase in routing weight, LR or updates.

The pilot is now complete and fails. Its technical gradient gate passes, but
all five oracle controls fail the `.85` feasibility floor and WAD remains at
`.50` worst-query balanced accuracy throughout. Rate coding adds limited third-
window activity and approximately doubles measured input-to-hidden events at
the matched center, without improving the primary endpoint; the last two
windows remain silent and learned query delays overlap. The full surface is
therefore closed. This result shifts the bottleneck from literal gradient
absence to credit quality/schedule identifiability under the shared model.

### 17A. Versioned rate-alignment repair

The failed rate pilot exposed an inherited burst-schedule mismatch: a ten-step
rate packet with `d_q=q*w` was not placed inside its output window. The new
`mixedop_rate_alignment_repair_v1` protocol tests a four-step late packet and
the analytically aligned schedule `d_q=3+q*w`. Fixed-oracle feasibility is a
hard prerequisite. Only if one event/N/w candidate passes three fresh seeds may
the project test a five-parameter per-query delay model. Task-only and explicit
routing-supervision arms remain distinct, with separate function and internal-
schedule gates. Per-synapse WAD, full surfaces and Pareto claims remain locked.

### 18A. Supervised temporal-interface repair outcome

The v2 formal recovery separates functional success from schedule recovery.
Its decoupled arrival-mass arm is perfect on classification but fails the
one-step schedule gate in all seeds because that objective drives q0 earlier
than the registered oracle. V3 preregisters the diagnosis and changes only the
explicit timing objective to output-window-centroid Huber. On three new seeds,
the matched mass-CE comparator again fails 0/3, while centroid Huber passes
function, activity and schedule jointly 3/3.

This is a valid supervised-interface repair, not learned temporal multiplexing:
delay parameters receive a query-to-window teaching signal. The next gate is a
new scaffold-withdrawal protocol (reduce/remove timing supervision while
preserving the same seeds policy and joint endpoint). K=8, per-synapse WAD,
sealed test and resource-Pareto surfaces remain closed until task-derived
routing survives withdrawal.

### 18B. Registered scaffold-withdrawal gate

`mixedop_temporal_wad_scaffold_withdrawal_v1` is the only authorized path from
the supervised V3 result toward task-derived timing. It first replicates the
scaffold on new seeds, then distinguishes frozen retention, abrupt task-only
retention, fixed auxiliary annealing, and local restoration after two declared
delay perturbations. Final checkpoints, not intermediate best checkpoints, are
the primary endpoints.

Success in W1 is only warm-start maintenance. A narrow local-identifiability
claim requires task-only W2 to restore both perturbations in all seeds while
meeting function, activity and schedule gates. Even that would not establish
from-scratch autonomous routing. Only implementation smoke is currently
authorized; all scientific stages remain locked behind their predecessor.

Smoke and W0 have now passed their gates. W0 reproduces the explicitly
supervised scaffold in all three new seeds and freezes accuracy-first source
checkpoints at updates 380/320/400. The 12-cell W1 matrix is authorized without
changing LR, update budget, source policy or endpoints. W2 remains conditional
on final-checkpoint retention in all seeds; no Pareto or K=8 work is unlocked.

W1 is now complete. Abrupt removal retains near-perfect classification but
fails the coordinate-schedule gate in two of three seeds, whereas the frozen
annealing curriculum passes all three after its final 100 task-only updates.
This authorizes the exact W2 local-restoration matrix, but establishes only
curriculum-dependent warm-start retention. W2 must start from untouched W0
sources; no K=8, Pareto, per-synapse or sealed-test work is unlocked.

W2 is now complete and rejects local task-only schedule restoration. The
centroid positive control restores both declared perturbations in all six
seed/perturbation cells, while task-only passes 0/6 despite near-perfect
classification. This establishes an identifiability problem, not a generic
optimizer or forward-interface failure. The withdrawal branch is closed; the
sealed test remains locked.

A future event8 N/T surface may still be useful as a supervisor-preview branch,
but its learned condition must be labelled curriculum-assisted. It must include
shared-d0 and fixed-oracle controls and cannot be used as evidence for
autonomous WAD, task-derived routing or learned temporal multiplexing. A paper-
level learned-delay claim now requires a new method that makes timing
identifiable, rather than a larger sweep of the same task BCE.

### 18C. Event8-aligned five-landmark surface v2

The descriptive supervisor-preview branch is now frozen as
`mixedop_event8_aligned_surface_v2`. Its five points are `(20,4)`, `(20,12)`,
`(240,4)`, `(240,12)`, and `(80,8)`, where `w` determines `T=10+5w`.
Every point contains independent spatial d0, shared d0, fixed event8 oracle,
task-only five-delay WAD, and a curriculum-assisted five-delay model. This
exposes ordinary shared representation and hand-scheduled routing alternatives
that were missing or conflated in earlier previews.

The curriculum arm costs 600 training updates versus 400 for ordinary arms;
that difference must be disclosed even though inference-resource comparisons
remain separate. Its timing scaffold is explicit supervision, and W2 already
shows that current task BCE cannot locally restore the schedule. Therefore a
positive curriculum surface is an assisted architectural result only. A
technical five-cell smoke is the sole authorized run. The 25-cell landmark is
conditional on smoke, and the 120-cell full grid is conditional on registered
oracle and curriculum landmark gates. Neither stage opens sealed test or
supports publication-grade factor laws.

The five invalid implementation-smoke cells pass their technical gate. This
changes execution authorization only: the frozen 25-cell landmark may run.
It does not add scientific evidence, relax the curriculum label, or unlock the
full surface. Landmark logs must report clipping as well as function, activity
and schedule endpoints.

The landmark gate is now negative and closes v2 expansion. More importantly,
the curriculum loss and oracle-vector endpoint diverge systematically as w
grows, so the result cannot isolate optimization from endpoint definition.
The remaining registered grid is also above the reliability transition:
spatial, oracle and assisted models already saturate at N=20. The next
publication-relevant protocol must first reconcile exact schedule, centroid
and packet-containment endpoints, then sample widths below 20. Re-running v2
with a post-hoc gate or completing its larger-N surface is inadmissible.

### 18D. K=6 explicitly centroid-supervised surface

`mixedop_k6_centroid_supervised_surface_v1` is a separate single-seed
supervisor-preview branch. It uses the same arrival-centroid definition for
training, checkpoint eligibility and the mechanism-valid boundary, covering
N=1--60 and T=34--166 for six fixed basic logic operations. Delays receive
only explicit centroid gradients; weights/readout receive only task BCE. Any
positive curve is assisted architectural feasibility, not autonomous WAD.

Only four extreme preflight cells are authorized. The 112-cell surface is
implemented but YAML-locked until all preflight final states reach maximum
centroid error <=0.5 step and pass technical/artifact gates. Accuracy and
activity are excluded from preflight. Formal boundary points additionally
require worst-query BAcc >=.90 and activity in all six windows. With no
spatial/shared-d0/oracle comparator, this protocol cannot support a Pareto claim.

The four-cell preflight is complete and passes. Final maximum centroid errors
are 0.018--0.347 step, all artifacts and numerical gates pass, and the exact
112-cell surface is authorized. Diagnostic accuracy spans near-chance at N=1
to perfect at N=60 for both latency extremes, suggesting the width grid crosses
a transition. This observation is not preflight evidence and does not yet
identify a T effect. Formal execution must remain single-seed, validation-only
and unchanged.

The complete 112-cell surface is now available and rejects the proposed
L-shaped time--width law. All cells are mechanism-valid, but raw N90 is
`[3,3,2,2,3,4,3,3]`; the hyperbolic fit has the wrong sign (`C<0`) and low
`R2=.129`. Descriptively, hidden width accounts for .811 of worst-BAcc grid SS,
T only .013 and residual interaction .176. This supports only a narrow single-
seed statement that width dominates this explicitly supervised fixed workload.
Two higher-width reversals show the boundary is not stable. Do not enlarge the
surface. The next admissible gate is a versioned multi-seed confirmation near
N=1--5, followed separately by operation-position counterbalancing and selected
spatial/shared controls.

### 18E. K=6 boundary-region fresh-seed confirmation

`mixedop_k6_boundary_multiseed_confirmation_v1` is preregistered and
implemented to distinguish a seed-3907 artifact from a reproducible boundary.
It freezes N=1--5, all eight parent T values and fresh seeds
`{4013,4027,4049,4061,4079}` for 200 cells. The parent seed is never pooled.
All training, mechanism and checkpoint settings are inherited exactly.

Primary decisions are frozen before execution: each named reversal requires at
least 2/5 fresh seeds to reproduce; overall non-monotonicity requires at least
3/5 non-monotonic seed curves. Robust 4/5 and strict 5/5 N90 are reported
without smoothing or censor imputation. Boundary stability additionally
requires a per-T seed range <=1. This is still explicitly supervised and cannot
unlock autonomous-WAD or Pareto claims.

The 200-cell confirmation is complete. Both named parent reversals occur in
0/5 fresh seeds and therefore do not reproduce. Nevertheless, raw N90 is
non-monotone in 3/5 fresh seeds, exactly meeting the registered confirmation
threshold, and the robust 4/5 boundary `[4,3,3,3,2,3,3,3]` contains one
increase. The boundary-stability gate fails because T=34 spans N90=2--4 across
seeds, and the robust curve differs from the parent at five of eight T values.
All cells remain centroid-mechanism-valid; the unstable N=2 transition is
driven mainly by XOR/XNOR classification. This closes the uncounterbalanced
surface branch without a time--width law. The next scientific gate is a new
operation-window counterbalancing protocol, followed separately by selected
matched spatial/shared-d0/oracle controls; neither is yet preregistered.

### 18F. Spatial/temporal capacity and SLAYER trainability

The K=6 centroid surface and fresh-seed confirmation are now a closed explicit-
supervision branch. They show capacity when a schedule is supplied, not that
task loss learns delays, and must not be enlarged or fit with a time--width law.

The replacement `spatial_temporal_capacity_slayer_v1` separates two evidence
chains: fixed-oracle/explicit-centroid architectural capacity and task-only
SLAYER trainability. The main workload is repeated XOR at
`K={1,2,3,4,6,8}` with deterministic `binary_one_hot_packet` input: four value
channels per query and exactly `8K` events independent of K, B and T. The main
endpoint is one shared MLP reused over disjoint hidden-count windows. Opponent
spiking output is conditional follow-up evidence and is never pooled with the
MLP scaling fit.

Lava-DL 0.6.0 is isolated in `D:/anaconda3/envs/snn_slayer` with Torch
2.3.1+cu121. One `slayer.axon.Delay` precedes the dense CUBA block, giving
exactly `4K` input-axonal delays. Because this Windows host lacks the MSVC
compiler used by Lava-DL's optional JIT CUDA kernels, the adapter records and
uses pure-PyTorch implementations of the same integer shift, time convolution
and CUBA recurrence on CPU and CUDA.

S0 is complete and permanently invalid for scientific claims. Seeds 99501 and
99502 pass early/late gradient direction, fixed arrival, event preservation,
BTC--NCT round trip, checkpoint recovery, and full CUBA CPU/CUDA parity. Logits
and hidden spikes are exact; maximum delay-gradient differences are `2.17e-7`
and `6.29e-8`. This authorizes S1 calibration by result only. Its YAML launch
remains locked; no calibration, confirmation, capacity surface or sealed test
has run.

After S1 and S2 pass, spatial capacity fixes B=48/T=58 and scans total hidden
width. Temporal capacity freezes the smallest 5/5 K=6 fixed-oracle width and
scans B={24,48,72,96}. Raw staircases, inverse required-resource curves and
power fits prohibit smoothing and censor imputation. “Better than linear”
requires at least four uncensored points and a bootstrap exponent CI whose
lower bound exceeds one.

### 18G. Deferred capacity-figure refinement TODO (postponed)

The completed spatial and temporal screens establish only the narrow
fixed-oracle architectural exchange result. Do not resume these extensions
until other active tasks are complete; they require a new preregistered grid
and retain the frozen three-seed / at-least-two joint reliability rule,
deterministic packet encoding, validation-only checkpoint selection, raw
staircases, censoring markers, and full resource ledgers.

1. **Spatial refinement.** Extend K above the present boundary only at the
   high-N transition region (principally N=24 and N=32), with enough K values
   to resolve whether the independent and fixed-oracle curves continue, plateau
   or are censored. Preserve the current B=48/T=58 and keep the independent
   spatial d0 versus shared fixed-oracle comparison. The aim is a denser
   inverse-N-versus-K compression curve, not smoothing or manufacturing an
   exponent.
2. **Temporal refinement.** Extend K above 8 for B>=48, where the completed
   screen is top-grid censored, and preregister B/K combinations whose windows
   are integral. Add intermediate B only if needed to resolve the observed
   B=24,K=4 to B=48,K=8 transition. The sole capacity curve remains the
   route-capable shared fixed-oracle condition at frozen N_ref=6; d0 is not a
   competing temporal-rate curve because it lacks a route to distinct output
   windows.
3. **Diagnostics and interpretation.** Retain one `diagnostic_panel.npz` per
   new temporal cell. Plot raw pass fractions, per-query accuracy/activity and
   delay-to-window alignment alongside capacity; never fit a temporal exponent
   until at least four uncensored B points exist. These remain supplied-schedule
   upper-bound results unless a later task-only SLAYER gate independently passes.
