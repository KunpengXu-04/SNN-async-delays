# Results: simultaneous spatial-control pilot v1

Date: 2026-07-13  
Protocol: `simultaneous_spatial_control_pilot_v1`  
Scientific status: validation-only exploratory spatial control; test sealed

## Audit and scope

All 36 preregistered cells completed: four delay conditions (`d0`, optimized
`scalar`, `fixed_matched`, `w_and_d`) by three endpoints (`linear`, `mlp`,
`opponent_parallel`) by seeds `{0,1,42}`. Every cell contains config, best and
last checkpoint, training log, validation result, exhaustive 64-pattern truth
table, diagnostic NPZ, and a diagnostic panel generated during the run.

The experiment uses dedicated input identities and dedicated spatial outputs.
It tests parallel representational/readout capacity. It does **not** test shared
temporal routing, temporal multiplexing, or a latency advantage.

## Primary reliability result

The preregistered primary endpoint is the linear readout and the primary metric
is worst-operation balanced accuracy.

| Condition | Validation worst balanced | Validation exact trial | Exhaustive worst balanced | Exhaustive exact trial |
|---|---:|---:|---:|---:|
| d0 | .961 ± .039 | .952 ± .052 | .964 ± .039 | .958 ± .048 |
| scalar | .944 ± .043 | .941 ± .047 | .943 ± .045 | .943 ± .045 |
| fixed matched | .960 ± .034 | .961 ± .040 | .958 ± .033 | .964 ± .036 |
| WAD | .956 ± .025 | .960 ± .030 | .953 ± .016 | .958 ± .024 |

WAD does not outperform the strongest non-learned control. Against d0, its
paired validation worst-balanced differences are `[-.030, +.038, -.021]`
(mean `-.0043`; one of three positive). Against fixed matched they are
`[-.054, +.040, +.004]` (mean `-.0034`). The exhaustive table gives the same
substantive result. These data reject a positive WAD-superiority interpretation
for the primary spatial endpoint; they do not establish equivalence because the
pilot has only three seeds and no equivalence margin was preregistered.

## Endpoint diagnosis

### MLP

All 12 MLP cells reach exactly 1.0 worst-balanced and 1.0 exact-trial accuracy
on both validation and the exhaustive truth table. This is a complete ceiling,
not evidence for a delay mechanism. The MLP endpoint is non-discriminative for
this task size and should not be used to rank delay conditions.

### Direct spiking opponent output

| Condition | Worst balanced | Exact trial | Silent | Tie | Collision |
|---|---:|---:|---:|---:|---:|
| d0 | .742 ± .081 | .710 ± .069 | .148 | .204 | .056 |
| scalar | .802 ± .059 | .777 ± .072 | .135 | .145 | .010 |
| fixed matched | .764 ± .051 | .749 ± .067 | .112 | .195 | .117 |
| WAD | .804 ± .138 | .887 ± .056 | .076 | .078 | .003 |

WAD versus scalar is essentially tied on the primary-style marginal endpoint:
mean worst-balanced difference `+.0014`, with seed differences
`[-.028, +.104, -.072]`. The WAD variance is also large. Therefore there is no
robust WAD reliability advantage here.

There is, however, one coherent exploratory signal: WAD improves exact-trial
accuracy over scalar in every seed by `[+.160, +.150, +.020]` (mean `+.110`),
and the exhaustive truth-table mean difference is `+.109`. Its lower silence,
tie, and collision rates provide a plausible output-interface explanation.
This signal is secondary, post-pilot interpretive evidence with only three
seeds. It must not be promoted to a confirmatory learned-delay claim.

## Resource interpretation

All endpoints use `6 input -> 50 hidden` and 20 simulation steps. Linear/MLP
use 1,000 neuron updates per trial; opponent output uses 1,120. Decoder MACs
are 150 for linear, 2,650 for MLP, and zero for the spike-count opponent
decoder, although the latter adds 300 hidden-output synapses and their events.

For opponent output, mean measured synaptic events per trial are approximately
527 (d0), 534 (scalar), 544 (fixed), and 548 (WAD). Mean hidden spikes are
12.85, 14.12, 15.70, and 16.45 respectively. WAD therefore obtains its
exploratory exact-trial signal with more activity/events than scalar and d0.
It does not demonstrate a resource-frontier or energy advantage. WAD stores
600 trainable delay values; scalar stores two. Hardware energy is not measured.

## Decision

1. The primary spatial result is negative for WAD superiority.
2. The MLP endpoint is saturated and scientifically uninformative at this task
   size.
3. The opponent exact-trial improvement is retained as an exploratory output-
   interface hypothesis, not as evidence of temporal routing.
4. No result here licenses the claim that delays temporally multiplex mixed
   operations.
5. Do not expand this spatial pilot or open test data. The next experiment is
   the already-preregistered six-cell temporal viability preflight. Its gates
   concern event support and trainability, not accuracy or WAD ranking.

## Artifacts

- Raw cells: `runs/exploratory/simultaneous_spatial_control_pilot_v1/`
- Aggregate cells and grouped statistics:
  `docs/generated/simultaneous_spatial_control_pilot_v1/`
- Machine-readable decision: `decision_summary.json`
- Paired seed deltas: `paired_wad_minus_controls.csv`
- Reliability, resource, output-failure and exhaustive-pattern figures are in
  the same generated directory.

