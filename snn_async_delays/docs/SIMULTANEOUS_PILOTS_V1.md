# Preregistration: simultaneous spatial-control and temporal-routing pilots

Both pilots use three fixed operation positions, XOR/NAND/NOR, with all A/B
query pairs injected simultaneously through six dedicated channels. This fixed
mapping avoids an impossible random mixed-op task without an operation cue.

The spatial control uses separate output heads or opponent pairs and measures
parallel multi-task capacity. It cannot support temporal-multiplexing language.
The temporal-routing pilot uses ordered output windows. Its linear/MLP probes
share one decoder across windows; its task-native endpoint reuses one opponent
output pair, with signed spike-count difference per window.

The WAD audit is complete and freezes hidden threshold .3, dmax 30, delay LR
.001, constant initialization and joint optimization. Output-interface
calibration v2 subsequently froze opponent-output threshold .2. V1's temporal
d0 arm is invalid; WAD outcomes did not participate in threshold selection.

The spatial task uses `6 input -> 50 hidden` neurons. Linear maps 50 counts to
three dedicated logits; MLP is a charged 50->50->3 upper bound; opponent output
uses six LIF neurons (one class-0/class-1 pair per operation). The temporal task
uses the same input/hidden layer. A single 50->1 linear or 50->50->1 MLP is
reused over three 10-step output windows, while the task-native endpoint reuses
one pair of LIF output neurons over those windows.

Raw pooled accuracy is descriptive only because XOR/NAND/NOR have different
class prevalence. Primary reliability is worst-operation balanced accuracy for
the spatial control and worst-window balanced accuracy for temporal routing.
Exact-trial accuracy is required. Every checkpoint is also evaluated on all 64
six-bit input patterns and records a 3x3 cross-target balanced-accuracy matrix,
routing selectivity, opponent silent/tie/collision rates, per-window spike
counts and signed temporal margins where defined.

The diagnostic panel now records the true seed and endpoint, labels six input
channels and opponent neurons, draws all output-window boundaries, places
non-spiking decisions at their actual window centres, annotates opponent counts,
and makes shared-window observation explicit in the mechanism plot. The NPZ
remains the source for regeneration. Aggregate scripts additionally produce
paired reliability, resource-frontier, output-interface failure and exhaustive
truth-table heatmaps. Single-sample panels remain illustrative, not mechanism
evidence.

Each pilot has 36 validation-only cells (four delay conditions, three endpoints,
three seeds), retains all failures and diagnostics, and keeps test sealed.

## Output-interface calibration v2 gate

Calibration v1 is stopped and must not be resumed. Its temporal d0 arm was
structurally silent because a feedforward zero-delay network had no events in
the requested post-input windows. Calibration v2 reuses only the nine valid
spatial-v1 cells and adds nine temporal-interface cells with a frozen timing
scaffold. The scaffold and a peak-voltage-to-spike training curriculum are
calibration devices, not pilot conditions and not delay-performance evidence.

The temporal pilot remains locked until v2 selects one threshold under both
interface viability gates. If v2 selects none, do not lower gates, choose the
best-looking threshold, or launch only the WAD temporal condition. Redesign the
output interface as a new version instead.

V2 completed and selected threshold 0.2 under the locked one-spike rule. This
unlocks the spatial control only. Temporal worst balanced accuracy was exactly
.5 for every threshold/seed; at .2, XOR and NOR stayed at chance while NAND was
learned inconsistently. The temporal matrix therefore remains locked for an
outcome-independent, matched-training preflight. See
`RESULTS_SIMULTANEOUS_OUTPUT_INTERFACE_CALIBRATION_V2.md`.

That preflight is now preregistered as
`simultaneous_temporal_viability_preflight_v1`: six seed-0 opponent cells add a
full-support frozen delay bank and scaffold positive control to the four pilot
conditions. Its gates use window event support and gradients, not accuracy.
The spatial 36-cell pilot is complete. The primary linear endpoint gives no WAD
advantage over d0 or fixed matched delay; the MLP endpoint is at a complete
ceiling. WAD improves opponent exact-trial accuracy over scalar in 3/3 seeds,
but worst-balanced accuracy is unstable and measured event cost is higher, so
this remains an exploratory interface signal. See
`RESULTS_SIMULTANEOUS_SPATIAL_CONTROL_PILOT_V1.md`. The temporal viability
preflight subsequently failed its hidden-emission gate. The checkpoint audit
showed that earlier WAD spikes still generate third-window output current in
70.3% of exhaustive trials, so hidden emission is not a valid arrival proxy.
Preflight v2 is complete and passes all output-current/realized-arrival/
activity/gradient gates in both held-out seeds. WAD nevertheless remains at
worst-window balanced `.5` because NOR in window 2 is not discriminated. Since
operation identity and temporal position are confounded, the old fixed-order
full pilot remains code-locked pending a counterbalanced amendment. See
`RESULTS_SIMULTANEOUS_TEMPORAL_CHECKPOINT_MECHANISM_AUDIT_V1.md` and
`RESULTS_SIMULTANEOUS_TEMPORAL_VIABILITY_PREFLIGHT_V2.md`.

The 45-cell counterbalanced amendment is now complete. No condition passes its
worst operation-position routing floor. WAD is strongest only on secondary
mean balanced and exact-trial metrics; it leaves late XOR and NOR at chance.
Full-support fixed delays learn late positions better but fail early, revealing
a timing-allocation trade-off. The test remains sealed. See
`RESULTS_SIMULTANEOUS_TEMPORAL_COUNTERBALANCED_PERFORMANCE_V1.md`.
