# Results: XOR delay control matrix v1

## Evidence boundary and completeness

All preregistered artifacts completed: 60 validation-only training cells,
60 same-checkpoint late-window probes, and 15 WAD delay-shuffle probes. All
135 training/intervention artifacts have NPZ and diagnostic panels. The test
split remained sealed. Validation selected checkpoints, so all conclusions are
exploratory and conditional on this protocol.

## Main results

Worst-query accuracy, mean ± sample SD across five paired seeds:

| Setting | Readout | d0 | Scalar | Fixed heterogeneous | WAD |
|---|---|---:|---:|---:|---:|
| Primary K3/N35 | Linear | .673±.026 | **.699±.019** | .644±.059 | .678±.036 |
| Primary K3/N35 | MLP | .775±.045 | .780±.046 | **.809±.094** | .770±.061 |
| Stress K4/N50 | Linear | .646±.028 | **.652±.030** | .648±.056 | .644±.018 |

Exact-trial means for the same groups are:

| Setting | Readout | d0 | Scalar | Fixed heterogeneous | WAD |
|---|---|---:|---:|---:|---:|
| Primary | Linear | .418 | .429 | .344 | **.438** |
| Primary | MLP | .567 | .635 | **.745** | .657 |
| Stress | Linear | .225 | .222 | .208 | .224 |

The preregistered primary comparator is the strongest non-learned control on
primary worst-query accuracy: optimized scalar. WAD−scalar paired differences
are `−.028, −.060, −.030, −.036, +.050`; mean `−.0208`, with only 1/5 positive.
The required margin was `+.03` with at least 4/5 positive. Both conditions fail.

**Decision:** reject the positive learned-delay superiority programme. WAD does
not beat the strongest simple delay control and does not improve stress-setting
performance. MLP does not rescue WAD; fixed heterogeneous has the largest MLP
mean but high variance.

## Shuffle intervention

Shuffling WAD delays while preserving weights, decoder, and the exact delay
multiset reduces worst-query accuracy in every checkpoint family:

| Setting/readout | Original WAD | Shuffled | Mean drop | Positive drops |
|---|---:|---:|---:|---:|
| Primary linear | .678 | .563 | .116 | 5/5 |
| Primary MLP | .770 | .628 | .142 | 5/5 |
| Stress linear | .644 | .547 | .098 | 5/5 |

This is strong evidence that a trained model is sensitive to the mapping
between delays and synapses. It is **not** evidence that heterogeneous learned
delays are superior: scalar performs better on the primary endpoint. The most
conservative interpretation is weight/decoder–delay co-adaptation. A shuffle
destroys a coordinated solution, but the coordinated solution is not better
than the simpler scalar solution.

## Late-window probe

Every condition loses worst-query accuracy when an all-time-trained checkpoint
is evaluated through the late window. Primary-linear drops are .172 (d0), .206
(scalar), .136 (fixed heterogeneous), and .208 (WAD), all 5/5 seeds positive.
Post-probe means are approximately .47–.51. Similar collapse occurs for MLP
and stress cells.

Therefore useful evidence is distributed before the final window for all
conditions. WAD shows no special robustness to late censoring. Because the
decoder was trained all-time, this probe establishes dependence on early
activity, not the achievable performance of a separately trained late-window
model.

## Resource result

At the primary linear setting:

| Resource | Scalar | WAD |
|---|---:|---:|
| Trainable parameters | 179 | 248 |
| Delay values stored | 1 | 70 |
| Trainable delay parameters | 1 | 70 |
| Total scalar storage | 179 | 248 |
| Neuron updates | 1,400 | 1,400 |
| Dense synapse MACs | 2,800 | 2,800 |
| Decoder MACs | 105 | 105 |
| Mean hidden spikes | 10.78 | 9.85 |

Scalar is better on the primary reliability endpoint and much smaller in delay
degrees of freedom/storage. WAD emits about .93 fewer hidden spikes, but dense
compute is identical and no hardware energy model exists. WAD has a tiny exact-
trial advantage (.438 vs .429), so strict all-metric Pareto dominance is not
established; nevertheless no credible WAD resource-frontier improvement exists.

## Delay-distribution audit

Primary-linear learned scalar delays converge tightly to about 3.0 steps. WAD
has mean about 3.4–3.6, SD about 1.0–1.5, and range roughly 1–9.5. The fixed
heterogeneous bank is uniform over 0–30, mean about 13–16 and SD about 8.

Thus fixed heterogeneous is not distribution-matched to WAD and spends much of
its support on long delays. Its weaker linear result cannot be used to infer
that delay learning beats a fair fixed heterogeneous structure. Its strong but
variable MLP result further shows that decoder capacity interacts with delay
distribution.

## Diagnostic-panel assessment

Corrected panels clearly separate decoder decisions from true output spikes,
show membrane dynamics, and expose query-level errors. They support the factual
observations that scalar is temporally uniform, WAD is moderately heterogeneous,
and uniform fixed banks create much later arrivals. They remain single-sample
illustrations. The causal evidence comes from aggregate shuffle effects, not
from visually attractive routing arcs.

## Claims supported, rejected, and unresolved

Supported narrowly:

- trained WAD networks depend causally on their specific delay placement;
- all-time-trained networks across all conditions depend on pre-final-window
  activity;
- a shared learned delay around three steps is a strong, compact baseline.

Rejected within this protocol:

- WAD is superior to the strongest non-learned delay control;
- heterogeneous learned delays improve scaling at K=4;
- lower WAD hidden spike count establishes energy or cost advantage;
- WAD uniquely solves late-window alignment.

Unresolved:

- whether a distribution-matched fixed heterogeneous bank can equal WAD;
- whether shuffled delays remain harmful after weights/decoder are retrained;
- whether learned placement transfers across seeds/tasks instead of merely
  co-adapting within a checkpoint;
- whether fixed-T packing changes the ranking.

## Next scientific step

Do not open the test set and do not run a larger K/N sweep. Preregister a compact
`xor_delay_causal_decomposition_v1` at primary K3/N35, all-time linear, with
five paired seeds:

1. optimized scalar;
2. fixed scalar at the cross-seed learned scalar value;
3. cross-fitted fixed heterogeneous bank drawn from another seed's WAD delay
   multiset, with weights/decoder trained from scratch;
4. shuffled WAD delays frozen, followed by weight/decoder reinitialization and
   retraining;
5. original WAD.

This separates global timing, distribution, placement, and co-adaptation. Add
a matched narrow random bank (roughly the observed 1–9 range) only if declared
before running. If WAD still fails to outperform cross-fitted/retrained fixed
controls, pivot the paper to the observation-confound and co-adaptation result.

## Generated artifacts

- `docs/generated/xor_delay_control_matrix_v1_run_level.csv`
- `docs/generated/xor_delay_control_matrix_v1_group_summary.csv`
- `docs/generated/xor_delay_control_matrix_v1_primary_paired_gate.csv`
- `docs/generated/xor_delay_control_matrix_v1_intervention_run_level.csv`
- `docs/generated/xor_delay_control_matrix_v1_intervention_summary.csv`
- `docs/generated/xor_delay_control_matrix_v1_delay_summary.csv`
- `docs/generated/xor_delay_control_matrix_v1_decision.json`
- `docs/generated/xor_delay_control_matrix_v1_performance.png`
- `docs/generated/xor_delay_control_matrix_v1_gate_and_shuffle.png`
