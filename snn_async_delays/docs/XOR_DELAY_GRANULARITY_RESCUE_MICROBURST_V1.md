# Dimension-aware XOR micro-burst rescue v1

**Protocol ID:** `xor_delay_granularity_rescue_microburst_v1`  
**Status:** complete. Fixed-d4 Stage B0 passes 5/5 and learned Stage B1 is
40/40 complete. See
`RESULTS_XOR_DELAY_GRANULARITY_RESCUE_MICROBURST_STAGE_B0.md` and
`RESULTS_XOR_DELAY_GRANULARITY_RESCUE_MICROBURST_STAGE_B1.md`.

## 1. Scientific question

Level 1B established that a fixed delay of four solves the declared
consecutive micro-burst interface, but its frozen learned matrix uses a mean
arrival loss whose gradient is diluted by `1/P`. Level 1B-R subsequently
showed on single-event XOR that setting `lambda_P=.01P` recovers global,
per-hidden and per-synapse delay parameterizations on independent sealed
seeds.

This protocol asks one new question: does that dimension-matched recipe retain
exact hard-spike XOR and coordinate-wise homogeneous delay recovery when each
selected input channel emits at both steps 8 and 9 rather than only step 9?

It does not amend or complete the original 60-cell Level-1B Stage-B matrix.
That matrix remains frozen, mechanically authorized and unrun.

## 2. Model and interface

The architecture and optimization are inherited without tuning:

- exhaustive K=1 XOR with value-identity inputs `[A0,A1,B0,B1]`;
- `4 -> 16 -> 2` hard-LIF class-opponent SNN;
- trainable sigmoid input-hidden delays with `dmax=8`;
- hidden-output delays fixed at d0;
- hidden/output thresholds `.2/.03` in simulator arbitrary units;
- Adam with weight LR `.01`, delay LR `.01`, 500 full-batch updates;
- final checkpoint only, with no best-epoch selection;
- one exact correct output spike at step 15 and no early or extra output
  spikes.

The consecutive micro-burst emits twice on each selected value channel at
steps `{8,9}`, giving exactly four input events per truth-table trial. The
arrival-centroid target for delay four is therefore `13.5` steps under the
production `emission + delay + 1` convention. Because one delay coordinate
shifts both events together, matching the centroid cannot independently warp
the two-event shape.

## 3. Dimension-aware loss

For `P` independent delay coordinates,

\[
L_{arr}^{mean}=\frac{1}{2P}\sum_{p=1}^{P}
[\mu_p(d_p)-\mu_p(4)]^2,
\qquad \lambda_P=.01P.
\]

Thus

\[
\lambda_P L_{arr}^{mean}
=.01\sum_{p=1}^{P}\frac12[\mu_p(d_p)-\mu_p(4)]^2.
\]

The effective weights are `.01`, `.16` and `.64` for global, per-hidden and
per-synapse delays. This fixes oracle-teacher strength per coordinate, so the
total teacher strength grows with `P`. It is not task-derived timing.

## 4. Frozen 45-cell design

Formal seeds are `{2333,2351,2371,2389,2411}` and are disjoint from Level 1A,
Level 1B and all Level 1B-R calibration/confirmation seeds. Both raw delay
initializations `-2/+2` are used in every learned candidate.

### Stage B0: fixed oracle replication — 5 cells

Fixed input-hidden delay four is trained on all five new seeds. All five cells
must pass the exact hard-spike interface. Failure keeps the learned stage
locked and is interpreted as an environment/interface replication failure,
not as evidence against delay optimization.

The old fixed-d0 negative control is not rerun: it already failed 0/5 and
emitted at step 11 in the parent protocol. This protocol changes neither the
target nor the timing convention.

### Stage B1: learned rescue — 40 cells

| Condition | P | Effective lambda | Role | Cells |
|---|---:|---:|---|---:|
| global anchor | 1 | .01 | required anchor | 10 |
| per-hidden dimension-matched | 16 | .16 | primary selected recipe | 10 |
| per-synapse dimension-matched | 64 | .64 | secondary complexity control | 10 |
| per-hidden task-only | 16 | 0 | diagnostic; not required | 10 |

The primary protocol passes only if fixed d4 is 5/5, global is 10/10 and the
selected per-hidden dimension-matched recipe is 10/10. Per-synapse success is
secondary and never displaces a passing per-hidden model. Task-only success or
failure is reported but cannot change the primary decision.

## 5. Per-cell gates

Every learned cell must satisfy all of the following simultaneously:

- balanced accuracy one and all four XOR classifications correct;
- output spike train exactly equals the target train;
- zero silence and collision, correct target-time rate one;
- exactly one output spike per trial and hidden activity in all four patterns;
- maximum error of every independent delay at most `.1` step;
- fraction of independent delays within tolerance equals one;
- every initial total raw-delay gradient is nonzero and target-directed.

Accuracy without the exact spike train, a correct mean delay with failed
coordinates, or a favourable subset of seeds is a failure.

## 6. Diagnostics and resources

Each cell must write during execution:

- config, final checkpoint and full training log;
- exhaustive truth-table results and strict metrics;
- static/measured resource ledger;
- runtime `diagnostic_data.npz`;
- runtime 12-panel diagnostic containing loss components, interface trace,
  delay range/coverage, coordinate gradients, clipping coefficient,
  micro-burst input raster, hidden/output raster, output pre-reset voltage,
  final delay map and arrival centroids.

The resource ledger separates 1/16/64 trainable delay parameters from the same
64 physical input-hidden synapses, and records neuron updates, dense MACs,
measured events and delay memory.

## 7. Decision and claim limits

A positive result supports only robust **oracle-supervised homogeneous delay
recovery under this micro-burst**. It does not show autonomous discovery,
heterogeneous routing, K>1 multiplexing, WAD superiority, compression or an
energy/Pareto law.

Even a full pass does not directly authorize K>1. It authorizes only a newly
preregistered scaffold-withdrawal/task-derived timing test. A failure triggers
a mechanism diagnosis of burst-induced task/teacher conflict; it must not be
repaired by changing LR, lambda, threshold or update budget inside this
protocol.

## 8. Seed and implementation hygiene

During implementation smoke testing, proposed seed `2309` was accidentally
materialized before the formal launch. Its smoke artifacts are retained and
classified invalid, and it was permanently removed from the formal seed set.
The replacement `2411` has not been run. The runner now forces all smoke runs
onto dedicated `99xxx` seeds.

Seven protocol-specific structural tests pass. Ten smoke cells (five original
seed-hygiene artifacts plus five dedicated-seed reruns) are implementation
evidence only and cannot be pooled with formal results. Every dedicated smoke
cell generated all eight mandatory artifacts, including the runtime NPZ and
panel. The full project test suite passes 104/104.

## 9. Execution order

```powershell
python -m scripts.run_xor_delay_granularity_rescue_microburst --stage control --device cuda
python -m scripts.run_xor_delay_granularity_rescue_microburst --stage learned --device cuda
```

Both commands are complete. The learned decision passes global, per-hidden and
per-synapse 10/10, while per-hidden task-only passes 0/10. No additional cell
may be added to this completed protocol.
