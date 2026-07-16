# Preregistration: spatial versus temporal Pareto Phase 0

> **Formal Stage-A outcome (2026-07-14): failed.** All 15 cells completed, but
> no hidden size passed every locked gate in all three seeds; no baseline width
> was selected and Stage B remains locked. See
> `docs/RESULTS_SPATIAL_VS_TEMPORAL_PARETO_PHASE0_STAGE_A.md`.

## Scientific objective

At matched multi-query reliability, determine whether a delay-enabled shared
hidden population can trade additional latency for lower neuron area relative
to `K` independent synchronous spatial modules, and whether any apparent
compression survives area-time, synaptic-compute, event and delay-memory
accounting.

Three claims are deliberately separated:

1. **area:** fewer physical hidden neurons;
2. **cost:** lower cost after latency and connectivity are charged;
3. **mechanism:** the same hidden neurons are causally reused at different
   times, rather than merely providing an ordinary shared representation.

The completed counterbalanced temporal experiment remains valid negative
evidence about the frozen WAD method. Phase 0 is a versioned pivot and does not
retroactively reinterpret that result.

## Architectures and mathematics

The synchronous spatial baseline is `K` independent block-diagonal modules,

\[
K\times[n_{in}\rightarrow h\rightarrow n_{out}],
\]

not one unnecessarily dense `(K n_in)->(K h)->(K n_out)` network. The temporal
candidate is

\[
(K n_{in})\rightarrow h'\rightarrow(K n_{out}),
\]

with simultaneous inputs, a shared hidden layer and query-specific opponent
outputs observed either simultaneously or in assigned temporal windows.

Hidden-neuron area compression is

\[
\rho_{area}=\frac{h'}{K h}.
\]

It is not a total-cost ratio. With spatial and temporal latencies `T_A,T_B`,

\[
\rho_{hidden\ update}
=\frac{h'T_B}{K hT_A}
=\rho_{area}\frac{T_B}{T_A}.
\]

For block-diagonal A and dense shared B, assuming identical per-query input
and output dimensions,

\[
C_A^{syn}\propto K h(n_{in}+n_{out})T_A,
\quad
C_B^{syn}\propto K h'(n_{in}+n_{out})T_B,
\]

so

\[
\rho_{dense\ synapse\ compute}=\frac{h'T_B}{hT_A}.
\]

Thus neuron-area compression does not imply compute compression. If
`T_B approximately K T_A`, hidden-update savings require `h'<h`, while dense
synaptic-compute savings require approximately `h'<h/K`. These analytic
ratios are implemented in `utils/pareto_cost.py`; the measured resource ledger
remains authoritative for instantiated models.

For later mixed-operation tasks the fair spatial baseline is

\[
H_A=\sum_{k=1}^{K}h_k^*(r),
\]

where each `h_k^*(r)` is the minimum width meeting reliability target `r` for
that operation. It is not automatically `K h`. Phase 0 therefore uses repeated
XOR before mixed operations.

## Encoding decision

Habashy et al. (2024) motivate two useful ideas: a single-burst impulse probe
and filtered spike-train output loss. Their variable zero-to-five input-spike
code is not copied into the primary experiment because it changes input events
with the bit value and can confound information with cost.

Phase 0 instead uses:

- four binary one-hot channels `[A0,A1,B0,B1]` per query, with a two-spike
  micro-burst on the selected A channel and another on the selected B channel
  at the same final input-window phase;
- two class-opponent output neurons per query;
- exactly one target output spike per query, in the neuron corresponding to
  class 0 or class 1;
- a causal exponential filter with `tau=5` steps followed by class/window-
  balanced MSE, combined with signed count BCE and the frozen membrane warmup.

This makes every query consume exactly four input events regardless of its bit
values, retains temporal target supervision, and equalizes output target events
between classes. It is stricter than copying the paper and avoids both a
rate/event shortcut and a silent-zero class. Evaluation uses actual output
spikes, never membrane logits. The Stage-A target is placed one step after the
start of the output window: under the simulator's two one-step d0 layer
transitions it is causally reachable from the final input timestep. An arbitrary
window-centre target would make the feedforward d0 baseline structurally unable
to wait and is therefore prohibited. Later temporal queries use the same
within-window offset. Target timing remains an explicit supervision signal and
requires all-time/count and window-permutation controls before a mechanism claim.

Surrogate-gradient training remains primary. The paper's population-based
evolution evaluates millions of tiny-network candidates and does not scale or
provide a matched optimizer budget here. Evolution may later be used only as a
low-dimensional optimization audit for delay-strata centres or time constants.

## Stage A: single-query interface and baseline calibration

Stage A contains 15 validation-only cells:

- XOR, `K=1`, d0 input and output delays;
- hidden sizes `{4,8,12,16,24}`;
- new seeds `{107,211,509}`;
- `4 -> h -> 2` spiking opponent architecture;
- 10 input plus 10 output steps;
- hidden/output thresholds `.2/.03`; both thresholds are re-calibrated for
  the new four-channel one-hot fan-in rather than inherited from the old burst
  protocol;
- 100 epochs, final checkpoint;
- exhaustive four-pattern truth table, resource ledger, NPZ and in-run panel.

The selected `h` is the smallest width that, in every seed, has balanced
accuracy at least `.90`, exact exhaustive truth-table completion, silent and
collision rates at most `.10`, target-timing hit rate at least `.90`, mean
absolute timing error at most two steps, and mean output spikes in `[.5,1.5]`.

Stage A cannot support a multiplexing or delay claim. Failure selects no width;
thresholds and gates may not be relaxed after inspection.

Stage A is an **all-d0 baseline**: `train_mode=weights_only` together with
`fixed_delay_value=0` fixes input-to-hidden delays, and `output_delay_mode=d0`
fixes hidden-to-output delays. The `input_to_hidden_only` phrase refers only to
the planned Stage-B delay intervention locus; it must not be read as saying
that Stage-A input delays were trainable. Early stored Stage-A `config.json`
files retained that misleading metadata label, but their instantiated delays
and diagnostic heatmaps correctly contain d0 on both layers.

### Pre-formal smoke amendment

Before any formal Stage-A cell existed, an h=24/d0 smoke at inherited hidden/
output thresholds `.3/.2` was structurally silent. The final maximum hidden
membrane was `.242`: gradients and weight movement were present, but no hidden
unit crossed `.3`. That value came from a different two-channel, unequal-event
burst protocol and is not transportable to the new four-channel one-hot fan-in.
Freezing hidden threshold `.2` restored sparse hidden activity, but output
remained silent because its maximum pre-reset voltage was only `.046` against
the inherited `.2` output threshold. Stage A therefore freezes hidden/output
thresholds `.2/.03` and uses two, rather than one, spikes on each selected
one-hot value channel before formal launch. Event count remains identical for
every input pattern. The formal silent, collision, one-spike and timing gates
can still reject an over-excitable interface. No formal artifact informed this
amendment.

## Locked Stage B construction

Stage B remains locked until Stage A materializes `h`. It uses repeated XOR at
`K=2`, shared sizes derived mechanically from `{.5,1,1.5,2}*h`, and compares:

1. spatial independent d0 (`2h` hidden, block diagonal, one output window);
2. shared spatial d0 (same shared topology, one output window);
3. shared temporal d0 (two output windows);
4. shared temporal fixed-full input delays;
5. query-scheduled fixed input delays as an oracle positive control;
6. shared temporal WAD with delays only on input-to-hidden synapses.

Hidden-to-output delays are d0 in every condition. This prevents the causal
location of temporal scheduling from drifting between layers.

Primary reliability is worst-query balanced accuracy plus exact-trial
accuracy. Full curves and iso-reliability contours are reported; `.90` is a
calibration summary, not the sole Pareto endpoint. Cost reports area, latency,
hidden updates, dense synapse MACs, measured synaptic events, parameters and
delay memory separately.

## Commands

```powershell
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_spatial_vs_temporal_pareto_phase0 --stage a --dry-run
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_spatial_vs_temporal_pareto_phase0 --stage a --smoke --hidden 8 --seed 107 --device cpu
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_spatial_vs_temporal_pareto_phase0 --stage a --device cuda
D:\anaconda3\envs\snn_async\python.exe -m scripts.summarize_spatial_vs_temporal_pareto_phase0
```
