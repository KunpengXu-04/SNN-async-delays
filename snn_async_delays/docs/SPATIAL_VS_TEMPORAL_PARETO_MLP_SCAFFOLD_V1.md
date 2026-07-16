# Exploratory MLP scaffold: spatial parallelism versus temporal routing

> **Superseded execution:** V1 was aborted after one completed cell because its
> repeated exhaustive batch created 2,400 redundant optimizer updates per cell.
> Do not pool V1 artifacts. The active runtime-amended protocol is
> `SPATIAL_VS_TEMPORAL_PARETO_MLP_SCAFFOLD_V2.md`.

## Why this branch exists

The formal direct-spiking-output Phase-0 Stage A failed its locked interface
gates. The MLP scaffold is therefore an explicitly separate exploratory branch
for answering a narrower question: **does the hidden dynamics implementation
produce a coherent latency-width-resource surface when output decoding is made
easy and differentiable?** It is useful for a supervisor progress discussion,
but it cannot be presented as a successful spiking-output system or used to
unlock formal Stage B.

## Locked comparison

All conditions receive two XOR queries simultaneously using four binary
one-hot channels per query. Each query has exactly four input events. The grid
is `T in {18,22,26,30}` and `h in {4,8,16,24}`, with two new exploratory seeds.

The five conditions are:

1. `spatial_independent_d0`: two independent `4 -> h` SNN modules and one
   shared per-query MLP decoder. Total hidden area is `2h`.
2. `shared_spatial_d0`: one dense `8 -> h` shared hidden layer and one all-time
   two-output MLP. This is an ordinary shared-representation control.
3. `shared_temporal_d0`: the same dense shared hidden topology, but one shared
   one-output MLP is applied to two successive hidden-count windows. It should
   expose the failure of unscheduled d0 activity to populate the second window.
4. `shared_temporal_oracle`: query 0 has fixed delay zero and query 1 has a
   fixed delay equal to one output-window width. It tests whether the simulator
   and readout can express the proposed routing mechanism at all.
5. `shared_temporal_wad`: input-to-hidden delays are learned over the same
   window-sized support.

There is no hidden-to-output synaptic delay in this branch because the endpoint
is a non-spiking MLP decoder. The intervention locus remains input-to-hidden.

## Fairness and mathematical interpretation

The plotted width `h` has different physical meanings that must be visible in
every caption:

\[
H_{spatial}=K h,\qquad H_{shared}=h'.
\]

At a matched reliability target `r`, area compression is

\[
\rho_{area}(r)=\frac{h'^*(r)}{K h^*(r)}.
\]

The hidden-update ratio is

\[
\rho_{update}(r)=\frac{h'^*(r)T_{temporal}}
                       {K h^*(r)T_{spatial}}.
\]

These ratios cannot be inferred from the `T x h` heatmap alone. The report must
also include instantiated hidden-neuron count, neuron updates, dense synaptic
MACs, measured synaptic events, decoder operations and delay storage. In
particular, the dense shared input connectivity can erase an apparent neuron
area saving.

## Readout choice

The temporal conditions use `windowed_shared`: a single MLP maps hidden counts
from each declared output window to one XOR logit. This preserves which window
generated the evidence. An all-time MLP would collapse the routing axis and is
therefore prohibited for the temporal candidates. The spatial-independent
baseline applies the same one-output MLP to each independent module, matching
decoder sharing. `shared_spatial_d0` uses an all-time two-output MLP by design;
its different decoder parameter/operation count is reported, not hidden.

## Data, training and evaluation

All 16 joint truth-table patterns are present in every optimizer epoch, repeated
eight times and grouped into balanced 16-pattern batches. This removes finite
sampling imbalance as an explanation for a weak query. Training uses BCE on
the two logits for 300 epochs. The preregistered selected checkpoint is the
highest exhaustive-validation pooled-accuracy checkpoint; this rule was fixed
before any formal exploratory cell and does not retroactively apply to the
failed spiking-output Stage A. Evaluation reports per-query balanced accuracy, worst-query
balanced accuracy, exact-trial accuracy, pooled accuracy, cross-target balanced
accuracy and routing-selectivity gap. Pooled accuracy is descriptive only.

Each cell must write both checkpoints, a training log, validation and exhaustive
truth-table JSON, a complete resource ledger, `diagnostic_data.npz`, and the
diagnostic panel during the run. The panel must label decoder decisions as
non-spiking outputs.

## Interpretation limits

- Success means that an MLP-decoded hidden-dynamics scaffold is viable; it does
  not solve output-spike timing, silence or collision.
- `shared_spatial_d0` success without temporal-oracle success argues for ordinary
  representation sharing, not time routing.
- Oracle success with WAD failure localizes the weakness to delay optimization.
- WAD success without oracle-vs-d0 separation is not evidence of causal routing.
- With only two seeds, the figures are progress diagnostics, not publication-
  quality uncertainty estimates.

## Smoke outcome and launch gate

The structural smoke produced checkpoints, exact predictions, resource ledgers,
NPZ files and panels for all five conditions, with finite metrics and nonzero
hidden activity. At an additional locked-grid viability point
`T=30,h=24,seed=307`, the selected spatial checkpoint reached worst-balanced
and exact-trial accuracy `1.0/1.0`; the temporal oracle reached
`.9375/.9375`. This checks that the easy endpoint and fixed routing scaffold are
capable of succeeding somewhere. It is not a reported surface result and did
not tune the formal grid. All 42 tests pass and the protocol status is
`preregistered_ready`.

## Commands

```powershell
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_spatial_vs_temporal_pareto_mlp_scaffold --dry-run
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_spatial_vs_temporal_pareto_mlp_scaffold --smoke --device cpu
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_spatial_vs_temporal_pareto_mlp_scaffold --device cuda
D:\anaconda3\envs\snn_async\python.exe -m scripts.summarize_spatial_vs_temporal_pareto_mlp_scaffold
```
