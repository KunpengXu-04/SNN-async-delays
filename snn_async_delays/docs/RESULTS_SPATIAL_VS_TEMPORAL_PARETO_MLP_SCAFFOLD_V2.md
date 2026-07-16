# Results: spatial versus temporal Pareto MLP scaffold v2

## Status and evidence boundary

All 160/160 preregistered exploratory cells completed. Every cell contains both
checkpoints, a training log, exhaustive truth-table predictions, a resource
ledger, `diagnostic_data.npz`, and an in-run diagnostic panel. No cell was
excluded and no sealed test split was opened.

This experiment succeeds as a **hidden-dynamics feasibility diagnosis** and
fails as evidence for the current learnable-delay method. It uses a non-spiking
MLP endpoint, two seeds and repeated XOR; it cannot repair the failed direct-
spiking-output Stage A, unlock formal Stage B, or support a publication claim.

## Primary result

At the locked robust target—both seeds must have worst-query balanced accuracy
and exact-trial accuracy at least `.90`—only two conditions are feasible:

| condition | selected T | surface h | total hidden | min-seed worstB | min-seed exact |
|---|---:|---:|---:|---:|---:|
| spatial independent d0 | 18 | 24 per query | 48 | 1.00 | 1.00 |
| shared temporal fixed oracle | 18 | 24 total | 24 | 1.00 | 1.00 |

The fixed oracle therefore realizes an area-compression ratio

\[
\rho_{area}=\frac{24}{2\times24}=0.5,
\]

and the same `0.5` ratio for hidden-neuron updates at matched `T=18`.
However, this is not a total-compute improvement: both instantiate 3,456 dense
input-hidden MACs, 192 measured synaptic events, 192 stored delay values and
1,200 decoder MACs per trial. The oracle needs 40 delay-buffer elements per
sample versus 8 for spatial d0 (`5x`). The supported statement is therefore
**half hidden area and hidden updates for a handcrafted schedule**, not half
compute or energy.

The ordinary shared-spatial all-time MLP reaches seed-mean worst-balanced
`.906` at `h=24`, but its minimum seed is at most `.875`; it does not pass the
locked robust `.90` rule. This makes the oracle result more than an all-time
shared-representation result within this small grid, but two seeds are far too
few for a stable superiority claim.

## WAD result: decisive negative under the frozen method

`shared_temporal_wad` has worst-query balanced accuracy exactly `.50` in all
32 cells. Its mean per-query balanced accuracies are `[.865, .500]`: query 0 is
learned, while query 1 is always chance. Temporal d0 has the same pattern
`[.867, .500]`. WAD does not improve the primary endpoint at any `T`, width or
seed.

The mechanism audit localizes this failure:

- WAD mean delays are only `.117` and `.119` output-window lengths for query 0
  and query 1 inputs, respectively;
- the mean query-conditioned delay gap is only `.013` timestep, with range
  `[-.052,.082]` across cells;
- zero WAD query-1 synapses in all 32 selected checkpoints reach even the
  `window_start-1` boundary;
- WAD mean hidden spikes are `1.307` in output window 0 and exactly `0` in
  output window 1; window-1 activity fraction is zero in every cell;
- final checkpoints give the same conclusion: no query-1 delay reaches the
  second window;
- gradients are not globally absent: mean delay-gradient norm is `.00297` and
  mean final absolute delay movement is `.108`, with zero saturation. The
  optimizer moves delays slightly but never breaks query symmetry or approaches
  the required schedule.

By contrast, the fixed oracle sets normalized delays `[0,1]`, produces mean
hidden spikes `[.910,.811]` in the two windows, and reaches perfect robust
reliability at `h=24` for every tested `T`. Thus the simulator, MLP interface and
shared hidden layer can express temporal routing; the present WAD
parameterization/initialization/objective does not learn it.

## What the T-by-h surface actually says

Hidden width, not latency, explains almost all variation among non-degenerate
conditions. On the seed-mean grid:

| condition | hidden marginal range | T marginal range | grid SS from hidden | grid SS from T |
|---|---:|---:|---:|---:|
| spatial independent d0 | .250 | .031 | 93.7% | 1.6% |
| shared spatial d0 | .313 | .031 | 97.7% | 0.9% |
| fixed temporal oracle | .398 | .008 | 99.6% | 0.05% |
| temporal d0 | 0 | 0 | undefined/flat | undefined/flat |
| temporal WAD | 0 | 0 | undefined/flat | undefined/flat |

This grid therefore does **not** identify a meaningful empirical accuracy law
in `T`. Increasing `T` simultaneously increases output-window length and delay
support, while the oracle delay is rescaled to remain exactly one window. The
task difficulty is almost invariant to `T`; for WAD and d0 the second window is
empty at every `T`. A fitted `acc=f(T,h)` curve would overstate what the data
contain. The defensible result is a width threshold at `h=24` plus a flat-
latency diagnosis.

## Claim decisions

- **Supported narrowly:** a fixed query-conditioned delay schedule can process
  two simultaneous XOR queries with half the hidden neurons and hidden updates
  of two independent modules at matched reliability and latency.
- **Not supported:** reduced dense synaptic compute, events, delay storage,
  total memory, energy, or hardware cost.
- **Contradicted for the frozen method:** unconstrained WAD learns the temporal
  routing schedule in this task.
- **Not tested:** direct spiking outputs, mixed operations, K>2, robustness to
  timing noise, generalization to unseen input timing, or external tasks.

## Recommended next decision

There are now two scientifically distinct paths; they should not be mixed in
one experiment.

1. If the intended contribution is **learnable delays**, first run one small
   oracle-to-WAD bridge at the single informative point `T=18,h=24`: compare
   ordinary low-delay initialization against query-stratified initialization
   and a preregistered delay-separation/coverage objective, retaining d0 and the
   fixed oracle. The outcome must be whether WAD can create second-window
   activity, not merely improve pooled accuracy.
2. If the intended contribution is **the spatial-versus-temporal architecture**,
   return to spiking-output training only at `T=18,h=24`, initially using
   spatial d0 and the fixed oracle. This isolates the output-spike interface
   from delay learning. Do not immediately repeat the full 160-cell surface.

The supervisor should decide which contribution is primary. Until then, the
most accurate progress statement is: *temporal reuse is feasible under an
explicit schedule and trades delay-buffer memory for hidden area, but the
current learned-delay method does not discover that schedule.*

## Generated evidence

- `docs/generated/spatial_vs_temporal_pareto_mlp_scaffold_v2/exploratory/cells.csv`
- `surface_summary.csv`, `factor_effect_summary.csv`
- `robust_feasible_points_90.csv`, `matched_90_resource_comparison.csv`
- `temporal_mechanism_cells.csv`, `temporal_mechanism_summary.csv`
- `figC_nhid_T_plane.png`, `figC_nhid_T_plane_worst_seed.png`
- `figC_exact_trial_T_plane.png`, `pareto_resource_frontiers.png`
- `fig_matched_90_resource_comparison.png`
- `fig_temporal_mechanism_summary.png`
- `decision.json`
