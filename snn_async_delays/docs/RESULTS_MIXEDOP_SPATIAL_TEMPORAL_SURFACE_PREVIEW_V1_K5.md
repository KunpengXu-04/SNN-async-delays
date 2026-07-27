# K=5 Mixed-Operation Surface Preview v1 — Results

Status: all 36 K=5 cells complete; K=8 is on hold and excluded from this
decision. This is a single-seed validation-only exploratory preview.

## Primary result

The intended width-versus-latency analysis is not identifiable from this grid.
Every independent-spatial d0 cell and every hand-scheduled oracle cell reaches
exactly 1.00 worst-query balanced accuracy and 1.00 exact-trial accuracy. Every
WAD cell remains exactly 0.50 on worst-query balanced accuracy.

| condition | cells | worst-query balanced | exact-trial | interpretation |
|---|---:|---:|---:|---|
| independent spatial d0 | 12/12 | 1.000 in every cell | 1.000 in every cell | ceiling over the entire tested grid |
| fixed temporal oracle | 12/12 | 1.000 in every cell | 1.000 in every cell | hand-scheduled feasibility, also ceilinged |
| shared temporal WAD | 12/12 | 0.500 in every cell | 0.1909–0.1997 | primary routing failure, not a width/T trade-off |

Consequently, both preregistered marginal accuracy ranges are zero for all
three conditions. The two-way sums-of-squares fractions are undefined because
there is no primary-metric variation. The registered rule does **not** permit
the statement that hidden neurons are more valuable than time. It also does
not support an interaction claim: the correct conclusion is insufficient
variation because of ceiling/floor effects.

The main result is
[`K5_T_total_hidden_plane.png`](generated/mixedop_spatial_temporal_surface_preview_v1/K5_gate/K5_T_total_hidden_plane.png),
and the explicit zero-variation diagnosis is
[`factor_value_summary.png`](generated/mixedop_spatial_temporal_surface_preview_v1/K5_gate/factor_value_summary.png).

## WAD failure is localized to missing temporal coverage

Pooled metrics conceal the failure. Across WAD cells, pooled accuracy is
0.7419–0.7496 and mean balanced accuracy is 0.6923–0.6992, but the mean
per-query balanced accuracies are approximately
`[.9954, .9870, .5000, .5000, .5000]`. Thus WAD solves the first two fixed
positions and leaves the final three at chance.

The activity diagnostics agree with the accuracy endpoint. Averaged over the
12 WAD cells, the fractions of validation trials with any hidden spike in the
five output windows are `[1.000, .967, .0145, 0, 0]`; mean hidden spike counts
are `[42.25, 3.54, .015, 0, 0]`. The fixed oracle has activity in essentially
every window (`[1,1,1,1,.9993]`).

WAD also fails to learn query-separated delays. Its query-mean delays are only
about `.45–.60` output-window lengths for every query, instead of the declared
oracle pattern `[0,1,2,3,4]` window lengths. Query-schedule MAE grows from
6.36 to 13.43 steps as the output window grows. This is direct evidence that
the current LR=.01/right-linear WAD recipe does not create the intended K=5
temporal queue. It is not evidence that no trainable-delay method can do so.

See
[`routing_heatmaps.png`](generated/mixedop_spatial_temporal_surface_preview_v1/K5_gate/routing_heatmaps.png)
and
[`hidden_window_activity.png`](generated/mixedop_spatial_temporal_surface_preview_v1/K5_gate/hidden_window_activity.png).

## Resource interpretation

At the smallest tested point, `N=40,T=30`, spatial d0 and the oracle both have
perfect reliability and both perform 1,200 hidden-neuron updates. Therefore the
tested grid shows no hidden-neuron compression: the matched-accuracy ratio is
`40/40 = 1`, and the true minimum lies below the tested lower boundary for both
conditions.

The oracle is not cheaper on the remaining recorded resources at that point:

| resource | spatial d0 | oracle | WAD |
|---|---:|---:|---:|
| trainable parameters | 561 | 2,481 | 3,281 |
| trainable delay parameters | 0 | 0 | 800 |
| decoder parameters | 401 | 1,681 | 1,681 |
| dense synaptic MACs/trial | 4,800 | 24,000 | 24,000 |
| measured synaptic events/trial | 160 | 800 | 800 |
| delay-value storage elements | 160 | 800 | 800 |
| delay-buffer elements/sample | 20 | 340 | 340 |

The fivefold temporal fan-in cost follows from the declared `K*n_in -> N`
dense architecture. Delay-buffer ratios are 17x, 25x, and 33x spatial at
window lengths 4, 6, and 8. `N_hidden*T` must therefore remain labelled a
neuron-update proxy; it is especially misleading as a scalar energy proxy in
this comparison.

## Decision on K=8

K=5 is a technical success and confirms that the hand schedule is executable,
but it fails the intended scientific gate for a useful factor surface:

1. spatial and oracle surfaces are fully ceilinged;
2. WAD is fully floored on the primary endpoint;
3. no width or latency effect can be estimated;
4. the smallest tested point already saturates both successful conditions;
5. WAD shows no late-window coverage or query-delay correspondence.

Therefore K=8 is **not automatically authorized**. The recommendation is to
pause and decide whether the immediate goal is merely to show the K=5
feasibility/failure contrast, or to obtain an informative cost law. For the
latter, continuing K=8 alone is not a principled repair: the grid first needs a
new calibration that brackets transition regions, while any WAD rescue must be
versioned and cannot be tuned on these results.

Five K=8 cells completed and two were interrupted before the researcher asked
for the K=5-first gate. They are excluded from every number above. The audit is
[`k8_exclusion_audit.json`](generated/mixedop_spatial_temporal_surface_preview_v1/K5_gate/k8_exclusion_audit.json).

## Evidence boundary

- Seed 307 is the only seed; no stability or significance language is allowed.
- Validation is a 2,048-sample marginally balanced joint workload, not the
  exhaustive K=5 truth table.
- The MLP output does not test exact output-spike timing.
- Fixed operation position and the missing full shared-spatial-d0 surface leave
  alternative shared-representation explanations open.
- The fixed oracle is an experimenter-scheduled reference, not proof that WAD
  can learn the schedule and not a mathematical upper bound on all schedules.
- No sealed test split was opened.

