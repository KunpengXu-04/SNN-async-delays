# Mixed-Op Spatial/Temporal Surface Preview v1

Status: the six invalid stability-smoke cells passed every technical gate and
all 36 K=5 cells completed. The researcher then requested a K=5-first result
gate, so K=8 is now on hold. Five K=8 cells that had already completed and two
interrupted K=8 cells are excluded from the K=5 analysis.

## Question and evidence boundary

This single-seed MLP preview asks how worst-query balanced accuracy changes
with total physical hidden neurons and latency for fixed K=5 and K=8 mixed
Boolean workloads. It compares independent spatial modules, a hand-scheduled
shared temporal oracle, and WAD. It does not test exact output-spike timing and
cannot establish learned temporal multiplexing.

The oracle is an architectural feasibility upper bound. WAD is retained even
if it fails, because its late-window activity and learned query-delay structure
are diagnostic. No outcome can authorize a publication claim or opening the
sealed test split.

## Frozen matrix

All cells use seed 307, binary-one-hot micro-bursts at steps 8 and 9, an MLP
decoder, LR 0.01 for weights/readout/delays, 200 optimizer updates, macro
query/class-balanced BCE, and 2,048 marginally balanced training plus 2,048
independent validation samples.

The plotted hidden axis is total physical hidden neurons `[40,80,120,160]`.
Spatial modules receive `N_total/K` neurons each; temporal conditions share
`N_total`. Output-window lengths `[4,6,8]` imply K=5 total latencies
`[30,40,50]` and K=8 latencies `[42,58,74]`. Temporal oracle delays are
`d_q=q*w` and both oracle and WAD use `dmax=(K-1)w`.

The complete grid contains 72 cells. Because the deadline-driven design has
one seed and omits a full shared-spatial-d0 surface, all factor statements are
descriptive for this fixed workload only.

The post-registration execution amendment stages the grid as 36 K=5 cells
followed by a researcher decision on K=8. It does not change any cell recipe or
retroactively select K=5 results. The K=5 result is fully degenerate on the
primary metric: spatial and oracle are 1.00 everywhere, while WAD is .50
everywhere. Read
[`RESULTS_MIXEDOP_SPATIAL_TEMPORAL_SURFACE_PREVIEW_V1_K5.md`](RESULTS_MIXEDOP_SPATIAL_TEMPORAL_SURFACE_PREVIEW_V1_K5.md)
before deciding whether K=8 should resume.

## Stability gate

Before the full grid, six invalid smoke cells run at total hidden 80 and
window length 6 for both K values and all three conditions. They must be
finite, active, artifact-complete, legal in delay support, exact in oracle
schedule, nonzero in WAD delay gradient, and not continuously clipped over the
last 20 updates. Accuracy is deliberately not a smoke gate. Any technical gate
failure stops the protocol; LR must not be changed after observing it.

The completed smoke audit is
[`docs/generated/mixedop_spatial_temporal_surface_preview_v1/smoke_decision.json`](generated/mixedop_spatial_temporal_surface_preview_v1/smoke_decision.json).
All six cells are finite, active, delay-legal, artifact-complete, and not
continuously clipped over their final 20 updates. Oracle schedules are exact
and both WAD cells have nonzero delay gradients. These cells remain invalid
for scientific claims.

The machine-readable source of truth is
[`configs/mixedop_spatial_temporal_surface_preview_v1.yaml`](../configs/mixedop_spatial_temporal_surface_preview_v1.yaml).

## Reproduction

The versioned runner uses optimizer-update counts rather than epochs and
implements the registered macro-balanced loss and checkpoint ordering directly:

```powershell
python -m scripts.run_mixedop_spatial_temporal_surface_preview --stage smoke --device cuda
python -m scripts.run_mixedop_spatial_temporal_surface_preview --stage formal --device cuda
python -m scripts.summarize_mixedop_spatial_temporal_surface_preview --stage k5
```

Completed cells are immutable. A nonempty incomplete directory causes a hard
failure and must be audited instead of silently resumed. The aggregate script
refuses to plot fewer than all 72 formal cells. Its factor sums of squares are
descriptive balanced-grid decompositions without replication or an inferential
error term.
