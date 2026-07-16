# MLP Pareto scaffold v2: runtime-budget amendment

This protocol inherits every scientific factor, architecture, seed, metric,
resource field and artifact rule from
`SPATIAL_VS_TEMPORAL_PARETO_MLP_SCAFFOLD_V1.md`. Only the training-step
accounting changes.

V1 repeated the same exhaustive 16-pattern batch eight times per epoch for 300
epochs. Those repeats contain no additional observations; they create 2,400
near-identical optimizer updates per cell and made the 160-cell sweep take
several hours because the simulator is Python-time-step bound. The launch was
stopped for this runtime reason before buffered cell output was inspected.
Artifact inspection afterward found one complete cell and one interrupted cell.
V1 is therefore classified as aborted provenance and must not be pooled with
V2.

V2 freezes one exhaustive 16-pattern batch per epoch and 200 epochs, hence 200
optimizer steps per cell. The earlier smoke used 160 optimizer steps and had
already shown that both the spatial and temporal-oracle paths can train at a
viable grid point. V2 repeated the five-condition structural smoke in its own
directory and produced all checkpoints, truth-table results, resource ledgers,
NPZ files and panels.

No task, grid point, seed, input encoding, threshold, delay support, condition,
decoder, checkpoint selection rule or outcome metric changed. The amendment was
made to remove redundant computation, not in response to the completed V1
cell's accuracy. The output remains a non-spiking MLP scaffold and cannot unlock
formal Phase-0 Stage B.

## Commands

```powershell
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_spatial_vs_temporal_pareto_mlp_scaffold --dry-run
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_spatial_vs_temporal_pareto_mlp_scaffold --device cpu
D:\anaconda3\envs\snn_async\python.exe -m scripts.summarize_spatial_vs_temporal_pareto_mlp_scaffold
```
