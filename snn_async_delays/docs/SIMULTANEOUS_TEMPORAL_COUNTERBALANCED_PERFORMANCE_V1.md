# Preregistration: counterbalanced temporal performance v1

**Completion note (2026-07-14):** all 45 cells completed and the test split
remained sealed. No condition passed the locked routing floor and WAD did not
pass the superiority rule. WAD was best on the secondary mean balanced and
exact-trial metrics, but retained chance-level late operation-position cells.
See `RESULTS_SIMULTANEOUS_TEMPORAL_COUNTERBALANCED_PERFORMANCE_V1.md`; this
completion note does not alter the preregistration below.

## Scientific question

After establishing output-arrival and gradient viability, can WAD or a matched
non-learned delay control solve mixed XOR/NAND/NOR routing through one shared
opponent output pair when operation identity is separated from temporal
position?

The previous fixed order mapped XOR/NAND/NOR permanently to windows 0/1/2.
That design cannot distinguish operation difficulty, temporal-position
difficulty and their interaction. This protocol uses a cyclic Latin-square
counterbalance:

| Order | Window 0 | Window 1 | Window 2 |
|---|---|---|---|
| cyclic_0 | XOR | NAND | NOR |
| cyclic_1 | NAND | NOR | XOR |
| cyclic_2 | NOR | XOR | NAND |

## Matrix

- Primary endpoint: `opponent_shared_windowed` only.
- Conditions: d0, scalar, narrow fixed `[1,9]`, full-support fixed `[0,30]`, WAD.
- New held-out paired seeds: `{7,19,73}`. Earlier seeds `{0,1,42}` are excluded.
- Three cyclic orders.
- Total: `5 × 3 × 3 = 45` validation-only cells.
- Test remains sealed.

All cells use `6 -> 50 -> 2` spiking neurons, 10-step input, three 10-step
output windows, burst encoding, output threshold `.2`, hidden threshold `.3`,
100 epochs, the frozen 20-epoch output-membrane curriculum and a window/class-
balanced objective. WAD keeps `dmax=30`, delay LR `.001`, constant
initialization and joint optimization.

## Estimands

For each condition and seed, the three trained orders yield nine
operation-position balanced accuracies. The primary estimand is their minimum:

\[
R_{c,s}=\min_{o\in\{XOR,NAND,NOR\},p\in\{0,1,2\}}
\operatorname{BAcc}_{c,s,o,p}.
\]

Secondary estimands are the mean over nine cells, mean exact-trial accuracy,
worst order, operation and position main effects, operation-by-position
interaction, output-current/arrival diagnostics and the complete resource
ledger. Pooled accuracy is descriptive.

## Locked interpretation rules

A condition is labelled as demonstrating exploratory counterbalanced routing
only if every seed has `R >= .55` and mean `R >= .60`.

Exploratory WAD superiority requires all of the following:

1. mean paired WAD-minus-strongest-nonlearned-control primary difference
   at least `.03`;
2. a positive primary difference in all three paired seeds;
3. mean exact-trial difference at least `-.02`.

The strongest non-learned control is chosen by its observed mean primary score,
which is conservative for WAD. These are pilot decision labels, not hypothesis-
test significance or publication-level confirmation.

No cell may be excluded for poor accuracy. Linear/MLP readouts are not included
in the primary matrix because the spatial pilot showed MLP saturation and this
experiment targets the task-native shared spiking interface.

## Required outputs

Every cell saves checkpoints, log, validation, exhaustive 64-pattern truth
table, resource ledger, NPZ and diagnostic panel during the run. The aggregate
analysis must produce:

- cell and operation-position long tables;
- condition/seed primary scores;
- paired WAD-control differences;
- operation-position heatmaps;
- reliability/resource summaries;
- mechanism support summaries reconstructed from final checkpoints.

## Commands

```powershell
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_simultaneous_temporal_counterbalanced_performance --dry-run
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_simultaneous_temporal_counterbalanced_performance --device cuda
D:\anaconda3\envs\snn_async\python.exe -m scripts.summarize_simultaneous_temporal_counterbalanced_performance
```
