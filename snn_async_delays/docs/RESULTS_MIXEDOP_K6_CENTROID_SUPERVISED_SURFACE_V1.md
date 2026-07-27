# K=6 centroid-supervised T x N_hid surface v1: results

## Executive decision

All 112 seed-3907 cells are complete and mechanism-valid. The experiment
supports a narrow descriptive conclusion: under explicit centroid supervision,
the tested reliability transition is dominated by hidden width, not by
simulation duration. It does **not** show the preregistered L-shaped
time--neuron tradeoff. It also does not establish autonomous WAD, output-spike
timing, spatial superiority or a publication-grade Pareto law.

## Completeness and mechanism audit

- 112/112 cells contain checkpoints, logs, fresh predictions NPZ, runtime
  diagnostic panel/data, 48-field resource ledger and completion marker.
- 112/112 have a mechanism-eligible checkpoint; no cell is masked as
  `mechanism_invalid`.
- Maximum selected-checkpoint centroid error over the whole grid is 0.4800
  step, below the frozen 0.5-step limit.
- 109/112 cells pass the all-window activity floor. The three activity failures
  are low-width cells that already fail the 0.90 accuracy gate, so activity
  does not change any boundary decision.
- 95/112 cells pass the joint mechanism/activity/90% gate.

This separates routing from task capacity: low-accuracy cells generally have a
valid experimenter-supplied schedule, so they cannot be dismissed as delay
training failures.

## Observed 90% boundary

| T | w | observed N90 | worst BAcc | exact trial | N90*T | dense MACs | measured events | delay-buffer elements |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 34 | 4 | 3 | 0.9775 | 0.9277 | 102 | 2,448 | 144.1 | 648 |
| 40 | 5 | 3 | 0.9932 | 0.9844 | 120 | 2,880 | 144.1 | 792 |
| 46 | 6 | 2 | 0.9961 | 0.9937 | 92 | 2,208 | 96.1 | 936 |
| 58 | 8 | 2 | 0.9956 | 0.9907 | 116 | 2,784 | 96.1 | 1,224 |
| 70 | 10 | 3 | 0.9987 | 0.9976 | 210 | 5,040 | 144.1 | 1,512 |
| 94 | 14 | 4 | 0.9980 | 0.9966 | 376 | 9,024 | 192.2 | 2,088 |
| 130 | 20 | 3 | 0.9980 | 0.9966 | 390 | 9,360 | 144.1 | 2,952 |
| 166 | 26 | 3 | 0.9995 | 0.9985 | 498 | 11,952 | 144.1 | 3,816 |

The raw boundary is `[3,3,2,2,3,4,3,3]`. It is not decreasing and therefore
is not an L-shaped time--space exchange. The preregistered descriptive fit

`N90(T)=N_infinity+C/(T-T0)`

returns `N_infinity=3.817`, `C=-148.65`, `T0=-87.94`, `R2=0.129`. The negative
`C` has the opposite sign from a decreasing time-for-width tradeoff, and the
fit explains little variation. `CV(N90*T)=0.629`, so constant `N*T` is also a
poor description.

The minimum observed neuron-update proxy is at T=46,N=2 (`N*T=92`), not at the
largest T. Longer simulation substantially increases updates, dense MACs and
buffer memory without a systematic reduction in N90.

## Width, time and interaction

The frozen complete grid permits a descriptive two-way sum-of-squares
decomposition, not inferential ANOVA because there is one seed and no within-
cell replication.

| metric | hidden marginal range | T marginal range | hidden SS fraction | T SS fraction | interaction residual |
|---|---:|---:|---:|---:|---:|
| worst-query BAcc | 0.4862 | 0.0528 | 0.811 | 0.013 | 0.176 |
| exact-trial | 0.7419 | 0.0917 | 0.837 | 0.015 | 0.148 |

Thus, in this exact single-seed explicitly supervised workload, reliability is
much more sensitive to hidden width than T. This is descriptive and cannot be
called significant, stable or universal.

The transition is very narrow. Mean worst-query BAcc across T is 0.514 at
N=1, 0.745 at N=2, 0.905 at N=3, 0.974 at N=4 and 0.997 at N=5. Nearly all
N>=5 cells are saturated, so most of the large-width grid contributes no
information about a cost law.

There are two explicit non-monotonic optimization anomalies: at T=34, N=3
passes but N=4 falls to 0.801; at T=46, N=2 passes but N=3 falls to 0.728.
Consequently `N90` is only the minimum observed passing cell, not a stable
capacity boundary. The preregistration forbids isotonic repair or post-hoc
smoothing, so both anomalies remain visible.

## Operation-specific mechanism

At N=1, mean BAcc across T is about 0.999/0.972/0.529/0.527/0.998/0.988 for
AND/OR/XOR/XNOR/NAND/NOR. At N=2, XOR and XNOR rise only to 0.745 and 0.816,
while the four linearly simpler operations remain near one. The bottleneck is
therefore nonlinear task representation, especially XOR/XNOR, rather than
missing windows or an invalid delay schedule.

This does not mean two or three neurons can solve arbitrary six-operation
workloads. Operation identity is fixed to temporal position, the decoder is an
MLP, and the centroid targets are supplied by the experimenter.

## Limitations and alternative explanations

1. `T` is not independently manipulated. Increasing T also increases output-
   window length, target delays, delay support, integration duration and buffer
   depth. A pure latency effect is not identified.
2. There is one seed. The two width reversals demonstrate that optimization
   variation is large enough to move the observed minimum boundary.
3. Checkpoints are selected on validation; sealed test remains closed.
4. Operation and window position are fixed/confounded. No counterbalanced
   operation-to-window assignment was tested.
5. Delay routing is explicitly supervised and never withdrawn. The result says
   nothing positive about autonomous task-derived delay learning.
6. Output is a windowed shared MLP logit, not a spiking output interface.
7. There is no shared-d0, fixed-oracle, independent-spatial or matched multi-
   seed surface. No spatial-versus-temporal Pareto comparison is possible.
8. `N*T` is only a neuron-update proxy. Dense MACs, events, buffer memory,
   parameter storage and decoder cost remain separate.

## Result files

The main figures and machine-readable tables are under
`docs/generated/mixedop_k6_centroid_supervised_surface_v1/surface/`, including
the worst-BAcc plane, joint pass plane, raw N90 curve, exact-trial and centroid
planes, per-query/activity/resource planes, delay correspondence,
`factor_diagnostics.json`, `N90_boundary_and_fit.json` and `surface_cells.csv`.

## Recommended next gate

Do not enlarge the full grid or claim a time-space law. The first useful next
experiment is a versioned multi-seed boundary confirmation restricted to
N={1,2,3,4,5} and the existing eight T values, with special attention to the
two reversals. Counterbalanced operation-to-window assignments should be a
separate factor. Only after boundary stability is established should selected
points receive shared-d0, fixed-oracle and independent-spatial controls.
