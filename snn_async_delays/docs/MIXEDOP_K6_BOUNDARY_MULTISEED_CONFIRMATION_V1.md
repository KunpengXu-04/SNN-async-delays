# K=6 boundary-region multi-seed confirmation v1

> Status: complete and analyzed. The binding result is in
> `RESULTS_MIXEDOP_K6_BOUNDARY_MULTISEED_CONFIRMATION_V1.md`. Formal reruns,
> extra seeds and sealed-test evaluation are not authorized.

## Purpose

The parent single-seed explicitly centroid-supervised surface produced a raw
N90 curve `[3,3,2,2,3,4,3,3]` and two higher-width reversals:

1. T=34: N=3 passed while N=4 failed;
2. T=46: N=2 passed while N=3 failed.

This protocol tests whether those reversals and the broader non-monotonic
boundary reproduce across five fresh full-pipeline seeds. It does not change
the model, delay teacher, loss, optimizer, decoder, dataset size, checkpoint
rule or mechanism gate.

## Frozen design

- K=6 fixed operations: AND, OR, XOR, XNOR, NAND, NOR.
- N hidden: `[1,2,3,4,5]`.
- Window lengths: `[4,5,6,8,10,14,20,26]`.
- T: `[34,40,46,58,70,94,130,166]`.
- Fresh seeds: `[4013,4027,4049,4061,4079]`.
- Total: `5 seeds x 5 widths x 8 T = 200 cells`.
- Parent seed 3907 and preflight seed 3899 are excluded from confirmation
  statistics. The parent boundary is displayed only as an external reference.
- Sealed test remains closed.

Each seed controls initialization, train/validation dataset generation and
stochastic rate encoding. This confirms the stability of the complete pipeline,
not an optimizer-only effect with data held fixed.

The inherited recipe remains six query-tied delays, explicit
`arrival_centroid_huber` delay-only credit, macro-balanced BCE for
weights/readout, LR=.01, clip=1, threshold=.2 arbitrary units, 600 updates,
validation every 25 updates, and a windowed shared MLP.

## Frozen endpoints

A cell passes only if:

- worst-query balanced accuracy >=.90;
- maximum centroid error <=.5 step;
- every output-window activity fraction >=.10.

For each seed and T, raw N90 is the smallest observed passing N. No passing
width is reported as `>5`; it is not imputed as six for trend tests.

Two consensus boundaries are reported:

- robust N90: smallest N passed by at least 4/5 fresh seeds;
- strict N90: smallest N passed by all 5/5 fresh seeds.

No isotonic repair, smoothing or post-hoc threshold adjustment is allowed.

## Preregistered decisions

A registered reversal is considered reproduced if it occurs in at least 2/5
fresh seeds. Exact seed flags and Wilson intervals are reported; the interval
is descriptive because five seeds are few.

A seed curve is non-monotonic if at least one adjacent observed N90 increases
as T increases. Non-monotonicity is confirmed if at least 3/5 fresh seed curves
are non-monotonic. Adjacent comparisons involving `>5` censoring are recorded
as unknown rather than silently converted to a number. The robust 4/5 boundary
is assessed separately.

Boundary stability requires robust N90 at all eight T and a fresh-seed N90
range <=1 at every T. Exact reproduction of the parent boundary requires the
robust curve to match all eight parent values; partial matches are reported
without relabelling the decision.

## Outputs and scientific boundary

Every cell writes the same runtime checkpoint, logs, NPZ, six-query diagnostic
panel and resource ledger as the parent. Aggregate outputs include cell and
per-seed tables, mean BAcc and pass-fraction planes, five raw N90 curves,
robust/strict/parent comparison, reversal counts and boundary variability.

Even a stable boundary remains explicitly supervised, validation-selected,
fixed-position and MLP-decoded. It cannot establish autonomous WAD, a pure T
effect, output-spike timing, spatial Pareto superiority or hardware energy.

## Execution

The exact 200-cell confirmation is authorized without an additional smoke
because the cell runner and artifacts are unchanged and already passed the
parent preflight and 112-cell surface.

```powershell
python -m scripts.run_mixedop_k6_boundary_multiseed_confirmation --device cuda
```

Do not add seeds, alter the grid or rerun failed cells with changed settings.
