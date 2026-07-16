# Results: counterbalanced temporal performance v1

## Scope and integrity

All 45 preregistered validation cells completed: five delay conditions, three
cyclic operation orders and paired seeds `{7,19,73}`. Every cell contains both
checkpoints, a 100-epoch log, validation and exhaustive truth-table results,
resource ledger, diagnostic NPZ and an in-run diagnostic panel. No cell was
excluded and the test split remains sealed.

This is an exploratory validation decision, not a confirmatory test. The
primary estimand and thresholds below were fixed before results were inspected.

## Primary result: no condition demonstrates reliable routing

The primary score is the minimum balanced accuracy over all nine
operation-by-position cells for each condition and seed. The routing gate
requires every seed to reach `.55` and their mean to reach `.60`.

| Condition | Seed 7 | Seed 19 | Seed 73 | Mean | Gate |
|---|---:|---:|---:|---:|---|
| d0 | .500 | .500 | .500 | .500 | fail |
| scalar | .500 | .500 | .500 | .500 | fail |
| fixed matched `[1,9]` | .500 | .500 | .500 | .500 | fail |
| fixed full support `[0,30]` | .500 | .486 | .482 | .489 | fail |
| WAD | .500 | .500 | .500 | .500 | fail |

Therefore neither WAD nor any non-learned control solves the declared
counterbalanced shared-output routing problem. WAD superiority is also false:
its paired primary difference is zero against d0, scalar and fixed-matched in
all three seeds. Those three controls tie on the primary score; reporting d0
alone as the "strongest" control would be an arbitrary list-order artefact.

The fixed-full values slightly below `.5` arise from empirical validation
balanced accuracy and do not indicate meaningful anti-learning. They do not
alter the gate decision.

## Secondary result: WAD is best on average, but not reliable

Mean operation-position balanced accuracy is `.500/.613/.628/.656/.734` for
d0/scalar/fixed-matched/fixed-full/WAD. Mean exact-trial accuracy is
`.094/.203/.192/.326/.411`. Relative to fixed-full, WAD gains `.0777` mean
balanced accuracy and `.0847` exact-trial accuracy, with positive differences
in all three seeds. WAD also has the highest measured synaptic-event count
(`488.1` per trial versus `482.8` for fixed-full and `472.7` for scalar).

This average advantage is scientifically interesting but cannot override the
preregistered minimum-cell endpoint. It says WAD distributes useful capacity
better on average in this implementation; it does not say WAD routes every
operation at every temporal position.

## Counterbalancing diagnosis

The seed-mean operation-by-position table changes the interpretation of the
fixed-order preflight:

- WAD position means fall sharply from `.920` to `.722` to `.559`.
- WAD learns NOR when it occurs early (`.78` at position 0, `.71` at position
  1) but NOR remains `.50` at position 2.
- WAD also leaves XOR at `.50` in position 2, while NAND reaches `.68` there.
- Thus the old NOR/window-2 failure is not explained by NOR identity alone.
  Temporal position is the dominant limitation, with a genuine
  operation-by-position interaction that makes NAND easier late.
- Scalar and narrow fixed delays are strong early but collapse to `.50` for
  every operation at position 2.
- Full-support fixed delays show the reverse tendency: position means are
  `.519/.666/.783`. They can learn late XOR/NAND/NOR but are weak at position
  0. This proves late routing is executable, while exposing a timing-support
  allocation trade-off rather than a universal best delay bank.

The mechanism traces agree with this diagnosis. In WAD, late XOR has zero
output-current support in every seed. Late NOR often receives output current
but remains non-discriminative, so event arrival is necessary but not
sufficient. WAD therefore combines a support failure for some late routes with
a class-separation failure for others.

## Scientific decision

The study rules out three tempting claims:

1. the fixed-order failure was merely a NOR-specific task difficulty;
2. mechanism viability implies routing performance;
3. WAD currently provides reliable or resource-superior temporal routing.

It supports a narrower descriptive statement: learned delays improve average
counterbalanced performance over the tested fixed banks, but the frozen method
does not guarantee temporal coverage or late-window discriminability. The
current architecture/objective appears to allocate timing capacity
asymmetrically rather than solve all routes.

Do not open the test split, run a K/N/T scaling surface, or promote the average
WAD advantage to a headline claim. The next gate should be a method-design
decision: either preregister a small coverage-constrained/stratified-delay
intervention against fixed-full support, or explicitly pivot the paper toward
the negative methodology result about support, position and worst-cell
evaluation. That decision must precede further training.

## Artifacts

- `docs/generated/simultaneous_temporal_counterbalanced_performance_v1/cells.csv`
- `docs/generated/simultaneous_temporal_counterbalanced_performance_v1/condition_seed_primary.csv`
- `docs/generated/simultaneous_temporal_counterbalanced_performance_v1/condition_summary.csv`
- `docs/generated/simultaneous_temporal_counterbalanced_performance_v1/marginal_effects.csv`
- `docs/generated/simultaneous_temporal_counterbalanced_performance_v1/paired_wad_minus_controls.csv`
- `docs/generated/simultaneous_temporal_counterbalanced_performance_v1/operation_position_heatmaps.png`
- `docs/generated/simultaneous_temporal_counterbalanced_performance_v1/decision.json`
