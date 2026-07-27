# Mixed-Operation Rate-WAD Surface Calibration v1

Status: smoke passed and all ten pilot cells completed. The pilot failed the
oracle-feasibility, WAD-performance and routing gates; the 30-cell full surface
is locked and must not run.

## Question

Does a temporally distributed Poisson rate input improve shared-temporal WAD
gradient health, late-window coverage, and worst-query reliability relative to
the failed K=5 binary-one-hot micro-burst experiment?

This is an encoding-rescue calibration, not a continuation of the frozen burst
surface. It keeps the same fixed operations, 4K value channels, model, WAD
parameterization, LR=.01 recipe, 200 updates, loss, checkpoint rule, dataset
patterns and seed. Only the spike-generation process changes.

## Important confound

The selected rate code is not event-count matched. Over the ten-step input
window, each query has two selected channels at 400 Hz and two unselected
channels at 10 Hz, for 8.2 expected input events per query versus exactly four
in the parent burst protocol. A positive result would show that this richer
rate input rescues performance under its measured resource cost; it would not
isolate temporal distribution from event count.

Training rate realizations are resampled every optimizer update. Validation
uses one fixed, preregistered RNG realization across every checkpoint so
checkpoint selection is not driven by changing Monte Carlo noise.

## Expanded search domain and pilot

The possible full WAD surface expands total hidden neurons to
`[20,40,80,120,160,240]` and output-window length to `[2,4,6,8,12]`, giving
T=`[20,30,40,50,70]`. It remains locked.

The first evidence stage contains only the four corners and center:

| point | total hidden | window | T |
|---|---:|---:|---:|
| lowN-lowT | 20 | 2 | 20 |
| lowN-highT | 20 | 12 | 70 |
| highN-lowT | 240 | 2 | 20 |
| highN-highT | 240 | 12 | 70 |
| center | 120 | 6 | 40 |

Each point runs a fixed-oracle feasibility control and WAD, for ten pilot
cells. Every cell saves its checkpoint, fixed validation predictions, input
event counts, runtime NPZ/diagnostic panel, update-level total and per-query
delay gradients, window activity, delay correspondence and resource ledger.

## Decision gates

The full 30-cell WAD surface is authorized only if all three gates pass:

1. technical: complete/finite artifacts, at least 90% nonzero delay-gradient
   updates for every WAD query, and median total delay-gradient norm above
   `1e-8`;
2. oracle feasibility: worst-query balanced accuracy at least .85 at every
   pilot point;
3. WAD performance: center worst-query balanced accuracy at least .60 and at
   least three of five points above .55.

Mechanism evidence has a separate, stricter gate at the center: windows Q2-Q4
must each be active in at least 10% of validation trials, query-mean delay must
have Spearman correlation at least .5 with query index, and its spread must be
at least one output-window length. Passing the performance gate without this
gate permits a descriptive rate-WAD surface but not a routing claim.

The machine-readable source of truth is
[`configs/mixedop_rate_wad_surface_calibration_v1.yaml`](../configs/mixedop_rate_wad_surface_calibration_v1.yaml).

## Smoke decision

Both center smoke cells are complete. Oracle and WAD are finite and active,
the oracle tensor is exact, every WAD query receives nonzero delay gradient,
delays remain legal, clipping is not continuously active in the final 20
updates, and repeated checkpoint evaluation reproduces the same fixed
validation realization. Runtime NPZ files, diagnostic panels and resource
ledgers are present. Accuracy was not used to pass this technical gate.

## Pilot decision

The technical gradient gate passes, but no fixed-oracle point reaches the
registered `.85` worst-query floor (range `.518–.769`). WAD worst-query
balanced accuracy is `.50` at all five points. At the center, every query has
nonzero delay gradient on every update, yet query-mean delays remain overlapped,
the last two output windows have zero hidden activity, and the routing gate
fails. The failure is therefore not literal gradient disappearance; available
gradient credit does not identify or build the required schedule.

Rate coding increases center input-to-hidden events from 2400 to 4917 while
leaving the primary endpoint unchanged relative to the burst parent. See
`RESULTS_MIXEDOP_RATE_WAD_SURFACE_CALIBRATION_V1.md`. No full sweep was run.
