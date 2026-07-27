# Spatial/temporal capacity scaling and SLAYER v1

## Scientific separation

This protocol separates two questions that previous surfaces mixed together.
The fixed-oracle and explicitly centroid-supervised arms measure architectural
capacity when a temporal schedule is supplied.  The task-only SLAYER arm tests
whether a different training system can discover delay-dependent routing
without that teacher.  A positive teacher-assisted result is never promoted to
autonomous delay learning.

The completed K=6 centroid surface and its fresh-seed confirmation close the
old uncounterbalanced `N_hid x T` branch: its boundary is unstable and
non-monotone, so it must not be enlarged, smoothed or fitted as a universal
time--width law.  Historical Level 0/1 runs and all result artifacts remain
immutable.

## Workload and interface

The primary workload repeats XOR at every query position with
`K=[1,2,3,4,6,8]`.  This makes K a query-load variable instead of changing the
Boolean operation at the same time.  Mixed-operation position effects are a
separate counterbalanced study.

Input is deterministic `binary_one_hot_packet`: each query owns
`[A0,A1,B0,B1]`; the selected A and B channels fire at steps 6--9.  Every trial
therefore contains exactly `8K` input events.  The primary output remains one
shared MLP applied to hidden spike counts in each output window.  A shared
opponent spiking output is conditional on the K=2 SLAYER gate and cannot be
pooled with the MLP endpoint.

## Gate order

1. **S0:** invalid smoke verifies tensor layout, bidirectional delay credit,
   actual event shift, checkpoint restoration and CPU/CUDA execution.
2. **S1:** K=1 XOR compares d0, correct fixed delay, current task-only,
   explicit-centroid positive control and task-only SLAYER.  SLAYER receives no
   centroid loss or schedule-eligible checkpoint selection.
3. **S2:** K=2 repeated XOR tests a shared hidden population and shared decoder.
   Learned routing must pass reliability, activity and delay-intervention
   gates.  Failure closes the learned-delay scaling arm.
4. **Scaling:** fixed-oracle capacity always remains eligible as an
   architectural upper bound.  Task-only SLAYER enters only after S2 passes.

The machine-readable grid, seeds, thresholds, fits and authorization state are
frozen in `configs/spatial_temporal_capacity_slayer_v1.yaml`.

## Scaling estimands

Spatial scaling holds total latency at 58 steps (`B=48`) and measures
`C_S(N)`, the largest reliable K at each total hidden width.  Temporal scaling
holds one validation-selected `N_ref` for every method and measures `C_T(B)`
at `B=[24,48,72,96]`.  Reliability requires both worst-query balanced
accuracy and exact-trial accuracy of at least .90.

Power-law exponents are secondary summaries.  A superlinear label requires at
least four uncensored capacity points and a 95% bootstrap interval whose lower
bound exceeds one.  Raw staircases, inverse required-resource curves and every
censored value remain visible; isotonic repair and censor imputation are
forbidden.

## Resource and claim discipline

Every result reports latency, dimensions, parameter/storage counts, axonal
delay values, delay buffer, neuron updates, dense MACs, measured events and
decoder operations.  `N*T` and spike count remain descriptive and are never
called energy.  The sealed test stays closed until the SLAYER recipe, all grids
and every decision gate are frozen.

## Current execution state (2026-07-21)

S0 is complete for invalid smoke seeds 99501 and 99502. Both pass the declared
arrival, gradient-direction, conversion, checkpoint and full-model CPU/CUDA
checks. The decision artifact is
`docs/generated/spatial_temporal_capacity_slayer_v1/s0_decision.json`.

S1 calibration is authorized by the S0 result but remains explicitly locked by
`authorization.s1_calibration_launch: false`. The 30 registered calibration
cells can be inspected with `--dry-run`; no calibration or later scientific
cell has been launched and sealed test remains closed.

The execution layer now covers both backends under the same artifact schema.
Current d0, fixed, task-only and explicit-centroid cells use compact per-input-
axon delay storage; the independent spatial d0 control uses the same frozen
granularity. S1/S2 decision scripts, K=6 fixed-oracle N_ref calibration,
spatial/temporal aggregation and mixed-operation counterbalance manifests are
implemented but remain gated.
