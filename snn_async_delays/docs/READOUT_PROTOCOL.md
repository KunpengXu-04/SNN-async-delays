# Readout interface protocol (v0.1)

The observation interface is part of the experimental treatment.  It must be
declared independently of the decoder (`linear`, `mlp`, or spiking output).

## Implemented observation modes

### `late_window`

Accumulate last-hidden-layer spikes only during
`[win_len, win_len + read_len)`.  This exactly preserves the historical Plan-D
implementation and checkpoint shapes.

Scientific role: delayed-alignment diagnostic.  It is not a neutral baseline,
because spikes occurring before the final window are censored.  A d0 failure
under this interface does not establish that no-delay temporal computation is
generally impossible.

### `all_time`

Accumulate last-hidden-layer spikes over the full trial `[0, T)`.  Decoder
feature dimension remains `n_last_hidden`, so late/all-time linear decoders
have equal parameter counts.

Scientific role: uncensored count-based control.  It removes the observation
window asymmetry but still discards precise spike time.  To answer K outputs
from one aggregate vector, the hidden population must spatialize query
information.

### `time_binned`

For sequential Plan D, expose K equal input-subwindow count vectors plus one
final-read-window count vector.  The decoder input dimension is

`(K + 1) * n_last_hidden`.

Scientific role: temporally informed upper/control condition.  It is not
resource matched to `late_window` or `all_time`; decoder parameters and
operations grow with K and must be reported.  `win_len` must be divisible by K.

## Decoder rules

Linear and MLP decoders may be compared only when their parameter and operation
counts are reported.  A frozen probe experiment and an end-to-end performance
experiment answer different questions and must not be combined.

Historical direct spiking output is currently exploratory only.  It uses a
non-negative spike count directly as a BCE logit, so a negative class cannot
produce a negative logit.  A confirmatory spiking decoder requires either:

- an opponent pair per query with `logit = count_positive - count_negative`;
- or a predeclared spike-count decision rule and compatible loss.

Direct spiking output supports `late_window` and `all_time`; `time_binned` is
rejected until an output-bin decision rule is defined.

## Required comparisons

The minimum readout-interface experiment is a factorial comparison of
`delay condition × observation mode × decoder`, holding the logical dataset,
encoding, training budget, seeds, and declared resource quantities fixed.

Do not evaluate a checkpoint trained for one observation mode under another
mode and interpret the result as a fair model comparison.  A frozen-checkpoint
cross-mode pass is only a sensitivity diagnostic; formal performance requires
retraining each condition under the declared interface.

## Required output metadata

Every result must record:

- `observation_mode`;
- `observation_steps`;
- `observation_bins`;
- `readout_feature_dim`;
- decoder type and trainable parameter count;
- output-neuron and output-event counts when a spiking decoder is used.
# Simultaneous readout amendment (2026-07-13)

Spatial control may use an all-time dedicated linear head because no temporal
output claim is made. Its MLP and six-neuron opponent output are secondary and
task-native endpoints respectively. Temporal-routing evidence forbids dedicated
query heads: one linear/MLP decoder must be reused over three output windows, or
one opponent pair must be reused in time. Ordinary all-time and final late-window
decoders collapse the routing variable and are diagnostic ablations only.

The opponent-output threshold is distinct from the hidden threshold. Calibration
v1 failed because its temporal d0 arm had no causal event route into post-input
windows. Version 2 reuses the valid spatial d0 evidence and uses a fixed timing
scaffold solely to expose the temporal opponent pair to events. WAD outcomes are
unavailable to selection. The scaffold is not an experimental delay baseline.

For spiking-output calibration, a flat count accuracy during early epochs is
not sufficient to diagnose missing learning: discrete predictions change only
after threshold crossings. Logs must separate peak-voltage curriculum loss from
spike-count loss, and panels must show the output membrane against the threshold.
All reported task metrics nevertheless use signed spike counts only.

Calibration v2 selected output threshold 0.2 by its preregistered activity
rule. This does not establish temporal readout adequacy: XOR and NOR remained
at .5 balanced accuracy across all three temporal seeds. Any temporal pilot
must retain this threshold, use matched output training across delay conditions,
and pass a preflight before the full matrix runs.

## Phase-0 target-spike amendment (2026-07-14)

Phase 0 uses two opponent output neurons per binary query and one target spike
in the correct-class neuron. Both classes therefore have exactly one target
event. Stage A places targets at a fixed, causally reachable offset in one
output window; temporal Stage B assigns query `k` to the same offset in window
`k`. An arbitrary late centre target is forbidden when a feedforward d0
baseline cannot wait until that time. A causal
exponential filter (`tau=5` steps) supplies a spike-train loss, while reported
predictions remain signed spike-count decisions. Target timing is an explicit
supervision signal, not neutral observation, and requires timing-hit/error,
silent/tie/collision and later window-permutation controls.
