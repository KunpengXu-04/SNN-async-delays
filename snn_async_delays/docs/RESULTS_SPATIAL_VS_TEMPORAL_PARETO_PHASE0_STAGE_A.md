# Results: spatial versus temporal Pareto Phase 0, Stage A

## Formal decision

Stage A **failed its preregistered selection rule**. All 15 cells and all
required artifacts are present, but no hidden size passes every gate in all
three seeds. Consequently:

- `selected_baseline_hidden_per_query = null`;
- `passing_hidden_sizes = []`;
- Stage B remains locked;
- no spatial-versus-temporal Pareto or delay advantage claim was tested;
- the test split remains sealed.

This Stage-A spatial baseline used d0 on **both** input-to-hidden and
hidden-to-output synapses. The stored `delay_placement=input_to_hidden_only`
field was an inaccurate metadata carry-over from the planned Stage-B
intervention locus; it did not make Stage-A input delays trainable. The actual
controls `train_mode=weights_only`, `fixed_delay_value=0`, and
`output_delay_mode=d0`, as well as both diagnostic delay heatmaps, confirm the
all-d0 implementation.

Six of 15 individual cells pass all locked gates, but the rule operates on a
hidden size across all seeds, not on cherry-picked cells.

## Formal results

| hidden | balanced accuracy by seed 107/211/509 | timing hit by seed | main interface failure |
|---:|---|---|---|
| 4 | `.749 / 1.000 / .746` | `.258 / .713 / .484` | severe silence; no seed passes the interface gates |
| 8 | `.751 / .743 / 1.000` | `.742 / .752 / 1.000` | two seeds are silent on one positive XOR pattern |
| 12 | `1.000 / .743 / 1.000` | `1.000 / 1.000 / 1.000` | seed 211 emits both opponent neurons on 53.5% of trials |
| 16 | `.751 / .757 / 1.000` | `.742 / .766 / 1.000` | seeds 107/211 are silent on one positive XOR pattern |
| 24 | `1.000 / .757 / 1.000` | `1.000 / 1.000 / 1.000` | seed 211 has 23.4% opponent collisions |

Per-cell values and explicit gate booleans are in
`docs/generated/spatial_vs_temporal_pareto_phase0/stage_a/cells.csv`.
The machine-readable decision is `decision.json` in the same directory.

## Diagnosis

This is not a simple insufficient-width result. Accuracy is non-monotonic in
width and the same seed can succeed at one width and fail at another. Increasing
from 12 to 24 hidden neurons does not remove the seed-211 collision mode.

Every failed exhaustive classification is one of the two asymmetric positive
XOR patterns, `(A=0,B=1)` or `(A=1,B=0)`, and its signed opponent count is
exactly zero. The model generally does not select the wrong class with a
negative margin. It instead reaches a tie through one of two failure modes:

1. **silence:** neither opponent neuron spikes;
2. **collision:** both opponent neurons spike within the observed trial.

The decoder uses `logit > 0`, so a zero tie defaults to class 0. This explains
why h=4/seed=211 attains balanced accuracy 1.0 while still having a 28.7% silent
rate: silence on a class-0 pattern is counted as a correct classification but
is not a valid one-target-spike output. The preregistered interface gates were
therefore necessary and prevented a misleading accuracy-only pass.

Conditional timing error is zero in every cell. Timing support itself is not
the primary bottleneck: when a correct-class spike exists it is at the declared
time. The unstable quantities are spike existence and opponent exclusivity.
For h=12/24 seed 211, timing hit is 1.0 despite classification failure because
the correct neuron fires at `t=11` but the incorrect opponent has already fired
at `t=10`. The subsequent read-only checkpoint audit established this timing;
the collision is not simultaneous target-time firing.

Training is also not stable under the frozen final-checkpoint rule. Eleven of
15 cells reached validation accuracy 1.0 at least once, yet several later
returned to approximately .75. This does not authorize retrospective best-epoch
selection; it identifies threshold-crossing/checkpoint stability as a problem
for a new protocol.

## Interpretation limits

The result contradicts the operational assumption that the current
equal-event input plus opponent target-spike recipe is a seed-robust baseline
interface. It does **not** show that spiking opponent outputs are impossible,
that XOR requires more than 24 hidden neurons, or that temporal multiplexing
fails. The formal experiment never reached the K=2 comparison needed to test
those claims.

The single pre-formal h=24/seed=107 smoke was not representative of
initialization robustness. Its use was legitimate for detecting structural
silence before launch, but it was insufficient evidence for a generally stable
output threshold.

## Next gate

Do not choose h=12 or h=24 from their means and do not launch Stage B. The
read-only mechanism audit is now complete; see
`docs/RESULTS_SPATIAL_VS_TEMPORAL_PARETO_PHASE0_STAGE_A_CHECKPOINT_AUDIT.md`.
Next preregister a small versioned interface-stability calibration using exact
truth-table-balanced training, explicit global one-target/no-early-spike
supervision, calibration seeds separated from held-out initialization seeds,
and an interface-stability criterion rather than post-hoc best-epoch selection.
Only a seed-robust pass may materialize Stage B.
