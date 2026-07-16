# Delay soft-trace credit — Level 0C v1

**Status:** complete; preregistered gate passed. Selected candidate is
production sigmoid + soft centroid + Adam `.05`. See
`RESULTS_DELAY_SOFT_TRACE_CREDIT_LEVEL0C_V1.md`.

## Scientific question

Level 0B separated a valid production delay buffer from two downstream credit
failures: causal filtered-trace loss sometimes points in the wrong direction,
and the centroid of one hard spike is often flat. Level 0C asks whether the
failure disappears when the training signal is computed from a continuous,
non-silent output trace, and whether any residual failure belongs to the loss
or to the sigmoid parameterization.

This is still a deterministic unit experiment. It contains no XOR labels,
trainable weights, readout or performance metric.

## Matched soft paths

One input spike occurs at `t=2`, with `T=16` and `dmax=8`.

1. `buffer_current` optimizes the current emitted by the real circular
   `DelayedSynapticLayer` with fixed weight 1.
2. `lif_membrane` sends that current into one production LIF neuron and
   optimizes its membrane trace. Weight 1 gives an integer-event voltage jump
   of `1-exp(-1/10)≈.0952`, below the fixed `.2 a.u.` threshold. Any hard spike
   therefore invalidates the cell instead of being silently used.

For each declared target delay `{1,5,7}`, the target current is a detached
one-hot trace at arrival `{4,8,10}`. The membrane target is the detached
response of the same subthreshold production LIF dynamics. Neither target nor
loss reads the trainable delay.

## Objectives

The preregistered negative control repeats raw causal filtered-trace MSE with
filter time constant 3.

The first candidate is a global soft centroid:

\[
\mu(z)=\frac{\sum_t t z_t}{\sum_t z_t+\epsilon},\qquad
L_{centroid}=\frac12(\mu(z)-\mu(y))^2.
\]

The second candidate normalizes traces into temporal mass distributions and
maximizes their global symmetric Laplace-kernel similarity:

\[
p_t=\frac{z_t}{\sum_s z_s+\epsilon},\quad
q_t=\frac{y_t}{\sum_s y_s+\epsilon},\quad
K_{ij}=e^{-|i-j|/4},\quad
L_{kernel}=-\log\left(\sum_{ij}p_iK_{ij}q_j+\epsilon\right).
\]

Unlike causal trace overlap, both candidates expose a global temporal order.
The objective-independent endpoint is normalized temporal Wasserstein-1:

\[
W_1(z,y)=\sum_t |CDF_p(t)-CDF_q(t)|.
\]

A cell recovers when final `W1 <= .1` step, trace mass is at least `.05`, and
the path validity rule holds.

## Parameterization control

The production `sigmoid` delay is compared with the existing `direct`
parameterization. Initial forward delays are exactly matched. Labels
`{-4,-2,0,2,4}` define functional values `8*sigmoid(label)`; the direct arm is
initialized at those values rather than at the labels. Thus initial traces are
identical and differences after optimization cannot be attributed to an easier
starting location.

Direct success alone does not pass Level 0C. It only identifies a sigmoid/raw-
coordinate limitation.

## Locked matrix and decision

- parameterizations: `{sigmoid,direct}`;
- paths: `{buffer_current,lif_membrane}`;
- objectives: `{causal_filtered_trace,soft_centroid,symmetric_kernel_alignment}`;
- target delays: `{1,5,7}`;
- matched initialization labels: `{-4,-2,0,2,4}`;
- Adam learning rates: `{.01,.05}`;
- 200 updates;
- `2×2×3×3×5×2 = 360` deterministic cells.

For objective selection, pair-specific LR mixing is prohibited. A production
candidate passes only if one common LR recovers all 30 sigmoid cells across
both paths and every initially misaligned pair has a correct, nonzero initial
gradient direction. The causal negative control is not eligible.

If both candidates pass, `symmetric_kernel_alignment` is selected because it
uses the full temporal mass distribution rather than only its first moment.
Within the selected objective, the lower common passing LR is selected.

## Pre-run amendment

Before any smoke or formal cell, the implementation unit test showed that the
original squared-CDF candidate has an exact zero gradient when the current
delay is an integer and the target is earlier. At such a point, the production
floor/ceil interpolation exposes only the right-hand mixture and the squared
CDF discrepancy is locally flat. This violates the preregistered directional
gate by construction. The candidate was therefore replaced before data
generation by the global Laplace-kernel objective above. The independent W1
endpoint, matrix, paths, LRs, targets and decision tolerance are unchanged.

Passing Level 0C authorizes only Level 0D: a hard-spike forward interface whose
delay is trained with the selected soft auxiliary. It does not authorize XOR,
pairwise WAD or an accuracy/resource claim.

## Required diagnostics

Every cell writes config, metrics, scalar state, NPZ and a runtime six-panel
diagnostic showing delay, soft coordinate, W1 endpoint, objective, raw gradient
and initial/final/target traces. Aggregation reports fixed-LR coverage,
objective/parameterization heatmaps and initial-gradient direction; failures
remain visible.

## Frozen outcome

All 360 cells completed. Sigmoid soft centroid at one common LR `.05` recovers
30/30 cells across current and membrane paths with 13/13 correct nonzero
directions on each, so Level 0C passes and Level 0D is authorized. Symmetric
kernel alignment fails all membrane cells, and direct centroid recovers 11/15
per path within the locked update budget. XOR remains unauthorized.
