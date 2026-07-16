# Delay hard-output / soft-credit bridge — Level 0D v1

**Status:** completed; preregistered gate passed (135/135 cells).

## Scientific question

Level 0C established bidirectional delay credit only while the optimized trace
remained continuous and subthreshold. Level 0D restores the suprathreshold LIF
forward path from Level 0B. Training may use a continuous auxiliary, but the
primary endpoint is now exclusively the final hard spike train.

This experiment asks both whether a soft auxiliary can place a hard spike and
whether that auxiliary remains effective when combined with the task-adjacent
hard filtered-spike loss. Soft-only success is insufficient for the formal
gate.

## Forward dynamics and target

One input spike occurs at `t=2`, `T=16`, `dmax=8`. The actual circular
`DelayedSynapticLayer` has fixed weight 4 and one trainable sigmoid delay. The
production LIF uses `tau=10`, threshold `.2 a.u.`, reset 0, refractory 2 and
surrogate beta 4. Target nominal delays `{1,5,7}` correspond to one-hot hard
target spikes at `{4,8,10}`.

The evaluated output must contain exactly one spike. With one spike, temporal
Wasserstein-1 is simply its absolute arrival-time error. A cell passes only at
final step when hard-spike `W1<=.1`; transient crossings do not count.

## Continuous auxiliary sources

`synaptic_current` uses the Level-0C centroid of the delayed current trace.

`pre_reset_membrane` uses

\[
v_t^{pre}=\alpha v_{t-1}^{post}+(1-\alpha)I_t\,1[r_t=0]
\]

before thresholding and reset. This is the same pre-reset quantity already
recorded by the production spiking-output model. Post-reset voltage is
prohibited because it becomes zero precisely at the threshold-crossing event.

For source `s`, the auxiliary is

\[
L_{aux}^{(s)}=\frac12\left(\mu(z^{(s)})-\mu(y^{(s)})\right)^2.
\]

The hard loss repeats Level 0B's causal filtered-spike MSE with `tau=3`. The
combined loss is

\[
L=L_{hard}+\lambda L_{aux},\qquad
\lambda\in\{.01,.1,1\}.
\]

## Locked conditions and matrix

Nine conditions are evaluated:

- hard filtered only;
- current centroid only;
- pre-reset centroid only;
- hard filtered plus current centroid at each of `.01/.1/1`;
- hard filtered plus pre-reset centroid at each of `.01/.1/1`.

All use production sigmoid delays, Adam `.05`, 200 updates, targets `{1,5,7}`
and raw initializations `{-4,-2,0,2,4}`: `9×3×5=135` deterministic cells.

## Decision gate

Every initially misaligned pair must receive a correct nonzero initial total
gradient, and all 15 target/init cells must end with exactly one hard spike at
the target time. Level 0D passes only if a **combined hard-plus-auxiliary**
condition satisfies both requirements. Soft-only controls cannot pass the
formal gate.

If several combined conditions pass, select pre-reset membrane before current,
then the smallest passing auxiliary weight. This favors the output-proximal
signal and minimizes auxiliary domination.

A pass authorizes only preregistration of a versioned K=1 XOR calibration. It
does not authorize an XOR result claim, pairwise WAD, per-neuron tying, K
scaling or a Pareto surface.

## Diagnostics

Each cell writes config, metrics, scalar state, NPZ and an eight-panel runtime
diagnostic: delay, hard arrival, hard W1, spike count, component losses,
component gradients, hard spike traces and the selected soft trace. Aggregate
outputs include per-condition hard-error heatmaps, recovery, gradient direction
and gradient-conflict summaries.

## Locked outcome

The selected compatibility condition is hard filtered-spike loss plus
synaptic-current centroid with `lambda=.1`. It recovers 15/15 cells and has
13/13 correct nonzero initial directions among initially misaligned pairs.
The `.01` current condition recovers only 10/15; `lambda=1` also passes but is
not selected under the smallest-passing-weight rule.

Pre-reset membrane credit fails its hypothesis: soft-only recovery is 3/15
with eight zero-gradient pairs, and no combined pre-reset condition exceeds
10/15. See `RESULTS_DELAY_HARD_OUTPUT_SOFT_CREDIT_LEVEL0D_V1.md` for the
mechanism diagnosis and claim limits.
