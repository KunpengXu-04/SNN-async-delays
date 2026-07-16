# Delay parameter recovery — Level 0A v1

## Purpose

This is a deterministic unit experiment on the production sigmoid delay
parameterization. It does not train an SNN and it does not ask a model to copy
an oracle. Its only question is whether one trainable delay parameter can move
to a declared numerical target under a fixed optimizer budget.

The experiment separates two failure modes that are currently confounded:

1. the raw sigmoid parameter or optimizer budget cannot move a delay far enough;
2. the parameter can move directly, but task loss cannot provide useful temporal
   credit through the spike buffer and LIF dynamics.

Level 0A tests only the first. Level 0B will be required for the second.

## Parameter under test

The experiment instantiates the actual `DelayedSynapticLayer(1, 1)` used by the
project and optimizes its single `delay_raw` parameter:

\[
d = 8\,\sigma(d_{\rm raw}).
\]

The direct diagnostic loss is

\[
\mathcal L_{0A}=\frac{1}{2}(d-d^*)^2.
\]

There are no input spikes, weights, LIF neurons, readout, labels or accuracy.
The buffer convention's effective `d+1` shift is recorded but is not exercised.

## Locked matrix

- initial raw values: `{-4,-2,0,2,4}`;
- interior targets: `{1,5,7}` nominal steps;
- boundary stress targets: `{0,8}` nominal steps;
- learning rates: `{0.001,0.01,0.05}`;
- optimizer: Adam, default betas/epsilon, no weight decay;
- optimizer steps: `200`;
- total cells: `5 x 5 x 3 = 75`;
- stochastic seeds: none, because the scalar delay trajectory is deterministic.

The `lr=.001`, `init_raw=-2`, `target=5` cell is the declared analogue of the
current low-initialization/long-routing recipe. The two larger learning rates
are preregistered diagnostic rescues, not post-hoc choices.

## Gates

A cell recovers its target when final absolute error is at most `.1` nominal
step. The current-recipe gate applies only to its declared cell.

Interior numerical recoverability passes only if every interior target and
initialization combination is recovered by at least one preregistered learning
rate. Targets `0` and `8` are descriptive boundary stresses because a finite
sigmoid raw value cannot equal either boundary exactly.

Passing Level 0A would not validate WAD, XOR routing, the oracle schedule or a
learning-rate change in the task model. It would only permit Level 0B: learning
arrival timing through the production buffer and then through an LIF neuron.

## Artifact rule

Each cell writes its config, metrics, final scalar state, NPZ trajectory and
diagnostic panel during execution. Aggregate error/convergence heatmaps and a
machine-readable decision are written only after all 75 cells are complete.
No per-cell result may be silently omitted.
