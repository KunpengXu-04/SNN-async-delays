# Delay temporal credit — Level 0B v1

**Status:** complete; preregistered strict gate failed. See
`RESULTS_DELAY_TEMPORAL_CREDIT_LEVEL0B_V1.md`.

## Purpose

Level 0A showed that the production sigmoid delay can recover an interior
numerical target under direct delay supervision, but that the current
`.001/200-step` budget is too weak for long movement. Level 0B removes direct
delay supervision and asks whether a loss computed from an output time trace
can move the delay through the actual simulator.

This is still not XOR and is not oracle imitation. The declared target is an
arrival time, not a desired delay parameter.

## Two forward paths

One input spike is emitted at `t=2`, with `dmax=8` and a 16-step trace.

1. `buffer_current`: the actual circular `DelayedSynapticLayer` emits its
   delayed synaptic-current trace. Its fixed weight is 1.
2. `lif_spike`: the same buffer drives one actual `LIFNeurons` unit through a
   fixed weight of 4. The LIF uses `tau=10`, threshold `.2 a.u.`, reset 0,
   refractory 2 and surrogate beta 4. An integer-delay event must produce one
   suprathreshold output spike.

The simulator reads before writing the input buffer, so a nominal delay `d`
has effective shift `d+1`. Target nominal delays `{1,5,7}` therefore correspond
to target arrival times `{4,8,10}`.

## Two timing objectives

`arrival_centroid` uses only the output trace:

\[
\hat t=\frac{\sum_t t\,z_t}{\sum_t z_t+\epsilon},\qquad
\mathcal L_{centroid}=\frac{1}{2}(\hat t-t^*)^2.
\]

It provides a global timing direction and is an auxiliary mechanism probe, not
an ordinary classification loss.

`filtered_trace` causally filters both output and one-hot target traces with
time constant 3 steps, then computes their mean squared error. This mirrors the
project's filtered spike-timing objective more closely and tests whether a
local trace loss can move across longer temporal gaps.

## Locked matrix and gates

- paths: `{buffer_current,lif_spike}`;
- objectives: `{arrival_centroid,filtered_trace}`;
- targets: `{1,5,7}` nominal steps;
- raw initializations: `{-4,-2,0,2,4}`;
- Adam learning rates: `{.001,.01,.05}`;
- 200 optimizer steps;
- 180 deterministic cells; no seeds.

A cell functionally recovers when its final output arrival centroid is within
`.1` step of target and output trace mass is at least `.5`. For each path/loss,
every target/initialization pair must be recovered by at least one preregistered
learning rate. The strict Level-0B gate requires all four path/loss combinations
to pass.

Passing centroid alone would establish only global timing credit. Passing the
filtered loss is required before using a timing objective in a later task
protocol. No Level-0B result authorizes an XOR learning-rate change.

## Required diagnostics

Every cell saves during execution: delay, output-centroid, loss, gradient,
trace mass, initial/final output traces, target trace, NPZ and a six-panel
diagnostic plot. Aggregation reports all cells and keeps centroid versus
filtered results separate.

## Frozen outcome

All 180 cells completed. Only `buffer_current + arrival_centroid` passes its
15/15 coverage gate; buffer filtered trace, LIF centroid and LIF filtered trace
recover 5/15, 3/15 and 10/15 pairs. The strict gate therefore fails and XOR
remains locked. This outcome does not amend the preregistered rule.
