# Results: delay hard-output / soft-credit Level 0D v1

## Formal decision

Level 0D **passes** its preregistered gate. All 135 deterministic cells are
complete, and every cell contains config, metrics, final scalar state, NPZ and
the diagnostic panel generated during the run. The selected bridge is:

```text
hard endpoint:       one suprathreshold production-LIF spike
hard loss:           causal filtered-spike MSE, tau=3
delay auxiliary:     synaptic-current centroid
auxiliary weight:    lambda=0.1
delay optimizer:     sigmoid parameterization, Adam 0.05, 200 updates
```

It recovers all 15 target/initialization cells and gives the correct nonzero
initial total-gradient direction in all 13 initially hard-misaligned pairs.
This authorizes preregistration of a K=1 XOR calibration only. It does not
establish XOR learning, WAD, routing, per-neuron tying or a Pareto advantage.

## Primary results

The locked endpoint requires exactly one final hard spike and hard-spike
temporal W1 `<=0.1` step.

| Condition | recovered | correct / zero / wrong initial directions | gradient conflicts | formal compatibility |
|---|---:|---:|---:|---:|
| hard filtered only | 10/15 | 8 / 0 / 5 | 0 | fail |
| current centroid only | **15/15** | **13 / 0 / 0** | 5 | soft-only control |
| pre-reset centroid only | 3/15 | 5 / 8 / 0 | 2 | soft-only control |
| hard + `.01` current | 10/15 | 9 / 0 / 4 | 5 | fail |
| hard + `.1` current | **15/15** | **13 / 0 / 0** | 5 | **pass; selected** |
| hard + `1` current | **15/15** | **13 / 0 / 0** | 5 | pass |
| hard + `.01` pre-reset | 10/15 | 9 / 0 / 4 | 2 | fail |
| hard + `.1` pre-reset | 10/15 | 10 / 0 / 3 | 2 | fail |
| hard + `1` pre-reset | 10/15 | 10 / 0 / 3 | 2 | fail |

There are two initially hard-aligned pairs in each condition; the direction
counts above therefore cover 13 pairs, not all 15. The selected condition's
median final nominal-delay error is `.000361` step and its median first hard-
arrival crossing is optimizer step 38. The `lambda=1` current candidate also
passes but loses the locked smallest-passing-weight tie-break.

## Mechanism diagnosis

### 1. Hard filtered-spike credit is directionally wrong, not merely weak

The hard-only condition repeats the Level-0B result: 10/15 recovery and five
wrong initial directions. Its five failures all start later than the target:
three target-1 cells initialized at raw `0/2/4`, and two target-5 cells at raw
`2/4`. Rather than moving earlier, the learned delays approach the upper bound
and finish with hard W1 of 7 or 3 steps.

This excludes the benign explanation that more iterations alone would fix the
failure. At those starting points the local objective sends the parameter in
the wrong direction.

### 2. Current centroid supplies the missing continuous coordinate

Current-centroid-only recovers 15/15 and has 13/13 correct directions. More
importantly, the hard and current gradients conflict in exactly five cells—the
same class of cells in which hard-only credit is problematic. At `lambda=.1`,
the auxiliary is large enough to reverse the total direction in all 13
misaligned pairs while retaining the hard-spike endpoint. Both `.1` and `1`
therefore recover 15/15.

At `lambda=.01`, four directions remain wrong and recovery stays 10/15. Thus
"add any small timing auxiliary" is false; the relative gradient scale matters.
The selected `.1` is a diagnostic-scale value fixed by this preregistered
one-synapse experiment, not a transferable XOR hyperparameter.

### 3. Pre-reset voltage is not a useful smooth bridge under hard reset

The pre-reset hypothesis fails. Pre-reset-only recovers 3/15 and has zero
initial delay gradient in 8/13 misaligned pairs. None of its three combined
weights improves beyond the hard-only 10/15 recovery.

Although pre-reset voltage is measured before the instantaneous reset, the
suprathreshold one-event trajectory is still segmented by the discrete spike
and refractory state. At integer-delay configurations its temporal centroid
can be locally flat. Being closer to the output threshold therefore does not
make it a better delay-credit signal. The more distal synaptic-current trace
is scientifically preferable here because it remains continuously sensitive
to fractional delay interpolation.

### 4. The bridge is real but extremely narrow

The evaluated forward output is now a genuine production hard spike, so this
is stronger than Level 0C's subthreshold trace result. However, the auxiliary
directly supervises the first temporal moment of one isolated current event.
There are no labels, trainable weights, competing events, multiple neurons,
truth-table generalization or query identities. Multiple-event traces can have
the same centroid, and task BCE may conflict with the timing auxiliary in ways
not represented by this unit experiment.

## Supported and unsupported conclusions

Supported narrowly:

- a production sigmoid delay can be trained bidirectionally while the final
  endpoint is a correctly timed suprathreshold hard LIF spike;
- current-centroid credit repairs the five wrong-direction cases of the hard
  filtered-spike objective when its locked weight is at least `.1` here;
- pre-reset voltage centroid is a poor auxiliary for this hard-reset path.

Not established:

- that `.1` is appropriate for XOR or any multi-synapse network;
- XOR learning, classification accuracy or held-out generalization;
- per-synapse versus per-neuron delay tying;
- learned temporal multiplexing, K scaling, resource savings or Pareto gains;
- robustness across thresholds, weights, time constants, multiple events or
  stochastic seeds.

## Next gate

Preregister **Level 1A: K=1 XOR calibration** before running it. Keep the
current-centroid timing term as a named auxiliary rather than silently folding
it into task loss. The protocol must specify its scale relative to task loss,
retain a hard-spiking output endpoint, include d0/fixed-delay controls, audit
delay and weight gradients separately, and require complete XOR truth-table
performance. It should remain small enough to distinguish failure of temporal
credit from failure of representation or output training.

Do not jump directly to K>1, pairwise WAD, per-neuron tying or the `n_hid x T`
surface. Those require a successful task-level bridge first.

## Artifacts

- `docs/generated/delay_hard_output_soft_credit_level0d_v1/decision.json`;
- `docs/generated/delay_hard_output_soft_credit_level0d_v1/cells.csv`;
- nine final hard-W1 heatmaps and three aggregate mechanism summaries;
- `runs/exploratory/delay_hard_output_soft_credit_level0d_v1/` with all 135
  cell-level configs, states, NPZ files and runtime diagnostic panels.
