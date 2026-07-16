# Results: delay temporal credit Level 0B v1

## Decision

The preregistered strict Level-0B gate **fails**. All 180 deterministic cells
completed with their config, metrics, final scalar state, NPZ trace and
runtime-generated diagnostic panel, but only one of the four path/objective
combinations recovered all 15 target/initialization pairs under at least one
preregistered learning rate.

Consequently:

- Level 1 / XOR is not authorized;
- no Level-0B learning rate is authorized for XOR;
- no learned-routing, accuracy or resource-frontier claim is authorized;
- the formal gate is not relaxed after observing that one direction or one
  objective works.

This result is validation-free and label-free: it is a deterministic mechanism
diagnostic, not a task-performance experiment.

## Recovery results

The primary endpoint is final output-arrival error `<= 0.1` step with output
trace mass `>= 0.5`. The table reports target/initialization pairs recovered by
at least one of Adam `{.001,.01,.05}`.

| Forward path | Timing objective | recovered pairs | per-LR recovered cells (`.001/.01/.05`) | gate |
|---|---:|---:|---:|---:|
| buffer current | arrival centroid | 15/15 | 2 / 6 / 15 | pass |
| buffer current | filtered trace | 5/15 | 2 / 5 / 5 | fail |
| hard LIF spike | arrival centroid | 3/15 | 2 / 3 / 3 | fail |
| hard LIF spike | filtered trace | 10/15 | 3 / 5 / 10 | fail |

All final trace masses are 1.0. Therefore the failures are not caused by
silence in this calibrated one-event setup.

The current-recipe analogue (`init_raw=-2`, `lr=.001`, target nominal delay 5)
fails in all four conditions. Final arrival errors are respectively `3.8611`,
`4.1823`, `4.0` and `4.0` steps in the table order. Merely passing the spike
through the production buffer and LIF therefore does not rescue the existing
`.001/200-step` delay recipe.

## Mechanism audit: the objective, not just the optimizer, is limiting credit

For the 13 target/initialization pairs that are not already temporally aligned,
the sign of the initial raw-delay gradient was audited. A sign is correct when
an Adam descent step moves the delay toward the target arrival.

| Forward path | Timing objective | correct | zero | wrong nonzero |
|---|---:|---:|---:|---:|
| buffer current | arrival centroid | 13/13 | 0 | 0 |
| buffer current | filtered trace | 8/13 | 0 | 5/13 |
| hard LIF spike | arrival centroid | 5/13 | 8/13 | 0 |
| hard LIF spike | filtered trace | 8/13 | 0 | 5/13 |

This separates three mechanisms:

1. **The production circular buffer is differentiable enough for long-range
   delay movement.** Buffer-current centroid supervision recovers 15/15, and
   all 13 initially misaligned pairs have the correct gradient sign. Together
   with Level 0A, this rules out a hard scalar-delay or circular-buffer gradient
   implementation bug in this one-synapse regime.
2. **The causal filtered-trace objective has a directional/local credit
   failure.** It recovers only 5/15 before the LIF and starts with the wrong
   gradient sign in 5/13 misaligned pairs. Raising LR from `.01` to `.05` does
   not increase buffer recovery beyond 5/15, so this is not explained by step
   size alone.
3. **A centroid of one hard output spike is a degenerate training signal.** Its
   time is piecewise constant; changing surrogate spike amplitude at the same
   hard time does not move the normalized centroid. Eight of 13 misaligned LIF
   cases therefore start with exactly zero raw-delay gradient. The 3/15
   recovered pairs are not evidence that hard-spike centroid learning is
   generally viable.

The LIF filtered-trace arm is the strongest task-adjacent arm but remains
asymmetric. It recovers 10/15: early spikes can often be pushed later, whereas
late spikes are not reliably pulled earlier. For example, target delay 5 from
`init_raw=-2` at LR `.05` moves the output from `t=4` to the correct `t=8`.
Conversely, target delay 1 from `init_raw=0` starts at `t=7` with gradient
`-0.1662`; the optimizer increases the delay and moves the spike farther away
to `t=11`. This is direct evidence against treating the current causal
filtered loss as a bidirectional delay-routing objective.

## What is and is not established

Supported narrowly:

- the simulator's effective arrival convention is `input step + delay + 1`;
- a loss that exposes a global temporal coordinate can train a scalar sigmoid
  delay through the actual buffer;
- surrogate credit through one LIF exists for some timing directions under a
  filtered-trace objective.

Contradicted:

- that the current `.001/200-step` recipe is sufficient;
- that the present causal filtered-trace loss supplies reliable bidirectional
  long-range credit;
- that hard-spike centroid loss is a viable general timing objective;
- that Level 0B is ready to transfer to XOR or full pairwise WAD.

Not tested:

- task BCE combined with a timing auxiliary;
- multiple synapses, trainable weights, query identities or XOR labels;
- per-neuron versus per-synapse delay tying;
- accuracy, generalization, resource use or a Pareto frontier.

## Next gate

Do **not** add XOR, K, mixed operations or a new `T x n_hid` surface yet. The
next justified experiment is a new preregistered **Level 0C objective and
parameterization audit** at the same one-event scale. It should test whether a
symmetric/global transport signal computed from a soft current or membrane
trace can provide correct gradients in both early-to-late and late-to-early
directions, before crossing one LIF hard spike. Per-neuron delay tying may then
be introduced as a separately controlled reduction in optimization dimension;
it cannot repair a wrong single-delay gradient sign by itself.

Level 0C must retain the initial-gradient sign audit and the same target/init
coverage gate. Only a bidirectional path that passes this gate should advance
to K=1 XOR.

## Artifacts

- machine-readable decision: `docs/generated/delay_temporal_credit_level0b_v1/decision.json`;
- all-cell table: `docs/generated/delay_temporal_credit_level0b_v1/cells.csv`;
- recovery plot: `docs/generated/delay_temporal_credit_level0b_v1/path_loss_recovery_summary.png`;
- initial-gradient audit: `docs/generated/delay_temporal_credit_level0b_v1/initial_gradient_direction_summary.png`;
- four final-error heatmaps in the same generated directory;
- immutable per-cell artifacts: `runs/exploratory/delay_temporal_credit_level0b_v1/`.
