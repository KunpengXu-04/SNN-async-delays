# Results — delay parameter recovery Level 0A v1

## Decision

All 75 preregistered deterministic cells are complete with config, metrics,
final state, NPZ trajectory and in-run diagnostic panel.

The result separates two statements:

1. **Current recipe fails direct long-range scalar recovery.** With
   `init_raw=-2`, `lr=.001`, target delay `5`, and 200 Adam steps, the nominal
   delay moves only from `.9536` to `1.1389`; final error is `3.8611` steps and
   the `.1`-step gate is never reached.
2. **The production sigmoid parameterization is numerically recoverable under
   a stronger direct optimizer.** The preregistered `lr=.05` arm recovers all
   15 interior target/initialization combinations within `.1` step. Its maximum
   final interior error is `.0781` step.

Therefore there is no evidence of a hard `get_delays()` implementation error.
There is strong evidence that the current `.001/200-step` raw-parameter budget
is inadequate for large movement, even under an unrealistically informative
direct delay-target loss.

## Complete matrix summary

| Adam LR | Interior recovered | Mean final interior error | Maximum final interior error |
|---:|---:|---:|---:|
| `.001` | 2/15 | 3.0833 | 6.8236 |
| `.01` | 6/15 | 1.5303 | 5.1399 |
| `.05` | 15/15 | .01988 | .07809 |

The two `.001` interior passes are cells already initialized near their target;
they do not demonstrate long-range recovery. For the declared routing-relevant
`init_raw=-2,target=5` cell:

| Adam LR | Final delay | Final absolute error | First step within `.1` |
|---:|---:|---:|---:|
| `.001` | 1.1389 | 3.8611 | never |
| `.01` | 3.8999 | 1.1001 | never |
| `.05` | 4.9997 | .00028 | 56 |

## Boundary stress

Targets `0` and `8` were descriptive because finite sigmoid raw parameters
cannot equal either boundary. At `lr=.05`, 4/10 boundary cells enter the `.1`
tolerance; the mean final boundary error is `.3426` and the maximum is `.6098`.
This confirms that placing an intended oracle/WAD solution exactly at `dmax`
creates avoidable saturation geometry. Boundary stress does not affect the
interior recoverability pass.

## What is and is not established

Supported narrowly:

- the actual production sigmoid delay parameter can recover short, middle and
  long *interior* numerical targets under direct supervision;
- `.001` for 200 Adam steps is too weak for the declared low-init to delay-5
  movement;
- exact-boundary targets are a poor design choice for a sigmoid delay.

Not established:

- that `.05` is safe or optimal for WAD, XOR or joint weight/readout training;
- that task BCE supplies a usable direction for delay movement;
- that a spike buffer or LIF neuron preserves the direct-recovery gradient;
- that per-query, per-input or pairwise delays can learn temporal routing;
- any reliability, resource, oracle-imitation or publication claim.

Level 0B must therefore keep the direct target hidden from the parameter and
test whether a target arrival/current trajectory can move a delayed event
through the production buffer, followed by a separate LIF extension. It should
include `.001`, `.01` and `.05` as preregistered diagnostic arms, not silently
replace the XOR learning rate.

## Artifacts

- `docs/generated/delay_parameter_recovery_level0a_v1/cells.csv`
- `docs/generated/delay_parameter_recovery_level0a_v1/decision.json`
- `docs/generated/delay_parameter_recovery_level0a_v1/final_error_heatmaps.png`
- `docs/generated/delay_parameter_recovery_level0a_v1/convergence_heatmaps.png`
- per-cell artifacts under
  `runs/exploratory/delay_parameter_recovery_level0a_v1/`
