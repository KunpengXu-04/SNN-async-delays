# Results: delay soft-trace credit Level 0C v1

## Formal decision

Level 0C **passes** its preregistered gate. All 360 deterministic cells
completed with config, metrics, final scalar state, NPZ and runtime diagnostic
panel. The selected production candidate is:

```text
delay parameterization: sigmoid
training trace:          buffer current or subthreshold LIF membrane
objective:               normalized soft-trace centroid
Adam learning rate:      0.05
updates:                 200
```

At this single common learning rate it recovers all 30 sigmoid cells across
both soft paths, and all 13 initially misaligned target/init pairs on each path
have a correct nonzero initial gradient direction. Level 0D is authorized.
Level 1 XOR, pairwise WAD, routing, accuracy and resource claims remain locked.

## Primary results

Recovery requires final normalized-trace Wasserstein-1 `<=.1` step, trace mass
`>=.05`, and zero hard spikes on the membrane path.

| Parameterization | Path | Objective | LR .01 | LR .05 | initial directions |
|---|---|---:|---:|---:|---:|
| sigmoid | buffer current | causal filtered | 5/15 | 5/15 | 8 correct, 5 wrong |
| sigmoid | buffer current | soft centroid | 6/15 | **15/15** | **13/13 correct** |
| sigmoid | buffer current | symmetric kernel | 6/15 | **15/15** | **13/13 correct** |
| sigmoid | LIF membrane | causal filtered | 6/15 | **15/15** | **13/13 correct** |
| sigmoid | LIF membrane | soft centroid | 6/15 | **15/15** | **13/13 correct** |
| sigmoid | LIF membrane | symmetric kernel | 0/15 | 0/15 | 12 correct, 1 wrong |
| direct | buffer current | causal filtered | 5/15 | 5/15 | 8 correct, 5 wrong |
| direct | buffer current | soft centroid | 5/15 | 11/15 | 13/13 correct |
| direct | buffer current | symmetric kernel | 6/15 | **15/15** | 13/13 correct |
| direct | LIF membrane | causal filtered | 5/15 | **15/15** | 13/13 correct |
| direct | LIF membrane | soft centroid | 5/15 | 11/15 | 13/13 correct |
| direct | LIF membrane | symmetric kernel | 0/15 | 0/15 | 12 correct, 1 wrong |

The bold sigmoid soft-centroid rows jointly satisfy the locked production
selection rule. Pair-specific LR mixing was not used.

For the selected arm:

| Path | recovered | maximum final W1 | mean final W1 | maximum delay error | latest first crossing | minimum trace mass |
|---|---:|---:|---:|---:|---:|---:|
| buffer current | 15/15 | .0781 | .0199 | .0781 | step 185 | 1.000 |
| LIF membrane | 15/15 | .0583 | .0128 | .0870 | step 168 | .451 |

Every Level-0C membrane cell remains subthreshold: maximum hard-spike count is
zero. Thus the pass is continuous-state temporal credit, not surrogate credit
through a hard threshold.

## Mechanism conclusions

### 1. A global soft temporal coordinate repairs the Level-0B direction problem

Soft centroid has 13/13 correct directions on both paths and both
parameterizations. With production sigmoid and LR `.05`, it recovers every
initial condition in both early-to-late and late-to-early directions. This
shows that the circular buffer and passive LIF membrane jointly preserve enough
continuous timing information for long-range delay optimization.

This does **not** mean centroid is a generally sufficient temporal-task loss.
It constrains only the first temporal moment. Distinct multi-event or
multimodal traces can share a centroid, so Level 0D must retain a hard-output
endpoint and must not infer correct spike structure from centroid alone.

### 2. The Level-0B filtered-loss failure is path dependent

Raw causal filtered trace repeats the Level-0B buffer result exactly: 5/15 and
five wrong initial directions. Yet after a subthreshold membrane low-pass, the
same loss reaches 15/15 with 13/13 correct directions at LR `.05`. Therefore
the defensible conclusion is not "causal filtered loss is always invalid". Its
credit geometry depends on the observed state: passive membrane tails create
temporal overlap that an isolated buffer impulse lacks.

This negative-control success remains selection-ineligible by preregistration.
It is nevertheless important evidence for designing the Level-0D auxiliary
source.

### 3. Global similarity is not equivalent to correct template alignment

The symmetric Laplace-kernel objective recovers 15/15 buffer cells but 0/15
membrane cells for both parameterizations. For target 5 it converges near delay
6; for target 1 it converges near delay 2. Its cross-correlation-like optimum is
biased by the causal, one-sided membrane tail and finite trial truncation. One
late target/init pair even begins with the wrong sign.

This is why the independent W1 endpoint was necessary. A decreasing global
similarity loss cannot itself be treated as proof of temporal alignment.

### 4. The direct control does not show that sigmoid is the bottleneck

Direct and sigmoid arms start from identical functional delays and identical
traces. Direct soft centroid has correct initial directions but recovers only
11/15 at LR `.05`; its eight path-specific failures are the same four extreme
target/init moves duplicated across current and membrane. Final W1 remains
approximately `.16` to `.61`.

This should not be interpreted as intrinsic sigmoid superiority. Equal Adam LR
in parameter coordinates does not match effective delay-space travel: the
sigmoid Jacobian can amplify middle-range delay movement, whereas direct LR
`.05` permits a smaller fixed-coordinate budget. The direct arm localizes no
sigmoid obstruction, but it is not a fair method comparison without an
effective-step-matched optimizer study.

## Pre-run amendment integrity

Before smoke or formal data generation, a unit test showed that the initially
proposed squared-CDF candidate has exact zero gradient at an integer delay when
the target is earlier. The amendment replacing it with symmetric Laplace-
kernel alignment was recorded in the protocol and experiment log before any
Level-0C run. Targets, paths, LRs, primary W1 endpoint and gates were unchanged.

## What is supported and what is not

Supported narrowly:

- production sigmoid delays can receive correct bidirectional long-range credit
  from a global soft centroid through the real buffer and passive LIF membrane;
- LR `.05` is sufficient for all declared one-event target/init pairs under
  this auxiliary and 200-step budget;
- observing continuous membrane state can qualitatively change temporal-credit
  geometry.

Not established:

- learning through a hard output spike or reset;
- compatibility of the auxiliary with classification/BCE;
- correctness for multiple spikes or multimodal traces;
- trainable weights, multiple synapses, query routing, XOR, generalization;
- per-neuron versus per-synapse delays;
- any resource or Pareto advantage.

## Next gate

The next experiment must be **Level 0D**, not XOR. One trainable production
sigmoid delay should drive the suprathreshold hard-LIF path used in Level 0B.
The evaluated output remains the hard spike, while delay learning receives the
selected soft-centroid auxiliary from a prespecified continuous state. The
protocol must distinguish current versus pre-reset membrane auxiliary because
the existing post-reset membrane erases the voltage exactly when a spike
occurs. It must require correct hard-spike arrival in both directions, preserve
the 15-pair coverage grid, and include an auxiliary-weight ablation.

Only a successful hard-output/soft-credit bridge should be considered for a
later K=1 XOR protocol. Per-neuron tying remains a later, separately controlled
optimization-dimension experiment.

## Artifacts

- `docs/generated/delay_soft_trace_credit_level0c_v1/decision.json`;
- `docs/generated/delay_soft_trace_credit_level0c_v1/cells.csv`;
- fixed-LR recovery and initial-gradient summaries in the same directory;
- twelve parameterization/path/objective W1 heatmaps;
- `runs/exploratory/delay_soft_trace_credit_level0c_v1/` with all 360 immutable
  cell artifacts.
