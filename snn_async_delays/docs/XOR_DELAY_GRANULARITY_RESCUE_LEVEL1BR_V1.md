# XOR dimension-aware delay rescue — Level 1B-R v1

**Status:** complete. R1 is 50/50 and sealed R3 is 30/30; R2 was not required.
See `RESULTS_XOR_DELAY_GRANULARITY_RESCUE_LEVEL1BR_R1.md` and
`RESULTS_XOR_DELAY_GRANULARITY_RESCUE_LEVEL1BR_R3.md`.

## 1. Motivation and scope

Level 1B Stage A produced a clean but limited result. The global delay-4
scaffold replicated 10/10, while per-hidden and per-synapse candidates passed
2/10 and 0/10. The registered scaffold was

\[
L_{arr}^{mean}=\frac{1}{2P}\sum_{p=1}^{P}
[\mu_p(d_p)-\mu_p(4)]^2,
\]

so its gradient on each coordinate scales as `1/P`. Empirically, the absolute
initial arrival gradient per coordinate was approximately `2.5588/P`. At fixed
`lambda=.01`, increasing delay dimension therefore weakened the timing teacher
relative to the task gradient.

Level 1B-R asks whether that normalization explains the higher-dimensional
failure. It does not alter or replace Level 1B. All previous cells and negative
candidate decisions remain immutable.

## 2. Frozen task, interface and model

The rescue retains the single-event K=1 exhaustive XOR task:

- four value-identity inputs `[A0,A1,B0,B1]`;
- 16 hard-LIF hidden neurons and two hard-LIF opponent outputs;
- one event on each selected value channel at step 9;
- input-hidden sigmoid delays with `dmax=8`;
- hidden-output d0 transmission;
- hidden/output thresholds `.2/.03` in simulator arbitrary units;
- one exact correct output spike at step 15;
- Adam, weight LR `.01`, initial delay LR `.01`, 500 updates and final
  checkpoint evaluation.

No micro-burst, K>1, mixed operation, resource-frontier or test-set experiment
is part of this protocol.

## 3. Dimension-matched scaffold

For `P` independent delay coordinates, the rescue keeps the same mean loss but
sets

\[
\lambda_P=.01P.
\]

Thus

\[
\lambda_P L_{arr}^{mean}
=.01\sum_{p=1}^{P}\frac12[\mu_p(d_p)-\mu_p(4)]^2.
\]

The effective lambdas are `.01`, `.16` and `.64` for global, per-hidden and
per-synapse delays. This is analytically fixed before any rescue run. It is not
a hyperparameter sweep and it does not use Level-1B failed seeds to select a
value.

The intervention changes the scientific resource convention from a fixed
total teacher budget to a fixed teacher strength per trainable delay
coordinate. Both conventions are legitimate but answer different questions.
Any positive result must explicitly state this stronger total oracle
supervision as `P` grows.

## 4. R1: normalization test

R1 uses five new calibration seeds `{1123,1229,1321,1427,1523}` and both raw
initializations `-2/+2`. Its five named conditions are:

| Condition | Granularity | P | Normalization factor | Effective lambda |
|---|---|---:|---:|---:|
| global anchor | global | 1 | 1 | .01 |
| per-hidden mean baseline | per hidden | 16 | 1 | .01 |
| per-hidden coordinate-matched | per hidden | 16 | 16 | .16 |
| per-synapse mean baseline | per synapse | 64 | 1 | .01 |
| per-synapse coordinate-matched | per synapse | 64 | 64 | .64 |

The fixed matrix contains 50 cells. Every named candidate spans ten cells and
passes only at 10/10 under the full Level-1B coordinate-wise gate. The global
anchor must also pass 10/10; otherwise the rescue environment is invalid.

The unscaled higher-dimensional conditions are new-seed baselines. They are
not eligible for method selection and cannot be pooled with Level 1B.

## 5. Why LR is conditional, not simultaneous

Changing delay LR cannot correct a wrong gradient direction. Therefore R2 is
allowed only when a coordinate-matched R1 candidate fails final recovery but
all ten cells have nonzero, correctly directed initial total gradients on
every coordinate.

If any R1 cell has a wrong or zero coordinate, LR/budget escalation is
mechanically forbidden. That outcome requires a new loss or task/teacher
compatibility study, not a faster optimizer.

## 6. R2: conditional LR and budget calibration

For each mechanically eligible granularity, R2 uses new calibration seeds
`{1601,1693,1789}` and tests exactly three interventions:

| Priority | Intervention | Delay LR | Updates | Interpretation |
|---:|---|---:|---:|---|
| 1 | LR only | .05 | 500 | larger step size |
| 2 | budget only | .01 | 1000 | longer optimization |
| 3 | LR and budget | .05 | 1000 | combined rescue |

Each intervention spans six cells. Selection takes the first 6/6 passing
intervention in the declared priority order. R2 is calibration only and cannot
support the final rescue claim.

## 7. R3: sealed confirmation

R3 uses sealed seeds `{2003,2011,2027,2039,2053}`. They may not be
materialized or inspected before R1 and, if needed, R2 have written complete
decisions.

R3 always reruns the global anchor and every provisional higher-dimensional
recipe selected directly by R1 or conditionally by R2. Every recipe spans ten
cells. Final rescue success requires:

1. global anchor confirmation 10/10; and
2. at least one higher-dimensional recipe confirmation 10/10.

If both higher-dimensional recipes pass, per-hidden is selected because it
uses 16 rather than 64 trainable delay values. This is a predeclared complexity
priority, not post-hoc preference.

## 8. Per-cell gates and diagnostics

Every cell must satisfy all of the following:

- balanced accuracy one and all four XOR classifications correct;
- output spike train exactly equals the target train;
- zero silence/collision and correct target-time rate one;
- one output event per trial and hidden activity in all four patterns;
- maximum independent delay error at most `.1` step;
- all independent delays within tolerance;
- every initial total raw-delay gradient nonzero and correctly directed.

The runtime NPZ and 12-panel diagnostic remain mandatory. In addition to the
Level-1B traces, the rescue records effective lambda, normalization factor,
unweighted and weighted arrival gradients, global pre-clip gradient norm and
clip coefficient. Any clip coefficient below `.999` is flagged because it can
couple the stronger teacher to weight optimization.

## 9. Interpretation limits

A positive result would show only that explicit per-coordinate oracle timing
supervision can optimize a named higher-dimensional delay parameterization.
It would not show that XOR labels discovered delay 4, that delays specialize,
or that a network performs temporal routing. Because total teacher strength
grows with `P`, it also cannot establish equal-supervision efficiency.

A negative result after correct-gradient dimension scaling would rule out a
simple normalization-only explanation. It would not prove that
higher-dimensional delays are universally untrainable.

## 10. Execution order

The versioned runner is
`scripts/run_xor_delay_granularity_rescue_level1br.py`.

```powershell
python -m scripts.run_xor_delay_granularity_rescue_level1br --stage r1 --device cpu
python -m scripts.run_xor_delay_granularity_rescue_level1br --stage r2 --device cpu
python -m scripts.run_xor_delay_granularity_rescue_level1br --stage r3 --device cpu
```

R1 is complete. Its mechanical decision directly authorizes R3 and makes R2
inapplicable. R3 reads the immutable R1 decision and refuses unauthorized
materialization. Formal subsets within a materialized candidate are
forbidden.

Implementation validation is complete. Nine Level-1B-R structural tests and
the full 97-test project suite pass. Two smoke cells from each stage produced
checkpoint, strict metrics, full log, exhaustive truth table, resource ledger,
runtime NPZ and diagnostic panel. Synthetic decisions were used only to test
the locked R2/R3 code paths. All six smoke cells are invalid for claims. The
50 formal R1 cells are complete and contain all eight required artifacts.

## 11. R1 outcome (2026-07-16)

The global anchor passes 10/10. The unscaled mean-loss baselines reproduce the
dimension failure: per-hidden passes 1/10 and per-synapse 0/10. In contrast,
the analytically coordinate-matched conditions pass 10/10 for both
granularities, with exact hard-spike interfaces, correct nonzero initial total
gradient direction on every coordinate, full final delay coverage, and no
gradient-clipping flag.

The intervention equalizes the initial weighted arrival-gradient magnitude
per coordinate at approximately `.025588`; the unscaled values are
approximately `.001599` (`P=16`) and `.0003998` (`P=64`). This is strong
evidence that the registered mean scaffold's `1/P` dilution was sufficient to
cause the observed Level-1B optimization failure in this oracle-supervised
homogeneous delay-recovery task. It is not evidence for autonomous delay
discovery or temporal routing.

R2 was mechanically skipped because both scaled candidates already satisfied
the 10/10 gate. The registered next gate was confirmation of global,
per-hidden and per-synapse recipes on 30 sealed-seed cells, with per-hidden the
predeclared preferred recipe if both higher-dimensional candidates confirmed.

## 12. R3 outcome (2026-07-16)

The sealed 30-cell confirmation is complete. Global, per-hidden and
per-synapse each pass 10/10 under every interface, coordinate-gradient and
delay-coverage gate. No cell triggers gradient clipping and all required
runtime artifacts are present. Both higher-dimensional recipes therefore
confirm, and the preregistered complexity rule selects per-hidden-neuron
because it uses 16 rather than 64 independent delay parameters.

Level 1B-R passes, but its claim remains restricted to homogeneous delay-4
recovery under explicit oracle strength proportional to the number of delay
coordinates. The decision explicitly leaves micro-burst and K>1 unauthorized.
