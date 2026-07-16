# Results: XOR difficulty calibration v1

## Status and evidence boundary

All 54 preregistered validation-only cells completed. Every cell has a final
`validation_results.json`, `diagnostic_data.npz`, and `diagnostic_panel.png`.
The test split was not evaluated. These results select difficulty; they do not
test WAD superiority because validation was also used for checkpoint selection,
only three seeds were run, and d0 is not the full causal control family.

## Main numerical result

Mean ± sample SD across three paired seeds:

| K | N | d0 worst | WAD worst | WAD−d0 | d0 exact | WAD exact | WAD−d0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 20 | .767±.073 | .845±.069 | +.078 | .674±.096 | .769±.088 | +.095 |
| 2 | 35 | .795±.025 | .873±.027 | +.077 | .734±.010 | .792±.043 | +.058 |
| 2 | 50 | .865±.096 | .878±.044 | +.013 | .847±.097 | .824±.046 | −.023 |
| 3 | 20 | .627±.022 | .664±.058 | +.037 | .365±.093 | .371±.077 | +.007 |
| 3 | 35 | .663±.031 | .676±.049 | +.013 | .386±.014 | .431±.018 | +.045 |
| 3 | 50 | .692±.020 | .661±.025 | −.031 | .464±.009 | .435±.032 | −.029 |
| 4 | 20 | .603±.031 | .599±.056 | −.003 | .168±.031 | .172±.033 | +.004 |
| 4 | 35 | .586±.020 | .604±.049 | +.018 | .171±.034 | .197±.033 | +.027 |
| 4 | 50 | .661±.025 | .650±.017 | −.011 | .231±.022 | .225±.041 | −.005 |

The dominant pattern is task degradation with K. Increasing N does not restore
exact-trial reliability monotonically or produce a stable WAD advantage. The
apparent WAD benefit at K=2 shrinks with N and K, reverses at K=3,N=50, and is
near zero/inconsistent at K=4. This calibration therefore provides no evidence
for a learned-delay scaling advantage.

Paired-seed differences reinforce the caution. At K=3,N=35 the worst-query
differences are `+.012, −.004, +.032`; at K=3,N=50 they are
`−.058, +.020, −.054`. With three validation-selected seeds, inferential testing
would be misleading.

## Preregistered calibration decision

Both conditions meet the target/variance rule at K=2,N=20/35/50;
K=3,N=35/50; and K=4,N=50 (WAD lies exactly at the lower boundary).

**Primary next setting:** `K=3, N=35, T=40, sub_win=10`. It is the smallest N
at K=3 where both conditions qualify; K=3,N=20 fails for d0. It is not selected
because WAD looks best: the WAD−d0 worst-query difference is only +.013 and one
of three paired seeds is negative.

**Stress setting:** `K=4, N=50, T=50, sub_win=10`. It is retained because both
conditions meet the rule at the lower boundary, but exact-trial accuracy is only
about .23, making it a severe joint-output regime rather than the sole primary
comparison.

## Resource interpretation

At K=3,N=35, d0/WAD have 178/248 trainable parameters; both store 248 scalar
elements and perform 1,400 neuron updates, 2,800 dense synapse MACs, and 105
decoder MACs. Mean hidden spikes are 10.68/9.89. The 0.79-spike difference is
not an energy gain: dense compute is unchanged and WAD optimizes 70 additional
delay values.

## Diagnostic-panel assessment

The regenerated panels are internally complete and correctly label all-time
observation. They show sparse hidden activity and heterogeneous WAD arrivals,
but do not show that learned timing is useful. Current limitations are:

- one synthetic trace per panel, not an aggregate distribution;
- spike-associated edge selection in the mechanism plot;
- no shuffle/replacement counterfactual;
- membrane/readout traces and query-wise predictions remain absent;
- “Output” raster semantics are unclear for a non-spiking linear readout;
- under all-time observation, late arrival is not the success criterion.

## Conclusions and next gate

Calibration succeeded, but WAD does not show a stable advantage across K/N.
K is the dominant observed difficulty factor, exact-trial accuracy falls sharply,
and more neurons do not yield clean monotonic recovery. The experiment also
cannot separate K from T because total T grows with K.

Next, preregister `xor_delay_control_matrix_v1` at K=3,N=35 as primary and
K=4,N=50 as stress. Compare d0, optimized scalar delay, fixed heterogeneous
delay, and WAD with at least five paired seeds. Keep all-time linear primary,
late-window diagnostic, and the test split sealed until tuning is frozen.
Before running, implement/unit-test the controls and improve panel semantics.

## Machine-readable artifacts

- `docs/generated/xor_difficulty_calibration_v1_run_level.csv`
- `docs/generated/xor_difficulty_calibration_v1_group_summary.csv`
- `docs/generated/xor_difficulty_calibration_v1_paired_deltas.csv`
- `docs/generated/xor_difficulty_calibration_v1_selection.json`
- `docs/generated/xor_difficulty_calibration_v1_accuracy.png`
- `docs/generated/xor_difficulty_calibration_v1_paired_deltas.png`
