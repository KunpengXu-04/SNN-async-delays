# Results: XOR readout-interface pilot v1

## Status and scope

All 36 preregistered cells completed: 3 seeds × WAD/d0 × three observation
modes × linear/MLP. This remains an exploratory mechanism pilot at one task
point (`K=2`, `h=50`, deterministic burst XOR, threshold 0.3); it is not a
capacity, energy, or generalization result.

## Primary result

The apparent WAD–d0 advantage is strongly interface-dependent.

| Decoder | Observation | d0 worst-query | WAD worst-query | Interpretation |
|---|---:|---:|---:|---|
| Linear | late window | 0.482 ± 0.011 | 0.676 ± 0.166 | WAD is better, but neither is reliable. |
| MLP | late window | 0.490 ± 0.017 | 0.727 ± 0.132 | Same alignment-sensitive pattern. |
| Linear | all time | 0.850 ± 0.090 | 0.877 ± 0.054 | The difference nearly vanishes under uncensored observation. |
| MLP | all time | 1.000 ± 0.000 | 1.000 ± 0.000 | d0 solves this pilot task when early spikes are observable. |
| Linear / MLP | time binned | 1.000 ± 0.000 | 1.000 ± 0.000 | Explicit temporal features make both conditions solve the task. |

Exact-trial accuracy follows the same pattern. In particular, d0 MLP rises
from `0.252 ± 0.024` under the late window to `1.000` under all-time
observation. WAD MLP rises from `0.637 ± 0.208` to `1.000`.

The correct interpretation is narrow but decisive: in this controlled XOR
pilot, the late window censors information that the d0 network has computed.
It therefore cannot serve as a neutral test of general delay-enabled temporal
multiplexing.

This does **not** prove that learnable delays are never useful. Fixed-delay
bank, shuffled-delay, recurrent/time-feature, other K values, other tasks, and
matched-resource confirmation are still absent.

## Resource interpretation

`late_window` and `all_time` have equal decoder feature size within a decoder
type. Thus the all-time d0 MLP result is not explained by a larger MLP.
`time_binned` is deliberately not resource matched:

- d0 MLP storage: 2,852 (`late_window`/`all_time`) → 23,152 (`time_binned`);
- WAD MLP storage: 2,852 → 23,152;
- WAD MLP decoder MACs/trial: 2,600 → 22,800.

Time-binned is consequently an upper/control condition, not evidence of
efficient temporal sharing. Measured input-fan-out event counts were nearly
identical across these readout conditions because the deterministic input and
hidden width were fixed; performance changes are attributable to observation
and decoder access, not a changed input spike budget.

## Diagnostic panel policy and findings

The training run set `no_diag: true` deliberately: generating 36 per-run
raster/mechanism panels during a factorial pilot would create a large,
potentially cherry-picked visual corpus before the aggregate result was known.

After all cells completed, six panels were generated using the fixed,
outcome-independent rule: **lowest preregistered seed (0), MLP, all delay
conditions, all observation modes**. The selection manifest is
`docs/generated/xor_readout_interface_pilot_v1_diagnostic_selection.json`.
These panels are illustrative only; the full 36-cell table is the evidence.

They show that d0 still produces early hidden activity, while late-window
training cannot turn that activity into reliable decisions. WAD displays
learned late arrivals, consistent with alignment; but all-time d0 performance
shows that alignment is not unique evidence of a superior general computation.

The diagnostic code now labels the observation mode and counts “observed”
rather than always calling final-window activity “readout”, avoiding a
misleading mechanism plot for `all_time`/`time_binned` conditions.

## Artifacts

- `docs/generated/xor_readout_interface_pilot_v1/group_summary.csv`
- `docs/generated/xor_readout_interface_pilot_v1/accuracy_summary.png`
- `docs/generated/xor_readout_interface_pilot_v1/resource_summary.png`
- `docs/generated/xor_readout_interface_pilot_v1_diagnostic_selection.json`
- `runs/canonical/xor_readout_interface_pilot_v1/*/plots/diagnostic_panel.png`

## Next scientific gate

Do not enlarge the old late-window sweep. The next protocol should compare
WAD, d0, optimized scalar fixed delay, fixed heterogeneous/random delay, and
shuffled learned delays under `all_time` as the main neutral interface and
`late_window` as an alignment diagnostic. It must use at least five seeds,
predeclared resources, and a sealed test protocol.
