# Preregistration: simultaneous temporal viability preflight v1

> Historical note: v1 formally failed its preregistered hidden-emission gate.
> The subsequent checkpoint audit showed that hidden emission was not a valid
> proxy for delayed output arrival: WAD retained third-window output current
> and realized arrivals. Do not reuse the v1 gate. See
> `RESULTS_SIMULTANEOUS_TEMPORAL_CHECKPOINT_MECHANISM_AUDIT_V1.md` and the
> preregistered v2 protocol.

## Question

Before the full temporal-routing matrix, does the frozen threshold-.2 shared
opponent interface receive events in all three windows and retain usable output
and delay gradients under the intended controls? This is an implementation and
support audit, not a performance comparison.

## Design

One fixed seed (`0`), one endpoint (`opponent_shared_windowed`) and six
conditions produce six validation-only cells:

1. d0 structural lower bound;
2. trainable scalar delay;
3. frozen heterogeneous narrow support `[1,9]` from the current pilot;
4. frozen heterogeneous full support `[0,30]`;
5. the calibration scaffold as a positive execution control;
6. WAD with the frozen Stage-B recipe.

All cells use hidden threshold .3, output threshold .2, 100 epochs, the same
20-epoch peak-voltage warm-up, 0.2 membrane auxiliary, burst encoding and final
checkpoint. No timing scaffold is applied outside its named positive-control
cell.

The full-support fixed bank is necessary because `[1,9]` cannot directly cover
the entire 30-step output horizon. It is added before preflight results and is
not selected by accuracy.

## Decision gates

Accuracy and exact-trial accuracy are descriptive only. The locked checks are:

- all six cells and artifacts complete with finite logs/results;
- scaffold: hidden activity in every window for at least 10% of trials and
  output silent rate below .5;
- fixed-full: at least 5% hidden-window activity in every window and nonzero
  output-weight gradient in at least 95% of epochs;
- WAD: the same window-support and output-gradient requirements, nonzero delay
  gradient in at least 95% of epochs, final mean delay movement at least .05,
  and final saturation below .95.

D0, scalar and narrow-fixed are geometry diagnostics and need not cover all
windows. No condition may be dropped based on its accuracy. Failure keeps the
temporal matrix locked and requires a versioned redesign; passing requires a
prelaunch amendment that retains full-support fixed delay and the matched
output curriculum.

## Required artifacts

Each cell must save config, best/final checkpoints, training log, validation,
exhaustive truth table, resource ledger, NPZ and diagnostic panel. The result
must additionally report per-window hidden spike count/activity, output
silent/tie/collision, output-weight gradient, and WAD delay gradient/movement.

## Commands

```powershell
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_simultaneous_temporal_viability_preflight --dry-run
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_simultaneous_temporal_viability_preflight --device cuda
D:\anaconda3\envs\snn_async\python.exe -m scripts.summarize_simultaneous_temporal_viability_preflight
```
