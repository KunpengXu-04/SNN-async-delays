# Simultaneous output-interface calibration v2

## Purpose and non-claim

This validation-only protocol asks a narrow engineering question: which output
LIF threshold, if any, yields a viable signed opponent-spike interface for both
the spatial and shared-windowed simultaneous tasks? It does not test whether
learned delays improve routing.

Version 1 is immutable and stopped. Its spatial d0 arm completed, but temporal
d0 was structurally unable to deliver activity after the input interval. The
six completed temporal-v1 cells therefore diagnose an invalid calibration arm,
not a bad threshold and not a WAD result.

## Evidence and new cells

V2 reuses the nine completed spatial-v1 cells: three thresholds
`{0.2,0.3,0.5}` by seeds `{0,1,42}`. It adds nine temporal cells with the same
threshold/seed matrix, a shared opponent pair, and fixed input-to-hidden delay
rows `[8,8,18,18,28,28]`. Hidden-to-output delay is zero. For bursts at phases
0.2 and 0.8 of a 10-step input window, the scaffold creates arrivals in the
three output windows. It supplies timing but is neither learned nor a pilot
control.

## Training and evaluation

The first 20 epochs optimize two class-wise peak pre-reset voltages relative to
the candidate threshold: the correct opponent is encouraged above threshold
and the incorrect opponent below it. Epochs 21–100 optimize signed spike-count
BCE plus a fixed `0.2` voltage auxiliary. Evaluation never uses membrane
voltage; its logit is always `count(class1)-count(class0)` in each window.

V2 evaluates `last_model.pt`, while retaining `best_model.pt` for audit.
Selecting by best pooled count accuracy during the voltage phase could silently
reload epoch 1. Training logs therefore include aggregate loss, count loss and
membrane loss separately.

Every formal cell must contain config, final/best checkpoints, train log,
validation result, exhaustive 64-pattern result, resource ledger, diagnostic
NPZ and panel. The panel shows true output spikes and output membrane against
threshold. Missing artifacts make a cell incomplete.

## Locked decision rule

For each candidate threshold, both spatial and temporal arms must have finite
metrics, mean worst balanced accuracy in `[.50,.95]`, silent rate `<.50`, tie
rate `<.50`, and collision rate `<.25`. Among thresholds passing both arms,
select the pooled mean output spikes per operation/window closest to one.
Accuracy is a gate, not the tie-breaker. If no threshold passes, select none and
keep both simultaneous pilots locked.

## Verification boundary

Core tests include peak-voltage backpropagation across all shared windows. A
40-epoch full-data-budget threshold-.3 smoke produced nonzero output spikes and
declining loss, showing that the corrected path is executable, but its worst
balanced accuracy remained `.5` and silent/tie rate was `.715`. It is not one
of the nine formal cells and provides no threshold or routing evidence.

## Commands

Dry expansion:

```powershell
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_simultaneous_output_calibration_v2 --dry-run
```

Formal validation-only run (only after confirming the preregistration file is
unchanged):

```powershell
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_simultaneous_output_calibration_v2 --device cuda
```

After all nine new cells are complete:

```powershell
D:\anaconda3\envs\snn_async\python.exe -m scripts.summarize_simultaneous_output_calibration_v2
```

## Completion decision (2026-07-13)

All nine new cells and required artifacts completed. Combined with the nine
reused spatial cells, the locked rule selects threshold 0.2. Threshold 0.5
fails temporal viability; 0.2 beats viable 0.3 only on distance to the declared
one-spike target. The temporal arm's worst balanced accuracy remains exactly
0.5 for every threshold and seed, with XOR and NOR at chance under threshold
0.2. Therefore this freezes excitability but does not validate routing. See
`RESULTS_SIMULTANEOUS_OUTPUT_INTERFACE_CALIBRATION_V2.md`.
