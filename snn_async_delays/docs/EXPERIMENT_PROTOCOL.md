# Confirmatory experiment protocol (v0.1)

## Unit of evidence

A registered experiment specifies: question, hypothesis, protocol version,
task, encoding, readout interface, delay condition, resource vector, training
budget, validation rule, test set, seeds, primary metric, and exclusion rule.

## Encoding

Deterministic burst encoding is the primary mechanism probe because its event
times and input-event budget are controlled.  Jittered burst and rate coding
are robustness conditions.  No conclusion transfers between encodings without
explicit evidence.

## Readout ablation

For a sequential task, compare: (a) all-window count, (b) late-window
alignment count, (c) query-local count, and (d) matched spiking-output
readout.  The late window begins only after the final input window; it must
never be described as a neutral readout.

## Resource matching

When comparing query loads, report actual `T`, input and output dimension,
hidden width, decoder parameters, delay-buffer size, and event counts.  Do not
label a comparison “fixed resource” if any of these are intentionally varied.

## Diagnostics

Raster plots use a predeclared example plus success/failure pairs, never a
post-hoc richest sample.  Mechanism figures must trace input time, effective
arrival time, hidden activity, and readout contribution, and must include a
shuffle or ablation control.
# Simultaneous-pilot amendment (2026-07-13)

Before launch, complete the versioned output-threshold calibration and freeze
one output threshold across all delay conditions. Calibration v1's temporal d0
arm is structurally invalid; use v2 and do not resume v1. Each simultaneous run
must save random-validation metrics, exhaustive 64-pattern truth-table results,
checkpoint, config with seed, train log, resource ledger, diagnostic NPZ and
window-aware panel. Missing exhaustive or diagnostic artifacts make a cell
incomplete. Aggregation must be paired by seed and report failures, ties and
silent opponent outputs rather than silently mapping zero logit to class 0.
