# Pre-registration: XOR readout-interface pilot v1

## Question

Under deterministic burst-encoded, sequential XOR queries, does the relative
behaviour of learned-delay WAD and d0 change when the observation interface is
changed from final-window censoring to all-time or time-binned observation?

## Fixed design

`K=2`, `h=50`, deterministic burst, `sub_win=10`, `read_len=10`, LIF threshold
`0.3`, no homeostasis, 100 epochs, and train/validation/test sizes
`2000/500/1000`.  XOR is balanced and avoids NAND's 75% majority shortcut.
The low threshold is predeclared because historical XOR/burst threshold 1.0
caused an all-silent cold start; it is not tuned within this pilot.

The full factorial has 36 cells:

`3 seeds × {WAD,d0} × {late_window,all_time,time_binned} × {linear,MLP}`.

`late_window` and `all_time` have equal decoder feature dimensions within a
decoder type.  `time_binned` intentionally has larger temporal features and is
therefore a resource-accounted upper/control condition, not a fair
same-capacity comparison.

## Outcomes and interpretation

Primary outputs are worst-query accuracy, exact-trial accuracy, balanced
accuracy, and `resource_ledger_v1`.  Pooled accuracy is descriptive only.

This pilot detects interface sensitivity and implementation failures.  It does
not establish a capacity advantage, energy advantage, delay necessity, or
publication-level result.  All 36 cells remain exploratory regardless of the
numeric outcome.

## Immutable artifacts

The machine-readable pre-registration is
`configs/pilot_xor_readout_v1.yaml`.  Outputs are written under
`runs/canonical/xor_readout_interface_pilot_v1/` but are labelled exploratory
in their metadata and summaries.
