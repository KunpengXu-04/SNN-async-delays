# Results: simultaneous temporal viability preflight v2

Date: 2026-07-13  
Protocol: `simultaneous_temporal_viability_preflight_v2`  
Decision: **passed all mechanism-valid viability gates**

## Completeness and design separation

All six held-out cells completed with finite logs and all required artifacts:
config, best/final checkpoints, training log, validation, exhaustive truth
table, resource ledger, diagnostic NPZ and run-time panel. Seed 0 was used only
for audit/design; v2 used seeds `{1,42}`. Every cell used the frozen
window/class-balanced loss. Test remains sealed.

Accuracy did not enter any pass/fail check.

## Locked gate result

All gates pass separately in both held-out seeds. The limiting WAD values are:

| Measure | Seed 1 | Seed 42 | Gate |
|---|---:|---:|---:|
| Minimum output-current support | .500 | .766 | >=.50 |
| Minimum output activity | .234 | .328 | >=.10 |
| Minimum realized arrival mass | .051 | .084 | >=.05 |
| Minimum window output-weight gradient | .150 | .110 | >1e-8 |
| Minimum window total delay gradient | .605 | .548 | >1e-8 |
| Final mean delay movement | 2.303 | 2.302 | >=.05 |
| Final delay saturation | 0 | 0 | <.95 |

Scaffold and full-support fixed controls also pass their output-current,
activity and output-gradient gates. The mechanism-valid shared opponent
interface is therefore executable for a matched performance experiment.

## What passed—and what did not

Viability passed; reliable routing did not.

WAD descriptive balanced accuracy by window is:

| Seed | XOR/window 0 | NAND/window 1 | NOR/window 2 | Exact trial |
|---|---:|---:|---:|---:|
| 1 | .983 | .891 | .500 | .604 |
| 42 | .931 | .819 | .500 | .474 |

The third-window signed-current class gaps are approximately `-.038` and
`-.069`, effectively zero, despite nonzero current, output spikes, arrival mass
and gradients. Balanced loss corrects objective weighting but does not make the
current WAD recipe learn the third target.

Full-support fixed delay reaches NOR balanced accuracy `.743/.746`, showing
that the interface can carry discriminative late information. Its worst-window
accuracy is unstable (`.702/.487`) because seed 42 fails XOR rather than NOR.
Scaffold remains an execution control and stays at worst-balanced `.5`.

## Critical remaining confound

The current order is always XOR in window 0, NAND in window 1 and NOR in window
2. Consequently, WAD's failure could be caused by late temporal position, NOR
operation identity, or their interaction. The present data cannot distinguish
these explanations. Launching the old fixed-order full matrix would preserve
this confound and is not acceptable for a serious routing claim.

The next full protocol must counterbalance operation and temporal position,
preferably with the three cyclic orders:

1. XOR, NAND, NOR;
2. NAND, NOR, XOR;
3. NOR, XOR, NAND.

Each operation then appears once in every window. The primary endpoint should
remain the shared opponent pair; fixed-full must be retained alongside d0,
scalar, narrow fixed and WAD. The balanced objective and mechanism diagnostics
must remain frozen. Linear/MLP probes should be diagnostic rather than multiply
the primary matrix.

## Decision

1. Mark preflight v2 as passed for viability only.
2. Do not claim successful temporal routing or WAD performance advantage.
3. Do not launch the old fixed-order temporal matrix.
4. Amend and preregister a counterbalanced temporal performance protocol before
   running new cells.
5. N/T resource-frontier and K=8 experiments remain premature until the
   counterbalanced K=3 temporal experiment establishes a meaningful reliability
   regime.

## Artifacts

- Raw: `runs/exploratory/simultaneous_temporal_viability_preflight_v2/`
- Decision/table/figure:
  `docs/generated/simultaneous_temporal_viability_preflight_v2/`

