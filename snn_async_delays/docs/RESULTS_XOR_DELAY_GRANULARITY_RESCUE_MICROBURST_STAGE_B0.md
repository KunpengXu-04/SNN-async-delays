# Results: dimension-aware XOR micro-burst rescue Stage B0

**Protocol:** `xor_delay_granularity_rescue_microburst_v1`  
**Date:** 2026-07-16  
**Status:** fixed-d4 control complete, 5/5; the 40-cell learned stage is
mechanically authorized.

## Result

The same-seed fixed-delay replication passes every preregistered interface
gate in all five fresh formal seeds `{2333,2351,2371,2389,2411}`.

| Gate | Result |
|---|---:|
| complete formal cells | 5/5 |
| balanced accuracy = 1 | 5/5 |
| exact four-pattern output spike trains | 5/5 |
| exact target time at step 15 | 5/5 |
| zero silence | 5/5 |
| zero collision | 5/5 |
| exactly one output spike per trial | 5/5 |
| hidden activity in all four patterns | 5/5 |
| complete eight-artifact bundle | 5/5 |

Every trial contains four input events, from the consecutive events at steps
8 and 9 on both selected value channels. Fixed input-hidden delay four remains
exactly four in every cell. No early or extra output spike is accepted by the
exact-train gate.

## Interpretation

The result replicates the parent fixed-oracle feasibility result on the new
formal seeds and current implementation environment. It rules out a broken
micro-burst interface as an explanation for a subsequent learned-stage
failure.

It does not provide evidence for learned delays, dimension normalization,
task-derived timing or routing. Delay four is supplied by the experimenter.

## Mechanical decision

`fixed_d4_gate_pass=true` and `learned_stage_authorized=true`. The complete
40-cell learned matrix is now launch-ready. It must be run without candidate
subsetting or seed selection.

## Artifacts

- Decision: `docs/generated/xor_delay_granularity_rescue_microburst_v1/stage_b0_fixed_control/decision.json`
- Aggregate cells: `docs/generated/xor_delay_granularity_rescue_microburst_v1/stage_b0_fixed_control/cells.csv`
- Formal runs: `runs/exploratory/xor_delay_granularity_rescue_microburst_v1/stage_b0_fixed_control/`

The test split remains unopened and K greater than one remains unauthorized.
