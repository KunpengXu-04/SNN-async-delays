# Results: mixed-op temporal WAD repair v2

Status: complete exploratory recovery; assisted withdrawal gate failed.

The 12 formal cells used three new seeds and evaluated both accuracy-best and
schedule-best checkpoints. Oracle passed all seeds with worst-query balanced
accuracy and exact-trial accuracy equal to 1.0 and delay schedule exactly
`[3,7,11,15,19]`.

Task-only WAD failed completely: worst-query balanced accuracy remained .50,
minimum output-window activity was zero, and maximum schedule error exceeded
10 steps. Joint calibrated supervision improved classification, reaching
worst-query balanced accuracy .780/.863/.980 across seeds at the accuracy-best
checkpoints, but schedule errors remained 8.71--9.91 steps.

The decoupled arrival-mass arm exposed the important failure mode. It achieved
perfect worst-query balanced and exact-trial accuracy at the accuracy-best
checkpoint for every seed, with activity in every window. Nevertheless its
maximum schedule errors were 1.96/2.06/1.96 steps, above the preregistered
one-step gate. The learned schedules were approximately
`[1.0,7.5,11.5,15.3,18.2]`. Schedule-best checkpoints still had errors near
1.90 steps. Thus functional success and the registered oracle schedule did not
coexist at a valid checkpoint.

This result falsifies the claim that v2 repaired the registered schedule. It
also shows that the arrival-mass CE was mismatched to the schedule gate: its
first-query optimum is earlier than delay 3. V3 tests a preregistered
arrival-centroid objective that is mathematically aligned with output-window
centers. Even if V3 succeeds, it will establish only explicitly supervised,
query-tied temporal routing—not autonomous learned time multiplexing.

Primary machine-readable evidence:
`docs/generated/mixedop_temporal_wad_repair_v2/formal_checkpoint_rows.csv` and
`docs/generated/mixedop_temporal_wad_repair_v2/formal_decision.json`.
