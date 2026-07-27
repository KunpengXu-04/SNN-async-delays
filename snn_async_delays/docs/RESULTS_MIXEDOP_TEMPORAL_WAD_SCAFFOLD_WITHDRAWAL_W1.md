# Results: temporal WAD scaffold withdrawal W1

Status: complete. Abrupt task-only retention fails the all-seed schedule gate;
the preregistered annealed withdrawal passes 3/3 and authorizes W2.

All 12 cells start from the corresponding frozen W0 source checkpoint, verify
its hash, and initialize a fresh optimizer. The final update-200 checkpoint is
primary. Every cell is finite, delay-legal, artifact-complete and active in all
five windows.

| Arm | Seeds passing final joint gate | Final maximum schedule errors |
|---|---:|---|
| delay frozen | 3/3 | .500/.501/.500 |
| centroid continue | 3/3 | .512/.510/.516 |
| abrupt task-only | 1/3 | .781/1.114/1.280 |
| annealed joint | 3/3 | .386/.897/.501 |

Function is not the limiting endpoint: every abrupt cell remains essentially
perfect on classification, including worst-query balanced and exact-trial
accuracy of 1.0 in two seeds. Nevertheless abrupt task-only moves internal
delays away from the registered schedule and fails seeds 3669/3691. This is
direct evidence that functional retention does not certify temporal-coordinate
retention.

The annealed arm uses the frozen schedule: lambda decreases linearly from 1 to
0 over updates 1--100, followed by 100 fully task-only updates. It passes every
seed at the final checkpoint, and maximum schedule error throughout the final
task-only half remains below the preregistered 1.5-step bound. Thus a curriculum
can retain an acceptable local schedule more reliably than abrupt removal.

This is warm-start retention, not autonomous discovery or restoration. W2 is
authorized to test whether task-only gradients can reverse two declared
physical-delay perturbations. Machine-readable evidence is in
`docs/generated/mixedop_temporal_wad_scaffold_withdrawal_v1/w1_decision.json`.
