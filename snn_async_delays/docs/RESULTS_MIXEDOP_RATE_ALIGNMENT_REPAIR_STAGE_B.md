# Results: Mixed-Operation Rate Alignment Repair v1, Stage B

Status: all nine cells are complete. The new-seed oracle passes 3/3;
task-only and routing-assisted per-query delay learning pass 0/3. Stage C is
not authorized. The sealed test remains closed.

## Registered endpoints

The selected interface is event8, N=120, w=4, T=30 with oracle schedule
`[3,7,11,15,19]`. Three new seeds compare a fixed oracle, five query-tied
task-only delays, and the same delays with `.10` population arrival-window
cross entropy. A learned cell needs both reliable function and maximum error at
most one step for every query delay.

## Results

| arm | seed | worst BAcc | exact-trial | minimum window activity | maximum delay error |
|---|---:|---:|---:|---:|---:|
| oracle | 3347 | 1.0 | 1.0 | 1.0 | 0 |
| oracle | 3359 | 1.0 | 1.0 | 1.0 | 0 |
| oracle | 3371 | 1.0 | 1.0 | 1.0 | 0 |
| task-only | 3347 | .50 | .222 | 0 | 12.73 |
| task-only | 3359 | .50 | .245 | 0 | 13.46 |
| task-only | 3371 | .50 | .189 | 0 | 11.93 |
| routing-assisted | 3347 | .50 | .170 | 0 | 14.21 |
| routing-assisted | 3359 | .50 | .214 | 0 | 8.76 |
| routing-assisted | 3371 | .50 | .387 | 0 | 6.73 |

Task-only delays remain severely collapsed. Only the first two queries are
reliable, query 2 is partial, and the final two remain at chance with silent
late windows.

Routing assistance is directionally active but insufficient. Its route loss
falls from about `8.82` to `2.72--3.43`; every query has substantial nonzero
delay gradients. At accuracy-selected checkpoints, learned schedules range
from `[1.36,4.45,2.97,4.28,4.79]` to
`[1.01,5.28,7.42,9.26,12.27]`. The strongest seed activates four windows and
raises query-2 BAcc to `.936`, but query 4 is still silent and at chance.

## Mechanism interpretation

This failure is not gradient disappearance and not an invalid interface:

1. fixed oracle is perfect on all new seeds;
2. median total delay-gradient norms are `.36--.48` for task-only and
   `.44--.55` for routing-assisted;
3. route loss decreases and moves later-query delays into additional windows.

Task BCE favours delays roughly two or more steps earlier than the explicit
arrival objective, so `.10` supervision does not control the shared optimum.
Also q4 starts near delay 2.26 but must approach the sigmoid support boundary
at 19; bounded sigmoid movement slows near that boundary. At update 200,
assisted q4 reaches only about `12.3` in all seeds. The checkpoint rule can
select an earlier accuracy peak before schedule loss finishes decreasing, but
last checkpoints still fail both function and schedule.

## Decision

Stage C scaffold withdrawal remains locked. Do not increase `.10`, LR or
updates inside v1. A future versioned repair must place every target delay in
the interior of its support and calibrate relative task/routing gradient scale
before a recovery study. A lower-dimensional affine schedule `d_q=b+qg` is a
justified comparator, but its strong structural prior must be disclosed.
Per-synapse WAD and full N/T sweeps remain unjustified.
