# Results: mixed-op temporal WAD repair v3

Status: complete exploratory supervised repair; centroid arm passes the frozen
function-plus-schedule gate on all three new seeds.

## Primary result

The nine formal cells compare fixed oracle, the v2 arrival-mass CE, and the new
arrival-centroid Huber objective on seeds 3557/3569/3591. Both learned arms use
the same per-query delay tying, delay-3 initialization, support `[0,23]`,
event8/N120/w4/T30 interface, LR .01, 400 updates, and delay-only routing
credit. Only the routing objective differs.

| arm | seeds passing joint gate | accuracy-best worst BAcc | accuracy-best exact | max schedule error |
|---|---:|---:|---:|---:|
| fixed oracle | 3/3 | 1.000/1.000/1.000 | 1.000/1.000/1.000 | 0/0/0 |
| arrival-mass CE | 0/3 | 1.000/1.000/1.000 | 1.000/1.000/1.000 | 2.016/2.012/1.897 |
| arrival-centroid Huber | 3/3 | 1.000/1.000/1.000 | 1.000/1.000/1.000 | .500/.501/.501 |

Rounded display hides only the oracle seed-3591 and centroid seed-3557 values,
which are both .99967 worst-query and .99951 exact. Every qualifying centroid
checkpoint has activity in all five windows. Its learned delays are near the
window-centering solution `[3.5,7.5,11.5,15.5,19.5]`; relative to the registered
integer oracle `[3,7,11,15,19]`, the maximum error remains about half a step.

## Interpretation

This is a clean objective ablation: the old mass CE is functionally perfect but
fails the schedule gate because it learns q0 near 1, whereas the centroid loss
is both functionally perfect and schedule-valid. The repair therefore resolves
the supervised temporal-interface optimization bug identified in v2.

It does **not** show that task loss discovers routing, that independent
per-synapse delays are identifiable, or that WAD beats spatial parallelism. The
five delays receive an explicit query-to-window teaching signal. The next
scientifically admissible experiment is a preregistered scaffold-withdrawal
study that reduces/removes this signal; directly opening K=8 or a Pareto surface
would overclaim the result.

Machine-readable evidence is in
`docs/generated/mixedop_temporal_wad_repair_v3/formal_checkpoint_rows.csv` and
`docs/generated/mixedop_temporal_wad_repair_v3/formal_decision.json`. The sealed
test was not opened.
