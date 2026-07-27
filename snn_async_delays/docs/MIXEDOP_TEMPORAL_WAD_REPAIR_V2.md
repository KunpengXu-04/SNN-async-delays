# Mixed-Operation Temporal WAD Repair v2

Status: complete. Oracle passes, but every learned arm fails the registered
joint function-plus-schedule gate. See
`RESULTS_MIXEDOP_TEMPORAL_WAD_REPAIR_V2.md`. Withdrawal remains locked.

## Failure addressed

V1 proves the event8/N120/w4 oracle interface is valid but rejects both
task-only and `.10` joint routing assistance. Routing gradients are nonzero and
partially expand coverage, yet task credit favours earlier delays and q4 must
approach the sigmoid boundary at 19 from an initial value near 2.26.

V2 changes the method, not merely a hyperparameter:

1. retain the oracle schedule `[3,7,11,15,19]` but expand support to `[0,23]`,
   placing every target at least four steps from the upper boundary;
2. initialize all five query delays exactly at 3 using the inverse sigmoid;
3. calibrate joint routing scale from task/route gradient norms before any
   training;
4. compare joint calibrated credit with a decoupled arm in which weights and
   readout receive task BCE gradients while delays receive only routing-loss
   gradients;
5. save both accuracy-best and schedule-best checkpoints.

All assisted conditions remain explicitly supervised. A success cannot be
called autonomous routing.

## Gradient preflight

For seeds `{3407,3413,3433}`, one balanced training batch produces separate raw
delay gradients `g_task` and `g_route`. Define

\[
\lambda_0=\operatorname{median}_s
\frac{\lVert g_{task}^{(s)}\rVert}
     {\lVert g_{route}^{(s)}\rVert}.
\]

Test only multipliers `{1,3,10}` and select the smallest for which
`g_task + m*lambda_0*g_route` is target-directed for q1--q4 in every seed. The
decoupled arm independently requires route-only target-directed gradients for
the same coordinates. q0 is excluded because its target equals its initial
delay. Training is authorized only if at least one assisted rule passes.

## Conditional recovery

After a four-arm invalid smoke, three fresh seeds compare oracle, task-only,
joint-calibrated and routing-only-for-delays for 400 updates. The 400-update
budget is fixed from the raw-coordinate distance: q4 must traverse about 3.43
raw sigmoid units at LR `.01`; it is not an extension selected from observed
accuracy.

Each learned seed needs worst-query BAcc `.85`, exact-trial `.70`, activity in
every window, and maximum error across all five delays at most one step. Either
the accuracy-best or schedule-best checkpoint may pass, but function and
schedule must hold at the same checkpoint. Withdrawal remains locked until an
assisted arm passes all three seeds.

Machine-readable source of truth:
[`configs/mixedop_temporal_wad_repair_v2.yaml`](../configs/mixedop_temporal_wad_repair_v2.yaml).

## Gradient-preflight decision

The median task/route norm balance is `.0048002`. Multiplier 1 fails the
three-seed direction gate; multipliers 3 and 10 pass. The registered smallest
choice is multiplier 3, giving joint lambda `.0144006`. Route-only gradients
are target-directed for q1--q4 in all three seeds, so the decoupled arm also
passes. Both training candidates proceed unchanged to technical smoke.

## Smoke decision

Oracle, task-only, joint-calibrated and routing-only-for-delays smoke cells are
finite, delay-legal and artifact-complete. Both accuracy and schedule
checkpoints, predictions, runtime NPZ/panel and resource ledger are present.
Accuracy was not a smoke gate. Formal recovery is authorized unchanged.
