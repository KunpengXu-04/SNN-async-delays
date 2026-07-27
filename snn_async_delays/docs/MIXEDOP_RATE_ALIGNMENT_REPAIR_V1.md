# Mixed-Operation Rate Alignment Repair v1

Status: Stages A and B are complete. New-seed oracle passes, but task-only and
routing-assisted five-delay arms both fail 0/3; Stage C remains locked.

## Failure being repaired

The previous rate pilot placed spikes throughout steps 0--9 but inherited the
burst schedule `d_q=q*w`. With the simulator's one-step buffer offset, an input
event at step `s` arrived at `s+q*w+1`, while output window q began at
`10+q*w`. Only the final input step directly entered its declared window. The
pilot therefore tested a misaligned rate packet.

This protocol preserves those negative artifacts and creates a new interface.
Rate events are restricted to steps 6--9. The fixed schedule is

\[
d_q=3+qw,
\]

so the packet arrives at steps `10+qw` through `13+qw`. All candidate windows
have length at least four and therefore contain the complete packet.

## Stage A: fixed-oracle feasibility

The K=5 workload remains `[AND,OR,XOR,XNOR,NAND]`. Stage A crosses two expected
event budgets (4 and 8 events/query), total hidden neurons 120/240, window
length 4/6/8/12, and three new seeds, for 48 oracle cells after two technical
smokes. Threshold `.2`, all learning rates `.01`, 200 updates, the balanced
BCE, fixed stochastic validation realization and MLP window decoder remain
unchanged.

A candidate `(event budget,N,w)` passes only if all three seeds achieve worst-
query balanced accuracy at least `.90`, exact-trial accuracy at least `.80`,
at least `.10` trial-level hidden activity in every output window, and an exact
oracle tensor. The selected candidate minimizes measured events, then `N*T`,
then N and T. If no candidate passes, Stage B is not run.

## Stage B: five-parameter delay test

Only a selected Stage-A interface can materialize Stage B. Input channels from
the same query share one trainable delay, reducing 2400 or 4800 independent
values to five. Three new seeds compare a fixed oracle control, task-only
learning, and routing-assisted learning.

The routing-assisted arm adds a population arrival-mass cross entropy that
places query q in output window q. Its weight is `.10` and temporal sigmoid
temperature `.75` step. This is explicit routing supervision. A success in
this arm cannot be described as autonomous temporal-routing discovery.

Function and internal schedule are separate gates. Each WAD seed must meet
worst-query `.85`, exact-trial `.70`, activity `.10` in every window, and
maximum error of every one of the five query delays at most one step. A later
scaffold-withdrawal stage remains locked.

The machine-readable source of truth is
[`configs/mixedop_rate_alignment_repair_v1.yaml`](../configs/mixedop_rate_alignment_repair_v1.yaml).

## Smoke decision

Both event-budget smoke cells are finite, active, delay-legal and artifact-
complete. Their fixed delay tensors exactly equal `3+q*w`; runtime NPZ,
diagnostic panel and resource ledger are present. Accuracy was not used for
this technical decision. Stage A is authorized unchanged.

## Stage-A decision

All eight event8 candidates pass every three-seed gate, while all eight event4
candidates fail exact-trial reliability. The registered resource ordering
selects event8, N=120, w=4, T=30. See
`RESULTS_MIXEDOP_RATE_ALIGNMENT_REPAIR_STAGE_A.md`. Stage B must use this one
interface unchanged.

## Stage-B decision

Oracle passes 3/3 with perfect function and activity. Task-only and routing-
assisted arms remain at `.50` worst-query BAcc in every seed and fail schedule
recovery. Routing assistance provides nonzero gradients, reduces its own loss
and moves late delays, but q4 never reaches its window and task/routing credit
remain in conflict. See `RESULTS_MIXEDOP_RATE_ALIGNMENT_REPAIR_STAGE_B.md`.
