# Mixed-op temporal WAD scaffold withdrawal v1

Status: complete. Smoke/W0 pass; W1 annealing retains the schedule, but W2
task-only restoration fails 0/6 while centroid restoration passes 6/6. The
sealed test remains locked.

Runner: `python -m scripts.run_mixedop_temporal_wad_scaffold_withdrawal`.

## Why task-only currently fails

The evidence does not support a simple gradient-vanishing diagnosis. V2
task-only ends at worst-query balanced accuracy .50 in every seed, has silent
output windows, and reaches maximum schedule errors above ten steps. Yet delay
gradients are measurable. The stronger explanation has three parts:

1. **The label loss does not identify a unique internal clock.** The MLP and
   weights can compensate for timing changes, and many delay vectors can produce
   the same Boolean labels. Classification correctness is therefore weaker than
   schedule correctness.
2. **Credit disappears or conflicts after coverage is lost.** When a late
   query produces no activity in its target window, BCE supplies a long,
   indirect gradient through shared hidden dynamics. Different queries can ask
   the same five delay coordinates to move in conflicting directions.
3. **The parameter geometry amplifies the problem.** All delays start together,
   while q4 must traverse much farther in sigmoid raw space. Near support edges,
   equal physical movement requires increasingly large raw-coordinate changes.

V3 proves that the forward architecture and optimizer can work when an explicit
query-to-window signal is supplied. It does not show that task BCE contains
enough information to discover that schedule.

## Questions separated by this protocol

- **W0 — scaffold reproducibility:** can V3's supervised interface be rebuilt
  on three untouched seeds?
- **W1 — retention:** after the correct interface exists, does task-only credit
  preserve it? This is maintenance, not discovery.
- **W2 — local restoration:** after a declared schedule perturbation, does task
  BCE move delays back, or do weights/readout merely compensate?

These questions must not be merged. A successful frozen-delay arm establishes
only that weights can keep classification working. A successful abrupt arm
establishes task-only retention from a warm start. Only W2 can support a narrow
claim about local schedule identifiability.

## Frozen design

All scientific settings remain those of V3: K=5 fixed mixed operations,
event8 rate packet, N=120, w=4, T=30, five query-tied sigmoid delays on `[0,23]`,
threshold .2 arbitrary units, shared per-window MLP, Adam and LR .01.

W0 trains three new seeds with centroid supervision for 400 updates. If any
seed lacks a single checkpoint jointly passing function, activity and <=1-step
schedule error, the protocol stops.

W1 starts from the corresponding W0 checkpoint and runs 200 updates with four
arms: frozen delays, continued centroid supervision, abrupt task-only credit,
and a fixed 100-update linear auxiliary anneal followed by 100 task-only
updates. The final checkpoint is primary; intermediate best checkpoints cannot
rescue a failed withdrawal endpoint.

W2 returns to the untouched W0 checkpoint and applies either a uniform two-step
early shift or the fixed offsets `[0,-1,-2,-3,-4]`. Frozen, centroid-restoration
and task-only-restoration arms run from identical perturbed states. Task-only
must pass function and schedule at the final update and reduce initial schedule
error by at least 50% for both perturbations in all three seeds.

## What could solve the task-only problem

This protocol tests the least invasive solution first: establish a temporal
scaffold, then anneal it away. If task-only retains and locally restores the
schedule, a curriculum may be defensible, provided the paper clearly reports
the training-time timing supervision.

If retention succeeds but restoration fails, the internal schedule is locally
non-identifiable under Boolean BCE. The next method should change the task or
architecture, not merely LR: enforce monotone base-plus-positive-spacing delay
coordinates, randomize query-to-window assignments with explicit cues, and use
counterfactual timing perturbations that make incorrect routing change the
task loss.

If even retention fails, task and timing gradients are actively incompatible.
Then a persistent multi-objective constraint or bilevel/projection method is
needed; calling that autonomous task-only WAD would be incorrect.

Per-synapse delays, K=8, Pareto surfaces and sealed test remain prohibited
throughout this protocol.

## Implementation smoke result

The four seed-3649 cells are artifact-complete, finite and delay-legal. Every
source checkpoint hash is valid, state dictionaries load strictly, and no
optimizer state is inherited. Abrupt task-only has nonzero delay gradients;
the annealed arm exactly follows the registered update-wise lambda schedule.
Runtime NPZ, diagnostic panel and resource ledger are present for every cell.

Smoke accuracy is not a decision variable and is invalid for scientific claims.
Its near-perfect values, including the apparent 20-update task-only schedule
improvement, cannot be reported as retention evidence because all arms reuse a
previous V3 checkpoint and the seed was selected only for implementation smoke.
The machine-readable decision is
`docs/generated/mixedop_temporal_wad_scaffold_withdrawal_v1/smoke_decision.json`.

## W0 result

The scaffold passes all three new seeds. Accuracy-first source checkpoints are
selected at updates 380/320/400 and jointly achieve worst-query balanced
accuracy 1.000/1.000/.99951, exact-trial 1.000/1.000/.99951, activity in every
window, and maximum schedule error .50034/.50143/.50000 steps. W1 is therefore
authorized without changing its arms, update budget, LR or gates. See
`RESULTS_MIXEDOP_TEMPORAL_WAD_SCAFFOLD_WITHDRAWAL_W0.md`.

## W1 result

Frozen and continued-centroid controls pass 3/3. Abrupt task-only retains
near-perfect classification but passes the schedule gate only 1/3, with final
maximum errors .781/1.114/1.280 steps. The fixed annealed arm passes 3/3 with
errors .386/.897/.501 after its final 100 task-only updates. This authorizes W2
but supports only curriculum-assisted warm-start retention. See
`RESULTS_MIXEDOP_TEMPORAL_WAD_SCAFFOLD_WITHDRAWAL_W1.md`.

## W2 result

The positive centroid control restores both perturbations in every seed.
Task-only passes 0/6: final schedule errors are 1.354--1.710 after uniform
damage and 1.884--2.199 after compressed spacing, despite essentially perfect
classification. This rejects local schedule identifiability under the current
Boolean BCE. See
`RESULTS_MIXEDOP_TEMPORAL_WAD_SCAFFOLD_WITHDRAWAL_W2.md`.
