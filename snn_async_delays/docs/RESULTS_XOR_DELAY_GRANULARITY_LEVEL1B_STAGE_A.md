# Results: XOR delay granularity Level 1B Stage A

**Protocol:** `xor_delay_granularity_level1b_v1`  
**Date:** 2026-07-16  
**Status:** 60/60 formal Stage-A cells complete; global replication passes;
higher-dimensional extension fails; fixed micro-burst controls authorized.

## 1. Scope

Stage A tests whether the narrow Level-1A scaffold-assisted K=1 XOR bridge
survives a change from one global input-hidden delay to one delay per hidden
neuron or one delay per input-hidden synapse. It uses the exhaustive four-row
XOR training domain, not a held-out task set. The five seeds are new relative
to Level 1A, so this is an initialization replication, not a generalization
test.

All models retain the `4 -> 16 -> 2` hard-spiking opponent interface, one event
per selected input channel at `t=9`, hidden-to-output d0 transmission and one
exact target output spike at `t=15`. The frozen loss is hard filtered-spike
loss plus either no timing teacher or `lambda=.01` times the per-coordinate
delay-4 arrival-centroid scaffold.

## 2. Completion and integrity

All 60 declared cells completed. Every cell contains:

- frozen cell config and final checkpoint;
- strict metrics and full 501-state training log;
- exhaustive truth-table output;
- resource ledger;
- runtime compressed NPZ; and
- runtime 12-panel diagnostic.

No test split was opened. No cell, seed, initialization direction or failed
candidate was excluded.

## 3. Preregistered candidate decisions

| Delay granularity | Trainable delay values | Task-only pass | Scaffold pass | Candidate decision |
|---|---:|---:|---:|---|
| global | 1 | 1/10 | **10/10** | global replication passes |
| per hidden neuron | 16 | 0/10 | 2/10 | extension fails |
| per synapse | 64 | 0/10 | 0/10 | extension fails |

The global scaffold is the only candidate passing the required 10/10 rule.
It succeeds in both initialization directions and every new seed. Its maximum
final delay error is `.001864` step; all ten cells retain balanced accuracy
one and an exact output spike train.

Task-only delay learning remains unreliable. Only one global cell passes, and
neither higher-dimensional task-only candidate passes any cell. This does not
reverse the Level-1A conclusion that task supervision alone is not a robust
delay-learning method.

## 4. Failure localization

The per-hidden scaffold solves the hard output interface in 10/10 cells, but
only 3/10 place every delay within `.1` step and only 4/10 have the required
correct initial total-gradient direction for every coordinate. The intersection
of all gates contains only 2/10 cells. Thus its primary failure is distributed
delay recovery, not XOR classification or output conversion.

The per-synapse scaffold is weaker: exact interface passes 7/10, delay coverage
passes 0/10 and all-coordinate initial direction passes 0/10. Every cell has a
nonzero gradient on every coordinate, so the failure is not global gradient
silence. Instead, task and scaffold components yield heterogeneous coordinate
directions and convergence rates.

Descriptively, per-hidden tying is substantially more stable than per-synapse
tying under the frozen recipe, but 2/10 versus 0/10 is not a positive
candidate result and is not sufficient to select per-hidden delays as a new
method.

## 5. Important normalization limitation

The preregistered scaffold is a mean over independent coordinates:

\[
L_{arr}=\frac{1}{2P}\sum_{p=1}^{P}[\mu_p(d_p)-\mu_p(4)]^2.
\]

Consequently, under matched homogeneous initialization the initial absolute
arrival gradient per coordinate is approximately `2.5588/P`: `2.5588` for the
global scalar, `.159925` for 16 per-hidden coordinates and `.039981` for 64
per-synapse coordinates. With fixed `lambda=.01`, delay LR `.01` and 500
updates, increasing delay dimension therefore also weakens the scaffold force
on each coordinate.

This is not an implementation error: it is the frozen mean-loss definition.
It does constrain interpretation. Stage A shows that the **frozen Level-1A
optimization recipe does not scale to higher-dimensional delay coordinates**.
It does not establish that per-neuron or per-synapse delays are intrinsically
untrainable, nor does it cleanly separate parameter dimension from
per-coordinate supervision strength. Any sum-normalized or dimension-scaled
teacher would be a new post-result protocol, not a reinterpretation of Stage A.

## 6. Resource accounting

The physical architecture and dynamic dense work are unchanged: all three
models have 64 input-hidden synapses, 360 neuron updates per trial, 1,920 dense
synaptic MACs per trial and 180 delay-buffer elements per sample. What changes
is parameter and delay-value storage:

| Granularity | Total trainable parameters | Trainable input-delay parameters | Total delay-value storage elements |
|---|---:|---:|---:|
| global | 97 | 1 | 33 |
| per hidden neuron | 112 | 16 | 48 |
| per synapse | 160 | 64 | 96 |

The storage totals include the 32 fixed d0 hidden-output delay entries. These
counts do not imply energy or compute savings.

## 7. Decision and claim boundary

The preregistered global held-out-seed replication gate passes 10/10, so the
ten fixed micro-burst controls are authorized. The higher-dimensional extension
gate fails. This means:

- proceed only to `stage_b_controls`;
- do not yet run the 60 learned micro-burst cells;
- do not select per-hidden tying from its two successful cells;
- do not claim heterogeneous delay specialization, temporal routing, WAD
  superiority, K scaling, compression or a Pareto advantage; and
- do not describe the global scaffold as autonomous task-derived timing.

The next formal action is the complete five-seed fixed-delay-4/fixed-d0
micro-burst control matrix. Learned Stage B remains mechanically locked until
fixed delay 4 passes 5/5.

## 8. Artifacts

- cells: `runs/exploratory/xor_delay_granularity_level1b_v1/stage_a/`
- decision: `docs/generated/xor_delay_granularity_level1b_v1/stage_a/decision.json`
- aggregate cells: `docs/generated/xor_delay_granularity_level1b_v1/stage_a/cells.csv`
- gate heatmap: `docs/generated/xor_delay_granularity_level1b_v1/stage_a/granularity_gate_heatmap.png`
