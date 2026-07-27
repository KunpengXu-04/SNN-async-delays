# Event8-aligned mixed-operation surface v2

Status: the frozen 25-cell landmark pilot is complete. Its preregistered full-
sweep gate is negative, so the full sweep and sealed test remain locked. A
post-run audit also identifies a centroid-loss/oracle-gate inconsistency; see
`RESULTS_MIXEDOP_EVENT8_ALIGNED_SURFACE_V2_LANDMARK.md`.

## Purpose and boundary

This is a single-seed exploratory supervisor-preview branch. It asks how total
hidden neurons and latency affect reliability after the input packet and fixed
oracle were analytically aligned. It does not repair the W2 identifiability
failure and cannot establish autonomous temporal multiplexing.

The event8 packet occupies steps 6--9. Therefore output windows shorter than
four steps are excluded. For K=5,

\[
T=10+5w,\qquad d_q=3+qw.
\]

The full frozen domain is

\[
N_{hid}\in\{20,40,80,120,160,240\},\quad
w\in\{4,6,8,12\}.
\]

## Five landmarks

The pilot runs `(20,4)`, `(20,12)`, `(240,4)`, `(240,12)` and `(80,8)`.
The last is called the geometric center because it is closest to the geometric
center of both registered axes; `(120,6)` is not silently retained as a center.

Every point compares:

1. independent spatial d0;
2. shared hidden/readout d0;
3. hand-scheduled event8 oracle;
4. from-scratch task-only five-delay WAD diagnostic;
5. curriculum-assisted five-delay temporal model.

The curriculum condition trains a 400-update centroid scaffold and then resets
the optimizer for the frozen W1 withdrawal schedule: 100 updates annealing
lambda from one to zero and 100 fully task-only updates. Its primary endpoint is
the final checkpoint. It must always be labelled curriculum-assisted.

## Gate

Technical smoke checks the new shared-d0 implementation, event8/oracle
schedule, task-only delay gradients, curriculum phase transition, numerical
health, and runtime artifacts. Accuracy is not a smoke gate.

The full sweep remains locked unless the center and both high-N oracle points
pass feasibility, the curriculum center passes, at least one high-N curriculum
point passes, and at least three curriculum landmarks jointly pass function,
activity and <=1-step schedule error. Task-only is diagnostic and is not used
to authorize the sweep.

Even a positive surface permits only a descriptive single-seed statement. It
does not justify significance, universality, hardware energy, per-synapse WAD,
or autonomous routing claims.

## Implementation and execution boundary

The runner is `scripts/run_mixedop_event8_aligned_surface_v2.py`. It writes a
runtime NPZ, diagnostic panel, validation predictions, checkpoint pair,
training and validation logs, and a resource ledger for every cell. The
curriculum arm additionally records its scaffold source and verifies that the
optimizer is recreated before withdrawal.

Dry-run enumeration is always safe and does not unlock training:

```powershell
python -m scripts.run_mixedop_event8_aligned_surface_v2 --stage landmark --dry-run
```

The completed five-cell implementation smoke was launched with:

```powershell
python -m scripts.run_mixedop_event8_aligned_surface_v2 --stage smoke --device cuda
```

Its independent artifact and numerical audit passed. The landmark was launched
with:

```powershell
python -m scripts.run_mixedop_event8_aligned_surface_v2 --stage landmark --device cuda
```

The runner still refuses the full surface because the registered landmark gate
failed. No change to the endpoint, loss, grid or authorization may be made
inside v2 after observing these results.
