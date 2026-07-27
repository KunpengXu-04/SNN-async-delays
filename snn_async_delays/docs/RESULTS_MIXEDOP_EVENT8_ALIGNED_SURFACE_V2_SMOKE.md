# Event8-aligned mixed-operation surface v2: smoke audit

Status: complete and technically passed. These five short cells are invalid
for scientific claims. They authorize only the frozen 25-cell landmark pilot.

## Technical decision

All five cells are complete, finite and delay-legal. Each contains checkpoints,
training and validation logs, fresh validation predictions, runtime diagnostic
NPZ/panel, resource ledger and completion marker. Shared d0 remains exactly
zero, the fixed oracle tensor is exactly `[3,11,19,27,35]`, and the task-only
arm has nonzero delay gradients. The curriculum cell contains both scaffold
and withdrawal phases and records a fresh optimizer before withdrawal.

The generated `smoke_decision.json` therefore passes. No accuracy threshold was
used in this decision.

## Diagnostic observations

The 20-update oracle reaches worst-query balanced accuracy `.998535`, exact-
trial accuracy `.996582`, and activity fraction `1.0` in every output window.
This is a useful interface sanity check, but remains invalid smoke evidence.

Task-only selects its initial accuracy-best checkpoint, so its reported delays
remain `[3,3,3,3,3]` even though every training update has a nonzero aggregate
delay gradient (`.00635` to `.53838`). This is not evidence that optimizer
updates were absent; it is a consequence of accuracy-based checkpoint
selection in a deliberately short run.

The short curriculum run moves the five delays to approximately
`[3.34,4.35,4.31,4.29,4.32]` and activates only its first window. Its scaffold
source does not pass the scientific joint gate, which is expected because the
smoke uses only 20 scaffold updates rather than the registered 400. Twelve of
40 updates exceed the clipping threshold before clipping; all post-update
parameters remain finite and legal. This must be monitored in landmark logs,
but is not numerical failure.

The runtime panels agree with the numeric audit: the oracle raster and routing
plot cover all five declared windows, whereas the two learned smoke arms remain
concentrated in the first window. Decoder markers are explicitly labelled as
nonspiking decisions.

## Authorization boundary

The exact 25-cell landmark matrix is authorized without changing seeds, rates,
learning rates, threshold, budgets, grid points, checkpoint rules or gates.
The full 120-cell surface, K=8, sealed test and publication claims remain
locked. A positive curriculum condition must still be described as explicitly
timing-assisted, never autonomous WAD.

The post-audit YAML SHA-256 is
`BA5BEF79A5DED407B8724C936F608A75F6792E9D76A9F60B5171E8D7CE905F97`;
relative to preregistration, the scientific matrix is unchanged and only the
recorded smoke/landmark authorization state has advanced.
