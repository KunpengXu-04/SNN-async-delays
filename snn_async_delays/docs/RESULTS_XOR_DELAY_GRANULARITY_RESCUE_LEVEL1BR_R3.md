# Results: XOR dimension-aware delay rescue Level 1B-R R3

**Protocol:** `xor_delay_granularity_rescue_level1br_v1`  
**Execution device:** NVIDIA GeForce RTX 4070 Laptop GPU  
**Date:** 2026-07-16  
**Status:** sealed R3 complete, 30/30; Level 1B-R passes.

## Confirmatory result

| Candidate | Independent delay parameters | Full pass | Exact interface | Correct initial direction on every coordinate | Full delay coverage | Max-error range (step) | Clipping flags |
|---|---:|---:|---:|---:|---:|---:|---:|
| global anchor | 1 | 10/10 | 10/10 | 10/10 | 10/10 | .000071-.017347 | 0/10 |
| per-hidden coordinate-matched | 16 | 10/10 | 10/10 | 10/10 | 10/10 | .000043-.000497 | 0/10 |
| per-synapse coordinate-matched | 64 | 10/10 | 10/10 | 10/10 | 10/10 | .000040-.000217 | 0/10 |

All candidates pass from both low and high initial delays on all five sealed
seeds. The initial weighted arrival-gradient magnitude per coordinate remains
approximately `.025588` in every candidate, as analytically predicted.

## Registered decision

- Global confirmation gate: pass.
- Passing higher-dimensional granularities: per-hidden-neuron and per-synapse.
- Selected rescued granularity: **per-hidden-neuron**.
- Selection reason: the preregistered priority chooses 16 independent delay
  parameters over 64 when both pass; this is not a post-hoc accuracy choice.
- R2: not run and not required.
- Test split: unopened.
- Micro-burst and K>1: still unauthorized by this protocol.

## Artifact audit

All 30 formal cell directories contain the checkpoint, configuration, strict
metrics, training log, exhaustive truth-table evaluation, resource ledger,
runtime diagnostic NPZ and runtime diagnostic panel. Missing artifacts: zero.

The aggregate table and mechanical decision are:

- `docs/generated/xor_delay_granularity_rescue_level1br_v1/r3/cells.csv`
- `docs/generated/xor_delay_granularity_rescue_level1br_v1/r3/decision.json`

## Scientific interpretation

Together, R1 and R3 establish a reproducible optimization result: the
Level-1B failure of the frozen mean scaffold is rescued by holding explicit
oracle-teacher strength constant per independent delay coordinate. A
per-hidden tying scheme is sufficient and is preferred to per-synapse tying
for this homogeneous target because the additional 48 delay parameters do not
improve an already exact endpoint.

This is not evidence that XOR supervision discovers timing, that heterogeneous
delays self-organize, or that temporal multiplexing reduces resources. The
teacher explicitly supplies the desired delay and its total strength grows
with delay dimension. Those questions require a new protocol and cannot reuse
these sealed confirmation seeds for tuning.
