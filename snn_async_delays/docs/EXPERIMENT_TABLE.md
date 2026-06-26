# Experiment Design Table

Each row is one experimental *configuration family* (a set of runs sharing the same structural design).
Runs within a family differ only in K, h, seed, or minor hyperparameters.

Folder naming convention (as of 2026-06):  
`runs/{characteristic_description}_(step_plan_code)/` — codes in parentheses are annotations.

## Column Glossary

| Column | Meaning |
|---|---|
| **Folder** | Directory under `runs/` |
| **K** | Queries per trial (temporal multiplexing load) |
| **#Ops** | Number of distinct boolean operations per trial |
| **h tested** | Hidden neuron counts evaluated |
| **Readout** | linear / MLP / spiking |
| **Status** | Done / TODO |

---

## Completed Experiments

| Folder | K | #Ops | h tested | Readout | Key finding | Status |
|--------|---|------|----------|---------|-------------|--------|
| `single_op_(step1)` | 1 | 1 (each run) | 10, 20, 50 | linear | wad dominates; delays_only collapses on XOR/XNOR | **Done** |
| `NAND_serial_slots_(step2_planA)` | 1–20 | 1 | 20 | linear | K=20+ achievable — delays do *time alignment*, not routing | **Done** |
| `NAND_simul_channels_(step2_planC)` | 1–12 | 1 | 20 | linear | Spatial channels handle separation; delay adds +2.7% | **Done** |
| `NAND_time_mux_(step2_planD)` | 1–12 | 1 | 20, 50 | linear + MLP | Linear: Max K@90%=2; MLP: Max K@90%=3. Delays sole necessary mechanism | **Done** |
| `NAND_depth_ablation` | 2–4 | 1 | L1-h50, L2-h25+25 | linear + MLP | Depth neutral for linear; MLP benefit scales with readout input size | **Done** |
| `NAND_timing_ablation` | 3–4 | 1 | 50 | MLP | Timing knobs (τ_m, sub_win, read_len) net zero or negative — ceiling is representational | **Done** |
| `NAND_neuron_sweep_(planD)` | 1–5 | 1 | 10, 20, 30, 50 | MLP | Capacity scales with h; h=10 collapses at K≥2 | **Done** |
| `8op_mixed_(step3)` | 1–4 | 8 | 50 | MLP | Delay gap +11-20%; data sparsity limits abs. accuracy | **Done** |
| `4op_mixed_1k_(step3)` | 1–4 | 4 | 50 | MLP | Multi-function learning bottleneck: K@90%=0 even at K=1 | **Done** |
| `4op_mixed_16k_(step3)` | 1–4 | 4 | 50 | MLP | Data hypothesis confirmed: K@90%=2 with 16k samples | **Done** |
| `4op_mixed_16k_h100_(step3)` | 1–4 | 4 | 100 | MLP | h=100 lifts K@90%=2→3, matching single-op ceiling | **Done** |
| `4op_mixed_16k_2layer_(step3)` | 1–4 | 4 | 50+50 | MLP | 2-layer at fixed 100-neuron budget drops K@90% back to 2 | **Done** |
| `NAND_spiking_out` | 1 | 1 | 50 | spiking LIF | 200 ep: wad=54%, d0=35%; BCELoss on spike counts is hard to train | **Done** |
| `NAND_spiking_out_1k` | 1 | 1 | 50 | spiking LIF | 1000 ep: wad=71.8%, d0=43.6%; delay advantage +28%; still far below linear (95%) | **Done** |
| `one_query_many_op_(step4)` | — | 1 broadcast → K_ops=2..4 | 50 | linear + MLP | Gap +14–19pp (flat, K_ops-invariant); linear>MLP (reversal); no ceiling at K_ops=4 | **Done** |
| `many_query_one_out_(step4)` | 1,3,5 | 1 (NAND) | 50 | linear + MLP | d0 BalAcc≈50% (trivial prior); majority Max K@BalAcc≥70%=3; AND Max K@BalAcc≥70%=5 | **Done** |

---

## Key Results Summary

| Experiment | Max K @ 90% | Max K @ 95% | Mechanism |
|---|---|---|---|
| Step 1 (single-op) | N/A (K=1) | 15/24 runs | Single-op solvability |
| Plan A (h=20) | K=20+ | K=20+ | Slot isolation (LIF decay), not routing |
| Plan C (h=20) | K≈5 | K=2 | Spatial channels + delay alignment |
| Plan D (h=20, linear) | K=2 | K=1 | Temporal routing |
| Plan D (h=50, linear) | K=2 | K=1 | Temporal routing |
| Plan D (h=50, MLP) | **K=3** | K=1 | Delay + nonlinear decoder |
| Plan D neuron sweep | K@90% breaks at h<20 | — | Capacity scales with h |
| Step 3: 4op 16k h=50 | K=2 | K=1 | Data volume + delays |
| Step 3: 4op 16k h=100 | **K=3** | K=1 | Capacity + delays |
| Step 3: 4op 16k 2-layer | K=2 | 0 | Depth costs effective capacity |
| Spiking output (1000 ep) | 0 | 0 | BCELoss on spike counts: hard to optimise |
| Step 4a: 1-query K_ops=4 (linear) | — | — | Gap flat +18pp (alignment only); linear>MLP; no ceiling found |
| Step 4b: K=3 NAND, majority (MLP) | K=3 (BalAcc=76.4%) | — | MLP>linear for aggregation; d0 BalAcc≈50% at K≥3 |
| Step 4b: K=5 NAND, AND-of-results | K=5 (BalAcc=75.0%) | — | AND-of-results gives cleanest d=0 structural-failure evidence |

---

## Pattern Taxonomy

```
Pattern name          Input structure           Output structure
─────────────────────────────────────────────────────────────────
one-query-one-op      1 (A,B) × 1 op            1 logit            [Step 1]
many-query-one-op     K (A,B) × same op          K logits           [Plan A/C]
many-query-one-op     K (A,B) × same op          1 shared MLP       [Plan D]  ← primary
many-query-multi-op   K (A,B) × K diff ops       K logits           [Step 3]
─────────────────────────────────────────────────────────────────
one-query-many-op     1 (A,B) × K_ops ops        K_ops logits       [Step 4a, Done]
many-query-one-out    K (A,B) × same op          1 aggregate        [Step 4b, Done]
many-many-one-out     K (A,B) × K diff ops       1 aggregate        [Step 4c, Done — Sec 26, Max K@90%=1]
```

---

## Notes

- **Plan A** max K is limited only by total trial length (K×35ms): delays serve for *time alignment*, not true multiplexing.
- **Plan D** is the only design where `d=0` structurally cannot solve the task: weights alone cannot distinguish sub-windows on shared channels.
- **Spiking output**: requires full 3-layer visualisation (input→hidden→output spike trains). Training is hard because spike counts as BCEWithLogitsLoss logits give weak gradients through the surrogate. Linear readout (95%+) remains the better readout for Plan D experiments.
- **Folder naming**: descriptive characteristic names first, experiment codes (step1/planD/etc.) in parentheses as annotations.
