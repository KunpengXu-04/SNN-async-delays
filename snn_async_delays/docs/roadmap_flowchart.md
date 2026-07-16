# Publication Roadmap Flowchart & Guide (Updated 2026-07-13)

Following the findings from `RESULTS_XOR_DELAY_CONTROL_MATRIX_V1.md` on **2026-07-13**, the publication roadmap has branched. **Gate 1 positive superiority failed** (an optimized scalar delay outperformed WAD on the primary XOR endpoint).

The active focus has shifted to a **causal decomposition / methodology-negative-result programme** to investigate whether WAD was optimization-limited and if WAD provides value in simultaneous routing.

---

## 1. Updated Phased Experimental Flowchart

```mermaid
graph TD
    %% Phase 0 & Phase 1A/B
    P0["Phase 0: Trustworthy Infrastructure<br>🔍 Complete"] --> P1A["Phase 1A: Difficulty Calibration<br>🔍 Complete"]
    P1A --> P1B["Phase 1B: Causal Matrix Sweep<br>🔍 Complete (Failed Superiority)"]
    
    %% Gate 1 Fail / Branching
    P1B --> G1{"Gate 1: WAD vs. Controls?<br>(Tested under all_time observation)"}
    G1 -- "FAILED: Optimized scalar beats WAD<br>(WAD is optimization-limited or lacks routing value)" --> BR_20260713["⚡ Branch Decision (2026-07-13)"]

    %% Active Branching Paths
    BR_20260713 --> P1C["1C: WAD Optimization Audit<br>(Screen thresholds {0.2, 0.3, 0.5}<br>& test schedules)<br>🔥 ACTIVE NEXT STEP"]
    BR_20260713 --> P1D["1D: Simultaneous Pilots<br>(Spatial Control vs. True Temporal Routing)<br>🔥 ACTIVE NEXT STEP"]

    %% Optimization Gate
    P1C --> G_OPT{"Optimization Gate"}
    G_OPT -- "Pass: Saturation fixed,<br>worst-query improves >= 0.03" --> P1C_Success["Update WAD configs<br>(Proceed to matching controls)"]
    G_OPT -- "Fail: Original WAD cannot be improved<br>or requires excessive tuning" --> G_OPT_Fail["Freeze original negative conclusion"]

    %% Simultaneous Gate
    P1D --> G_SIM{"Simultaneous Gate"}
    G_SIM -- "Pass: WAD beats scalar on routing,<br>4/5 seeds agree" --> P1D_Success["Escalate temporal routing claim"]
    G_SIM -- "Fail" --> G_SIM_Fail["Acknowledge spatial capacity only"]

    %% Reintegration to publication paths
    G_OPT_Fail & G_SIM_Fail & G1_Fail_Window --> PUB_METHOD["📝 Methodology Paper / Negative Result<br>(Motivated by XOR pilot / audit outcomes)<br>🎯 TARGET PATH"]
    P1C_Success & P1D_Success --> P2["Phase 2: Scaling & Frontier<br>(K, N, T Pareto analysis)<br>🔒 Gated"]

    %% Post Phase 2 flows (preserved structure)
    P2 --> P3["Phase 3: Causal Mechanism Tests<br>(Shuffle/replacement interventions)"]
    P3 --> P4["Phase 4: Robustness Verification<br>(Burst timing noise & jitter)"]
    P4 --> P5["Phase 5: External Benchmark<br>(SHD / SSC under same discipline)"]
    P5 --> P6["Phase 6: Confirmation & Manuscript<br>(Rerun all headline cells from clean CLI)"]
    P6 --> PUB_CONF["🏆 Major SNN/AI Conference Paper<br>(Method/empirical paper)"]

    %% Specific sub-branches
    G1 -- "WAD matches fixed heterogeneous but beats d0" --> PUB_STRUCT["📝 Delay Structure Paper<br>(No learning novelty)"]
    G1 -- "Advantage only in late_window" --> G1_Fail_Window["Methodology paper on censoring artifacts"]
```

---

## 2. Updated Scientific Gates & Branching Actions (2026-07-13)

### 2.1 1C: WAD Optimization Audit (`wad_optimization_audit_v1`)
*   **Objective**: Diagnose if WAD failed Gate 1 due to optimization constraints.
*   **Stage A (Screening)**: Test threshold values $\{0.2, 0.3, 0.5\}$ checking for firing and gradient viability.
*   **Stage B (Optimization Schedules)**: Grid-search combinations of $d_{\max}$, delay learning rates, and warm-up/alternating training schedules on viable thresholds.
*   **Optimization Gate**: WAD must improve validation worst-query accuracy by $\ge 0.03$ over the original baseline in at least 2/3 of seeds without getting more tuning budget than the scalar delay. Otherwise, the original negative conclusion is frozen.

### 2.2 1D: Simultaneous-Input Pilots
*   **Spatial Control (`simultaneous_spatial_control_pilot_v1`)**: Multiple simultaneous query inputs mapped to independent outputs. Measures spatial parallel multitask capacity, not temporal routing.
*   **True Temporal Routing (`simultaneous_temporal_routing_pilot_v1`)**: Shared hidden layer, simultaneous inputs, and a shared opponent output pair reused across ordered output windows. Measures true delay-based routing.
*   **Simultaneous Gate**: WAD must outperform the scalar delay and fixed-delay controls on worst-window and exact-trial reliability in at least 4/5 paired seeds.

---

> [!WARNING]
> **Data Integrity Constraint**:
> Under the current negative-result branch, **do not open the sealed test split** or expand parameters ($K/N$) to avoid p-hacking. All current calibration and audit runs must remain validation-only.
