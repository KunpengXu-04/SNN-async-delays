# Results: simultaneous temporal viability preflight v1

Date: 2026-07-13  
Protocol: `simultaneous_temporal_viability_preflight_v1`  
Decision: **failed; full temporal-routing matrix remains locked**

## Completeness and decision discipline

All six preregistered seed-0 cells completed with finite logs/results. Every
cell contains config, best/final checkpoints, training log, validation,
exhaustive 64-pattern truth table, resource ledger, diagnostic NPZ and a panel
generated during the run. Test remains sealed.

Accuracy was descriptive and did not enter the decision. The pass/fail result
uses only the preregistered event-support and gradient gates.

## Locked gate results

| Gate | Observed | Result |
|---|---|---|
| All six cells complete and finite | yes | pass |
| Scaffold hidden activity fraction in each window >= .10 | `[1.00,1.00,.75]` | pass |
| Scaffold output silent rate < .50 | `.216` | pass |
| Fixed-full hidden activity fraction in each window >= .05 | `[1.00,1.00,1.00]` | pass |
| Fixed-full nonzero output-gradient epoch fraction >= .95 | `1.00` | pass |
| WAD hidden activity fraction in each window >= .05 | `[1.00,.606,0.00]` | **fail** |
| WAD nonzero output-gradient epoch fraction >= .95 | `1.00` | pass |
| WAD nonzero delay-gradient epoch fraction >= .95 | `1.00` | pass |
| WAD final mean delay movement >= .05 | `2.572` | pass |
| WAD final delay saturation fraction < .95 | `0.00` | pass |

The preflight therefore fails exactly one locked gate: WAD provides no hidden
event support in the third output window.

## Geometry diagnosis

The controls localize the failure:

- d0 has no hidden activity in any post-input window and is fully silent;
- scalar has activity only in window 1 (`[.728,0,0]`);
- narrow fixed `[1,9]` has activity only in window 1 (`[1,0,0]`);
- full-support fixed `[0,30]` covers all windows (`[1,1,1]`);
- the scaffold covers all windows (`[1,1,.75]`);
- WAD covers windows 1 and 2 but not window 3 (`[1,.606,0]`).

Thus the simulator, window metrics, shared opponent output and training path can
represent late support: both scaffold and frozen full-support control do so.
The WAD failure is not explained by a global implementation bug, an entirely
silent output layer, zero output gradient, zero delay gradient, or delay
saturation.

The learned effective WAD delays are concentrated near the beginning of the
horizon. Input-to-hidden delays have median `3.43`, 75th percentile `7.26`, max
`20.13`, and only `0.67%` are at least 20 steps. Hidden-to-output delays have
median `3.90`, 75th percentile `5.30`, max `14.20`, and none reach 20. This is
consistent with an early/middle-window solution that never creates active
paths into window 3. It is evidence of support collapse under the current
objective/initialization, not proof that learnable delays can never route late.

## Descriptive accuracy warning

WAD balanced accuracy by window is approximately `[.948,.878,.500]`; exact-
trial accuracy is `.528`. The apparently high pooled/exact values do not rescue
the protocol: the third NOR window is at balanced chance, and a silent/default
majority prediction benefits from NOR's class imbalance. Fixed-full is lower on
exact trial (`.414`) but is the only admissible non-scaffold condition with
event support in all windows. Preflight accuracy cannot select either method.

## Post-hoc mechanism-audit correction

The later read-only checkpoint audit found that hidden emission inside a window
was a mislocated proxy for delayed event arrival. WAD has zero new hidden spikes
in window 3 but nonzero output synaptic current in 70.3% of exhaustive trials,
output spikes in 34.4%, and 11.7% of normalized realized hidden-to-output
arrival mass in that window. Window-3 WAD delay/output gradients are also
nonzero. Thus the preregistered v1 gate formally failed, but it does **not**
establish absence of output event support. See
`RESULTS_SIMULTANEOUS_TEMPORAL_CHECKPOINT_MECHANISM_AUDIT_V1.md`.

## Scientific decision

1. Do not launch `simultaneous_temporal_routing_pilot_v1` in its current form.
2. Do not interpret the WAD exact-trial value as successful three-window
   routing; one window has zero hidden support and balanced accuracy .5.
3. V1 does not establish viability because its locked proxy was mechanistically
   invalid; performance nevertheless remains unsuccessful in window 3.
4. A new version must gate output arrival/current and window-specific gradients.
   Any objective change must apply to all conditions and be preregistered.
5. An N/T resource frontier for temporal routing is premature until a method
   passes support viability. A separately preregistered spatial resource study
   remains possible, but answers a different question.

## Artifacts

- Raw cells: `runs/exploratory/simultaneous_temporal_viability_preflight_v1/`
- Machine-readable decision:
  `docs/generated/simultaneous_temporal_viability_preflight_v1/decision.json`
