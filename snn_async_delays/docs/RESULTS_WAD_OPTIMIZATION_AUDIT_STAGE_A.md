# WAD optimization audit v1 — Stage A result

Date: 2026-07-13. Status: exploratory validation-only audit; test split sealed.

## Completeness and integrity

All 18/18 preregistered cells completed. Each cell contains
`validation_results.json`, `config.json`, `train_log.csv`, `best_model.pt`, a
diagnostic NPZ and a diagnostic panel. The summary was generated from fresh
validation results and all 100 logged epochs; no historical result was edited.

## Primary threshold decision

| threshold | scalar worst-query | WAD worst-query | scalar hidden spikes | WAD hidden spikes | viability |
|---:|---:|---:|---:|---:|:---|
| 0.2 | 0.712 | 0.687 | 17.513 | 17.128 | pass |
| 0.3 | 0.689 | 0.676 | 10.238 | 9.894 | pass |
| 0.5 | 0.537 | 0.597 | 2.859 | 2.495 | fail |

Threshold 0.3 is selected for Stage B. Across both conditions and all seeds its
mean hidden spike count is 10.066, only 0.066 from the preregistered target of
10. Threshold 0.2 is viable but averages 17.321 spikes (distance 7.321).
Threshold 0.5 fails the per-cell worst-query gate: scalar seed 1 is 0.466 and
WAD seed 1 is 0.548, below the preregistered 0.55 lower bound.

This is a dynamics/optimization selection, not a performance selection.
Threshold 0.3 does not maximize WAD accuracy.

## Optimization diagnostics

At threshold 0.3, all WAD epochs have finite, nonzero delay gradients. Mean
delay-gradient norm across training is 0.0449; final mean absolute effective
delay movement averages 0.989 steps (minimum 0.883), and final saturation is
0 in every seed. Thus the baseline WAD failure cannot be attributed to exactly
zero delay gradients, immobile delays, or boundary saturation.

The threshold strongly changes the firing regime: across-condition activity
falls from 17.321 spikes at 0.2 to 10.066 at 0.3 and 2.677 at 0.5. The diagnostic
panels qualitatively agree, but fixed-sample rasters and mechanism plots are
illustrative only and played no role in selection.

## Reliability interpretation

At threshold 0.3, paired WAD minus scalar worst-query differences are
`-0.028, -0.060, +0.050` for seeds `0,1,42` (mean `-0.0127`; 1/3 positive).
WAD therefore still has no Stage-A superiority signal. Exact-trial differences
are `+0.048,+0.004,+0.016`, showing that pooled joint success and worst-query
reliability can rank interfaces differently. The preregistered primary metric
remains worst-query accuracy.

Threshold 0.5 superficially favors WAD in two seeds, but this occurs in a
low-activity, near-floor regime that fails the preregistered gate. Treating it
as positive WAD evidence would be cherry-picking.

## Decision

Proceed to the locked Stage-B optimization audit at threshold 0.3. Stage B is
diagnostic: it may identify a better WAD optimization recipe, but it cannot
retroactively turn the completed negative superiority experiment into a
positive confirmatory result.

Generated evidence:

- `docs/generated/wad_optimization_audit_v1/stage_a_cells.csv`
- `docs/generated/wad_optimization_audit_v1/stage_a_grouped.csv`
- `docs/generated/wad_optimization_audit_v1/stage_a_decision.json`
