# WAD optimization audit v1 — Stage B result

Date: 2026-07-13. Status: completed exploratory optimization audit; test sealed.

All 42/42 cells contain validation results, config, 100-epoch train log,
checkpoint, diagnostic NPZ and panel. No variant passed the preregistered gate.

| variant | WAD worst | scalar worst | WAD−baseline | WAD−scalar | retained |
|---|---:|---:|---:|---:|:---:|
| baseline | .676 | .689 | .000 | -.0127 | no |
| narrow_d10 | .638 | .647 | -.038 | -.0093 | no |
| lr_low | .654 | .692 | -.022 | -.0380 | no |
| lr_high | .607 | .700 | -.069 | -.0933 | no |
| scalar_noise | .654 | .689 | -.022 | -.0347 | no |
| warmup20 | .624 | .644 | -.052 | -.0200 | no |
| alternating | .641 | .645 | -.0347 | -.0033 | no |

Every intervention lowers mean WAD worst-query relative to the fresh baseline.
No variant has mean gain >=.03 or gains >=.03 in 2/3 seeds. Delay motion is
material and saturation is zero, so failure is not explained by frozen or
boundary-clamped delays. `lr_high` moves delays most but performs worst,
arguing against the simple claim that WAD only needed stronger delay updates.

The optimization-rescue hypothesis fails. Baseline is frozen for later pilots:
hidden threshold .3, dmax 30, delay LR .001, constant raw initialization -2,
and joint weight/delay optimization. This does not establish that every
possible optimizer would fail; it establishes that none of the preregistered
compact, matched interventions rescues the result.

Across the three baseline WAD checkpoints, effective input-hidden delays span
.993–8.271 steps (5th–95th percentile 1.736–6.806; mean 3.566). The pilot's
fixed heterogeneous support [1,9] therefore covers the observed baseline range;
distribution matching remains approximate rather than exact.

Artifacts: `docs/generated/wad_optimization_audit_v1/stage_b_cells.csv`,
`stage_b_decisions.csv`, and `stage_b_decision.json`.
