# Results: temporal WAD scaffold withdrawal W2

Status: complete. Explicit centroid restoration passes 6/6; task-only local
schedule restoration passes 0/6. The withdrawal protocol therefore rejects
local delay identifiability under the current Boolean task BCE.

All 18 cells start from the corresponding untouched W0 source, verify its hash,
reset optimizer state, and apply the declared perturbation in physical delay
space before optimizer creation. Every cell is finite, delay-legal and
artifact-complete. The final update-200 checkpoint is primary.

## Frozen decision

| Perturbation | Arm | Joint passes | Final max schedule error | Error reduction |
|---|---|---:|---|---|
| uniform early -2 | frozen | 0/3 | 1.567/1.980/1.536 | approximately 0 |
| uniform early -2 | centroid restore | 3/3 | .502/.502/.501 | .680/.746/.674 |
| uniform early -2 | task-only restore | 0/3 | 1.685/1.710/1.354 | -.075/.137/.119 |
| compressed spacing | frozen | 0/3 | 3.567/3.980/3.536 | approximately 0 |
| compressed spacing | centroid restore | 3/3 | .507/.507/.501 | .858/.873/.858 |
| compressed spacing | task-only restore | 0/3 | 2.061/2.199/1.884 | .422/.448/.467 |

Centroid restoration returns both perturbations near the window-centering
solution and preserves near-perfect function and all-window activity. The
positive control therefore rules out insufficient update budget, an incapable
forward architecture, or a generic delay-optimizer failure.

Task-only restoration retains or regains essentially perfect classification in
all six cells, but never reaches the one-step schedule gate. Under uniform
damage it barely improves, and one seed moves farther from the schedule. Under
compressed spacing it reduces error by 42--47%, still below the registered 50%
minimum and with 1.88--2.20 steps of residual error.

The frozen uniform controls are especially diagnostic: classification remains
near perfect despite 1.5--2.0-step schedule errors. Thus the task loss has
little reason to restore absolute timing. With compressed spacing, weights and
the MLP readout again recover classification without recovering the registered
delay coordinates. Across task-only trajectories, task-versus-centroid delay
gradients are frequently conflicting; for uniform damage their cosine is
negative in 57.5% of updates.

## Scientific conclusion

The current task BCE does not locally identify the internal delay schedule.
Annealed W1 success is curriculum-assisted retention inside a scaffolded basin,
not evidence that task loss can repair timing. Claims of autonomous WAD,
task-derived time routing or learned temporal multiplexing remain unsupported.

Any subsequent N/T surface may include the annealed model only under the label
`curriculum-assisted temporal model`. It cannot be presented as autonomous
WAD. A method-level repair would need to change identifiability—for example a
monotone low-dimensional delay parameterization or a counterfactual task in
which incorrect timing necessarily changes the task loss—not merely tune LR,
threshold or updates.

Machine-readable evidence:
`docs/generated/mixedop_temporal_wad_scaffold_withdrawal_v1/w2_decision.json`.
The sealed test remains unopened.
