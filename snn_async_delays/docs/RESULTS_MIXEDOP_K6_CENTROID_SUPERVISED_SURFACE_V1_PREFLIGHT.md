# K=6 centroid-supervised surface v1: preflight result

## Decision

All four seed-3899 extreme preflight cells pass the frozen technical and
mechanism gate. The exact 112-cell seed-3907 surface is authorized. Preflight
accuracy remains invalid for claims, the sealed test remains closed, and no
spatial/autonomous-WAD/Pareto claim is unlocked.

| N hidden | w | T | final max centroid error | worst-query BAcc (diagnostic) | exact trial (diagnostic) | minimum window activity | decision |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | 34 | 0.0184 | 0.5010 | 0.2104 | 0.1963 | pass |
| 1 | 26 | 166 | 0.3472 | 0.5059 | 0.2671 | 0.2505 | pass |
| 60 | 4 | 34 | 0.0184 | 1.0000 | 1.0000 | 1.0000 | pass |
| 60 | 26 | 166 | 0.3212 | 1.0000 | 1.0000 | 1.0000 | pass |

Accuracy and activity columns are reported only to reveal the landscape; they
were not used in the preflight decision.

## Gate audit

- All four cells contain every required checkpoint, log, fresh predictions
  NPZ, diagnostic-data NPZ, diagnostic panel, 48-field resource ledger and
  completion marker.
- Losses, gradients, parameters and delays remain finite; delays remain inside
  their declared support.
- q0 has a finite gradient and is not required to be nonzero because its common
  initialization equals its target. q1--q5 all have nonzero, target-directed
  initial routing gradients.
- Every final maximum centroid error is below 0.5 step. The largest is 0.3472
  at N=1,T=166.
- The N=60,T=166 cell clips the pre-clipping global norm in 49/600 updates, but
  clipping is not continuous at the end and no numerical or support failure is
  observed. This should remain a formal-surface monitoring diagnostic.
- Runtime panels are structurally valid at K=6 and T=166. They correctly label
  MLP decisions as decoder markers rather than output spikes and show all six
  windows and operation labels.

## Scientific interpretation

The preflight establishes that the explicit centroid teacher can recover the
declared six-delay routing schedule at both width and latency extremes. It does
not show that task loss can discover or retain that schedule.

The exploratory accuracy contrast is useful for grid adequacy: both N=1
extremes remain near chance on the primary worst-query metric, whereas both
N=60 extremes reach 1.0. Therefore the registered width range plausibly spans
a reliability transition instead of lying wholly on a ceiling. At the two
extreme widths, changing T does not visibly change the primary endpoint; the
formal intermediate grid is still required before any factor statement.

The low-width failures are operation-specific rather than total inactivity:
AND/OR/NAND/NOR are often strong, while XOR/XNOR remain near chance. This is
consistent with insufficient nonlinear representational capacity at N=1, not
a routing failure, because centroid and all-window activity gates pass.

## Resource sanity check

The ledgers correctly separate six trainable delay values from latency-driven
buffer memory. Buffer elements increase from 648 at T=34 to 3816 at T=166;
neuron updates scale from 34 to 166 for N=1 and from 2040 to 9960 for N=60.
Dense MACs and measured synaptic events scale strongly with N. These quantities
must remain separate; `N*T` is only a neuron-update proxy.

## Authorization

The YAML now authorizes exactly the frozen 112-cell surface at seed 3907. No
seed, LR, update count, threshold, support, grid, endpoint or decoder change is
permitted. The surface should be launched with:

```powershell
python -m scripts.run_mixedop_k6_centroid_supervised_surface --stage surface --device cuda
```
