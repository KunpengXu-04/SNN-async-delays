# K=6 centroid-supervised T x N_hid surface v1

## Scientific status

This is a preregistered, single-seed, validation-only supervisor-preview
experiment. It asks whether an explicitly supplied temporal routing target can
produce a descriptive reliability boundary over simulation duration and total
shared hidden width. It does **not** test autonomous WAD, task-derived delay
discovery, a spatial baseline, or a spatial-versus-temporal Pareto advantage.

The six simultaneous fixed-position queries are `AND, OR, XOR, XNOR, NAND,
NOR`. The output remains a shared non-spiking MLP applied separately to six
output windows. Consequently, the study tests hidden temporal routing plus a
windowed decoder; it does not establish exact output-spike timing.

## Frozen design

- `K=6`, 24 input channels and six output windows.
- Total shared hidden width:
  `[1,2,3,4,5,6,8,10,12,16,20,30,40,60]`.
- Window length `w=[4,5,6,8,10,14,20,26]`.
- Total simulation length `T=10+6w=[34,40,46,58,70,94,130,166]`.
- Formal grid: 112 cells at seed 3907.
- Preflight: `(N,w) in {1,60} x {4,26}`, seed 3899, four cells.
- Train and validation each contain 2048 independently generated marginally
  balanced samples. The sealed test split stays closed.
- Event8 packet occupies input steps 6--9. The production buffer adds one step,
  so its pre-delay arrival centroid is 8.5.
- Hidden threshold is 0.2 in arbitrary simulation units, not volts or mV.
- Adam learning rates are 0.01 for weights, readout and delays; batch size 128,
  clip norm 1.0, 600 optimizer updates and validation every 25 updates.

There are six query-tied continuous input-to-hidden delays. The declared target
is `d_q*=1.5+w/2+qw`, which places the arrival centroid at the center of output
window q. All coordinates start from common value `d_init=1.5+w/2`, so the
initialization contains no query-specific schedule. Support is `0<=d<=6w+2`.

Weights and MLP receive only query-by-class macro-balanced BCE. Delays receive
only `arrival_centroid_huber`. No task gradient enters delay updates and no
withdrawal occurs. The correct label is **explicitly supervised temporal
routing**.

## Necessary preflight-gradient reconciliation

The common initialization equals q0's target exactly. Requiring a nonzero
initial routing gradient for all six queries would contradict the frozen
objective. The executable gate treats q0 as finite and potentially stationary,
and requires q1--q5 to have nonzero, target-directed initial gradients. This is
a correction of an internally impossible gate, not an outcome-dependent change.

Preflight accuracy and hidden activity are recorded but are not gates. Every
cell must be finite and delay-legal, write all runtime artifacts, pass the
q1--q5 direction gate, and finish with maximum centroid error <=0.5 step. A
passing machine-readable decision is necessary but not sufficient for the
surface: `formal_surface_launch` must also be explicitly unlocked in YAML.

## Checkpoint and boundary rules

A checkpoint is mechanism-eligible only when maximum centroid error is <=0.5
step. Among eligible validation checkpoints, selection uses highest worst-query
balanced accuracy, then exact-trial accuracy, then earlier update. If none is
eligible, the final state is retained for diagnosis and marked invalid.

A 90% pass requires worst-query balanced accuracy >=0.90, centroid error <=0.5,
and hidden-activity fraction >=0.10 in every output window. `N90(T)` is the
smallest tested passing width. N=1 is reported `<=1`; no pass is `>60`; no
extrapolation or monotonic repair is allowed. The descriptive fit
`N90(T)=N_infinity+C/(T-T0)` is attempted only with at least four non-censored
columns, with raw points, residuals, parameters and CV of `N90*T` reported.

## Runtime outputs

Every cell directly writes config, best/final checkpoints, update/validation
logs, fresh predictions NPZ, resource ledger, diagnostic-data NPZ, six-query
diagnostic panel and completion marker. Aggregate plots cover worst BAcc,
exact-trial, centroid error, each query, each activity window, N90, accuracy
versus `N*T`, delay/arrival correspondence, neuron updates, dense MACs,
synaptic events, delay buffer and parameter storage. Axes use true nonuniform
T/N coordinates. `N*T` is a neuron-update proxy, not energy.

## Execution

```powershell
# Authorized invalid preflight only
python -m scripts.run_mixedop_k6_centroid_supervised_surface --stage preflight --device cuda

# Path audit only; formal execution remains locked
python -m scripts.run_mixedop_k6_centroid_supervised_surface --stage surface --dry-run
```

Do not change seed, LR, updates, threshold, support, grid, gate or decoder after
observing results. Spatial comparison or multi-seed confirmation needs a new
protocol.
