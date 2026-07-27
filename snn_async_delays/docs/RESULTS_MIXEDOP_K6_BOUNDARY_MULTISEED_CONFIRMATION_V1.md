# Results: K=6 boundary-region multi-seed confirmation v1

## Binding outcome

All 200 registered cells are complete and artifact-valid. All 200 satisfy the
centroid mechanism gate; 134/200 satisfy the joint 90% rule. The sealed test
was not opened, and parent seed 3907 is excluded from all confirmation
statistics.

The two exact seed-3907 higher-width reversals do **not** reproduce:

| Registered reversal | Fresh seeds reproducing | Decision |
|---|---:|---|
| T=34: N=3 pass, N=4 fail | 0/5 | not reproduced |
| T=46: N=2 pass, N=3 fail | 0/5 | not reproduced |

For each reversal, the descriptive Wilson 95% interval is `[0,.434]`. This
does not prove that the event is impossible; it rejects the registered
`>=2/5` reproduction criterion.

The broader non-monotonicity endpoint does reproduce. Three of five fresh-seed
raw N90 curves contain at least one adjacent increase as T grows, exactly
meeting the preregistered `>=3/5` threshold. The robust 4/5 curve also has one
increase. This is evidence against a monotone L-shaped boundary under the
frozen pipeline, not evidence for a particular adverse causal effect of time.

## Boundary estimates

For T=`[34,40,46,58,70,94,130,166]`:

| Estimate | N90 |
|---|---|
| seed 4013 | `[4,3,3,2,2,2,3,3]` |
| seed 4027 | `[3,3,3,2,2,3,3,3]` |
| seed 4049 | `[4,3,3,3,2,2,2,2]` |
| seed 4061 | `[3,3,3,3,3,3,2,3]` |
| seed 4079 | `[2,2,2,2,2,2,2,2]` |
| robust 4/5 | `[4,3,3,3,2,3,3,3]` |
| strict 5/5 | `[5,3,3,3,3,3,3,3]` |
| parent seed 3907, reference only | `[3,3,2,2,3,4,3,3]` |

The registered boundary-stability rule fails. Robust N90 is observed at all
eight T values, but the seed range at T=34 is two neurons (`2--4`), exceeding
the allowed range of one. The robust boundary also differs from the parent at
five of eight T values. Therefore the parent curve is not a stable capacity
boundary.

## Where the instability occurs

The instability is confined mainly to the low-width transition:

| N | Mean worst-query BAcc | Joint-pass fraction | Mean exact-trial |
|---:|---:|---:|---:|
| 1 | .5068 | 0/40 | .2588 |
| 2 | .8204 | 18/40 | .7213 |
| 3 | .9777 | 37/40 | .9656 |
| 4 | .9940 | 39/40 | .9887 |
| 5 | .9976 | 40/40 | .9944 |

At N=2, the fresh-seed pass fraction over increasing T is
`[.2,.2,.2,.6,.8,.6,.6,.4]`. Thus longer simulation does not yield a monotone
reliability gain. N=3 or above is nearly saturated except at T=34.

The low-width bottleneck remains XOR/XNOR representation rather than routing.
Across all fresh-seed N=1 cells, mean BAcc for
`[AND,OR,XOR,XNOR,NAND,NOR]` is
`[.999,.972,.518,.524,.998,.992]`; at N=2 it is
`[.997,.995,.829,.847,.992,.996]`. Among the 40 N=2 cells, XOR or XNOR is the
worst operation in 36. All 200 cells are mechanism-valid, and the largest
selected maximum centroid error is `.4794` step, below the `.5` gate.

## Interpretation

The strict conclusion is negative:

1. The two visually striking parent reversals were seed-specific events, not
   replicated structural reversals.
2. The more general claim that the measured N90 boundary is non-monotone is
   reproduced by the registered rule.
3. The boundary is not stable enough to support a quantitative time--width law.
4. Width has a clear capacity threshold in this fixed interface: N=1 fails,
   N=2 is initialization/data-sensitive, and N>=3 is usually sufficient.
5. No pure T effect is identified. T is changed together with output-window
   length, delay targets/support, recurrent integration duration and decoder
   input statistics. The N=2 non-monotonicity can therefore reflect
   optimization or representation sensitivity rather than time itself.

This branch uses an explicit arrival-centroid teacher, fixed
operation-to-window assignment and an MLP readout. It provides neither
autonomous learned-delay evidence nor a spatial-versus-temporal Pareto result.
It also does not test spiking-output timing or hardware energy.

## Next scientific gate

Do not add more seeds to the same confounded surface or fit a smoothed
hyperbola. Before any time-cost claim, the next versioned study must separate
operation identity from temporal position by counterbalancing/permuting the six
operations across windows. After that, selected boundary points should receive
matched shared-d0, fixed-oracle and spatial controls with the complete resource
vector. This is a new protocol, not an extension authorized by v1.

## Artifacts

- Machine-readable decision: `generated/mixedop_k6_boundary_multiseed_confirmation_v1/boundary_confirmation_decision.json`
- Cell table: `generated/mixedop_k6_boundary_multiseed_confirmation_v1/confirmation_cells.csv`
- Per-seed N90: `generated/mixedop_k6_boundary_multiseed_confirmation_v1/per_seed_N90.csv`
- Pass-fraction and boundary figures: `generated/mixedop_k6_boundary_multiseed_confirmation_v1/`
