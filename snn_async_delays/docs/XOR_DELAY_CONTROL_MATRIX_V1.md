# Preregistration: XOR delay control matrix v1

## Question

At non-saturated burst-XOR settings chosen before this matrix, does learned
heterogeneous delay placement improve validation worst-query reliability over
the strongest of zero delay, an optimized shared scalar delay, and a fixed
random heterogeneous delay bank under all-time observation?

This is a validation-stage causal matrix. It does not open the test split and
cannot by itself support a confirmatory publication claim.

## Training matrix

- Primary: K=3, N=35, T=40; linear primary and MLP secondary.
- Stress: K=4, N=50, T=50; linear only.
- Seeds: 0, 1, 2, 3, 42, paired across conditions.
- Conditions: d0, optimized scalar, fixed uniform heterogeneous, WAD.
- Total: 60 training cells, each with NPZ and the corrected diagnostic panel.

The fixed bank uses `[0,d_max]` and a local generator seeded by the training
seed. The local generator does not perturb weight initialization. Shared scalar
uses one trainable raw-delay parameter broadcast to all input-hidden synapses.

## Post-training interventions

1. Every checkpoint is evaluated with its trained decoder unchanged but hidden
   activity accumulated only in the late window. This is an alignment/censoring
   probe, not a separately optimized late-window model.
2. Every WAD checkpoint has its learned raw-delay values deterministically
   permuted (`10000 + training_seed`). This preserves weights, decoder, and the
   exact delay multiset while destroying placement.

Intervention results and diagnostics live below each source run in
`interventions/<name>/`; the source checkpoint is never overwritten.

## Endpoints and gate

Primary endpoint is validation worst-query accuracy at K=3,N=35, all-time
linear. WAD is compared against the best non-learned delay control, not d0
alone. A positive programme requires all of:

- mean paired advantage at least 0.03;
- positive paired difference in at least four of five seeds;
- no domination in the declared resource vector;
- shuffle degradation in at least four of five WAD seeds.

Failure triggers a delay-structure, observation-confound, or negative-result
framing. Exact-trial, balanced accuracy, stress-setting results, MLP, resources,
and late-window interaction are secondary.

## Diagnostics

Every training/intervention panel stores its NPZ and identifies the fixed trace
seed. The corrected panel distinguishes true output spikes from decoder
decisions and includes hidden membrane dynamics plus per-query label,
prediction, and logit. Panels remain illustrative and cannot replace aggregate
intervention statistics.

## Commands

```text
python -m scripts.run_xor_delay_control_matrix --device cuda --dry-run
python -m scripts.run_xor_delay_control_matrix --device cuda
python -m scripts.run_xor_delay_control_interventions --device cuda
```
