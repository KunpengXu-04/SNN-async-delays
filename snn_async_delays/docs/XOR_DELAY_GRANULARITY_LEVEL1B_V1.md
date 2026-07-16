# XOR delay granularity and micro-burst — Level 1B v1

**Status:** Stage A and fixed micro-burst controls complete (70 formal cells).
Global scaffold replication passes 10/10, the higher-dimensional extension
fails, fixed delay 4 passes its micro-burst gate 5/5 and fixed d0 passes 0/5.
The 60 learned micro-burst cells are mechanically authorized but not launched.

## 1. Scientific question

Level 1A established a narrow result: one global input-hidden delay can reach a
prespecified delay-4 schedule while trainable weights solve exhaustive K=1 XOR
with exact hard output spikes, but only robustly when an explicit arrival-time
teacher is present. Task-only delay learning passed at most 1/10.

Level 1B asks two downstream questions without increasing K:

1. does that scaffold-assisted bridge remain reliable when the independent
   delay dimension grows from 1 to 16 or 64; and
2. does it survive the consecutive two-event-per-selected-channel micro-burst
   that previously produced early output collisions?

It does not test heterogeneous routing. Every independent delay still has the
same declared target, four steps. Increasing parameter dimension is an
optimization/credit-assignment intervention, not a new temporal code.

## 2. Frozen task and model

Every update uses all four XOR patterns. Inputs remain four value-identity
channels `[A0,A1,B0,B1]`; the network remains

\[
4\rightarrow16\text{ hard LIF}\rightarrow2\text{ opponent hard LIF}.
\]

The hidden and output thresholds remain `.2` and `.03` simulator arbitrary
units, membrane constants remain ten steps, reset is zero, refractory duration
is two steps and hidden-output delays remain d0. `eta=0`, weight LR `.01`,
delay LR `.01`, 500 full-batch updates and final-checkpoint-only evaluation are
frozen from Level 1A. Five new initialization seeds `{607,709,811,919,1021}`
are disjoint from the Level-1A selection seeds.

## 3. Delay granularities

For input channel `i` and hidden neuron `j`, the tested parameterizations are:

\[
\begin{aligned}
\text{global: }&d_{ij}=d &&(1\text{ independent scalar}),\\
\text{per-hidden: }&d_{ij}=d_j &&(16\text{ independent scalars}),\\
\text{per-synapse: }&d_{ij}\text{ free} &&(64\text{ independent scalars}).
\end{aligned}
\]

Every scalar uses `d=8 sigmoid(r)` and starts homogeneously from `r=-2` or
`r=+2`, corresponding to approximately `.954` or `7.046` steps. Constant
initialization makes the functional starting point matched; weight asymmetry
is allowed to break delay symmetry during task training.

## 4. Per-parameter arrival scaffold

A single global centroid would be invalid for higher-dimensional delays:
parameters moving equally early and late could preserve the mean. Level 1B
therefore defines one unweighted input-arrival trace for every independent
delay coordinate.

Let `B_p(t)` be the presynaptic event mass relevant to coordinate `p`, summed
over the exhaustive truth table. Production interpolation gives

\[
A_p(t;d_p)=(1-\alpha_p)B_p(t-1-\lfloor d_p\rfloor)
+\alpha_p B_p(t-1-\lceil d_p\rceil),
\]

where `alpha_p=d_p-floor(d_p)`. Its centroid is

\[
\mu_p(d_p)=\frac{\sum_t tA_p(t;d_p)}{\sum_t A_p(t;d_p)+\epsilon}.
\]

The explicit scaffold is

\[
L_{arr}=\frac{1}{2P}\sum_{p=1}^{P}
[\mu_p(d_p)-\mu_p(4)]^2.
\]

The total objective is

\[
L=L_{filtered\ hard\ spike}+\lambda L_{arr},
\qquad\lambda\in\{0,.01\}.
\]

`lambda=0` is the prespecified task-only comparator. `lambda=.01` is the
Level-1A scaffold. Because `L_arr` explicitly contains the delay-4 target, its
success cannot be described as autonomous timing discovery.

## 5. Stage A: held-out-seed granularity matrix

The single-event code places one spike on each selected A/B value channel at
step 9: two input events per trial. The matrix is

\[
3\text{ granularities}\times2\lambda\times2\text{ initial directions}
\times5\text{ new seeds}=60\text{ cells}.
\]

A `(granularity,lambda)` candidate spans ten cells and passes only if all ten
pass. The global scaffold must reproduce Level 1A in 10/10 new-seed cells.
Granularity extension requires at least one higher-dimensional scaffold
candidate to pass 10/10. Task-only conditions are retained as negative
comparators and are not required to pass.

Only the global held-out replication gate unlocks the micro-burst controls.
This prevents a failed base replication from being hidden by a new encoding.

## 6. Stage B: micro-burst robustness

The micro-burst places two consecutive spikes at steps 8 and 9 on each selected
value channel: four input events per trial. The evaluated output remains one
and only one correct opponent spike at step 15. A spike at step 14 is an error,
not partial success.

First run ten fixed controls:

- fixed delay 4, five seeds: feasibility gate;
- fixed d0 with the delayed target, five seeds: timing-specificity negative.

The 60 learned micro-burst cells are mechanically locked unless the fixed
delay-4 interface passes 5/5. If unlocked, the learned matrix repeats all three
granularities, both lambda conditions, both initialization directions and five
seeds. It is fixed in advance; no granularity may be selectively omitted after
Stage-A results are seen.

## 7. Per-cell gates

Every cell must have balanced accuracy one, all four classifications correct,
an output spike train exactly equal to the target train, zero silence and
collision, one output event per pattern at the declared time, and hidden
activity in all four patterns.

For learned cells, let `P` be the number of independent delay scalars. Every
coordinate must finish within `.1` step of four:

\[
\max_p|d_p-4|\le .1.
\]

At initialization, every coordinate must also have a nonzero correct total
gradient direction. For gradient descent this is

\[
\frac{\partial L}{\partial r_p}(d_{p,0}-4)>0\quad\forall p.
\]

Mean delay, mean accuracy or a favourable subset cannot compensate for one
failed coordinate or seed.

## 8. Diagnostics and resources

Every run saves checkpoint, strict JSON, exhaustive truth-table output,
resource ledger, runtime NPZ and a runtime 12-panel diagnostic. The panel is
granularity-aware: it shows delay mean/min/max, spread, fraction within target,
per-coordinate direction/nonzero fractions, task/arrival/total gradient norms,
input/hidden/output rasters, output pre-reset voltage, the final independent
delay map and initial/final/target arrival-centroid distributions.

Resource reporting keeps trainable delay parameters separate from physical
synapses and buffer storage. Global, per-hidden and per-synapse models therefore
have 1, 16 and 64 trainable delay values while retaining the same 64 physical
input-hidden synapses.

## 9. Decision boundary

A positive Level 1B result supports only scaffold-assisted K=1 optimization
and/or micro-burst robustness for named granularities. K greater than one,
learned temporal multiplexing, heterogeneous schedules, WAD superiority and
Pareto claims remain locked regardless of the outcome. A task-only success is
reported separately and would require replication before changing that claim.

## 10. Implementation and launch order

The versioned runner is `scripts/run_xor_delay_granularity_level1b.py`. It
creates the checkpoint, strict metrics, exhaustive truth-table output,
resource ledger, compressed NPZ and diagnostic panel inside each cell during
the run. Formal subsets are forbidden.

Run order is mechanical:

```powershell
python -m scripts.run_xor_delay_granularity_level1b --stage a --device cpu
python -m scripts.run_xor_delay_granularity_level1b --stage b_controls --device cpu
python -m scripts.run_xor_delay_granularity_level1b --stage b_learned --device cpu
```

Stage A and the second command are complete. Fixed delay 4 passes 5/5, so the
third command now clears the registered mechanical gate but has not been
launched. Dry expansion is 60, 10 and 60 formal cells respectively.

Implementation validation completed before formal launch: ten Level-1B tests
and the full 88-test suite pass. Two smoke cells from each stage exercised all
artifact and lock paths. Smoke-only synthetic gate decisions were used for
the two downstream code paths; all six smoke cells are invalid for claims and
preceded the subsequently completed formal Stage A.

Formal Stage-A results are reported separately in
`docs/RESULTS_XOR_DELAY_GRANULARITY_LEVEL1B_STAGE_A.md`; the preregistered
protocol above remains the source for frozen hypotheses and gates.
Fixed-control results are in
`docs/RESULTS_XOR_DELAY_GRANULARITY_LEVEL1B_STAGE_B_CONTROLS.md`.
