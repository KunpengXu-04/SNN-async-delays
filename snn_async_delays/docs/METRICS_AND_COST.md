# Metrics and cost protocol

## Primary reliability metrics

For predictions `p[i,k]` and labels `y[i,k]`:

- `pooled_accuracy`: mean correctness over all trials and positions;
- `per_query_accuracy[k]`: correctness for position `k`;
- `worst_query_accuracy`: `min_k per_query_accuracy[k]`;
- `exact_trial_accuracy`: fraction of trials for which all `K` answers are
  correct;
- `balanced_accuracy`: mean of sensitivity and specificity, reported whenever
  the task is class imbalanced.

The primary thresholded endpoint is

`max K such that worst_query_accuracy >= tau`.

`tau = 0.90` and `tau = 0.95` are summaries, not the sole result.  They must be
declared before a confirmatory run.  Pooled accuracy alone is never sufficient
for a multiplexing claim.

## Cost reporting

No result may be called hardware energy without a validated hardware model.
Report the vector:

`(T, n_input, n_hidden, n_output, parameters, delay-buffer memory,
 neuron_updates, input_events, hidden_events, output_events, decoder_ops)`.

An optional proxy cost is

`C_proxy = alpha*T*n_hidden + beta*input_events*n_hidden +
 gamma*hidden_events*n_output + delta*delay_memory + decoder_cost`.

The coefficients must be declared.  Until then, `hidden spikes` and `K/spike`
are descriptive quantities only; they are not energy or throughput claims.

The executable field definitions and simulator/event-count distinction are in
`RESOURCE_LEDGER.md`.  New simultaneous/Plan-D evaluations save a nested
`resource_ledger` using schema `resource_ledger_v1`.

## Selection and test discipline

Choose hyperparameters, architecture, and resource level using validation data.
Evaluate a sealed test set once per predeclared configuration.  For stochastic
encodings, record the evaluation seed and use paired realizations for methods
being compared.

Every summary reports seed-level mean, standard deviation, confidence interval,
and all seed values.  A crossing of a threshold by one seed is not a Max-K
claim.
# Simultaneous fixed-operation metrics

For fixed `[XOR,NAND,NOR]` positions, `per_query` is retained as a machine-
readable compatibility name but must be reported semantically as per-operation
(spatial) or per-output-window (temporal). Because operation prevalence differs,
the primary statistic is

`worst_query_balanced_accuracy = min_k balanced_accuracy_k`.

Pooled accuracy cannot gate a claim. Exact-trial accuracy requires all three
outputs correct. The deterministic burst protocol additionally evaluates all
`2^(2K)=64` input patterns and records exact truth-table completion and detailed
pattern failures.

Opponent outputs require silent, tie, both-neuron collision, per-window output
spikes and signed margin `m=(2y-1)(n_plus-n_minus)`. Temporal routing additionally
requires a 3x3 output-window-versus-target balanced-accuracy matrix and its
diagonal-minus-off-diagonal selectivity gap. These metrics supplement rather
than replace resource-ledger latency, neuron updates, synaptic events, decoder
operations and delay memory.

## Spatial-versus-temporal analytic ratios

For `K` independent spatial modules of width `h` and a shared temporal network
of width `h'`, report at minimum

`rho_area = h' / (K*h)`,

`rho_hidden_update = h'*T_B / (K*h*T_A)`, and

`rho_dense_synapse_compute = h'*T_B / (h*T_A)`.

The last expression assumes block-diagonal spatial modules and a dense shared
network connected to all `K` input/output groups. It exposes that neuron-area
compression can coexist with higher synaptic compute. None of these ratios may
replace the executable resource vector. Mixed-operation spatial width is the
sum of independently calibrated per-operation widths, not automatically `K*h`.
