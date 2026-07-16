# Project scope and research question

## Canonical question

For a temporal multi-query task with query load `K`, can a network with
learnable synaptic delays attain a better resource--reliability Pareto frontier
than matched non-learned-delay alternatives?

The resource vector is

`R = (latency T, hidden neurons, trainable parameters, state/delay memory,
neuron updates, synaptic events, input/output spikes, decoder cost)`.

Reliability is evaluated primarily by the worst query position and exact-trial
success, not only pooled binary accuracy.

## What the project is and is not testing

The sequential-subwindow task tests delayed retention and alignment of early
events into a later decision time.  It is a useful diagnostic, but it is not by
itself evidence for general temporal multiplexing: `K`, total duration,
readout width, and delay range can otherwise grow together.

The main programme therefore requires three task families:

1. sequential subwindows (retention/alignment);
2. simultaneous queries (removes serial ageing);
3. fixed-total-duration packed sequences (separates query load from latency).

## Readout-interface policy

The current final-window spike-count readout is named **late-window
alignment readout**.  It is retained as a diagnostic condition, not a neutral
default, because it can discard early spikes from a zero-delay network.

Every mechanism claim must be checked against at least one symmetric
alternative:

- all-window spike-count readout;
- query-local/subwindow readout;
- spiking-output readout with a declared decision rule.

The repository already contains an experimental `use_output_spikes` path.  It
is a valid ablation only when its output-neuron count, threshold, synaptic
delays, decision window, and output-spike cost are matched and reported.

## Baselines required before a positive claim

- d0 / no learned delay with a fair observation interface;
- optimized scalar fixed delay;
- fixed heterogeneous or random delay bank;
- learned-delay permutation/shuffle control;
- a non-delay temporal baseline (for example recurrence or explicit time
  features) where applicable;
- matched decoder capacity.

## Current claim boundary

Existing results are exploratory.  They may motivate hypotheses about
alignment, recency, and burst timing, but do not yet support claims of fixed
resource capacity gain, energy savings, general temporal routing, or neuron
reuse.
# Current experimental branch (2026-07-13)

The sequential XOR programme did not show WAD superiority over optimized scalar
delay, and a matched seven-variant optimization audit did not rescue WAD. The
active question is therefore conditional: does heterogeneous learned timing
become useful when simultaneous fixed-operation inputs must use a genuinely
shared temporal output interface? The spatial control and temporal-routing task
must remain distinct. Neither fixed position-to-operation mapping nor dedicated
spatial outputs establishes general operation routing or temporal multiplexing.

## Spatial-versus-temporal Pareto pivot (2026-07-14)

The next programme compares `K` independent block-diagonal synchronous modules
against one delay-enabled shared hidden population. The primary question is no
longer whether WAD improves one fixed architecture, but whether temporal reuse
changes the matched reliability-resource frontier. Area, area-time, synaptic
compute and causal reuse are separate claims. Dedicated output heads alone do
not prove multiplexing; shared-spatial d0, fixed-delay, window permutation and
delay intervention controls are required. Phase 0 uses repeated XOR before any
mixed-operation expansion. See `SPATIAL_VS_TEMPORAL_PARETO_PHASE0.md`.

Because the direct-spiking-output baseline failed, the project also permits one
versioned **exploratory MLP scaffold** to diagnose hidden-dynamics capacity and
resource scaling. It is not a replacement endpoint for the paper claim. A
temporal MLP candidate must retain query identity through declared output
windows (`windowed_shared`); an all-time MLP is only an ordinary shared-spatial
control. Any MLP `T x h` result must carry a visible non-spiking-output caveat
and cannot unlock the formal spiking-output protocol.
