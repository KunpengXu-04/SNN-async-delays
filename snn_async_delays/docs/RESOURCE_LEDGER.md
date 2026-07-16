# Resource ledger specification (v1)

The ledger reports raw resource counts for one sample and one inference trial.
It does not combine them into energy.  A scalar cost requires separately
declared hardware coefficients.

Shared-scalar controls store one delay value per delayed layer; learned and
fixed heterogeneous controls store one value per delayed synapse. The ledger
reports actual delay-value storage separately from delayed-synapse count. A
scalar control is not charged for a full heterogeneous delay matrix merely to
make parameter counts look matched.

## Static structure

- latency in simulation timesteps;
- input, hidden, and output dimensions;
- delayed synapse counts per layer;
- trainable parameter count;
- stored weight and delay values, including frozen matrices;
- decoder parameter count and feature dimension.

`parameter_tensors_total_elements` counts `nn.Parameter` tensors.  Frozen SNN
weights/delays are registered buffers, so fair storage comparison should use
`synaptic_weight_storage_elements`, `delay_value_storage_elements`, and
`model_scalar_storage_elements` instead.

## Runtime state and delay memory

Delay-buffer memory is the actual circular-buffer allocation per sample:

`(d_max + 1) * sum(presynaptic layer widths)`.

The sum includes input history, layer-1 history for a two-hidden-layer model,
and last-hidden history for a spiking output layer.  FP32 byte counts are
reported for transparency.  Neuron-state memory includes membrane voltage and
refractory state; framework/autograd overhead is not included.

## Operation counts

- `neuron_updates_per_trial`: number of LIF neuron/timestep updates;
- `dense_synapse_macs_per_trial`: dense weighted delayed connections evaluated
  by the current simulator;
- `delay_buffer_reads_per_trial`: two reads per synapse/timestep for floor/ceil
  interpolation;
- `delay_interpolation_elementwise_ops_per_trial`: three scalar interpolation
  operations per synapse/timestep under the current formula;
- decoder weight MACs, bias additions, and activation operations;
- spike-count accumulation additions for the declared observation window.

These are operation counts, not measured wall-clock time or joules.

## Dynamic event counts

Evaluation measures mean input, layer-1 hidden, layer-2 hidden, and output
spikes per trial.  Potential event-driven synaptic events are

- input spikes × first-hidden fan-out;
- layer-1 spikes × second-hidden fan-out;
- last-hidden spikes × spiking-output fan-out.

A dense linear/MLP readout is accounted through decoder operations rather than
hidden-spike fan-out.  The event counts describe a conventional fan-out event
model; routing, multicast, memory access, and hardware-specific costs require
explicit coefficients later.

## Comparison rules

Compare methods first as a resource vector.  `late_window` and `all_time` use
the same linear/MLP decoder dimension, but differ in accumulation operations.
`time_binned` has `(K+1) * n_last_hidden` decoder features, so its larger
decoder cost cannot be ignored.  Spiking output adds delayed synapses, delay
buffer, output-neuron updates, and output events.
