"""Transparent resource accounting for SNNSimultaneousModel.

Counts are intentionally reported as a vector.  No hardware-energy weights are
assumed.  Static counts describe one inference trial; dynamic counts use mean
spike/event totals measured over an evaluation set.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


SCHEMA_VERSION = "resource_ledger_v1"


def _linear_decoder_ops(module: nn.Module | None) -> tuple[int, int, int]:
    """Return (weight MACs, bias adds, pointwise activation ops) per trial."""
    if module is None:
        return 0, 0, 0
    macs = 0
    bias_adds = 0
    activations = 0
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            macs += layer.in_features * layer.out_features
            bias_adds += layer.out_features if layer.bias is not None else 0
        elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
            # The preceding Linear output determines pointwise activation size.
            # For the current readout MLP, this is available as the previous
            # Linear's out_features.
            previous = [m for m in module.modules() if isinstance(m, nn.Linear)]
            if previous:
                activations += previous[0].out_features
    return macs, bias_adds, activations


def static_resource_ledger(model: Any) -> dict[str, int | float | str]:
    """Architecture/simulator counts for one sample and one full trial."""
    h1 = int(model.n_hidden)
    h2 = int(model.n_hidden2) if model.n_hidden2 is not None else 0
    last_hidden = h2 if h2 else h1
    n_input = int(model.n_input)
    n_output = int(model.n_output_neurons) if model.use_output_spikes else int(model.n_queries)
    T = int(model.T)
    delay_depth = int(model.d_max) + 1

    spatial_modules = getattr(model, "syn_ih_modules", None)
    syn_ih = (
        sum(int(layer.n_pre) * int(layer.n_post) for layer in spatial_modules)
        if spatial_modules is not None else n_input * h1
    )
    syn_h1h2 = h1 * h2 if h2 else 0
    syn_ho = last_hidden * n_output if model.use_output_spikes else 0
    delayed_synapses = syn_ih + syn_h1h2 + syn_ho

    # Circular buffers hold presynaptic histories.  Counts are per sample;
    # multiply by batch size for runtime allocation.
    delay_buffer_elements = delay_depth * n_input
    if h2:
        delay_buffer_elements += delay_depth * h1
    if model.use_output_spikes:
        delay_buffer_elements += delay_depth * last_hidden

    spiking_neurons = h1 + h2 + (n_output if model.use_output_spikes else 0)
    neuron_state_elements = 2 * spiking_neurons  # membrane voltage + refractory state
    neuron_updates = T * spiking_neurons
    dense_synapse_macs = T * delayed_synapses
    delay_buffer_reads = 2 * T * delayed_synapses  # floor + ceil gather
    delay_interpolation_elementwise_ops = 3 * T * delayed_synapses

    decoder_module = model.syn_ho if model.use_output_spikes else model.readout
    dense_decoder = None if model.use_output_spikes else model.readout
    decoder_macs, decoder_bias_adds, decoder_activation_ops = _linear_decoder_ops(dense_decoder)
    decoder_repetitions = int(getattr(
        model, "decoder_repetitions",
        model.n_queries if (not model.use_output_spikes
                            and model.observation_mode == "windowed_shared") else 1,
    ))
    if not model.use_output_spikes and decoder_repetitions > 1:
        decoder_macs *= decoder_repetitions
        decoder_bias_adds *= decoder_repetitions
        decoder_activation_ops *= decoder_repetitions
    decoder_parameters = sum(p.numel() for p in decoder_module.parameters())
    decoder_trainable = sum(
        p.numel() for p in decoder_module.parameters() if p.requires_grad
    )

    observation_width = n_output if model.use_output_spikes else last_hidden
    observation_steps = (
        int(model.read_len) if model.observation_mode in {"late_window", "windowed_shared"} else T
    )
    readout_accumulation_adds = observation_steps * observation_width

    # Actual implementation stores one weight and one raw delay value per
    # delayed synapse, including frozen d0 matrices.  This is distinct from
    # trainable degrees of freedom.
    synaptic_weight_storage = delayed_synapses
    synapse_layers = (
        list(spatial_modules) if spatial_modules is not None else [model.syn_ih]
    )
    if h2:
        synapse_layers.append(model.syn_h1h2)
    if model.use_output_spikes:
        synapse_layers.append(model.syn_ho)
    delay_value_storage = sum(
        (layer.fixed_delay_tensor.numel()
         if getattr(layer, "fixed_delay_tensor", None) is not None
         else layer.delay_raw.numel())
        for layer in synapse_layers
    )
    decoder_storage = 2 * syn_ho if model.use_output_spikes else decoder_parameters
    nonspiking_decoder_storage = 0 if model.use_output_spikes else decoder_parameters
    model_scalar_storage = (
        synaptic_weight_storage + delay_value_storage + nonspiking_decoder_storage
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "topology_type": getattr(model, "topology_type", "shared_dense"),
        "latency_steps": T,
        "input_channels": n_input,
        "hidden_neurons_layer1": h1,
        "hidden_neurons_layer2": h2,
        "hidden_neurons_total": h1 + h2,
        "output_neurons_or_logits": n_output,
        "spiking_neurons_total": spiking_neurons,
        "synapses_input_hidden": syn_ih,
        "synapses_hidden_hidden": syn_h1h2,
        "synapses_hidden_output": syn_ho,
        "delayed_synapses_total": delayed_synapses,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_tensors_total_elements": sum(p.numel() for p in model.parameters()),
        "synaptic_weight_storage_elements": synaptic_weight_storage,
        "delay_value_storage_elements": delay_value_storage,
        "trainable_delay_parameters": sum(p.numel() for p in model.delay_params()),
        "decoder_parameters": decoder_parameters,
        "decoder_trainable_parameters": decoder_trainable,
        "decoder_storage_elements": decoder_storage,
        "model_scalar_storage_elements": model_scalar_storage,
        "model_scalar_storage_bytes_fp32": 4 * model_scalar_storage,
        "delay_buffer_depth": delay_depth,
        "delay_buffer_elements_per_sample": delay_buffer_elements,
        "delay_buffer_bytes_per_sample_fp32": 4 * delay_buffer_elements,
        "neuron_state_elements_per_sample": neuron_state_elements,
        "neuron_state_bytes_per_sample_fp32": 4 * neuron_state_elements,
        "neuron_updates_per_trial": neuron_updates,
        "dense_synapse_macs_per_trial": dense_synapse_macs,
        "delay_buffer_reads_per_trial": delay_buffer_reads,
        "delay_interpolation_elementwise_ops_per_trial": delay_interpolation_elementwise_ops,
        "decoder_weight_macs_per_trial": decoder_macs,
        "decoder_bias_adds_per_trial": decoder_bias_adds,
        "decoder_activation_ops_per_trial": decoder_activation_ops,
        "readout_accumulation_adds_per_trial": readout_accumulation_adds,
        "observation_mode": model.observation_mode,
        "observation_steps": observation_steps,
        "observation_bins": int(model.n_observation_bins),
        "readout_feature_dim": int(model.readout_feature_dim),
    }


def dynamic_resource_ledger(
    model: Any,
    *,
    mean_input_spikes: float,
    mean_hidden1_spikes: float,
    mean_hidden2_spikes: float | None = None,
    mean_output_spikes: float | None = None,
) -> dict[str, float | None]:
    """Mean event counts per sample/trial from an evaluation dataset."""
    h2 = int(model.n_hidden2) if model.n_hidden2 is not None else 0
    last_hidden_spikes = (
        float(mean_hidden2_spikes) if h2 and mean_hidden2_spikes is not None
        else float(mean_hidden1_spikes)
    )
    events_ih = float(mean_input_spikes) * int(
        getattr(model, "input_event_fanout", model.n_hidden)
    )
    events_h1h2 = float(mean_hidden1_spikes) * h2 if h2 else 0.0
    events_ho = (
        last_hidden_spikes * int(model.n_output_neurons)
        if model.use_output_spikes else 0.0
    )
    return {
        "mean_input_spikes_per_trial": float(mean_input_spikes),
        "mean_hidden1_spikes_per_trial": float(mean_hidden1_spikes),
        "mean_hidden2_spikes_per_trial": (
            float(mean_hidden2_spikes) if mean_hidden2_spikes is not None else None
        ),
        "mean_output_spikes_per_trial": (
            float(mean_output_spikes) if mean_output_spikes is not None else None
        ),
        "mean_synaptic_events_input_hidden": events_ih,
        "mean_synaptic_events_hidden_hidden": events_h1h2,
        "mean_synaptic_events_hidden_output": events_ho,
        "mean_synaptic_events_total": events_ih + events_h1h2 + events_ho,
    }
