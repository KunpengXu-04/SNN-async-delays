"""Optional Lava-DL SLAYER backend for task-only axonal-delay experiments.

The historical simulator accepts BTC tensors. Lava-DL accepts NCT tensors,
so conversion is kept at this boundary and tested independently. The delay
module is deliberately placed before the dense block: ``slayer.block``'s
built-in delay is per output channel, whereas this protocol registers one
axonal delay per input value channel (4K delays).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _portable_shift(
    input_tensor: torch.Tensor, shift_value: torch.Tensor | float | int,
    sampling_time: float = 1,
) -> torch.Tensor:
    """Semantic fallback for SLAYER's JIT CUDA integer time shift.

    Lava-DL's CPU implementation performs the same channel loop. This version
    works on either device and avoids requiring a local MSVC toolchain solely
    to compile SLAYER's optional CUDA acceleration extension.
    """
    values = torch.as_tensor(shift_value, device=input_tensor.device).flatten()
    flattened = input_tensor.reshape(input_tensor.shape[0], -1, input_tensor.shape[-1])
    if values.numel() not in {1, flattened.shape[1]}:
        raise ValueError("delay count must be scalar or equal to the channel count")
    if values.numel() == 1:
        values = values.expand(flattened.shape[1])
    outputs = []
    steps = flattened.shape[-1]
    for channel in range(flattened.shape[1]):
        shift_blocks = int(values[channel].detach().cpu().item() / sampling_time)
        source = flattened[:, channel:channel + 1, :]
        if shift_blocks == 0:
            shifted = source
        elif shift_blocks >= steps or shift_blocks <= -steps:
            shifted = torch.zeros_like(source)
        elif shift_blocks > 0:
            shifted = torch.nn.functional.pad(source[..., :-shift_blocks], (shift_blocks, 0))
        else:
            shifted = torch.nn.functional.pad(source[..., -shift_blocks:], (0, -shift_blocks))
        outputs.append(shifted)
    return torch.cat(outputs, dim=1).reshape(input_tensor.shape)


def configure_slayer_portable_shift() -> str:
    """Install pure-Torch primitives where SLAYER otherwise JIT-compiles CUDA."""
    import importlib

    delay_implementation = importlib.import_module("lava.lib.dl.slayer.axon.delay")
    convolution = importlib.import_module("lava.lib.dl.slayer.utils.filter.conv")
    leaky = importlib.import_module(
        "lava.lib.dl.slayer.neuron.dynamics.leaky_integrator"
    )
    delay_implementation.shift = _portable_shift
    delay_implementation.conv = lambda values, kernel, sampling_time=1: convolution._fwd(
        values, kernel.flatten(), sampling_time
    )

    def portable_leaky_dynamics(
        inputs: torch.Tensor, decay: torch.Tensor, state: torch.Tensor,
        w_scale: int, threshold: float | None = None, debug: bool = False,
    ) -> torch.Tensor:
        del debug
        threshold_value = -1 if threshold is None else threshold
        if torch.numel(state) == 1:
            state = state * torch.ones(inputs.shape[:-1], device=inputs.device)
        leaky._LIDynamics.DEBUG = False
        return leaky._LIDynamics.apply(
            inputs, decay, state, threshold_value, w_scale
        )

    leaky.dynamics = portable_leaky_dynamics
    return "slayer_portable_torch_delay_and_cuba_dynamics"


def btc_to_nct(spikes: torch.Tensor) -> torch.Tensor:
    """Convert [batch,time,channel] to Lava-DL [batch,channel,time]."""
    if spikes.ndim != 3:
        raise ValueError(f"BTC input must have three dimensions, got {spikes.shape}")
    return spikes.transpose(1, 2).contiguous()


def nct_to_btc(spikes: torch.Tensor) -> torch.Tensor:
    """Convert Lava-DL [batch,channel,time] to [batch,time,channel]."""
    if spikes.ndim != 3:
        raise ValueError(f"NCT input must have three dimensions, got {spikes.shape}")
    return spikes.transpose(1, 2).contiguous()


def _load_slayer() -> Any:
    try:
        from lava.lib.dl import slayer
    except ImportError as exc:  # pragma: no cover - main env intentionally lacks Lava-DL
        raise RuntimeError(
            "Lava-DL is optional. Run this backend with the isolated "
            "snn_slayer environment (lava-dl==0.6.0)."
        ) from exc
    configure_slayer_portable_shift()
    return slayer


class SlayerNativeWindowedModel(nn.Module):
    """One-layer CUBA/LIF model with 4K input-axonal delays."""

    topology_type = "shared_dense"
    model_backend = "slayer_native"
    delay_granularity = "input_axon_per_channel"
    delay_operator = "lava.lib.dl.slayer.axon.Delay+portable_torch_time_primitives"

    def __init__(
        self,
        *,
        n_queries: int,
        n_hidden: int,
        win_len: int,
        read_len: int,
        d_max: int = 47,
        delay_method: str = "task_only_slayer",
        delay_init: float = 0.0,
        fixed_delays: torch.Tensor | None = None,
        threshold: float = 1.0,
        current_decay: float = 0.25,
        voltage_decay: float = 0.03,
        tau_grad: float = 1.0,
        scale_grad: float = 1.0,
        weight_scale: float = 1.0,
        readout_type: str = "mlp",
    ) -> None:
        super().__init__()
        if delay_method not in {"d0", "fixed", "task_only_slayer"}:
            raise ValueError("SLAYER delay_method must be d0, fixed, or task_only_slayer")
        if read_len % n_queries:
            raise ValueError("read_len must divide exactly into K output windows")
        self.n_queries = int(n_queries)
        self.n_input = 4 * self.n_queries
        self.n_hidden = int(n_hidden)
        self.n_hidden2 = None
        self.n_hidden_total = self.n_hidden
        self.win_len = int(win_len)
        self.read_len = int(read_len)
        self.output_window_len = self.read_len // self.n_queries
        self.T = self.win_len + self.read_len
        self.d_max = int(d_max)
        self.delay_method = delay_method
        self.observation_mode = "windowed_shared"
        self.n_observation_bins = 1
        self.readout_feature_dim = self.n_hidden
        self.decoder_repetitions = self.n_queries
        self.use_output_spikes = False
        self.n_output_neurons = 0
        self.input_event_fanout = self.n_hidden

        slayer = _load_slayer()
        self.axon_delay = slayer.axon.Delay(max_delay=self.d_max, grad_scale=1)
        if fixed_delays is not None:
            initial = torch.as_tensor(fixed_delays, dtype=torch.float32).flatten()
            if initial.numel() != self.n_input:
                raise ValueError(f"fixed_delays must contain exactly {self.n_input} values")
        else:
            initial_value = 0.0 if delay_method == "d0" else float(delay_init)
            initial = torch.full((self.n_input,), initial_value, dtype=torch.float32)
        self.axon_delay.delay = nn.Parameter(
            initial, requires_grad=(delay_method == "task_only_slayer")
        )
        self.axon_delay.init = True
        self.axon_delay.clamp()

        neuron_params = {
            "threshold": float(threshold),
            "current_decay": float(current_decay),
            "voltage_decay": float(voltage_decay),
            "tau_grad": float(tau_grad),
            "scale_grad": float(scale_grad),
        }
        self.snn = slayer.block.cuba.Dense(
            neuron_params, self.n_input, self.n_hidden,
            weight_scale=float(weight_scale), delay=False, delay_shift=False,
        )
        if readout_type == "mlp":
            decoder_hidden = max(self.n_hidden, self.n_queries * 8)
            self.readout = nn.Sequential(
                nn.Linear(self.n_hidden, decoder_hidden), nn.ReLU(),
                nn.Linear(decoder_hidden, 1),
            )
        elif readout_type == "linear":
            self.readout = nn.Linear(self.n_hidden, 1)
        else:
            raise ValueError("readout_type must be linear or mlp")
        self.readout_type = readout_type

        # Explicit overrides consumed by the shared resource ledger.
        self.input_hidden_synapse_count = self.n_input * self.n_hidden
        self.delay_value_storage_elements_override = self.n_input
        self.delay_buffer_elements_per_sample_override = (self.d_max + 1) * self.n_input
        # SLAYER forwards an integer time shift (one selected time value per
        # axon); its surrogate delay derivative must not be charged as linear
        # interpolation in inference accounting.
        self.delay_buffer_reads_per_trial_override = self.T * self.n_input
        self.delay_interpolation_ops_per_trial_override = 0

    def weight_params(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.snn.parameters() if parameter.requires_grad]

    def delay_params(self) -> list[nn.Parameter]:
        return [self.axon_delay.delay] if self.axon_delay.delay.requires_grad else []

    def readout_params(self) -> list[nn.Parameter]:
        return list(self.readout.parameters())

    def get_delays(self) -> dict[str, torch.Tensor]:
        self.axon_delay.clamp()
        return {"input_axons": self.axon_delay.delay}

    def delay_regularization(self) -> torch.Tensor:
        return self.get_delays()["input_axons"].mean()

    def forward(
        self, spike_input: torch.Tensor, record: bool = False,
        return_output_spike_train: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if return_output_spike_train:
            raise ValueError("windowed_shared_mlp has no output spike train")
        if spike_input.shape[1:] != (self.T, self.n_input):
            raise ValueError(
                f"expected BTC [B,{self.T},{self.n_input}], got {tuple(spike_input.shape)}"
            )
        delayed_nct = self.axon_delay(btc_to_nct(spike_input))
        hidden_nct = self.snn(delayed_nct)
        hidden_btc = nct_to_btc(hidden_nct)
        window_counts = []
        for query in range(self.n_queries):
            start = self.win_len + query * self.output_window_len
            stop = start + self.output_window_len
            window_counts.append(hidden_btc[:, start:stop, :].sum(dim=1))
        readout_features = torch.stack(window_counts, dim=1)
        logits = torch.stack(
            [self.readout(window_counts[k]).squeeze(-1) for k in range(self.n_queries)],
            dim=1,
        )
        total_hidden = hidden_nct.sum(dim=(1, 2))
        active_hidden = (hidden_nct > 0).any(dim=2).sum(dim=1)
        info: dict[str, Any] = {
            "total_hidden_spikes": total_hidden,
            "active_hidden_neurons": active_hidden,
            "active_hidden_fraction": active_hidden / float(self.n_hidden),
            "hidden_rate": hidden_nct.mean(dim=(0, 2)),
            "trial_steps": self.T,
            "hidden_window_counts": readout_features.detach().cpu(),
            "observation_mode": self.observation_mode,
            "observation_steps": self.read_len,
            "observation_bins": self.n_observation_bins,
            "readout_feature_dim": self.readout_feature_dim,
            "decoder_type": self.readout_type,
            "decoder_parameters": sum(p.numel() for p in self.readout.parameters()),
            "decoder_trainable_parameters": sum(
                p.numel() for p in self.readout.parameters() if p.requires_grad
            ),
            "output_neurons": self.n_queries,
            "model_backend": self.model_backend,
        }
        if record:
            info["hidden_spike_train"] = hidden_btc.detach().cpu()
            info["readout_features"] = readout_features.detach().cpu()
            info["delayed_input_spike_train"] = nct_to_btc(delayed_nct).detach().cpu()
        return logits, info
