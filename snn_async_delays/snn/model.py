"""
SNNModel: single hidden-layer SNN with trainable delays.

Architecture
------------
  Input spikes  -> DelayedSynapticLayer(W_ih, D_ih)
                -> LIFNeurons(N_h)
                -> [optional] DelayedSynapticLayer(W_ho, D_ho)
                            -> LIFNeurons(N_o)
                -> Linear readout (always trainable)

train_mode controls W_ih / D_ih (and W_ho / D_ho if use_output_layer=True):
  'weights_only'       : weights trained,  delays frozen
  'delays_only'        : weights frozen,   delays trained
  'weights_and_delays' : both trained

The linear readout is always trained regardless of train_mode.

Slot-based readout
------------------
forward() receives a list of SlotBoundaries. During each slot's
readout window [read_start, read_end), hidden (or output) spikes are
accumulated. After the full T-step simulation, the accumulated counts
are fed into the linear readout to produce one logit per slot.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn

from .neurons import LIFNeurons
from .synapses import DelayedSynapticLayer


@dataclass
class SlotBoundaries:
    win_start: int
    win_end: int
    read_start: int
    read_end: int


def make_slots(
    K: int,
    win_len: int,
    read_len: int,
    gap_len: int = 0,
) -> List[SlotBoundaries]:
    """Build K equally-spaced temporal slots."""
    slots = []
    slot_len = win_len + read_len + gap_len
    for k in range(K):
        off = k * slot_len
        slots.append(
            SlotBoundaries(
                win_start=off,
                win_end=off + win_len,
                read_start=off + win_len,
                read_end=off + win_len + read_len,
            )
        )
    return slots


class SNNModel(nn.Module):
    """
    Args:
        n_input          : input neuron count (2 for A/B; 2+n_ops for Step 3)
        n_hidden         : hidden neuron count
        d_max            : maximum delay in timesteps
        train_mode       : 'weights_only' | 'delays_only' | 'weights_and_delays'
        delay_param_type : 'sigmoid' | 'direct' | 'quantized'
        delay_step       : delay quantization step when delay_param_type='quantized'
        fixed_delay_value: fixed delay value for frozen-delay regimes
        use_output_layer : if True, add a hidden-to-output LIF layer
        n_output         : output neurons (only used when use_output_layer=True)
        readout_source   : 'hidden' (default) | 'output'
        lif_*            : LIF hyperparameters
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        d_max: int = 50,
        train_mode: str = "weights_and_delays",
        delay_param_type: str = "sigmoid",
        delay_step: float = 1.0,
        fixed_delay_value: float | None = None,
        fixed_delay_distribution: str | None = None,
        fixed_delay_seed: int = 0,
        fixed_delay_low: float = 0.0,
        fixed_delay_high: float | None = None,
        shared_delay: bool = False,
        delay_tying: str | None = None,
        delay_init_mode: str = "constant",
        delay_init_raw: float = -2.0,
        delay_init_std: float = 0.25,
        use_output_layer: bool = False,
        n_output: int = 1,
        readout_source: str = "hidden",
        lif_tau_m: float = 10.0,
        lif_threshold: float = 1.0,
        lif_reset: float = 0.0,
        lif_refractory: int = 2,
        dt: float = 1.0,
        surrogate_beta: float = 4.0,
    ):
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.d_max = d_max
        self.train_mode = train_mode
        self.use_output_layer = use_output_layer
        self.readout_source = readout_source

        train_w = train_mode in ("weights_only", "weights_and_delays")
        train_d = train_mode in ("delays_only", "weights_and_delays")

        # Make synchronous baseline explicit: weights_only means fixed zero delays
        if (not train_d) and (fixed_delay_value is None) and fixed_delay_distribution is None:
            fixed_delay_value = 0.0

        lif_kw = dict(
            tau_m=lif_tau_m,
            v_threshold=lif_threshold,
            v_reset=lif_reset,
            refractory_steps=lif_refractory,
            dt=dt,
            surrogate_beta=surrogate_beta,
        )

        self.syn_ih = DelayedSynapticLayer(
            n_input,
            n_hidden,
            d_max=d_max,
            delay_param_type=delay_param_type,
            delay_step=delay_step,
            fixed_delay_value=fixed_delay_value,
            fixed_delay_distribution=fixed_delay_distribution,
            fixed_delay_seed=fixed_delay_seed,
            fixed_delay_low=fixed_delay_low,
            fixed_delay_high=fixed_delay_high,
            shared_delay=shared_delay,
            delay_init_mode=delay_init_mode,
            delay_init_raw=delay_init_raw,
            delay_init_std=delay_init_std,
            train_weights=train_w,
            train_delays=train_d,
        )
        self.lif_h = LIFNeurons(n_hidden, **lif_kw)

        if use_output_layer:
            self.syn_ho = DelayedSynapticLayer(
                n_hidden,
                n_output,
                d_max=d_max,
                delay_param_type=delay_param_type,
                delay_step=delay_step,
                fixed_delay_value=fixed_delay_value,
                fixed_delay_distribution=fixed_delay_distribution,
                fixed_delay_seed=fixed_delay_seed + 1,
                fixed_delay_low=fixed_delay_low,
                fixed_delay_high=fixed_delay_high,
                shared_delay=shared_delay,
                delay_init_mode=delay_init_mode,
                delay_init_raw=delay_init_raw,
                delay_init_std=delay_init_std,
                train_weights=train_w,
                train_delays=train_d,
            )
            self.lif_o = LIFNeurons(n_output, **lif_kw)

        readout_in = n_output if (use_output_layer and readout_source == "output") else n_hidden
        self.readout = nn.Linear(readout_in, 1)

    def weight_params(self):
        """Trainable weight parameters (excludes delays and readout)."""
        return [
            p for name, p in self.named_parameters()
            if "weight" in name and "readout" not in name and p.requires_grad
        ]

    def delay_params(self):
        """Trainable delay parameters."""
        return [
            p for name, p in self.named_parameters()
            if "delay_raw" in name and p.requires_grad
        ]

    def readout_params(self):
        return list(self.readout.parameters())

    def get_delays(self) -> Dict[str, torch.Tensor]:
        d = {"ih": self.syn_ih.get_delays()}
        if self.use_output_layer:
            d["ho"] = self.syn_ho.get_delays()
        return d

    def delay_regularization(self) -> torch.Tensor:
        """Mean delay (L1 penalty proxy)."""
        reg = self.syn_ih.get_delays().mean()
        if self.use_output_layer:
            reg = reg + self.syn_ho.get_delays().mean()
        return reg

    def forward(
        self,
        spike_input: torch.Tensor,
        slots: List[SlotBoundaries],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            spike_input : [B, T, N_input] pre-generated spike trains
            slots       : list of SlotBoundaries (readout windows)

        Returns:
            logits : [B, K]
            info   : dict with spike/activity stats
        """
        B, T, _ = spike_input.shape
        device = spike_input.device
        K = len(slots)

        v_h, ref_h = self.lif_h.init_state(B, device)
        if self.use_output_layer:
            v_o, ref_o = self.lif_o.init_state(B, device)

        # Circular delay buffers (in-place writes; no torch.cat allocation per step)
        d_max_1 = self.d_max + 1
        buf_in  = torch.zeros(B, d_max_1, self.n_input,  device=device)
        buf_ptr = 0
        if self.use_output_layer:
            buf_h   = torch.zeros(B, d_max_1, self.n_hidden, device=device)
            buf_h_ptr = 0

        # Pre-compute delays once per forward pass (constant across all T steps)
        d_cont_ih = self.syn_ih.get_delays()
        if self.use_output_layer:
            d_cont_ho = self.syn_ho.get_delays()

        hidden_acc = [torch.zeros(B, self.n_hidden, device=device) for _ in range(K)]
        if self.use_output_layer:
            output_acc = [torch.zeros(B, self.n_output, device=device) for _ in range(K)]

        readout_at: List[List[int]] = [[] for _ in range(T)]
        for k, slot in enumerate(slots):
            for t in range(slot.read_start, min(slot.read_end, T)):
                readout_at[t].append(k)

        total_h_spk = torch.zeros(B, device=device)
        total_o_spk = torch.zeros(B, device=device)
        hidden_fired_any = torch.zeros(B, self.n_hidden, device=device)

        for t in range(T):
            x_t = spike_input[:, t, :]

            # Compute synaptic currents from circular buffer (reads happen before write)
            I_h = self.syn_ih(buf_in, d_cont_ih, buf_ptr)
            spike_h, v_h, ref_h = self.lif_h(I_h, v_h, ref_h)

            if self.use_output_layer:
                I_o = self.syn_ho(buf_h, d_cont_ho, buf_h_ptr)
                spike_o, v_o, ref_o = self.lif_o(I_o, v_o, ref_o)

            # In-place circular buffer update (no memory allocation)
            buf_in[:, buf_ptr, :] = x_t
            buf_ptr = (buf_ptr + 1) % d_max_1
            if self.use_output_layer:
                buf_h[:, buf_h_ptr, :] = spike_h
                buf_h_ptr = (buf_h_ptr + 1) % d_max_1

            total_h_spk = total_h_spk + spike_h.sum(dim=1)
            hidden_fired_any = torch.maximum(hidden_fired_any, (spike_h > 0).float())
            if self.use_output_layer:
                total_o_spk = total_o_spk + spike_o.sum(dim=1)

            for k in readout_at[t]:
                hidden_acc[k] = hidden_acc[k] + spike_h
                if self.use_output_layer:
                    output_acc[k] = output_acc[k] + spike_o

        use_output = self.use_output_layer and self.readout_source == "output"
        acc_list = output_acc if use_output else hidden_acc

        logits = torch.stack(
            [self.readout(acc_list[k]).squeeze(-1) for k in range(K)],
            dim=1,
        )

        active_hidden_neurons = hidden_fired_any.sum(dim=1)
        info = {
            "total_hidden_spikes": total_h_spk,
            "total_output_spikes": total_o_spk,
            "active_hidden_neurons": active_hidden_neurons,
            "active_hidden_fraction": active_hidden_neurons / float(self.n_hidden),
            "trial_steps": T,
        }
        return logits, info


class SNNSimultaneousModel(nn.Module):
    """
    SNN for TRUE temporal multiplexing (Plan C / Plan D).

    Architecture (1 hidden layer, default)
    ---------------------------------------
      n_input channels
        -> DelayedSynapticLayer(W_ih, D_ih)   [n_input -> n_hidden]
        -> LIFNeurons(n_hidden)
        -> Linear/MLP readout [n_hidden -> K]

    Architecture (2 hidden layers, num_hidden_layers=2)
    ---------------------------------------------------
      n_input channels
        -> DelayedSynapticLayer(W_ih, D_ih)    [n_input -> h1]
        -> LIFNeurons(h1)
        -> DelayedSynapticLayer(W_h1h2, D_h1h2) [h1 -> h2]
        -> LIFNeurons(h2)
        -> Linear/MLP readout [h2 -> K]

    ``observation_mode`` makes the formerly implicit readout window explicit:
    ``late_window`` preserves the historical final-window accumulation,
    ``all_time`` accumulates the complete trial, and ``time_binned`` exposes K
    sequential input bins plus the final read window to the decoder.

    Parameters
    ----------
    num_hidden_layers : 1 (default, identical to original) or 2
    hidden_sizes      : list[int] e.g. [25, 25] or [50, 50].
                        When given, overrides n_hidden for each layer.
                        For 1-layer models, [n_hidden] or None both work.
    """

    def __init__(
        self,
        n_queries: int,
        n_hidden: int,
        win_len: int,
        read_len: int,
        d_max: int = 49,
        train_mode: str = "weights_and_delays",
        delay_param_type: str = "sigmoid",
        delay_step: float = 1.0,
        fixed_delay_value: float | None = None,
        fixed_delay_distribution: str | None = None,
        fixed_delay_seed: int = 0,
        fixed_delay_low: float = 0.0,
        fixed_delay_high: float | None = None,
        shared_delay: bool = False,
        delay_tying: str | None = None,
        delay_init_mode: str = "constant",
        delay_init_raw: float = -2.0,
        delay_init_std: float = 0.25,
        lif_tau_m: float = 10.0,
        lif_threshold: float = 1.0,
        lif_reset: float = 0.0,
        lif_refractory: int = 2,
        dt: float = 1.0,
        surrogate_beta: float = 4.0,
        n_input_channels: int | None = None,
        readout_type: str = "linear",
        num_hidden_layers: int = 1,
        hidden_sizes: List[int] | None = None,
        use_output_spikes: bool = False,
        n_output_neurons: int | None = None,
        lif_output_threshold: float | None = None,
        observation_mode: str = "late_window",
        opponent_output_mode: str | None = None,
        output_window_len: int | None = None,
        output_delay_mode: str = "inherit",
    ):
        super().__init__()

        self.n_queries = n_queries
        # Plan C default: 2K dedicated channels per query.
        # Plan D override: n_input_channels=2 (shared channels, sequential injection).
        self.n_input    = n_input_channels if n_input_channels is not None else 2 * n_queries
        self.d_max      = d_max
        self.win_len    = win_len
        self.read_len   = read_len
        self.T          = win_len + read_len
        self.train_mode = train_mode

        valid_observation_modes = {"late_window", "all_time", "time_binned", "windowed_shared"}
        if observation_mode not in valid_observation_modes:
            raise ValueError(
                f"observation_mode must be one of {sorted(valid_observation_modes)}, "
                f"got {observation_mode!r}"
            )
        if observation_mode == "time_binned" and win_len % n_queries != 0:
            raise ValueError(
                "time_binned requires win_len divisible by n_queries so each "
                "Plan-D query receives one unambiguous input bin"
            )
        if use_output_spikes and observation_mode == "time_binned":
            raise ValueError(
                "time_binned is not defined for direct spiking output: declare "
                "an output-bin decision rule before enabling this combination"
            )
        self.observation_mode = observation_mode
        self.n_observation_bins = n_queries + 1 if observation_mode == "time_binned" else 1
        if opponent_output_mode not in {None, "parallel_pairs", "shared_windowed"}:
            raise ValueError("opponent_output_mode must be None, parallel_pairs, or shared_windowed")
        if opponent_output_mode is not None and not use_output_spikes:
            raise ValueError("opponent_output_mode requires use_output_spikes=True")
        self.opponent_output_mode = opponent_output_mode
        if output_delay_mode not in {"inherit", "d0"}:
            raise ValueError("output_delay_mode must be 'inherit' or 'd0'")
        self.output_delay_mode = output_delay_mode
        self.output_window_len = int(output_window_len or (read_len // n_queries))
        if (opponent_output_mode == "shared_windowed" or observation_mode == "windowed_shared"):
            if read_len != n_queries * self.output_window_len:
                raise ValueError("windowed modes require read_len == n_queries * output_window_len")

        # Resolve layer sizes
        if num_hidden_layers not in (1, 2):
            raise ValueError(f"num_hidden_layers must be 1 or 2, got {num_hidden_layers}")
        self._num_layers = num_hidden_layers

        if hidden_sizes is not None:
            if len(hidden_sizes) != num_hidden_layers:
                raise ValueError(
                    f"hidden_sizes length ({len(hidden_sizes)}) must match "
                    f"num_hidden_layers ({num_hidden_layers})"
                )
            h1 = hidden_sizes[0]
            h2 = hidden_sizes[1] if num_hidden_layers == 2 else None
        else:
            h1 = n_hidden
            h2 = n_hidden if num_hidden_layers == 2 else None

        self.n_hidden  = h1   # first (and only, for 1-layer) hidden size
        self.n_hidden2 = h2   # second hidden size (None for 1-layer)
        # Total hidden neuron count (for energy / ops_per_neuron metrics)
        self.n_hidden_total = h1 + (h2 if h2 is not None else 0)

        train_w = train_mode in ("weights_only", "weights_and_delays")
        train_d = train_mode in ("delays_only", "weights_and_delays")

        if (not train_d) and (fixed_delay_value is None) and fixed_delay_distribution is None:
            fixed_delay_value = 0.0

        lif_kw = dict(
            tau_m=lif_tau_m, v_threshold=lif_threshold, v_reset=lif_reset,
            refractory_steps=lif_refractory, dt=dt, surrogate_beta=surrogate_beta,
        )

        syn_kw = dict(
            d_max=d_max, delay_param_type=delay_param_type, delay_step=delay_step,
            fixed_delay_value=fixed_delay_value, train_weights=train_w, train_delays=train_d,
            fixed_delay_distribution=fixed_delay_distribution,
            fixed_delay_seed=fixed_delay_seed, shared_delay=shared_delay,
            delay_tying=delay_tying,
            fixed_delay_low=fixed_delay_low, fixed_delay_high=fixed_delay_high,
            delay_init_mode=delay_init_mode, delay_init_raw=delay_init_raw,
            delay_init_std=delay_init_std,
        )

        # ── Layer 1: input -> h1 ──
        self.syn_ih = DelayedSynapticLayer(self.n_input, h1, **syn_kw)
        self.lif_h  = LIFNeurons(h1, **lif_kw)   # always named lif_h for compat

        # ── Layer 2 (optional): h1 -> h2 ──
        if num_hidden_layers == 2:
            self.syn_h1h2 = DelayedSynapticLayer(h1, h2, **syn_kw)
            self.lif_h2   = LIFNeurons(h2, **lif_kw)

        # ── Readout ──
        readout_in = h2 if (num_hidden_layers == 2) else h1
        self.readout_in    = readout_in
        self.readout_type  = readout_type
        self.use_output_spikes = use_output_spikes
        self.readout_feature_dim = (
            readout_in * self.n_observation_bins
            if observation_mode == "time_binned" else readout_in
        )

        if use_output_spikes:
            # Spiking output layer: hidden → LIF output → spike counts = logits
            expected = (2 * n_queries if opponent_output_mode == "parallel_pairs"
                        else (2 if opponent_output_mode == "shared_windowed" else n_queries))
            n_out = n_output_neurons if n_output_neurons is not None else expected
            if opponent_output_mode is not None and n_out != expected:
                raise ValueError(f"{opponent_output_mode} requires {expected} output neurons")
            self.n_output_neurons = n_out
            self.readout_feature_dim = n_out
            output_syn_kw = dict(syn_kw)
            if output_delay_mode == "d0":
                output_syn_kw.update({
                    "fixed_delay_value": 0.0,
                    "fixed_delay_distribution": None,
                    "shared_delay": False,
                    "delay_tying": "pair",
                    "train_delays": False,
                })
            self.syn_ho = DelayedSynapticLayer(readout_in, n_out, **output_syn_kw)
            # Output neurons can use a separate (usually lower) threshold so that
            # sparse hidden activity (e.g. from burst encoding) can drive them.
            lif_o_kw = dict(lif_kw)
            if lif_output_threshold is not None:
                lif_o_kw["v_threshold"] = lif_output_threshold
            self.lif_o  = LIFNeurons(n_out, **lif_o_kw)
            self.readout = None   # not used; set to None to avoid confusion
        else:
            self.n_output_neurons = 0
            decoder_outputs = 1 if observation_mode == "windowed_shared" else n_queries
            if readout_type == "mlp":
                hidden_r = max(self.readout_feature_dim, n_queries * 8)
                self.readout = nn.Sequential(
                    nn.Linear(self.readout_feature_dim, hidden_r),
                    nn.ReLU(),
                    nn.Linear(hidden_r, decoder_outputs),
                )
            else:
                self.readout = nn.Linear(self.readout_feature_dim, decoder_outputs)

    def _observation_bin(self, t: int) -> int:
        """Return the Plan-D bin for a timestep in ``time_binned`` mode."""
        if t >= self.win_len:
            return self.n_queries
        sub_win = self.win_len // self.n_queries
        return min(t // sub_win, self.n_queries - 1)

    def observation_metadata(self) -> Dict[str, int | str]:
        """Resource-relevant description of the observation interface."""
        observed_steps = (self.read_len if self.observation_mode in {"late_window", "windowed_shared"}
                          else self.T)
        decoder_module = self.syn_ho if self.use_output_spikes else self.readout
        decoder_parameters = sum(p.numel() for p in decoder_module.parameters())
        decoder_trainable_parameters = sum(
            p.numel() for p in decoder_module.parameters() if p.requires_grad
        )
        return {
            "observation_mode": self.observation_mode,
            "observation_steps": observed_steps,
            "observation_bins": self.n_observation_bins,
            "readout_feature_dim": self.readout_feature_dim,
            "decoder_type": "spiking_output" if self.use_output_spikes else self.readout_type,
            "decoder_parameters": decoder_parameters,
            "decoder_trainable_parameters": decoder_trainable_parameters,
            "output_neurons": self.n_output_neurons if self.use_output_spikes else self.n_queries,
        }

    def weight_params(self):
        # Captures syn_ih.weight and syn_h1h2.weight (if present)
        return [p for name, p in self.named_parameters()
                if "weight" in name and "readout" not in name and p.requires_grad]

    def delay_params(self):
        # Captures syn_ih.delay_raw and syn_h1h2.delay_raw (if present)
        return [p for name, p in self.named_parameters()
                if "delay_raw" in name and p.requires_grad]

    def readout_params(self):
        if self.use_output_spikes or self.readout is None:
            return []
        return list(self.readout.parameters())

    def get_delays(self) -> Dict[str, torch.Tensor]:
        d = {"ih": self.syn_ih.get_delays()}
        if self._num_layers == 2:
            d["h1h2"] = self.syn_h1h2.get_delays()
        if self.use_output_spikes:
            d["ho"] = self.syn_ho.get_delays()
        return d

    def delay_regularization(self) -> torch.Tensor:
        reg = self.syn_ih.get_delays().mean()
        if self._num_layers == 2:
            reg = reg + self.syn_h1h2.get_delays().mean()
        if self.use_output_spikes:
            reg = reg + self.syn_ho.get_delays().mean()
        return reg

    def forward(
        self,
        spike_input: torch.Tensor,  # [B, T, n_input]
        record: bool = False,
        return_output_spike_train: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            spike_input : [B, T, n_input]  where T = win_len + read_len

        Returns:
            logits : [B, K]   (one logit per query)
            info   : dict with spike stats (layer-wise when num_hidden_layers=2)
        """
        B, T, _ = spike_input.shape
        device = spike_input.device
        d_max_1 = self.d_max + 1

        # ── Shared: input buffer & layer-1 state ──
        v_h, ref_h = self.lif_h.init_state(B, device)
        buf_in  = torch.zeros(B, d_max_1, self.n_input, device=device)
        buf_ptr = 0
        d_cont_ih = self.syn_ih.get_delays()

        # ── 2-layer extras ──
        if self._num_layers == 2:
            v_h2, ref_h2 = self.lif_h2.init_state(B, device)
            buf_h1       = torch.zeros(B, d_max_1, self.n_hidden, device=device)
            buf_h1_ptr   = 0
            d_cont_h1h2  = self.syn_h1h2.get_delays()
            last_acc     = torch.zeros(B, self.n_hidden2, device=device)
            total_h1_spk = torch.zeros(B, device=device)
            total_h2_spk = torch.zeros(B, device=device)
            h1_fired_any = torch.zeros(B, self.n_hidden,  device=device)
            h2_fired_any = torch.zeros(B, self.n_hidden2, device=device)
        else:
            last_acc         = torch.zeros(B, self.n_hidden, device=device)
            total_h_spk      = torch.zeros(B, device=device)
            hidden_fired_any = torch.zeros(B, self.n_hidden, device=device)

        time_binned_acc = None
        if self.observation_mode == "time_binned":
            time_binned_acc = [
                torch.zeros(B, self.readout_in, device=device)
                for _ in range(self.n_observation_bins)
            ]
        windowed_shared_acc = None
        if self.observation_mode == "windowed_shared":
            windowed_shared_acc = [
                torch.zeros(B, self.readout_in, device=device)
                for _ in range(self.n_queries)
            ]

        # Differentiable per-neuron spike accumulators (for homeostatic firing-rate reg)
        spk_pn1 = torch.zeros(B, self.n_hidden, device=device)
        spk_pn2 = (torch.zeros(B, self.n_hidden2, device=device)
                   if self._num_layers == 2 else None)

        # Optional full spike-train recording (last hidden layer + h1 for 2-layer)
        hidden_train: list = [] if record else None  # type: ignore[assignment]
        membrane_train: list = [] if record else None  # type: ignore[assignment]
        h1_train:     list = [] if (record and self._num_layers == 2) else None  # type: ignore[assignment]

        # Spiking output layer extras
        if self.use_output_spikes:
            v_o, ref_o    = self.lif_o.init_state(B, device)
            buf_ho        = torch.zeros(B, d_max_1, self.readout_in, device=device)
            buf_ho_ptr    = 0
            d_cont_ho     = self.syn_ho.get_delays()
            output_acc    = torch.zeros(B, self.n_output_neurons, device=device)
            output_membrane_acc = torch.zeros(B, self.n_output_neurons, device=device)
            output_membrane_peak = torch.full(
                (B, self.n_output_neurons), -1e6, device=device
            )
            output_window_acc = None
            output_window_membrane_acc = None
            output_window_membrane_peak = None
            if self.opponent_output_mode == "shared_windowed":
                # Keep each window as a separate tensor while unrolling time.
                # In-place writes to slices of a shared [B,K,2] tensor break
                # autograd once peak-voltage logits retain earlier versions.
                output_window_acc = [
                    torch.zeros(B, 2, device=device) for _ in range(self.n_queries)
                ]
                output_window_membrane_acc = [
                    torch.zeros(B, 2, device=device) for _ in range(self.n_queries)
                ]
                output_window_membrane_peak = [
                    torch.full((B, 2), -1e6, device=device)
                    for _ in range(self.n_queries)
                ]
            total_o_spk   = torch.zeros(B, device=device)
            output_train: list = [] if (record or return_output_spike_train) else None  # type: ignore[assignment]
            output_pre_reset_train: list = [] if return_output_spike_train else None  # type: ignore[assignment]
            output_membrane_train: list = [] if record else None  # type: ignore[assignment]
            output_current_train: list = [] if record else None  # type: ignore[assignment]

        for t in range(T):
            x_t   = spike_input[:, t, :]
            I_h   = self.syn_ih(buf_in, d_cont_ih, buf_ptr)
            spike_h, v_h, ref_h = self.lif_h(I_h, v_h, ref_h)

            # Write input to buffer AFTER reading (maintains correct delay semantics)
            buf_in[:, buf_ptr, :] = x_t
            buf_ptr = (buf_ptr + 1) % d_max_1

            if self._num_layers == 2:
                I_h2    = self.syn_h1h2(buf_h1, d_cont_h1h2, buf_h1_ptr)
                spike_h2, v_h2, ref_h2 = self.lif_h2(I_h2, v_h2, ref_h2)

                buf_h1[:, buf_h1_ptr, :] = spike_h
                buf_h1_ptr = (buf_h1_ptr + 1) % d_max_1

                total_h1_spk = total_h1_spk + spike_h.sum(dim=1)
                total_h2_spk = total_h2_spk + spike_h2.sum(dim=1)
                spk_pn1 = spk_pn1 + spike_h
                spk_pn2 = spk_pn2 + spike_h2
                h1_fired_any = torch.maximum(h1_fired_any, (spike_h  > 0).float())
                h2_fired_any = torch.maximum(h2_fired_any, (spike_h2 > 0).float())

                if record:
                    h1_train.append(spike_h.detach().cpu())     # layer 1
                    hidden_train.append(spike_h2.detach().cpu()) # layer 2
                    membrane_train.append(v_h2.detach().cpu())

                spike_last_h = spike_h2
            else:
                total_h_spk      = total_h_spk + spike_h.sum(dim=1)
                spk_pn1          = spk_pn1 + spike_h
                hidden_fired_any = torch.maximum(hidden_fired_any, (spike_h > 0).float())

                if record:
                    hidden_train.append(spike_h.detach().cpu())
                    membrane_train.append(v_h.detach().cpu())

                spike_last_h = spike_h

            if self.observation_mode == "time_binned":
                bin_index = self._observation_bin(t)
                time_binned_acc[bin_index] = time_binned_acc[bin_index] + spike_last_h
            elif self.observation_mode == "windowed_shared":
                if t >= self.win_len:
                    window_index = min((t - self.win_len) // self.output_window_len,
                                       self.n_queries - 1)
                    windowed_shared_acc[window_index] = windowed_shared_acc[window_index] + spike_last_h
            elif self.observation_mode == "all_time" or t >= self.win_len:
                last_acc = last_acc + spike_last_h

            # Spiking output layer (after last hidden layer, every timestep)
            if self.use_output_spikes:
                I_o = self.syn_ho(buf_ho, d_cont_ho, buf_ho_ptr)
                not_ref_o = (ref_o <= 0.0).float()
                v_o_pre = (self.lif_o.decay * v_o
                           + (1.0 - self.lif_o.decay) * I_o * not_ref_o)
                spike_o, v_o, ref_o = self.lif_o(I_o, v_o, ref_o)
                buf_ho[:, buf_ho_ptr, :] = spike_last_h
                buf_ho_ptr = (buf_ho_ptr + 1) % d_max_1
                total_o_spk = total_o_spk + spike_o.sum(dim=1)
                if self.observation_mode == "all_time" or t >= self.win_len:
                    output_acc = output_acc + spike_o
                    output_membrane_acc = output_membrane_acc + v_o
                    output_membrane_peak = torch.maximum(output_membrane_peak, v_o_pre)
                if self.opponent_output_mode == "shared_windowed" and t >= self.win_len:
                    window_index = min((t - self.win_len) // self.output_window_len,
                                       self.n_queries - 1)
                    output_window_acc[window_index] = (
                        output_window_acc[window_index] + spike_o
                    )
                    output_window_membrane_acc[window_index] = (
                        output_window_membrane_acc[window_index] + v_o
                    )
                    output_window_membrane_peak[window_index] = torch.maximum(
                        output_window_membrane_peak[window_index], v_o_pre
                    )
                if record or return_output_spike_train:
                    output_train.append(
                        spike_o if return_output_spike_train else spike_o.detach().cpu()
                    )
                if return_output_spike_train:
                    output_pre_reset_train.append(v_o_pre)
                if record:
                    output_current_train.append(I_o.detach().cpu())
                    # Record pre-reset voltage so a diagnostic can show actual
                    # threshold crossings; post-reset voltage would erase the
                    # very event the panel is meant to diagnose.
                    output_membrane_train.append(v_o_pre.detach().cpu())

        if self.use_output_spikes:
            if self.opponent_output_mode == "parallel_pairs":
                pair_counts = output_acc.reshape(B, self.n_queries, 2)
                logits = pair_counts[:, :, 1] - pair_counts[:, :, 0]
                membrane_pairs = output_membrane_acc.reshape(B, self.n_queries, 2)
                membrane_logits = membrane_pairs[:, :, 1] - membrane_pairs[:, :, 0]
                membrane_peak_pairs = output_membrane_peak.reshape(B, self.n_queries, 2)
                readout_features = pair_counts
            elif self.opponent_output_mode == "shared_windowed":
                output_window_acc_tensor = torch.stack(output_window_acc, dim=1)
                output_window_membrane_acc_tensor = torch.stack(
                    output_window_membrane_acc, dim=1
                )
                output_window_membrane_peak_tensor = torch.stack(
                    output_window_membrane_peak, dim=1
                )
                logits = (output_window_acc_tensor[:, :, 1]
                          - output_window_acc_tensor[:, :, 0])
                membrane_logits = (output_window_membrane_acc_tensor[:, :, 1]
                                   - output_window_membrane_acc_tensor[:, :, 0])
                membrane_peak_pairs = output_window_membrane_peak_tensor
                readout_features = output_window_acc_tensor
            else:
                logits = output_acc
                membrane_logits = output_membrane_acc
                readout_features = output_acc
        else:
            if self.observation_mode == "time_binned":
                readout_features = torch.stack(time_binned_acc, dim=1).reshape(B, -1)
                logits = self.readout(readout_features)
            elif self.observation_mode == "windowed_shared":
                readout_features = torch.stack(windowed_shared_acc, dim=1)
                logits = torch.stack([
                    self.readout(windowed_shared_acc[k]).squeeze(-1)
                    for k in range(self.n_queries)
                ], dim=1)
            else:
                readout_features = last_acc
                logits = self.readout(readout_features)

        if self._num_layers == 2:
            active_h2 = h2_fired_any.sum(dim=1)
            info = {
                "total_hidden_spikes":    total_h1_spk + total_h2_spk,
                "layer1_hidden_spikes":   total_h1_spk,
                "layer2_hidden_spikes":   total_h2_spk,
                "active_hidden_neurons":  active_h2,
                "active_hidden_fraction": active_h2 / float(self.n_hidden2),
                "trial_steps":            T,
                # differentiable per-neuron firing rate (both layers) for homeo reg
                "hidden_rate": torch.cat([spk_pn1.mean(dim=0), spk_pn2.mean(dim=0)]) / float(T),
            }
        else:
            active_hidden_neurons = hidden_fired_any.sum(dim=1)
            info = {
                "total_hidden_spikes":    total_h_spk,
                "active_hidden_neurons":  active_hidden_neurons,
                "active_hidden_fraction": active_hidden_neurons / float(self.n_hidden),
                "trial_steps":            T,
                # differentiable per-neuron firing rate [n_hid] for homeo reg
                "hidden_rate": spk_pn1.mean(dim=0) / float(T),
            }

        if self.use_output_spikes:
            info["total_output_spikes"] = total_o_spk
            info["output_membrane_logits"] = membrane_logits
            if self.opponent_output_mode is not None:
                info["output_membrane_class_logits"] = (
                    self.lif_o.surrogate_beta
                    * (membrane_peak_pairs - self.lif_o.v_threshold)
                )
            if self.opponent_output_mode == "parallel_pairs":
                info["output_pair_counts"] = pair_counts.detach().cpu()
            if self.opponent_output_mode == "shared_windowed":
                info["output_window_counts"] = output_window_acc_tensor.detach().cpu()

        if self.observation_mode == "windowed_shared":
            info["hidden_window_counts"] = torch.stack(
                windowed_shared_acc, dim=1
            ).detach().cpu()

        info.update(self.observation_metadata())

        if record:
            # [B, T, last_hidden_size] — always present
            info["hidden_spike_train"] = torch.stack(hidden_train, dim=1)
            info["hidden_membrane_train"] = torch.stack(membrane_train, dim=1)
            if self._num_layers == 2:
                # [B, T, h1] — first hidden layer (only for 2-layer models)
                info["hidden1_spike_train"] = torch.stack(h1_train, dim=1)
            if self.use_output_spikes:
                info["output_spike_train"] = torch.stack(output_train, dim=1)  # [B,T,n_out]
                info["output_membrane_train"] = torch.stack(output_membrane_train, dim=1)
                info["output_synaptic_current_train"] = torch.stack(
                    output_current_train, dim=1
                )
            info["readout_features"] = readout_features.detach().cpu()

        if return_output_spike_train and not record:
            info["output_spike_train"] = torch.stack(output_train, dim=1)
            info["output_pre_reset_train"] = torch.stack(output_pre_reset_train, dim=1)

        return logits, info


class SNNSpatialParallelModel(nn.Module):
    """K independent d0 SNN modules with one shared per-query decoder.

    Each query owns four binary-one-hot input channels and ``hidden_per_query``
    LIF neurons. There are no cross-query synapses. The same Linear/MLP decoder
    is applied to every module's accumulated hidden spike counts, matching the
    decoder sharing used by ``windowed_shared`` temporal models while retaining
    genuine spatial hidden replication.

    This class is intentionally restricted to the non-spiking decoder scaffold;
    it is not a replacement for the direct-spiking-output architecture.
    """

    topology_type = "spatial_independent_shared_decoder"

    def __init__(
        self,
        *,
        n_queries: int,
        hidden_per_query: int,
        win_len: int,
        read_len: int,
        d_max: int,
        train_mode: str = "weights_only",
        fixed_delay_value: float = 0.0,
        lif_tau_m: float = 10.0,
        lif_threshold: float = 0.2,
        lif_reset: float = 0.0,
        lif_refractory: int = 2,
        dt: float = 1.0,
        surrogate_beta: float = 4.0,
        readout_type: str = "mlp",
    ):
        super().__init__()
        if n_queries < 1 or hidden_per_query < 1:
            raise ValueError("n_queries and hidden_per_query must be positive")
        if train_mode not in {"weights_only", "weights_and_delays", "delays_only"}:
            raise ValueError("unsupported train_mode")
        if readout_type not in {"linear", "mlp"}:
            raise ValueError("readout_type must be linear or mlp")

        self.n_queries = int(n_queries)
        self.hidden_per_query = int(hidden_per_query)
        self.n_hidden = self.n_queries * self.hidden_per_query
        self.n_hidden_total = self.n_hidden
        self.n_hidden2 = None
        self.n_input = 4 * self.n_queries
        self.win_len = int(win_len)
        self.read_len = int(read_len)
        self.T = self.win_len + self.read_len
        self.d_max = int(d_max)
        self.train_mode = train_mode
        self.observation_mode = "all_time"
        self.n_observation_bins = 1
        self.readout_feature_dim = self.hidden_per_query
        self.readout_type = readout_type
        self.use_output_spikes = False
        self.n_output_neurons = 0
        self.decoder_repetitions = self.n_queries
        self.input_event_fanout = self.hidden_per_query

        train_w = train_mode in {"weights_only", "weights_and_delays"}
        train_d = train_mode in {"delays_only", "weights_and_delays"}
        syn_kw = dict(
            d_max=self.d_max,
            fixed_delay_value=fixed_delay_value,
            train_weights=train_w,
            train_delays=train_d,
        )
        lif_kw = dict(
            tau_m=lif_tau_m, v_threshold=lif_threshold, v_reset=lif_reset,
            refractory_steps=lif_refractory, dt=dt, surrogate_beta=surrogate_beta,
        )
        self.syn_ih_modules = nn.ModuleList([
            DelayedSynapticLayer(4, self.hidden_per_query, **syn_kw)
            for _ in range(self.n_queries)
        ])
        self.lif_h_modules = nn.ModuleList([
            LIFNeurons(self.hidden_per_query, **lif_kw)
            for _ in range(self.n_queries)
        ])
        if readout_type == "mlp":
            hidden_r = max(self.hidden_per_query, self.n_queries * 8)
            self.readout = nn.Sequential(
                nn.Linear(self.hidden_per_query, hidden_r),
                nn.ReLU(),
                nn.Linear(hidden_r, 1),
            )
        else:
            self.readout = nn.Linear(self.hidden_per_query, 1)

    def weight_params(self):
        return [
            layer.weight for layer in self.syn_ih_modules
            if layer.weight.requires_grad
        ]

    def delay_params(self):
        return [
            layer.delay_raw for layer in self.syn_ih_modules
            if layer.delay_raw.requires_grad
        ]

    def readout_params(self):
        return list(self.readout.parameters())

    def get_delays(self) -> Dict[str, torch.Tensor]:
        return {
            f"ih_q{query}": layer.get_delays()
            for query, layer in enumerate(self.syn_ih_modules)
        }

    def delay_regularization(self) -> torch.Tensor:
        return torch.stack([
            layer.get_delays().mean() for layer in self.syn_ih_modules
        ]).mean()

    def observation_metadata(self) -> Dict[str, int | str]:
        decoder_parameters = sum(p.numel() for p in self.readout.parameters())
        return {
            "observation_mode": self.observation_mode,
            "observation_steps": self.T,
            "observation_bins": 1,
            "readout_feature_dim": self.hidden_per_query,
            "decoder_type": self.readout_type,
            "decoder_parameters": decoder_parameters,
            "decoder_trainable_parameters": sum(
                p.numel() for p in self.readout.parameters() if p.requires_grad
            ),
            "output_neurons": self.n_queries,
        }

    def forward(
        self,
        spike_input: torch.Tensor,
        record: bool = False,
        return_output_spike_train: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if return_output_spike_train:
            raise ValueError("spatial MLP scaffold has no spiking output train")
        batch, steps, channels = spike_input.shape
        if steps != self.T or channels != self.n_input:
            raise ValueError(
                f"expected input [B,{self.T},{self.n_input}], got {tuple(spike_input.shape)}"
            )
        device = spike_input.device
        depth = self.d_max + 1
        voltages, refractory, buffers, pointers = [], [], [], []
        delay_tensors = []
        accumulators = []
        spike_totals = []
        fired_any = []
        per_neuron_totals = []
        for query in range(self.n_queries):
            voltage, ref = self.lif_h_modules[query].init_state(batch, device)
            voltages.append(voltage)
            refractory.append(ref)
            buffers.append(torch.zeros(batch, depth, 4, device=device))
            pointers.append(0)
            delay_tensors.append(self.syn_ih_modules[query].get_delays())
            accumulators.append(torch.zeros(batch, self.hidden_per_query, device=device))
            spike_totals.append(torch.zeros(batch, device=device))
            fired_any.append(torch.zeros(batch, self.hidden_per_query, device=device))
            per_neuron_totals.append(torch.zeros(batch, self.hidden_per_query, device=device))

        recorded_spikes = [] if record else None
        recorded_membrane = [] if record else None
        for time in range(self.T):
            step_spikes, step_membrane = [], []
            for query in range(self.n_queries):
                current = self.syn_ih_modules[query](
                    buffers[query], delay_tensors[query], pointers[query]
                )
                spikes, voltages[query], refractory[query] = self.lif_h_modules[query](
                    current, voltages[query], refractory[query]
                )
                start = 4 * query
                buffers[query][:, pointers[query], :] = spike_input[:, time, start:start + 4]
                pointers[query] = (pointers[query] + 1) % depth
                accumulators[query] = accumulators[query] + spikes
                spike_totals[query] = spike_totals[query] + spikes.sum(dim=1)
                fired_any[query] = torch.maximum(fired_any[query], (spikes > 0).float())
                per_neuron_totals[query] = per_neuron_totals[query] + spikes
                if record:
                    step_spikes.append(spikes.detach().cpu())
                    step_membrane.append(voltages[query].detach().cpu())
            if record:
                recorded_spikes.append(torch.cat(step_spikes, dim=1))
                recorded_membrane.append(torch.cat(step_membrane, dim=1))

        features = torch.stack(accumulators, dim=1)  # [B,K,h]
        logits = self.readout(features).squeeze(-1)  # shared decoder -> [B,K]
        total_hidden_spikes = torch.stack(spike_totals, dim=1).sum(dim=1)
        active_hidden = torch.cat(fired_any, dim=1).sum(dim=1)
        info: Dict[str, torch.Tensor] = {
            "total_hidden_spikes": total_hidden_spikes,
            "active_hidden_neurons": active_hidden,
            "active_hidden_fraction": active_hidden / float(self.n_hidden),
            "trial_steps": self.T,
            "hidden_rate": torch.cat(per_neuron_totals, dim=1).mean(dim=0) / float(self.T),
        }
        info.update(self.observation_metadata())
        if record:
            info["hidden_spike_train"] = torch.stack(recorded_spikes, dim=1)
            info["hidden_membrane_train"] = torch.stack(recorded_membrane, dim=1)
            info["readout_features"] = features.detach().cpu()
        return logits, info
