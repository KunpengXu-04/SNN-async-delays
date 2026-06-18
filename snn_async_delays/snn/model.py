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
        if (not train_d) and (fixed_delay_value is None):
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

    The readout always accumulates spikes from the last hidden layer
    during [win_len, win_len + read_len).

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

        if (not train_d) and (fixed_delay_value is None):
            fixed_delay_value = 0.0

        lif_kw = dict(
            tau_m=lif_tau_m, v_threshold=lif_threshold, v_reset=lif_reset,
            refractory_steps=lif_refractory, dt=dt, surrogate_beta=surrogate_beta,
        )

        syn_kw = dict(
            d_max=d_max, delay_param_type=delay_param_type, delay_step=delay_step,
            fixed_delay_value=fixed_delay_value, train_weights=train_w, train_delays=train_d,
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

        if use_output_spikes:
            # Spiking output layer: hidden → LIF output → spike counts = logits
            n_out = n_output_neurons if n_output_neurons is not None else n_queries
            self.n_output_neurons = n_out
            self.syn_ho = DelayedSynapticLayer(readout_in, n_out, **syn_kw)
            self.lif_o  = LIFNeurons(n_out, **lif_kw)
            self.readout = None   # not used; set to None to avoid confusion
        else:
            self.n_output_neurons = 0
            if readout_type == "mlp":
                hidden_r = max(readout_in, n_queries * 8)
                self.readout = nn.Sequential(
                    nn.Linear(readout_in, hidden_r),
                    nn.ReLU(),
                    nn.Linear(hidden_r, n_queries),
                )
            else:
                self.readout = nn.Linear(readout_in, n_queries)

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

        # Optional full spike-train recording (last hidden layer + h1 for 2-layer)
        hidden_train: list = [] if record else None  # type: ignore[assignment]
        h1_train:     list = [] if (record and self._num_layers == 2) else None  # type: ignore[assignment]

        # Spiking output layer extras
        if self.use_output_spikes:
            v_o, ref_o    = self.lif_o.init_state(B, device)
            buf_ho        = torch.zeros(B, d_max_1, self.readout_in, device=device)
            buf_ho_ptr    = 0
            d_cont_ho     = self.syn_ho.get_delays()
            output_acc    = torch.zeros(B, self.n_output_neurons, device=device)
            total_o_spk   = torch.zeros(B, device=device)
            output_train: list = [] if record else None  # type: ignore[assignment]

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
                h1_fired_any = torch.maximum(h1_fired_any, (spike_h  > 0).float())
                h2_fired_any = torch.maximum(h2_fired_any, (spike_h2 > 0).float())

                if t >= self.win_len:
                    last_acc = last_acc + spike_h2

                if record:
                    h1_train.append(spike_h.detach().cpu())     # layer 1
                    hidden_train.append(spike_h2.detach().cpu()) # layer 2

                if self.use_output_spikes:
                    spike_last_h = spike_h2
            else:
                total_h_spk      = total_h_spk + spike_h.sum(dim=1)
                hidden_fired_any = torch.maximum(hidden_fired_any, (spike_h > 0).float())

                if t >= self.win_len:
                    last_acc = last_acc + spike_h

                if record:
                    hidden_train.append(spike_h.detach().cpu())

                if self.use_output_spikes:
                    spike_last_h = spike_h

            # Spiking output layer (after last hidden layer, every timestep)
            if self.use_output_spikes:
                I_o = self.syn_ho(buf_ho, d_cont_ho, buf_ho_ptr)
                spike_o, v_o, ref_o = self.lif_o(I_o, v_o, ref_o)
                buf_ho[:, buf_ho_ptr, :] = spike_last_h
                buf_ho_ptr = (buf_ho_ptr + 1) % d_max_1
                total_o_spk = total_o_spk + spike_o.sum(dim=1)
                if t >= self.win_len:
                    output_acc = output_acc + spike_o
                if record:
                    output_train.append(spike_o.detach().cpu())

        if self.use_output_spikes:
            logits = output_acc   # [B, n_output_neurons]  spike counts as logits
        else:
            logits = self.readout(last_acc)  # [B, K]

        if self._num_layers == 2:
            active_h2 = h2_fired_any.sum(dim=1)
            info = {
                "total_hidden_spikes":    total_h1_spk + total_h2_spk,
                "layer1_hidden_spikes":   total_h1_spk,
                "layer2_hidden_spikes":   total_h2_spk,
                "active_hidden_neurons":  active_h2,
                "active_hidden_fraction": active_h2 / float(self.n_hidden2),
                "trial_steps":            T,
            }
        else:
            active_hidden_neurons = hidden_fired_any.sum(dim=1)
            info = {
                "total_hidden_spikes":    total_h_spk,
                "active_hidden_neurons":  active_hidden_neurons,
                "active_hidden_fraction": active_hidden_neurons / float(self.n_hidden),
                "trial_steps":            T,
            }

        if self.use_output_spikes:
            info["total_output_spikes"] = total_o_spk

        if record:
            # [B, T, last_hidden_size] — always present
            info["hidden_spike_train"] = torch.stack(hidden_train, dim=1)
            if self._num_layers == 2:
                # [B, T, h1] — first hidden layer (only for 2-layer models)
                info["hidden1_spike_train"] = torch.stack(h1_train, dim=1)
            if self.use_output_spikes:
                info["output_spike_train"] = torch.stack(output_train, dim=1)  # [B,T,n_out]

        return logits, info
