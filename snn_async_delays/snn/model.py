"""
SNNModel: single hidden-layer SNN with trainable delays.

Architecture
------------
  Input spikes  →  DelayedSynapticLayer(W_ih, D_ih)
                →  LIFNeurons(N_h)
                →  [optional] DelayedSynapticLayer(W_ho, D_ho)
                            → LIFNeurons(N_o)
                →  Linear readout (always trainable)

train_mode controls W_ih / D_ih (and W_ho / D_ho if use_output_layer=True):
  'weights_only'       : weights trained,  delays frozen
  'delays_only'        : weights frozen,   delays trained
  'weights_and_delays' : both trained

The linear readout is always trained regardless of train_mode.

Slot-based readout
------------------
forward() receives a list of SlotBoundaries.  During each slot's
readout window [read_start, read_end), hidden (or output) spikes are
accumulated.  After the full T-step simulation, the accumulated counts
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
    win_start:  int   # start of input-encoding window [timestep, inclusive]
    win_end:    int   # end   of input-encoding window [exclusive]
    read_start: int   # start of readout window [inclusive]
    read_end:   int   # end   of readout window [exclusive]


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


# ---------------------------------------------------------------------------

class SNNModel(nn.Module):
    """
    Args:
        n_input          : input neuron count (2 for A/B; 2+n_ops for Step 3)
        n_hidden         : hidden neuron count
        d_max            : maximum delay in timesteps
        train_mode       : 'weights_only' | 'delays_only' | 'weights_and_delays'
        delay_param_type : 'sigmoid' | 'direct'
        use_output_layer : if True, add a hidden→output LIF layer
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

        lif_kw = dict(
            tau_m=lif_tau_m,
            v_threshold=lif_threshold,
            v_reset=lif_reset,
            refractory_steps=lif_refractory,
            dt=dt,
            surrogate_beta=surrogate_beta,
        )

        # Input → Hidden
        self.syn_ih = DelayedSynapticLayer(
            n_input, n_hidden, d_max=d_max,
            delay_param_type=delay_param_type,
            train_weights=train_w, train_delays=train_d,
        )
        self.lif_h = LIFNeurons(n_hidden, **lif_kw)

        # Hidden → Output (optional)
        if use_output_layer:
            self.syn_ho = DelayedSynapticLayer(
                n_hidden, n_output, d_max=d_max,
                delay_param_type=delay_param_type,
                train_weights=train_w, train_delays=train_d,
            )
            self.lif_o = LIFNeurons(n_output, **lif_kw)

        # Linear readout (always trained)
        readout_in = n_output if (use_output_layer and readout_source == "output") else n_hidden
        self.readout = nn.Linear(readout_in, 1)

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def forward(
        self,
        spike_input: torch.Tensor,
        slots: List[SlotBoundaries],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            spike_input : [B, T, N_input]  pre-generated spike trains
            slots       : list of SlotBoundaries (readout windows)

        Returns:
            logits : [B, K]
            info   : dict with 'total_hidden_spikes', 'total_output_spikes'
        """
        B, T, _ = spike_input.shape
        device = spike_input.device
        K = len(slots)

        # Initialise neuron states
        v_h, ref_h = self.lif_h.init_state(B, device)
        if self.use_output_layer:
            v_o, ref_o = self.lif_o.init_state(B, device)

        # Spike buffers: buf[:, d, :] = spikes from (d+1) steps ago
        buf_in = torch.zeros(B, self.d_max + 1, self.n_input,  device=device)
        if self.use_output_layer:
            buf_h  = torch.zeros(B, self.d_max + 1, self.n_hidden, device=device)

        # Accumulators for each readout window
        hidden_acc = [torch.zeros(B, self.n_hidden, device=device) for _ in range(K)]
        if self.use_output_layer:
            output_acc = [torch.zeros(B, self.n_output, device=device) for _ in range(K)]

        # Pre-build readout lookup: t → list of active slot indices
        readout_at: List[List[int]] = [[] for _ in range(T)]
        for k, slot in enumerate(slots):
            for t in range(slot.read_start, min(slot.read_end, T)):
                readout_at[t].append(k)

        total_h_spk = torch.zeros(B, device=device)
        total_o_spk = torch.zeros(B, device=device)

        # ---- Main simulation loop ----------------------------------------
        for t in range(T):
            x_t = spike_input[:, t, :]           # [B, N_input]

            # Layer 1: input → hidden
            I_h = self.syn_ih(buf_in)             # [B, N_hidden]
            spike_h, v_h, ref_h = self.lif_h(I_h, v_h, ref_h)

            # Layer 2: hidden → output (optional)
            if self.use_output_layer:
                I_o = self.syn_ho(buf_h)          # [B, N_output]
                spike_o, v_o, ref_o = self.lif_o(I_o, v_o, ref_o)

            # Update buffers (shift right, prepend current spikes)
            buf_in = torch.cat([x_t.unsqueeze(1), buf_in[:, :-1, :]], dim=1)
            if self.use_output_layer:
                buf_h = torch.cat([spike_h.unsqueeze(1), buf_h[:, :-1, :]], dim=1)

            # Spike counters
            total_h_spk = total_h_spk + spike_h.sum(dim=1)
            if self.use_output_layer:
                total_o_spk = total_o_spk + spike_o.sum(dim=1)

            # Accumulate in active readout windows
            for k in readout_at[t]:
                hidden_acc[k] = hidden_acc[k] + spike_h
                if self.use_output_layer:
                    output_acc[k] = output_acc[k] + spike_o

        # ---- Readout -----------------------------------------------------
        use_output = self.use_output_layer and self.readout_source == "output"
        acc_list = output_acc if use_output else hidden_acc

        logits = torch.stack(
            [self.readout(acc_list[k]).squeeze(-1) for k in range(K)],
            dim=1,
        )  # [B, K]

        info = {
            "total_hidden_spikes": total_h_spk,
            "total_output_spikes": total_o_spk,
        }
        return logits, info
