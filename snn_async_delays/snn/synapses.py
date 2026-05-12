"""
Delayed synaptic layer.

Timing convention
-----------------
The spike buffer stores past activity as:
    buf[:, d, :]  =  spikes emitted  (d+1)  timesteps ago
so delay parameter d_param=0 gives an effective 1-step delay (minimum).
The buffer is updated AFTER the I_syn computation each timestep:
    new_buf = concat([x_t.unsqueeze(1), old_buf[:, :-1, :]], dim=1)

Gradient through delays
-----------------------
Continuous delay: d_cont = d_max * sigmoid(d_raw) in [0, d_max]
Forward uses linear interpolation between floor and ceil indices:
    s_delayed = (1-alpha)*buf[floor] + alpha*buf[ceil]
This gives a smooth gradient through d_cont (and hence d_raw) whenever
the spike trains at floor and ceil differ.
"""

import math
import torch
import torch.nn as nn


class DelayedSynapticLayer(nn.Module):
    """
    Fully-connected synaptic layer with per-synapse trainable delays.

    Args:
        n_pre            : number of pre-synaptic neurons
        n_post           : number of post-synaptic neurons
        d_max            : maximum delay index (buffer size = d_max+1)
        delay_param_type : 'sigmoid' | 'direct' | 'quantized'
        delay_step       : quantization step (used when delay_param_type='quantized')
        fixed_delay_value: if set and delays are frozen, force all delays to this value
        train_weights    : whether weights are trained
        train_delays     : whether delays are trained
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        d_max: int = 50,
        delay_param_type: str = "sigmoid",
        delay_step: float = 1.0,
        fixed_delay_value: float | None = None,
        train_weights: bool = True,
        train_delays: bool = True,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.d_max = d_max
        self.delay_param_type = delay_param_type
        self.delay_step = float(delay_step)
        self.fixed_delay_value = fixed_delay_value
        self.train_delays = train_delays

        # ------ Weights ------------------------------------------------
        w_init = torch.randn(n_pre, n_post) * math.sqrt(2.0 / n_pre)
        if train_weights:
            self.weight = nn.Parameter(w_init)
        else:
            self.register_buffer("weight", w_init)

        # ------ Delays -------------------------------------------------
        # sigmoid(-2) ~= 0.12 => initial delays ~= 0.12 * d_max (small)
        d_init = torch.full((n_pre, n_post), -2.0)
        if train_delays:
            self.delay_raw = nn.Parameter(d_init)
        else:
            self.register_buffer("delay_raw", d_init)

    # ------------------------------------------------------------------
    def get_delays(self) -> torch.Tensor:
        """Delay values in [0, d_max] (continuous or quantized via STE)."""
        if (not self.train_delays) and (self.fixed_delay_value is not None):
            d = torch.full_like(self.delay_raw, float(self.fixed_delay_value))
            return torch.clamp(d, 0.0, float(self.d_max))

        if self.delay_param_type == "sigmoid":
            d_cont = self.d_max * torch.sigmoid(self.delay_raw)
            return torch.clamp(d_cont, 0.0, float(self.d_max))

        if self.delay_param_type == "direct":
            return torch.clamp(self.delay_raw, 0.0, float(self.d_max))

        if self.delay_param_type == "quantized":
            d_cont = self.d_max * torch.sigmoid(self.delay_raw)
            step = max(self.delay_step, 1e-6)
            d_quant = torch.round(d_cont / step) * step
            # Straight-through estimator: forward uses quantized value,
            # backward behaves like identity on d_cont.
            d_ste = d_cont + (d_quant - d_cont).detach()
            return torch.clamp(d_ste, 0.0, float(self.d_max))

        raise ValueError(f"Unsupported delay_param_type: {self.delay_param_type}")

    # ------------------------------------------------------------------
    def forward(
        self,
        buf: torch.Tensor,
        d_cont: torch.Tensor | None = None,
        buf_ptr: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            buf     : [B, d_max+1, N_pre]
                      Shift-register mode  (buf_ptr=None): buf[:, d, :] = spikes from (d+1) steps ago.
                      Circular-buffer mode (buf_ptr=int):  write head; buf[:, (ptr-1-d)%(d_max+1), :]
                      = spikes from (d+1) steps ago.
            d_cont  : optional pre-computed delays [N_pre, N_post] in [0, d_max].
                      If None, calls get_delays() internally (slower for T-step loops).
            buf_ptr : circular buffer write-head position (int).  None → shift-register mode.

        Returns:
            I_syn : [B, N_post]
        """
        B = buf.shape[0]

        if d_cont is None:
            d_cont = self.get_delays()                                   # [N_pre, N_post]

        d_floor = torch.clamp(d_cont.detach().floor().long(), 0, self.d_max)
        d_ceil  = torch.clamp(d_floor + 1, 0, self.d_max)
        alpha   = d_cont - d_floor.float()                               # grad flows

        # buf transposed: [B, N_pre, d_max+1]  (contiguous for gather)
        buf_t = buf.permute(0, 2, 1).contiguous()

        if buf_ptr is not None:
            d_max_1 = self.d_max + 1
            idx_f = ((buf_ptr - 1 - d_floor) % d_max_1).unsqueeze(0).expand(B, -1, -1)
            idx_c = ((buf_ptr - 1 - d_ceil)  % d_max_1).unsqueeze(0).expand(B, -1, -1)
        else:
            idx_f = d_floor.unsqueeze(0).expand(B, -1, -1)              # [B, N_pre, N_post]
            idx_c = d_ceil.unsqueeze(0).expand(B, -1, -1)

        s_f   = torch.gather(buf_t, 2, idx_f)                           # [B, N_pre, N_post]
        s_c   = torch.gather(buf_t, 2, idx_c)
        s_del = (1.0 - alpha) * s_f + alpha * s_c                       # broadcast over B

        # weighted sum over N_pre  →  [B, N_post]
        I_syn = (s_del * self.weight).sum(dim=1)
        return I_syn
