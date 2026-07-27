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
The historical ``right_linear`` backward therefore takes the right-hand
slope at exact integer delays.  Optional backward-only estimators can replace
that ambiguous integer subgradient without changing the forward computation.
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
        fixed_delay_distribution: str | None = None,
        fixed_delay_seed: int = 0,
        fixed_delay_low: float = 0.0,
        fixed_delay_high: float | None = None,
        shared_delay: bool = False,
        delay_tying: str | None = None,
        delay_group_index: list[int] | torch.Tensor | None = None,
        delay_init_mode: str = "constant",
        delay_init_raw: float = -2.0,
        delay_init_std: float = 0.25,
        delay_gradient_mode: str = "right_linear",
        delay_gradient_sigma: float = 0.75,
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
        self.fixed_delay_distribution = fixed_delay_distribution
        if delay_tying is None:
            delay_tying = "global" if shared_delay else "pair"
        if delay_tying not in {"global", "post_neuron", "pre_group", "pair"}:
            raise ValueError(
                "delay_tying must be global, post_neuron, pre_group, or pair"
            )
        if shared_delay and delay_tying != "global":
            raise ValueError("legacy shared_delay=True is compatible only with global tying")
        self.delay_tying = str(delay_tying)
        self.shared_delay = self.delay_tying == "global"
        if self.delay_tying == "pre_group":
            if delay_group_index is None:
                raise ValueError("pre_group tying requires delay_group_index")
            group_index = torch.as_tensor(delay_group_index, dtype=torch.long)
            if group_index.ndim != 1 or group_index.numel() != n_pre:
                raise ValueError("delay_group_index must contain one group per pre neuron")
            if int(group_index.min().item()) < 0:
                raise ValueError("delay groups must be non-negative")
            unique = torch.unique(group_index, sorted=True)
            expected = torch.arange(unique.numel(), dtype=torch.long)
            if not torch.equal(unique, expected):
                raise ValueError("delay groups must be contiguous from zero")
            self.register_buffer("delay_group_index", group_index)
            self.delay_group_count = int(unique.numel())
        else:
            if delay_group_index is not None:
                raise ValueError("delay_group_index is valid only for pre_group tying")
            self.register_buffer("delay_group_index", None)
            self.delay_group_count = 0
        self.train_delays = train_delays
        if delay_gradient_mode not in {"right_linear", "symmetric_integer", "gaussian_ste"}:
            raise ValueError(
                "delay_gradient_mode must be right_linear, symmetric_integer, or gaussian_ste"
            )
        if float(delay_gradient_sigma) <= 0.0:
            raise ValueError("delay_gradient_sigma must be positive")
        self.delay_gradient_mode = str(delay_gradient_mode)
        self.delay_gradient_sigma = float(delay_gradient_sigma)
        if delay_init_mode not in {"constant", "scalar_noise"}:
            raise ValueError("delay_init_mode must be 'constant' or 'scalar_noise'")
        # Compact tied layouts are also valid for frozen d0/fixed controls.
        # Their stored delay degrees of freedom must match trainable arms even
        # when requires_grad=False.
        if fixed_delay_distribution not in {None, "uniform"}:
            raise ValueError("fixed_delay_distribution must be None or 'uniform'")
        if train_delays and fixed_delay_distribution is not None:
            raise ValueError("fixed heterogeneous delays cannot also be trainable")

        # ------ Weights ------------------------------------------------
        w_init = torch.randn(n_pre, n_post) * math.sqrt(2.0 / n_pre)
        if train_weights:
            self.weight = nn.Parameter(w_init)
        else:
            self.register_buffer("weight", w_init)

        # ------ Delays -------------------------------------------------
        # sigmoid(-2) ~= 0.12 => initial delays ~= 0.12 * d_max (small)
        tying_shape = {
            "global": (1, 1),
            "post_neuron": (1, n_post),
            "pre_group": (self.delay_group_count, 1),
            "pair": (n_pre, n_post),
        }[self.delay_tying]
        d_shape = (0,) if fixed_delay_distribution is not None else tying_shape
        d_init = torch.full(d_shape, float(delay_init_raw))
        if delay_init_mode == "scalar_noise" and d_init.numel() > 1:
            d_init = d_init + torch.randn_like(d_init) * float(delay_init_std)
        if train_delays:
            self.delay_raw = nn.Parameter(d_init)
        else:
            self.register_buffer("delay_raw", d_init)
        if fixed_delay_distribution == "uniform":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(fixed_delay_seed))
            low = max(0.0, float(fixed_delay_low))
            high = min(float(d_max), float(fixed_delay_high if fixed_delay_high is not None else d_max))
            if high < low:
                raise ValueError("fixed_delay_high must be >= fixed_delay_low")
            fixed = low + torch.rand((n_pre, n_post), generator=generator) * (high - low)
            self.register_buffer("fixed_delay_tensor", fixed)
        else:
            self.register_buffer("fixed_delay_tensor", None)

    # ------------------------------------------------------------------
    def get_delays(self) -> torch.Tensor:
        """Delay values in [0, d_max] (continuous or quantized via STE)."""
        if self.fixed_delay_tensor is not None:
            return self.fixed_delay_tensor
        if (not self.train_delays) and (self.fixed_delay_value is not None):
            d = torch.full_like(self.delay_raw, float(self.fixed_delay_value))
            return torch.clamp(d, 0.0, float(self.d_max))

        if self.delay_param_type == "sigmoid":
            d_cont = self.d_max * torch.sigmoid(self.delay_raw)
            d_cont = torch.clamp(d_cont, 0.0, float(self.d_max))
            return self._expand_tied_delays(d_cont)

        if self.delay_param_type == "direct":
            d_cont = torch.clamp(self.delay_raw, 0.0, float(self.d_max))
            return self._expand_tied_delays(d_cont)

        if self.delay_param_type == "quantized":
            d_cont = self.d_max * torch.sigmoid(self.delay_raw)
            step = max(self.delay_step, 1e-6)
            d_quant = torch.round(d_cont / step) * step
            # Straight-through estimator: forward uses quantized value,
            # backward behaves like identity on d_cont.
            d_ste = d_cont + (d_quant - d_cont).detach()
            d_ste = torch.clamp(d_ste, 0.0, float(self.d_max))
            return self._expand_tied_delays(d_ste)

        raise ValueError(f"Unsupported delay_param_type: {self.delay_param_type}")

    def _expand_tied_delays(self, values: torch.Tensor) -> torch.Tensor:
        if self.delay_tying == "pair":
            return values
        if self.delay_tying == "pre_group":
            return values.index_select(0, self.delay_group_index).expand(
                self.n_pre, self.n_post
            )
        return values.expand(self.n_pre, self.n_post)

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

        # Keep the historical hard/linear forward exactly unchanged while
        # optionally replacing only d(s_del)/d(d_cont).  The correction is
        # zero-valued in the forward pass, but its derivative is one.
        if self.delay_gradient_mode != "right_linear":
            legacy_slope = s_c - s_f

            if self.delay_gradient_mode == "symmetric_integer":
                d_prev = torch.clamp(d_floor - 1, 0, self.d_max)
                d_next = torch.clamp(d_floor + 1, 0, self.d_max)
                if buf_ptr is not None:
                    idx_prev = ((buf_ptr - 1 - d_prev) % (self.d_max + 1)).unsqueeze(0).expand(B, -1, -1)
                    idx_next = ((buf_ptr - 1 - d_next) % (self.d_max + 1)).unsqueeze(0).expand(B, -1, -1)
                else:
                    idx_prev = d_prev.unsqueeze(0).expand(B, -1, -1)
                    idx_next = d_next.unsqueeze(0).expand(B, -1, -1)
                s_prev = torch.gather(buf_t, 2, idx_prev)
                s_next = torch.gather(buf_t, 2, idx_next)
                denominator = (d_next - d_prev).clamp_min(1).to(s_next.dtype)
                central_slope = (s_next - s_prev) / denominator
                at_integer = alpha.detach().abs() <= 1e-6
                desired_slope = torch.where(at_integer.unsqueeze(0), central_slope, legacy_slope)
            else:  # gaussian_ste
                taps = torch.arange(
                    self.d_max + 1,
                    device=d_cont.device,
                    dtype=d_cont.dtype,
                )
                distance = taps.view(1, 1, -1) - d_cont.unsqueeze(-1)
                soft_weights = torch.softmax(
                    -0.5 * (distance / self.delay_gradient_sigma).square(),
                    dim=-1,
                )
                mean_tap = (soft_weights * taps.view(1, 1, -1)).sum(dim=-1, keepdim=True)
                weight_derivative = soft_weights * (
                    taps.view(1, 1, -1) - mean_tap
                ) / (self.delay_gradient_sigma ** 2)
                if buf_ptr is not None:
                    physical_taps = (buf_ptr - 1 - taps.long()) % (self.d_max + 1)
                    logical_buf = buf_t.index_select(2, physical_taps)
                else:
                    logical_buf = buf_t
                desired_slope = torch.einsum(
                    "bpd,pod->bpo",
                    logical_buf,
                    weight_derivative,
                )

            s_del = s_del + (
                d_cont - d_cont.detach()
            ).unsqueeze(0) * (desired_slope - legacy_slope).detach()

        # weighted sum over N_pre  →  [B, N_post]
        I_syn = (s_del * self.weight).sum(dim=1)
        return I_syn
