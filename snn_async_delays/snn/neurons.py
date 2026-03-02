"""
LIF neuron layer with surrogate-gradient spike function.

Effective pipeline per timestep:
  1. Gate input with refractory mask
  2. Euler-decay membrane update: V <- decay*V + (1-decay)*I_syn
  3. Spike = Heaviside(V - threshold)   [forward]
           = fast-sigmoid surrogate    [backward]
  4. Reset voltage and increment refractory counter (detached from graph)
"""

import torch
import torch.nn as nn


class _SurrogateSpike(torch.autograd.Function):
    """
    Forward : Heaviside(x)   where x = V - threshold
    Backward: beta / (1 + beta*|x|)^2   (fast-sigmoid derivative)
    """

    @staticmethod
    def forward(ctx, x, beta):
        ctx.save_for_backward(x)
        ctx.beta = beta
        return (x >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        beta = ctx.beta
        sg = beta / (1.0 + beta * x.abs()) ** 2
        return grad_out * sg, None


def spike_fn(v_minus_thr: torch.Tensor, beta: float = 4.0) -> torch.Tensor:
    return _SurrogateSpike.apply(v_minus_thr, beta)


class LIFNeurons(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer.

    Args:
        n_neurons        : number of neurons in this layer
        tau_m            : membrane time constant [ms]
        v_threshold      : spike threshold (default 1.0)
        v_reset          : reset potential after spike (default 0.0)
        refractory_steps : refractory period [timesteps]
        dt               : simulation timestep [ms]
        surrogate_beta   : sharpness of the surrogate gradient
    """

    def __init__(
        self,
        n_neurons: int,
        tau_m: float = 10.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        refractory_steps: int = 2,
        dt: float = 1.0,
        surrogate_beta: float = 4.0,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.refractory_steps = float(refractory_steps)
        self.surrogate_beta = surrogate_beta
        decay_val = float(torch.exp(torch.tensor(-dt / tau_m)).item())
        self.register_buffer("decay", torch.tensor(decay_val))

    # ------------------------------------------------------------------
    def init_state(self, batch_size: int, device="cpu"):
        """Return (v, ref) tensors initialised to zero."""
        v = torch.zeros(batch_size, self.n_neurons, device=device)
        ref = torch.zeros(batch_size, self.n_neurons, device=device)
        return v, ref

    # ------------------------------------------------------------------
    def forward(self, I_syn, v, ref):
        """
        Args:
            I_syn : [B, N]  synaptic current
            v     : [B, N]  membrane voltage
            ref   : [B, N]  refractory countdown (float >= 0)

        Returns:
            spike   : [B, N]
            v_new   : [B, N]
            ref_new : [B, N]
        """
        not_ref = (ref <= 0.0).float()
        v_new = self.decay * v + (1.0 - self.decay) * I_syn * not_ref

        spike = spike_fn(v_new - self.v_threshold, self.surrogate_beta)

        # Reset and refractory update — detached so they don't interfere
        # with the surrogate-gradient path through 'spike'
        spike_d = spike.detach()
        v_new = v_new * (1.0 - spike_d) + self.v_reset * spike_d
        ref_new = torch.clamp(ref - 1.0, min=0.0) + spike_d * self.refractory_steps

        return spike, v_new, ref_new
