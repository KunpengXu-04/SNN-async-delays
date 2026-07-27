import pytest
import torch

from snn.synapses import DelayedSynapticLayer


def _layer(mode: str, sigma: float = 0.75) -> DelayedSynapticLayer:
    layer = DelayedSynapticLayer(
        1,
        1,
        d_max=4,
        delay_param_type="direct",
        delay_gradient_mode=mode,
        delay_gradient_sigma=sigma,
    )
    with torch.no_grad():
        layer.weight.fill_(1.0)
    return layer


@pytest.mark.parametrize("delay", [0.0, 1.25, 2.0, 3.75, 4.0])
def test_backward_modes_preserve_shift_register_forward(delay: float) -> None:
    buf = torch.tensor([[[0.0], [1.0], [4.0], [9.0], [16.0]]])
    outputs = []
    for mode in ("right_linear", "symmetric_integer", "gaussian_ste"):
        d_cont = torch.tensor([[delay]], requires_grad=True)
        outputs.append(_layer(mode)(buf, d_cont=d_cont))
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


def test_backward_modes_preserve_circular_buffer_forward() -> None:
    # With ptr=2, logical delay taps d=0..4 map to physical indices 1,0,4,3,2.
    buf = torch.tensor([[[1.0], [0.0], [16.0], [9.0], [4.0]]])
    outputs = []
    for mode in ("right_linear", "symmetric_integer", "gaussian_ste"):
        d_cont = torch.tensor([[2.25]], requires_grad=True)
        outputs.append(_layer(mode)(buf, d_cont=d_cont, buf_ptr=2))
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])
    assert outputs[0].item() == pytest.approx(5.25)


def test_symmetric_integer_uses_central_subgradient_only_at_integer() -> None:
    buf = torch.tensor([[[0.0], [1.0], [4.0], [9.0], [16.0]]])

    integer_delay = torch.tensor([[2.0]], requires_grad=True)
    _layer("symmetric_integer")(buf, d_cont=integer_delay).backward()
    assert integer_delay.grad.item() == pytest.approx((9.0 - 1.0) / 2.0)

    fractional_delay = torch.tensor([[2.25]], requires_grad=True)
    _layer("symmetric_integer")(buf, d_cont=fractional_delay).backward()
    assert fractional_delay.grad.item() == pytest.approx(9.0 - 4.0)


def test_gaussian_ste_backward_matches_soft_kernel_derivative() -> None:
    sigma = 0.75
    values = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0])
    buf = values.view(1, 5, 1)

    delay = torch.tensor([[2.0]], requires_grad=True)
    _layer("gaussian_ste", sigma)(buf, d_cont=delay).backward()
    actual = delay.grad.detach().clone()

    reference_delay = torch.tensor(2.0, requires_grad=True)
    taps = torch.arange(5, dtype=reference_delay.dtype)
    weights = torch.softmax(-0.5 * ((taps - reference_delay) / sigma).square(), dim=0)
    (weights * values).sum().backward()
    assert actual.item() == pytest.approx(reference_delay.grad.item(), rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("mode", ["unknown", "central"])
def test_invalid_delay_gradient_mode_is_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="delay_gradient_mode"):
        DelayedSynapticLayer(1, 1, delay_gradient_mode=mode)


def test_nonpositive_gaussian_sigma_is_rejected() -> None:
    with pytest.raises(ValueError, match="delay_gradient_sigma"):
        DelayedSynapticLayer(1, 1, delay_gradient_sigma=0.0)
