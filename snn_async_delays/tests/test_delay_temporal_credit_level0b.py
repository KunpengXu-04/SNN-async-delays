import math
import unittest

import torch

from scripts.run_delay_temporal_credit_level0b import (
    expected_cells,
    load_protocol,
    make_delay_layer,
    render_temporal_trace,
    target_arrival_step,
    temporal_loss,
)


class DelayTemporalCreditLevel0BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def fixed_delay_layer(self, delay_value: float, weight: float):
        layer = make_delay_layer(d_max=8, initial_raw=0.0, fixed_weight=weight)
        raw = math.log(delay_value / (8.0 - delay_value))
        with torch.no_grad():
            layer.delay_raw.fill_(raw)
        return layer

    def test_preregistered_grid_has_180_cells(self):
        self.assertEqual(expected_cells(self.protocol), 180)

    def test_target_arrival_includes_buffer_one_step_offset(self):
        self.assertEqual(target_arrival_step(self.protocol, 1.0), 4.0)
        self.assertEqual(target_arrival_step(self.protocol, 5.0), 8.0)

    def test_buffer_integer_delay_peaks_at_declared_arrival(self):
        layer = self.fixed_delay_layer(5.0, 1.0)
        rendered = render_temporal_trace(
            layer, path="buffer_current", total_steps=16, input_spike_step=2,
            lif_config=self.protocol["lif"],
        )
        self.assertEqual(int(rendered["output"].argmax().item()), 8)
        self.assertAlmostEqual(float(rendered["output"].sum().item()), 1.0, places=6)

    def test_fixed_weight_lif_emits_one_spike_at_declared_arrival(self):
        layer = self.fixed_delay_layer(5.0, 4.0)
        rendered = render_temporal_trace(
            layer, path="lif_spike", total_steps=16, input_spike_step=2,
            lif_config=self.protocol["lif"],
        )
        self.assertEqual(int(rendered["output"].argmax().item()), 8)
        self.assertAlmostEqual(float(rendered["output"].sum().item()), 1.0, places=6)

    def test_both_losses_backpropagate_finite_delay_gradient(self):
        for path, weight in (("buffer_current", 1.0), ("lif_spike", 4.0)):
            for loss_mode in ("arrival_centroid", "filtered_trace"):
                layer = make_delay_layer(d_max=8, initial_raw=-2.0, fixed_weight=weight)
                rendered = render_temporal_trace(
                    layer, path=path, total_steps=16, input_spike_step=2,
                    lif_config=self.protocol["lif"],
                )
                loss, _, _ = temporal_loss(
                    rendered["output"], loss_mode=loss_mode,
                    target_arrival=8, filter_tau_steps=3.0,
                )
                loss.backward()
                self.assertIsNotNone(layer.delay_raw.grad)
                self.assertTrue(torch.isfinite(layer.delay_raw.grad).all())


if __name__ == "__main__":
    unittest.main()
