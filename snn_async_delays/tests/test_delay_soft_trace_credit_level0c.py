import math
import unittest

import torch

from scripts.run_delay_soft_trace_credit_level0c import (
    expected_cells,
    initial_parameter_value,
    lif_response,
    load_protocol,
    make_delay_layer,
    matched_initial_delay,
    objective_loss,
    render_soft_trace,
    target_template,
    trace_w1_steps,
)


class DelaySoftTraceCreditLevel0CTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_preregistered_grid_has_360_cells(self):
        self.assertEqual(expected_cells(self.protocol), 360)

    def test_sigmoid_and_direct_start_at_matched_functional_delay(self):
        for label in self.protocol["delay"]["matched_initialization_labels_raw"]:
            expected = matched_initial_delay(self.protocol, float(label))
            sigmoid = make_delay_layer(
                self.protocol,
                parameterization="sigmoid",
                initialization_label=float(label),
                fixed_weight=1.0,
            )
            direct = make_delay_layer(
                self.protocol,
                parameterization="direct",
                initialization_label=float(label),
                fixed_weight=1.0,
            )
            self.assertAlmostEqual(float(sigmoid.get_delays().item()), expected, places=6)
            self.assertAlmostEqual(float(direct.get_delays().item()), expected, places=6)
            self.assertAlmostEqual(
                initial_parameter_value(self.protocol, "direct", float(label)),
                expected,
                places=8,
            )

    def test_lif_membrane_template_is_subthreshold_and_non_silent(self):
        template = target_template(
            self.protocol, path="lif_membrane", target_nominal_delay=5.0
        )
        self.assertEqual(float(template["spike"].sum().item()), 0.0)
        self.assertGreater(float(template["trace"].sum().item()), 0.05)
        self.assertEqual(int(template["trace"].argmax().item()), 8)
        self.assertLess(
            float(template["trace"].max().item()),
            float(self.protocol["lif"]["threshold_au"]),
        )

    def test_w1_equals_integer_shift_for_point_traces(self):
        left = torch.zeros(16)
        right = torch.zeros(16)
        left[4] = 1.0
        right[10] = 1.0
        self.assertAlmostEqual(float(trace_w1_steps(left, right).item()), 6.0, places=6)

    def test_candidate_gradients_point_both_directions_on_both_paths(self):
        cases = ((5.0, -2.0), (1.0, 0.0))
        for parameterization in ("sigmoid", "direct"):
            for path in ("buffer_current", "lif_membrane"):
                for objective in ("soft_centroid", "symmetric_kernel_alignment"):
                    for target, label in cases:
                        layer = make_delay_layer(
                            self.protocol,
                            parameterization=parameterization,
                            initialization_label=label,
                            fixed_weight=1.0,
                        )
                        rendered = render_soft_trace(
                            layer,
                            path=path,
                            total_steps=int(self.protocol["timeline"]["total_steps"]),
                            input_spike_step=int(self.protocol["timeline"]["input_spike_step"]),
                            lif_config=self.protocol["lif"],
                        )
                        template = target_template(
                            self.protocol, path=path, target_nominal_delay=target
                        )
                        loss = objective_loss(
                            rendered["trace"],
                            template["trace"],
                            objective=objective,
                            filter_tau_steps=3.0,
                            kernel_tau_steps=4.0,
                        )
                        loss.backward()
                        gradient = float(layer.delay_raw.grad.item())
                        initial_delay = float(layer.get_delays().detach().item())
                        self.assertTrue(math.isfinite(gradient))
                        self.assertGreater(
                            abs(gradient), 1e-12,
                            msg=(parameterization, path, objective, target, label, gradient),
                        )
                        self.assertGreater(
                            gradient * (initial_delay - target),
                            0.0,
                            msg=(parameterization, path, objective, target, label, gradient),
                        )

    def test_matched_parameterizations_have_identical_initial_traces(self):
        for path in ("buffer_current", "lif_membrane"):
            traces = []
            for parameterization in ("sigmoid", "direct"):
                layer = make_delay_layer(
                    self.protocol,
                    parameterization=parameterization,
                    initialization_label=-2.0,
                    fixed_weight=1.0,
                )
                traces.append(render_soft_trace(
                    layer,
                    path=path,
                    total_steps=int(self.protocol["timeline"]["total_steps"]),
                    input_spike_step=int(self.protocol["timeline"]["input_spike_step"]),
                    lif_config=self.protocol["lif"],
                )["trace"])
            self.assertTrue(torch.allclose(traces[0], traces[1], atol=1e-7, rtol=1e-7))


if __name__ == "__main__":
    unittest.main()
