import math
import unittest

import torch

from scripts.run_delay_hard_output_soft_credit_level0d import (
    auxiliary_trace,
    condition_map,
    expected_cells,
    hard_filtered_loss,
    lif_response_with_pre_reset,
    load_protocol,
    make_delay_layer,
    optimize_bridge,
    render_hard_path,
    soft_centroid_loss,
    target_arrival_step,
    target_templates,
    trace_centroid,
    trace_w1_steps,
)


class DelayHardOutputSoftCreditLevel0DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_preregistered_grid_has_135_cells(self):
        self.assertEqual(expected_cells(self.protocol), 135)

    def test_target_templates_are_single_suprathreshold_spikes(self):
        for target_delay in self.protocol["timeline"]["target_nominal_delays_steps"]:
            target = target_templates(self.protocol, float(target_delay))
            arrival = target_arrival_step(self.protocol, float(target_delay))
            self.assertEqual(float(target["spike"].sum().item()), 1.0)
            self.assertEqual(int(target["spike"].argmax().item()), arrival)
            self.assertGreaterEqual(
                float(target["pre_reset_membrane"][arrival].item()),
                float(self.protocol["lif"]["threshold_au"]),
            )
            self.assertEqual(float(target["post_reset_membrane"][arrival].item()), 0.0)

    def test_pre_reset_state_matches_the_production_lif_update(self):
        current = torch.zeros(int(self.protocol["timeline"]["total_steps"]))
        current[4] = float(self.protocol["lif"]["fixed_synaptic_weight"])
        rendered = lif_response_with_pre_reset(current, self.protocol["lif"])
        decay = math.exp(
            -float(self.protocol["lif"]["dt_steps"])
            / float(self.protocol["lif"]["tau_m_steps"])
        )
        expected_pre_reset = (1.0 - decay) * float(
            self.protocol["lif"]["fixed_synaptic_weight"]
        )
        self.assertAlmostEqual(
            float(rendered["pre_reset_membrane"][4].item()),
            expected_pre_reset,
            places=6,
        )
        self.assertEqual(float(rendered["spike"][4].item()), 1.0)
        self.assertEqual(float(rendered["post_reset_membrane"][4].item()), 0.0)

    def test_current_centroid_credit_points_both_temporal_directions(self):
        for target_delay, initial_raw in ((5.0, -2.0), (1.0, 0.0)):
            layer = make_delay_layer(self.protocol, initial_raw=initial_raw)
            rendered = render_hard_path(
                layer,
                total_steps=int(self.protocol["timeline"]["total_steps"]),
                input_spike_step=int(self.protocol["timeline"]["input_spike_step"]),
                lif_config=self.protocol["lif"],
            )
            target = target_templates(self.protocol, target_delay)
            loss = soft_centroid_loss(rendered["current"], target["current"])
            loss.backward()
            gradient = float(layer.delay_raw.grad.item())
            initial_delay = float(layer.get_delays().detach().item())
            self.assertTrue(math.isfinite(gradient))
            self.assertGreater(abs(gradient), 1e-12)
            self.assertGreater(
                gradient * (initial_delay - target_delay),
                0.0,
                msg=(target_delay, initial_raw, initial_delay, gradient),
            )

    def test_pre_reset_credit_exposes_integer_boundary_zero_gradient(self):
        gradients = []
        for target_delay, initial_raw in ((5.0, -2.0), (1.0, 0.0)):
            layer = make_delay_layer(self.protocol, initial_raw=initial_raw)
            rendered = render_hard_path(
                layer,
                total_steps=int(self.protocol["timeline"]["total_steps"]),
                input_spike_step=int(self.protocol["timeline"]["input_spike_step"]),
                lif_config=self.protocol["lif"],
            )
            target = target_templates(self.protocol, target_delay)
            loss = soft_centroid_loss(
                rendered["pre_reset_membrane"], target["pre_reset_membrane"]
            )
            loss.backward()
            gradient = float(layer.delay_raw.grad.item())
            self.assertTrue(math.isfinite(gradient))
            gradients.append(gradient)
        self.assertGreater(abs(gradients[0]), 1e-12)
        self.assertEqual(gradients[1], 0.0)

    def test_weighted_component_gradients_equal_total_gradient(self):
        condition = condition_map(self.protocol)["hard_plus_current_lam_0p1"]
        layer = make_delay_layer(self.protocol, initial_raw=-2.0)
        rendered = render_hard_path(
            layer,
            total_steps=int(self.protocol["timeline"]["total_steps"]),
            input_spike_step=int(self.protocol["timeline"]["input_spike_step"]),
            lif_config=self.protocol["lif"],
        )
        target = target_templates(self.protocol, 5.0)
        hard = hard_filtered_loss(
            rendered["spike"],
            target["spike"],
            float(self.protocol["losses"]["hard_filtered"]["filter_tau_steps"]),
        )
        auxiliary = soft_centroid_loss(
            auxiliary_trace(rendered, str(condition["auxiliary_source"])),
            auxiliary_trace(target, str(condition["auxiliary_source"])),
        )
        hard_gradient = torch.autograd.grad(
            hard, layer.delay_raw, retain_graph=True
        )[0]
        auxiliary_gradient = torch.autograd.grad(
            auxiliary, layer.delay_raw, retain_graph=True
        )[0]
        total = (
            float(condition["hard_weight"]) * hard
            + float(condition["auxiliary_weight"]) * auxiliary
        )
        total.backward()
        expected = (
            float(condition["hard_weight"]) * hard_gradient
            + float(condition["auxiliary_weight"]) * auxiliary_gradient
        )
        self.assertTrue(
            torch.allclose(layer.delay_raw.grad, expected, atol=1e-7, rtol=1e-6)
        )

    def test_current_soft_bridge_recovers_a_representative_misaligned_cell(self):
        result = optimize_bridge(
            self.protocol,
            condition_name="current_soft_only",
            target_nominal_delay=5.0,
            initial_raw=-2.0,
        )
        self.assertTrue(result["recovered"])
        self.assertEqual(float(result["spike_counts"][-1]), 1.0)
        self.assertLessEqual(
            float(result["hard_w1_errors"][-1]),
            float(self.protocol["metrics"]["recovery_tolerance_steps"]),
        )
        self.assertLessEqual(
            float(
                trace_w1_steps(
                    torch.as_tensor(result["final_rendered"]["spike"]),
                    torch.as_tensor(result["target"]["spike"]),
                ).item()
            ),
            float(self.protocol["metrics"]["recovery_tolerance_steps"]),
        )
        self.assertAlmostEqual(
            float(result["hard_centroids"][-1]),
            float(trace_centroid(torch.as_tensor(result["target"]["spike"])).item()),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
