import copy
import unittest

import torch

from scripts.run_xor_task_bridge_level1a import (
    arrival_centroid_loss,
    build_model,
    build_optimizer,
    encode_exact_truth,
    exact_interface_metrics,
    exact_truth_batch,
    expected_stage_i_cells,
    expected_stage_ii_cells,
    global_voltage_envelope_loss,
    load_protocol,
    per_cell_interface_pass,
    shared_arrival_trace,
    stage_i_specs,
    stage_ii_specs,
    target_output_step,
    trace_centroid_batch,
)
from train.trainer import opponent_target_spike_train


class XORTaskBridgeLevel1ATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_preregistered_stage_counts(self):
        self.assertEqual(expected_stage_i_cells(self.protocol), 90)
        self.assertEqual(expected_stage_ii_cells(self.protocol), 85)
        self.assertEqual(len(stage_i_specs(self.protocol)), 90)

    def test_stage_ii_is_locked_without_passing_stage_i(self):
        with self.assertRaises(RuntimeError):
            stage_ii_specs(
                self.protocol,
                {"stage_ii_authorized": False, "selected_interface_candidate": None},
            )

    def test_stage_ii_materializes_only_locked_selected_interface_values(self):
        decision = {
            "stage_i_pass": True,
            "stage_ii_authorized": True,
            "selected_interface_candidate": {
                "voltage_envelope_weight": 0.1,
                "weight_learning_rate": 0.003,
            },
        }
        specs = stage_ii_specs(self.protocol, decision)
        self.assertEqual(len(specs), 85)
        self.assertTrue(all(spec["voltage_envelope_weight"] == 0.1 for spec in specs))
        self.assertTrue(all(spec["weight_learning_rate"] == 0.003 for spec in specs))
        self.assertEqual(sum(bool(spec["learned_delay"]) for spec in specs), 80)

    def test_binary_one_hot_encoding_has_two_events_at_the_locked_step(self):
        A, B, _, _ = exact_truth_batch("cpu")
        encoded = encode_exact_truth(self.protocol, A, B, device="cpu")
        self.assertEqual(tuple(encoded.shape), (4, 20, 4))
        self.assertTrue(torch.equal(encoded.sum(dim=(1, 2)), torch.full((4,), 2.0)))
        self.assertTrue(torch.equal(encoded[:, :9].sum(dim=(1, 2)), torch.zeros(4)))
        self.assertTrue(torch.equal(encoded[:, 10:].sum(dim=(1, 2)), torch.zeros(4)))
        for row in range(4):
            a_channel = int(A[row, 0].item())
            b_channel = 2 + int(B[row, 0].item())
            self.assertEqual(float(encoded[row, 9, a_channel].item()), 1.0)
            self.assertEqual(float(encoded[row, 9, b_channel].item()), 1.0)

    def test_locked_causal_target_steps_include_both_buffer_offsets(self):
        self.assertEqual(target_output_step(self.protocol, 0.0), 11)
        self.assertEqual(target_output_step(self.protocol, 4.0), 15)

    def test_learned_model_has_one_shared_input_delay_and_d0_output_delays(self):
        spec = next(
            spec for spec in stage_ii_specs(
                self.protocol,
                {
                    "stage_i_pass": True,
                    "stage_ii_authorized": True,
                    "selected_interface_candidate": {
                        "voltage_envelope_weight": 0.1,
                        "weight_learning_rate": 0.003,
                    },
                },
            ) if spec["learned_delay"]
        )
        model = build_model(self.protocol, spec)
        self.assertEqual(model.syn_ih.delay_raw.numel(), 1)
        self.assertEqual(tuple(model.syn_ih.get_delays().shape), (4, 16))
        self.assertTrue(torch.equal(model.syn_ho.get_delays(), torch.zeros(16, 2)))
        self.assertEqual(sum(parameter.numel() for parameter in model.delay_params()), 1)

    def test_optimizer_keeps_weight_and_delay_learning_rates_separate(self):
        decision = {
            "stage_i_pass": True,
            "stage_ii_authorized": True,
            "selected_interface_candidate": {
                "voltage_envelope_weight": 0.1,
                "weight_learning_rate": 0.003,
            },
        }
        spec = next(spec for spec in stage_ii_specs(self.protocol, decision) if spec["learned_delay"])
        model = build_model(self.protocol, spec)
        optimizer = build_optimizer(model, spec)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.003, 0.01])

    def test_arrival_trace_matches_the_production_d_plus_one_convention(self):
        A, B, _, _ = exact_truth_batch("cpu")
        encoded = encode_exact_truth(self.protocol, A, B, device="cpu")
        d0 = shared_arrival_trace(encoded, torch.tensor(0.0), d_max=8)
        d4 = shared_arrival_trace(encoded, torch.tensor(4.0), d_max=8)
        self.assertTrue(torch.allclose(trace_centroid_batch(d0), torch.full((4,), 10.0)))
        self.assertTrue(torch.allclose(trace_centroid_batch(d4), torch.full((4,), 14.0)))
        self.assertTrue(torch.allclose(d0.sum(dim=1), torch.full((4,), 2.0)))
        self.assertTrue(torch.allclose(d4.sum(dim=1), torch.full((4,), 2.0)))

    def test_arrival_auxiliary_points_both_delay_directions(self):
        A, B, _, _ = exact_truth_batch("cpu")
        encoded = encode_exact_truth(self.protocol, A, B, device="cpu")
        for raw in (-2.0, 2.0):
            raw_parameter = torch.tensor(raw, requires_grad=True)
            delay = 8.0 * torch.sigmoid(raw_parameter)
            loss, _, _, _, _ = arrival_centroid_loss(
                encoded, delay, target_delay_steps=4.0, d_max=8
            )
            loss.backward()
            gradient = float(raw_parameter.grad.item())
            self.assertGreater(abs(gradient), 1e-12)
            self.assertGreater(gradient * (float(delay.detach().item()) - 4.0), 0.0)

    def test_global_voltage_envelope_penalizes_an_early_wrong_spike_state(self):
        labels = torch.tensor([[0.0], [1.0]])
        target = opponent_target_spike_train(
            labels,
            total_steps=20,
            input_steps=10,
            output_window_len=10,
            timing_mode="simultaneous_offset",
            target_offset_steps=5,
        )
        threshold = 0.03
        beta = 4.0
        good = torch.full_like(target, threshold - 1.0)
        good[target > 0.5] = threshold + 1.0
        bad = good.clone()
        bad[0, 10, 1] = threshold + 1.0
        good_loss = global_voltage_envelope_loss(
            good, target, threshold=threshold, surrogate_beta=beta
        )
        bad_loss = global_voltage_envelope_loss(
            bad, target, threshold=threshold, surrogate_beta=beta
        )
        self.assertLess(float(good_loss.item()), float(bad_loss.item()))

    def test_exact_interface_gate_requires_the_full_target_spike_train(self):
        A, B, _, labels = exact_truth_batch("cpu")
        target = opponent_target_spike_train(
            labels,
            total_steps=20,
            input_steps=10,
            output_window_len=10,
            timing_mode="simultaneous_offset",
            target_offset_steps=5,
        )
        hidden = torch.ones(4)
        metrics = exact_interface_metrics(target, target, labels, hidden)
        self.assertTrue(per_cell_interface_pass(metrics))
        collided = target.clone()
        collided[0, 10, 1] = 1.0
        failed = exact_interface_metrics(collided, target, labels, hidden)
        self.assertFalse(per_cell_interface_pass(failed))
        self.assertGreater(failed["collision_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
