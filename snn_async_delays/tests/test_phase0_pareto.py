import unittest

import torch

from data.encoding import encode_simultaneous_trial
from snn.model import SNNSimultaneousModel
from train.trainer import (
    filtered_opponent_spike_train_loss,
    opponent_target_membrane_loss,
    opponent_target_spike_train,
)
from utils.pareto_cost import mixed_operation_spatial_hidden_total, spatial_temporal_ratios


class Phase0ParetoTests(unittest.TestCase):
    def test_binary_one_hot_microburst_has_constant_input_cost(self):
        A = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        B = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        encoded = encode_simultaneous_trial(
            A, B, win_len=10, read_len=10,
            encoding_mode="binary_one_hot", one_hot_phase=1.0,
            one_hot_n_spikes=2,
        )
        self.assertTrue(torch.equal(encoded.sum(dim=(1, 2)), torch.tensor([8.0, 8.0])))

    def test_opponent_targets_use_one_event_and_declared_windows(self):
        labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        target = opponent_target_spike_train(
            labels, total_steps=30, input_steps=10, output_window_len=10,
            timing_mode="sequential_offsets", target_offset_steps=1,
        )
        self.assertTrue(torch.equal(target.sum(dim=(1, 2)), torch.tensor([2.0, 2.0])))
        self.assertEqual(float(target[0, 11, 0]), 1.0)
        self.assertEqual(float(target[0, 21, 3]), 1.0)

    def test_filtered_target_loss_is_zero_only_for_matching_train(self):
        labels = torch.tensor([[0.0, 1.0]])
        target = opponent_target_spike_train(
            labels, total_steps=30, input_steps=10, output_window_len=10,
            timing_mode="sequential_centers",
        )
        exact = filtered_opponent_spike_train_loss(target, target, labels, tau_steps=5.0)
        shifted = torch.roll(target, shifts=3, dims=1)
        wrong = filtered_opponent_spike_train_loss(shifted, target, labels, tau_steps=5.0)
        self.assertEqual(float(exact), 0.0)
        self.assertGreater(float(wrong), 0.0)

    def test_target_time_membrane_loss_backpropagates(self):
        labels = torch.tensor([[0.0], [1.0]])
        voltage = torch.zeros(2, 20, 2, requires_grad=True)
        loss = opponent_target_membrane_loss(
            voltage, labels, input_steps=10, output_window_len=10,
            timing_mode="simultaneous_offset", target_offset_steps=1,
            threshold=.03, surrogate_beta=4.0,
        )
        loss.backward()
        self.assertIsNotNone(voltage.grad)
        self.assertGreater(float(voltage.grad[:, 11].abs().sum()), 0.0)
        self.assertEqual(float(voltage.grad[:, :11].abs().sum()), 0.0)

    def test_input_only_delay_mode_freezes_output_delays_at_zero(self):
        model = SNNSimultaneousModel(
            n_queries=2, n_hidden=5, win_len=10, read_len=10,
            d_max=20, train_mode="weights_and_delays", n_input_channels=4,
            use_output_spikes=True, n_output_neurons=4,
            opponent_output_mode="parallel_pairs", observation_mode="all_time",
            output_delay_mode="d0",
        )
        self.assertEqual(len(model.delay_params()), 1)
        self.assertTrue(torch.equal(model.get_delays()["ho"], torch.zeros(5, 4)))

    def test_declared_pareto_ratios_do_not_conflate_area_and_compute(self):
        ratios = spatial_temporal_ratios(
            K=2, baseline_hidden_per_query=10, shared_hidden=10,
            spatial_steps=20, temporal_steps=30,
        )
        self.assertAlmostEqual(ratios["hidden_area_ratio"], .5)
        self.assertAlmostEqual(ratios["latency_ratio"], 1.5)
        self.assertAlmostEqual(ratios["hidden_update_ratio"], .75)
        self.assertAlmostEqual(ratios["dense_synapse_compute_ratio"], 1.5)
        self.assertEqual(mixed_operation_spatial_hidden_total([4, 7, 3]), 14)


if __name__ == "__main__":
    unittest.main()
