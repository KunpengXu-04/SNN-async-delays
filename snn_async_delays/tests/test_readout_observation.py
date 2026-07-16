import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from snn.model import SNNSimultaneousModel
from data.boolean_dataset import ExhaustiveFixedOperationQueryDataset


def make_model(mode="late_window", use_output_spikes=False):
    return SNNSimultaneousModel(
        n_queries=2,
        n_hidden=6,
        win_len=8,
        read_len=4,
        d_max=8,
        train_mode="weights_only",
        fixed_delay_value=0.0,
        lif_threshold=0.5,
        n_input_channels=2,
        readout_type="linear",
        use_output_spikes=use_output_spikes,
        n_output_neurons=2 if use_output_spikes else None,
        observation_mode=mode,
    )


class ReadoutObservationTests(unittest.TestCase):
    def test_exhaustive_fixed_operation_dataset_covers_all_six_bit_patterns(self):
        dataset = ExhaustiveFixedOperationQueryDataset(["XOR", "NAND", "NOR"])
        self.assertEqual(len(dataset), 64)
        patterns = {(tuple(a.tolist()), tuple(b.tolist())) for a, b, _, _ in dataset}
        self.assertEqual(len(patterns), 64)

    def test_windowed_shared_uses_one_decoder_across_query_windows(self):
        model = SNNSimultaneousModel(
            n_queries=3, n_hidden=6, win_len=8, read_len=12,
            n_input_channels=2, train_mode="weights_only", fixed_delay_value=0.0,
            observation_mode="windowed_shared", output_window_len=4,
        )
        x = torch.randint(0, 2, (2, 20, 2)).float()
        logits, info = model(x, record=True)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(info["readout_features"].shape), (2, 3, 6))
        self.assertEqual(model.readout.out_features, 1)
        self.assertEqual(info["decoder_parameters"], 7)
        logits.sum().backward()
        self.assertIsNotNone(model.readout.weight.grad)

    def test_opponent_output_interfaces_produce_signed_logits(self):
        parallel = SNNSimultaneousModel(
            n_queries=3, n_hidden=6, win_len=8, read_len=4,
            n_input_channels=2, train_mode="weights_only", fixed_delay_value=0.0,
            use_output_spikes=True, n_output_neurons=6,
            opponent_output_mode="parallel_pairs", observation_mode="late_window",
        )
        x_parallel = torch.randint(0, 2, (2, 12, 2)).float()
        logits_parallel, _ = parallel(x_parallel, record=True)
        self.assertEqual(tuple(logits_parallel.shape), (2, 3))
        logits_parallel.sum().backward()
        self.assertIsNotNone(parallel.syn_ho.weight.grad)

        routed = SNNSimultaneousModel(
            n_queries=3, n_hidden=6, win_len=8, read_len=12,
            n_input_channels=2, train_mode="weights_only", fixed_delay_value=0.0,
            use_output_spikes=True, n_output_neurons=2,
            opponent_output_mode="shared_windowed", observation_mode="windowed_shared",
            output_window_len=4,
        )
        x_routed = torch.randint(0, 2, (2, 20, 2)).float()
        logits_routed, info = routed(x_routed, record=True)
        self.assertEqual(tuple(logits_routed.shape), (2, 3))
        self.assertEqual(tuple(info["output_window_counts"].shape), (2, 3, 2))
        self.assertEqual(tuple(info["output_membrane_logits"].shape), (2, 3))
        self.assertEqual(tuple(info["output_membrane_class_logits"].shape), (2, 3, 2))
        self.assertEqual(tuple(info["hidden_window_counts"].shape), (2, 3, 6))
        self.assertEqual(tuple(info["output_synaptic_current_train"].shape), (2, 20, 2))
        # Regression: peak-voltage warm-up must backpropagate through all
        # shared output windows without in-place autograd version errors.
        info["output_membrane_class_logits"].sum().backward()
        self.assertIsNotNone(routed.syn_ho.weight.grad)

    def test_default_is_exactly_explicit_late_window(self):
        torch.manual_seed(1)
        default = SNNSimultaneousModel(
            n_queries=2, n_hidden=6, win_len=8, read_len=4,
            n_input_channels=2, train_mode="weights_only", fixed_delay_value=0.0,
        )
        explicit = SNNSimultaneousModel(
            n_queries=2, n_hidden=6, win_len=8, read_len=4,
            n_input_channels=2, train_mode="weights_only", fixed_delay_value=0.0,
            observation_mode="late_window",
        )
        explicit.load_state_dict(default.state_dict())
        x = torch.randint(0, 2, (3, 12, 2)).float()
        y_default, i_default = default(x, record=True)
        y_explicit, i_explicit = explicit(x, record=True)
        self.assertTrue(torch.equal(y_default, y_explicit))
        self.assertTrue(torch.equal(i_default["readout_features"], i_explicit["readout_features"]))

    def test_all_time_is_uncensored_and_parameter_matched(self):
        late = make_model("late_window")
        all_time = make_model("all_time")
        all_time.load_state_dict(late.state_dict())
        x = torch.randint(0, 2, (4, 12, 2)).float()
        _, late_info = late(x, record=True)
        _, all_info = all_time(x, record=True)
        self.assertEqual(late_info["readout_feature_dim"], all_info["readout_feature_dim"])
        self.assertEqual(late_info["decoder_parameters"], all_info["decoder_parameters"])
        self.assertTrue(torch.all(all_info["readout_features"] >= late_info["readout_features"]))
        self.assertEqual(all_info["observation_steps"], 12)
        self.assertEqual(late_info["observation_steps"], 4)

    def test_time_binned_features_partition_all_time_counts_and_backpropagate(self):
        all_time = make_model("all_time")
        binned = make_model("time_binned")
        binned.syn_ih.load_state_dict(all_time.syn_ih.state_dict())
        x = torch.randint(0, 2, (4, 12, 2)).float()
        _, all_info = all_time(x, record=True)
        logits, binned_info = binned(x, record=True)
        features = binned_info["readout_features"].reshape(4, 3, 6)
        self.assertTrue(torch.equal(features.sum(dim=1), all_info["readout_features"]))
        self.assertEqual(binned_info["readout_feature_dim"], 18)
        self.assertGreater(binned_info["decoder_parameters"], all_time.observation_metadata()["decoder_parameters"])
        self.assertEqual(binned.readout.in_features, 18)
        logits.sum().backward()
        self.assertIsNotNone(binned.readout.weight.grad)

    def test_undefined_observation_combinations_fail_loudly(self):
        with self.assertRaises(ValueError):
            make_model("unknown")
        with self.assertRaises(ValueError):
            make_model("time_binned", use_output_spikes=True)
        with self.assertRaises(ValueError):
            SNNSimultaneousModel(
                n_queries=3, n_hidden=6, win_len=10, read_len=4,
                n_input_channels=2, observation_mode="time_binned",
            )


if __name__ == "__main__":
    unittest.main()
