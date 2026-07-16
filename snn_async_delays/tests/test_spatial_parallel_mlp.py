import unittest

import torch

from snn.model import SNNSpatialParallelModel, SNNSimultaneousModel
from utils.resource_ledger import static_resource_ledger


class SpatialParallelMLPTests(unittest.TestCase):
    def test_independent_modules_share_decoder_and_report_effective_resources(self):
        model = SNNSpatialParallelModel(
            n_queries=2,
            hidden_per_query=3,
            win_len=4,
            read_len=4,
            d_max=2,
            train_mode="weights_only",
            fixed_delay_value=0.0,
            lif_threshold=0.2,
            readout_type="mlp",
        )
        logits, info = model(torch.zeros(5, 8, 8), record=True)
        self.assertEqual(tuple(logits.shape), (5, 2))
        self.assertEqual(tuple(info["hidden_spike_train"].shape), (5, 8, 6))
        self.assertEqual(len(model.syn_ih_modules), 2)
        self.assertEqual(sum(1 for _ in model.readout.modules()), 4)
        self.assertTrue(all(torch.equal(delay, torch.zeros_like(delay))
                            for delay in model.get_delays().values()))

        ledger = static_resource_ledger(model)
        self.assertEqual(ledger["topology_type"], "spatial_independent_shared_decoder")
        self.assertEqual(ledger["hidden_neurons_total"], 6)
        self.assertEqual(ledger["synapses_input_hidden"], 2 * 4 * 3)
        self.assertEqual(ledger["dense_synapse_macs_per_trial"], 8 * 2 * 4 * 3)

    def test_decoder_architecture_matches_windowed_temporal_model(self):
        spatial = SNNSpatialParallelModel(
            n_queries=2, hidden_per_query=8, win_len=10, read_len=8,
            d_max=4, readout_type="mlp",
        )
        temporal = SNNSimultaneousModel(
            n_queries=2, n_hidden=8, win_len=10, read_len=8, d_max=4,
            train_mode="weights_only", fixed_delay_value=0.0,
            n_input_channels=8, readout_type="mlp",
            observation_mode="windowed_shared", output_window_len=4,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in spatial.readout.parameters()),
            sum(parameter.numel() for parameter in temporal.readout.parameters()),
        )
        self.assertEqual(
            static_resource_ledger(spatial)["decoder_weight_macs_per_trial"],
            static_resource_ledger(temporal)["decoder_weight_macs_per_trial"],
        )


if __name__ == "__main__":
    unittest.main()
