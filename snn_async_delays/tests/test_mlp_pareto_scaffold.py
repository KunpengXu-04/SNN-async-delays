import unittest

import torch
import yaml

from scripts.run_spatial_vs_temporal_pareto_mlp_scaffold import (
    _grid,
    _loaders,
    build_config,
    build_model,
    load_protocol,
)


class MLPParetoScaffoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_locked_grid_has_160_cells(self):
        conditions, latencies, widths, seeds = _grid(self.protocol, smoke=False)
        self.assertEqual(
            len(conditions) * len(latencies) * len(widths) * len(seeds), 160
        )
        self.assertEqual(self.protocol["training"]["epochs"], 200)
        self.assertEqual(
            self.protocol["training"]["exact_truth_table_repeats_per_epoch"], 1
        )

    def test_spatial_width_means_per_query_but_shared_width_means_total(self):
        spatial = build_config(
            self.protocol, "spatial_independent_d0", 26, 8, 307, False
        )
        shared = build_config(
            self.protocol, "shared_spatial_d0", 26, 8, 307, False
        )
        self.assertEqual(spatial["n_hidden_total"], 16)
        self.assertEqual(shared["n_hidden_total"], 8)
        self.assertEqual(spatial["d_max"], 0)
        self.assertEqual(shared["d_max"], 0)

    def test_temporal_oracle_uses_one_window_query_schedule(self):
        cfg = build_config(
            self.protocol, "shared_temporal_oracle", 26, 8, 307, False
        )
        model = build_model(cfg)
        delays = model.get_delays()["ih"]
        self.assertTrue(torch.equal(delays[:4], torch.zeros_like(delays[:4])))
        self.assertTrue(torch.equal(
            delays[4:], torch.full_like(delays[4:], cfg["output_window_len"])
        ))
        self.assertEqual(cfg["observation_mode"], "windowed_shared")

    def test_wad_is_trainable_and_exact_batches_are_balanced(self):
        cfg = build_config(
            self.protocol, "shared_temporal_wad", 26, 8, 307, False
        )
        model = build_model(cfg)
        self.assertGreater(sum(parameter.numel() for parameter in model.delay_params()), 0)
        train, validation = _loaders(cfg)
        self.assertEqual(len(validation.dataset), 16)
        for _, _, _, labels in train:
            self.assertEqual(tuple(labels.shape), (16, 2))
            self.assertTrue(torch.equal(labels.sum(dim=0), torch.tensor([8.0, 8.0])))


if __name__ == "__main__":
    unittest.main()
