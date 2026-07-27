import unittest

import torch

from data.encoding import encode_simultaneous_trial
from scripts.run_mixedop_rate_wad_surface_calibration import (
    load_protocol,
    specs,
    stage_protocol,
)
from scripts.run_mixedop_spatial_temporal_surface_preview import (
    build_config,
    build_model,
    encode_fn,
    loaders,
    validation_snapshot,
)


class MixedOperationRateWADCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_rate_code_preserves_four_value_channels_per_query(self):
        A = torch.tensor([[0, 1], [1, 0]])
        B = torch.tensor([[1, 0], [0, 1]])
        spikes = encode_simultaneous_trial(
            A, B, K=2, win_len=10, read_len=5, dt=1.0,
            r_on=1000.0, r_off=0.0,
            encoding_mode="binary_one_hot_rate", device="cpu",
        )
        self.assertEqual(tuple(spikes.shape), (2, 15, 8))
        expected = torch.tensor([
            [10, 0, 0, 10, 0, 10, 10, 0],
            [0, 10, 10, 0, 10, 0, 0, 10],
        ], dtype=spikes.dtype)
        self.assertTrue(torch.equal(spikes[:, :10].sum(dim=1), expected))
        self.assertEqual(float(spikes[:, 10:].sum()), 0.0)

    def test_rate_probability_validation(self):
        A = torch.zeros((1, 1), dtype=torch.long)
        with self.assertRaises(ValueError):
            encode_simultaneous_trial(
                A, A, K=1, win_len=2, read_len=1, dt=1.0,
                r_on=1001.0, r_off=0.0,
                encoding_mode="binary_one_hot_rate", device="cpu",
            )

    def test_preregistered_stage_sizes_and_points(self):
        self.assertEqual(len(specs(self.protocol, "smoke")), 2)
        self.assertEqual(len(specs(self.protocol, "pilot")), 10)
        self.assertEqual(len(specs(self.protocol, "full")), 30)
        pilot_points = {
            (item["total_hidden"], item["output_window_len"])
            for item in specs(self.protocol, "pilot")
        }
        self.assertEqual(pilot_points, {
            (20, 2), (20, 12), (240, 2), (240, 12), (120, 6),
        })

    def test_rate_config_and_oracle_schedule(self):
        oracle_spec = next(
            item for item in specs(self.protocol, "pilot")
            if item["point_label"] == "center"
            and item["condition"] == "shared_temporal_oracle"
        )
        cfg = build_config(self.protocol, oracle_spec)
        self.assertEqual(cfg["encoding_mode"], "binary_one_hot_rate")
        self.assertEqual(cfg["n_input"], 20)
        self.assertEqual(cfg["T"], 40)
        self.assertEqual(cfg["d_max"], 24)
        self.assertEqual(cfg["validation_encoding_seed"], 4307)
        model = build_model(cfg)
        delays = model.syn_ih.get_delays().detach()
        for query in range(5):
            block = delays[4 * query:4 * (query + 1)]
            self.assertTrue(torch.equal(
                block, torch.full_like(block, query * 6),
            ))

    def test_validation_rate_realization_is_fixed(self):
        wad_spec = next(
            item for item in specs(self.protocol, "smoke")
            if item["condition"] == "shared_temporal_wad"
        )
        cfg = build_config(self.protocol, wad_spec)
        cfg["n_val"] = 32
        cfg["batch_size"] = 16
        model = build_model(cfg)
        _, validation = loaders(cfg)
        encoder = encode_fn(cfg)
        _, first = validation_snapshot(
            model, validation, cfg, "cpu", encoder, collect=True,
        )
        _, second = validation_snapshot(
            model, validation, cfg, "cpu", encoder, collect=True,
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        for key in ("input_events_per_query", "logits", "predictions"):
            self.assertTrue(torch.equal(
                torch.from_numpy(first[key]), torch.from_numpy(second[key]),
            ))

    def test_full_stage_uses_separate_root(self):
        full = stage_protocol(self.protocol, "full")
        self.assertEqual(
            full["execution"]["formal_root"],
            self.protocol["execution"]["full_root"],
        )


if __name__ == "__main__":
    unittest.main()
