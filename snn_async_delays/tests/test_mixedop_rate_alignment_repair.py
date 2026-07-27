import unittest

import torch

from data.encoding import encode_simultaneous_trial
from scripts.run_mixedop_rate_alignment_repair import load_protocol, specs
from scripts.run_mixedop_spatial_temporal_surface_preview import (
    build_config,
    build_model,
    routing_alignment_loss,
)
from utils.resource_ledger import static_resource_ledger


class MixedOperationRateAlignmentRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_late_rate_packet_is_confined_to_steps_six_through_nine(self):
        A = torch.tensor([[0]])
        B = torch.tensor([[1]])
        spikes = encode_simultaneous_trial(
            A, B, K=1, win_len=10, read_len=4, dt=1.0,
            r_on=1000.0, r_off=0.0,
            encoding_mode="binary_one_hot_rate", rate_start_step=6,
            rate_steps=4, device="cpu",
        )
        self.assertEqual(float(spikes[:, :6].sum()), 0.0)
        self.assertEqual(float(spikes[:, 6:10].sum()), 8.0)
        self.assertEqual(float(spikes[:, 10:].sum()), 0.0)

    def test_packet_bounds_fail_loudly(self):
        A = torch.zeros((1, 1), dtype=torch.long)
        with self.assertRaises(ValueError):
            encode_simultaneous_trial(
                A, A, K=1, win_len=10, read_len=4, dt=1.0,
                r_on=400.0, r_off=10.0,
                encoding_mode="binary_one_hot_rate", rate_start_step=8,
                rate_steps=4, device="cpu",
            )

    def test_stage_a_grid_and_paths_are_unique(self):
        self.assertEqual(len(specs(self.protocol, "smoke")), 2)
        stage_a = specs(self.protocol, "stage_a")
        self.assertEqual(len(stage_a), 48)
        paths = set()
        from scripts.run_mixedop_rate_alignment_repair import _run_dir
        for spec in stage_a:
            paths.add(str(_run_dir(self.protocol, spec)))
        self.assertEqual(len(paths), 48)

    def test_oracle_packet_fits_every_declared_window(self):
        for spec in specs(self.protocol, "stage_a"):
            cfg = build_config(self.protocol, spec)
            self.assertEqual(cfg["oracle_base_delay_steps"], 3)
            self.assertEqual(cfg["d_max"], 3 + 4 * cfg["output_window_len"])
            for query, delay in enumerate(cfg["oracle_delay_schedule"]):
                first_arrival = 6 + delay + 1
                last_arrival = 9 + delay + 1
                window_start = 10 + query * cfg["output_window_len"]
                window_stop = window_start + cfg["output_window_len"]
                self.assertGreaterEqual(first_arrival, window_start)
                self.assertLess(last_arrival, window_stop)

    def test_per_query_wad_has_five_delay_parameters(self):
        spec = {
            "K": 5, "condition": "shared_temporal_wad",
            "training_arm": "task_only", "total_hidden": 120,
            "output_window_len": 6, "seed": 3347, "updates": 2,
            "stage": "stage_b_per_query_wad", "r_on_hz": 490.0,
        }
        cfg = build_config(self.protocol, spec)
        model = build_model(cfg)
        self.assertEqual(cfg["delay_tying"], "pre_group")
        self.assertEqual(tuple(model.syn_ih.delay_raw.shape), (5, 1))
        self.assertEqual(static_resource_ledger(model)["trainable_delay_parameters"], 5)
        delays = model.syn_ih.get_delays()
        for query in range(5):
            block = delays[4 * query:4 * (query + 1)]
            self.assertEqual(int(torch.unique(block).numel()), 1)

    def test_routing_loss_prefers_aligned_query_delays(self):
        spec = {
            "K": 5, "condition": "shared_temporal_wad",
            "training_arm": "routing_assisted", "total_hidden": 20,
            "output_window_len": 6, "seed": 3347, "updates": 2,
            "stage": "stage_b_per_query_wad", "r_on_hz": 1000.0,
            "routing_loss_weight": .1, "routing_loss_temperature": .75,
        }
        cfg = build_config(self.protocol, spec)
        model = build_model(cfg)
        A = torch.zeros((8, 5), dtype=torch.long)
        B = torch.ones((8, 5), dtype=torch.long)
        spikes = encode_simultaneous_trial(
            A, B, K=5, win_len=10, read_len=30, dt=1.0,
            r_on=1000.0, r_off=0.0,
            encoding_mode="binary_one_hot_rate", rate_start_step=6,
            rate_steps=4, device="cpu",
        )
        with torch.no_grad():
            targets = torch.tensor([3, 9, 15, 21, 27.0]).view(5, 1)
            model.syn_ih.delay_raw.copy_(torch.logit(targets / cfg["d_max"]))
        aligned = float(routing_alignment_loss(spikes, model, cfg))
        with torch.no_grad():
            collapsed = torch.full((5, 1), 3.0)
            model.syn_ih.delay_raw.copy_(torch.logit(collapsed / cfg["d_max"]))
        unaligned = float(routing_alignment_loss(spikes, model, cfg))
        self.assertLess(aligned, unaligned)


if __name__ == "__main__":
    unittest.main()
