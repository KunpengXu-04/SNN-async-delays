import unittest

import torch

from scripts import run_mixedop_event8_aligned_surface_v2 as surface
from scripts.run_mixedop_temporal_wad_scaffold_withdrawal import (
    lambda_at_update,
    physical_query_delays,
)
from utils.resource_ledger import static_resource_ledger


class Event8AlignedSurfaceV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = surface.load_protocol()

    def _spec(self, stage, condition, label=None):
        return next(
            item for item in surface.specs(self.protocol, stage)
            if item["surface_condition"] == condition
            and (label is None or item["point_label"] == label)
        )

    def test_preregistered_cell_counts_and_unique_paths(self):
        for stage, count in {"smoke": 5, "landmark": 25, "full": 120}.items():
            cells = surface.specs(self.protocol, stage)
            self.assertEqual(len(cells), count)
            paths = [surface.run_dir(self.protocol, item) for item in cells]
            self.assertEqual(len(paths), len(set(paths)))

    def test_full_surface_reuses_landmarks_and_adds_only_ninety_five_cells(self):
        cells = surface.specs(self.protocol, "full")
        self.assertEqual(sum(item["stage"] == "landmark" for item in cells), 25)
        self.assertEqual(sum(item["stage"] == "full" for item in cells), 95)
        landmark_paths = {
            surface.run_dir(self.protocol, item)
            for item in surface.specs(self.protocol, "landmark")
        }
        reused_paths = {
            surface.run_dir(self.protocol, item)
            for item in cells if item["stage"] == "landmark"
        }
        self.assertEqual(reused_paths, landmark_paths)

    def test_landmarks_are_four_corners_and_locked_center(self):
        points = {
            (item["total_hidden"], item["output_window_len"])
            for item in surface.specs(self.protocol, "landmark")
        }
        self.assertEqual(
            points, {(20, 4), (20, 12), (240, 4), (240, 12), (80, 8)}
        )

    def test_event8_packet_fits_every_window_and_latency_formula(self):
        self.assertEqual(self.protocol["encoding"]["packet_support_steps"], [6, 9])
        for item in surface.specs(self.protocol, "full"):
            cfg = surface.build_config(self.protocol, item)
            self.assertGreaterEqual(cfg["output_window_len"], 4)
            self.assertEqual(cfg["T"], 10 + 5 * cfg["output_window_len"])

    def test_shared_d0_is_fixed_zero_and_has_no_delay_parameters(self):
        cfg = surface.build_config(self.protocol, self._spec("smoke", "shared_d0"))
        model = surface.core.build_model(cfg)
        self.assertEqual(cfg["surface_condition"], "shared_d0")
        self.assertIs(cfg["event8_aligned"], True)
        self.assertEqual(cfg["d_max"], 0)
        self.assertEqual(cfg["oracle_delay_schedule"], [0.0] * 5)
        self.assertEqual(sum(p.numel() for p in model.delay_params()), 0)
        self.assertTrue(torch.equal(model.get_delays()["ih"], torch.zeros(20, 80)))
        self.assertEqual(static_resource_ledger(model)["trainable_delay_parameters"], 0)

    def test_oracle_schedule_is_event8_aligned_and_exactly_installed(self):
        for label, expected, d_max in (
            ("lowN_lowT", [3, 7, 11, 15, 19], 23),
            ("lowN_highT", [3, 15, 27, 39, 51], 55),
        ):
            cfg = surface.build_config(
                self.protocol,
                self._spec("landmark", "shared_temporal_oracle", label),
            )
            model = surface.core.build_model(cfg)
            self.assertEqual(cfg["oracle_delay_schedule"], expected)
            self.assertEqual(cfg["d_max"], d_max)
            delays = model.get_delays()["ih"]
            for query, delay in enumerate(expected):
                block = delays[query * 4 : (query + 1) * 4]
                self.assertTrue(torch.equal(block, torch.full_like(block, delay)))

    def test_task_only_wad_has_five_query_tied_parameters_at_delay_three(self):
        cfg = surface.build_config(
            self.protocol,
            self._spec("landmark", "shared_temporal_task_only", "geometric_center"),
        )
        model = surface.core.build_model(cfg)
        self.assertEqual(cfg["surface_condition"], "shared_temporal_task_only")
        self.assertIs(cfg["event8_aligned"], True)
        self.assertEqual(cfg["delay_tying"], "pre_group")
        self.assertEqual(sum(p.numel() for p in model.delay_params()), 5)
        self.assertTrue(
            torch.allclose(
                torch.tensor(physical_query_delays(model)),
                torch.full((5,), 3.0),
                atol=1e-5,
            )
        )

    def test_curriculum_budget_and_withdrawal_lambda_are_locked(self):
        item = self._spec("landmark", "shared_temporal_annealed", "geometric_center")
        self.assertEqual(item["updates"], 600)
        self.assertEqual(surface._curriculum_updates(self.protocol, "landmark"), (400, 200))
        self.assertEqual(lambda_at_update("annealed_joint", 1), 1.0)
        self.assertEqual(lambda_at_update("annealed_joint", 100), 0.0)
        self.assertEqual(lambda_at_update("annealed_joint", 200), 0.0)

    def test_passing_smoke_unlocks_only_landmark(self):
        authorization = self.protocol["authorization"]
        self.assertIs(authorization["smoke_launch"], True)
        self.assertIs(authorization["landmark_launch"], True)
        self.assertIs(authorization["full_sweep_launch"], False)
        surface.preconditions(self.protocol, "landmark")
        with self.assertRaises(SystemExit):
            surface.preconditions(self.protocol, "full")


if __name__ == "__main__":
    unittest.main()
