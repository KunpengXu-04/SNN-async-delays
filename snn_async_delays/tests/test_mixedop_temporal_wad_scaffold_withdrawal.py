import unittest

import torch

from scripts.run_mixedop_temporal_wad_scaffold_withdrawal import (
    apply_perturbation,
    build_config,
    compose_delay_gradient,
    lambda_at_update,
    load_protocol,
    physical_query_delays,
    run_dir,
    set_physical_query_delays,
    specs,
)
import scripts.run_mixedop_spatial_temporal_surface_preview as core


class MixedOpTemporalWADScaffoldWithdrawalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_frozen_stage_sizes_and_unique_paths(self):
        expected = {"smoke": 4, "w0": 3, "w1": 12, "w2": 18}
        for stage, count in expected.items():
            cells = specs(self.protocol, stage)
            paths = [run_dir(self.protocol, spec) for spec in cells]
            self.assertEqual(len(cells), count)
            self.assertEqual(len(paths), len(set(paths)))

    def test_anneal_schedule_is_exact_and_then_zero(self):
        self.assertEqual(lambda_at_update("annealed_joint", 1), 1.0)
        self.assertAlmostEqual(lambda_at_update("annealed_joint", 50), 50 / 99)
        self.assertEqual(lambda_at_update("annealed_joint", 100), 0.0)
        self.assertEqual(lambda_at_update("annealed_joint", 101), 0.0)
        self.assertEqual(lambda_at_update("abrupt_task_only", 1), 0.0)
        self.assertEqual(lambda_at_update("centroid_continue", 200), 1.0)

    def test_physical_delay_round_trip_and_registered_perturbations(self):
        spec = specs(self.protocol, "w0")[0]
        cfg = build_config(self.protocol, spec)
        model = core.build_model(cfg)
        nominal = [3.5, 7.5, 11.5, 15.5, 19.5]
        set_physical_query_delays(model, nominal)
        self.assertTrue(torch.allclose(
            torch.tensor(physical_query_delays(model)), torch.tensor(nominal), atol=1e-5
        ))
        before, early = apply_perturbation(
            self.protocol, model, "uniform_early_2_steps"
        )
        self.assertTrue(torch.allclose(torch.tensor(before), torch.tensor(nominal), atol=1e-5))
        self.assertTrue(torch.allclose(
            torch.tensor(early), torch.tensor([1.5, 5.5, 9.5, 13.5, 17.5]), atol=1e-5
        ))
        set_physical_query_delays(model, nominal)
        _, compressed = apply_perturbation(
            self.protocol, model, "compressed_spacing"
        )
        self.assertTrue(torch.allclose(
            torch.tensor(compressed), torch.tensor([3.5, 6.5, 9.5, 12.5, 15.5]), atol=1e-5
        ))

    def test_w2_is_unlocked_after_annealed_w1_passes(self):
        authorization = self.protocol["authorization"]
        self.assertTrue(authorization["implementation_smoke_launch"])
        self.assertTrue(authorization["w0_launch"])
        self.assertTrue(authorization["w1_launch"])
        self.assertTrue(authorization["w2_launch"])
        self.assertFalse(authorization["sealed_test"])

    def test_delay_credit_arms_are_separated_exactly(self):
        task = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        route = torch.tensor([[-1.0], [-2.0], [-3.0], [-4.0], [-5.0]])
        self.assertTrue(torch.equal(
            compose_delay_gradient("abrupt_task_only", task, route, 0.0), task
        ))
        self.assertTrue(torch.equal(
            compose_delay_gradient("centroid_continue", task, route, 1.0), route
        ))
        self.assertTrue(torch.equal(
            compose_delay_gradient("delay_frozen", task, route, 0.0), torch.zeros_like(task)
        ))
        self.assertTrue(torch.equal(
            compose_delay_gradient("annealed_joint", task, route, 0.25),
            task + 0.25 * route,
        ))


if __name__ == "__main__":
    unittest.main()
