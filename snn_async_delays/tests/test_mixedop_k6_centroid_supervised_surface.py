import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from scripts import run_mixedop_k6_centroid_supervised_surface as runner
from scripts import run_mixedop_spatial_temporal_surface_preview as core
from train.trainer import window_class_balanced_bce


class K6CentroidSupervisedSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = runner.load_protocol()

    def test_grid_counts_paths_and_operation_order(self):
        preflight = runner.specs(self.protocol, "preflight")
        surface = runner.specs(self.protocol, "surface")
        self.assertEqual(len(preflight), 4)
        self.assertEqual(len(surface), 112)
        self.assertEqual(len({str(runner.run_dir(self.protocol, s)) for s in preflight}), 4)
        self.assertEqual(len({str(runner.run_dir(self.protocol, s)) for s in surface}), 112)
        self.assertEqual(tuple(self.protocol["workloads"][6]), runner.OPS)

    def test_w_T_targets_support_and_common_initialization(self):
        expected_t = [34, 40, 46, 58, 70, 94, 130, 166]
        windows = self.protocol["surface"]["output_window_lengths"]
        self.assertEqual([10 + 6 * w for w in windows], expected_t)
        for w in windows:
            targets = runner.target_delays(w)
            self.assertEqual(targets[0], runner.initial_delay(w))
            self.assertTrue(all(0 < value < runner.delay_support_max(w) for value in targets))
            self.assertEqual(len(set(targets)), 6)
            # The common initialization itself contains no query schedule.
            self.assertEqual(len(set([runner.initial_delay(w)] * 6)), 1)

    def test_temporal_N1_is_legal_but_spatial_divisibility_remains(self):
        spec = runner.specs(self.protocol, "surface")[0]
        cfg = runner.build_config(self.protocol, spec)
        self.assertEqual(cfg["surface_total_hidden"], 1)
        model = core.build_model(cfg)
        self.assertEqual(model.n_hidden, 1)
        spatial = dict(spec, condition="spatial_independent_d0")
        with self.assertRaises(ValueError):
            core.build_config(runner._core_protocol(self.protocol), spatial)

    def test_delay_and_task_credit_are_decoupled(self):
        spec = runner.specs(self.protocol, "surface")[0]
        cfg = runner.build_config(self.protocol, spec)
        cfg.update({"surface_total_hidden": 2, "n_hidden": 2, "n_hidden_total": 2})
        model = core.build_model(cfg)
        train, _ = core.loaders(cfg)
        batch = next(iter(train))
        _, _, _, labels, spikes = core._batch_input(batch, cfg, "cpu", core.encode_fn(cfg))
        logits, _ = model(spikes)
        task = window_class_balanced_bce(logits, labels)
        route = core.routing_alignment_loss(spikes, model, cfg)
        delay_params = list(model.delay_params())
        weight_params = list(model.weight_params()) + list(model.readout_params())
        task_delay = torch.autograd.grad(task, delay_params, retain_graph=True, allow_unused=True)
        route_delay = torch.autograd.grad(route, delay_params, retain_graph=True)
        task_weight = torch.autograd.grad(task, weight_params, retain_graph=True, allow_unused=True)
        route_weight = torch.autograd.grad(route, weight_params, allow_unused=True)
        self.assertTrue(any(g is not None and torch.isfinite(g).all() for g in task_weight))
        self.assertTrue(all(g is None for g in route_weight))
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in route_delay))
        # The runner overwrites task-delay gradients with route-delay gradients.
        self.assertEqual(cfg["delay_credit_mode"], "routing_only_for_delays")
        self.assertTrue(any(g is not None for g in task_delay))

    def test_checkpoint_selection(self):
        rows = [
            {"update": 25, "delay_query_schedule_max_abs_error_steps": .7,
             "worst_query_balanced_accuracy": .99, "exact_trial_accuracy": .99},
            {"update": 50, "delay_query_schedule_max_abs_error_steps": .4,
             "worst_query_balanced_accuracy": .91, "exact_trial_accuracy": .80},
            {"update": 75, "delay_query_schedule_max_abs_error_steps": .3,
             "worst_query_balanced_accuracy": .91, "exact_trial_accuracy": .80},
        ]
        self.assertEqual(runner.select_mechanism_checkpoint_rows(rows)["update"], 50)
        self.assertIsNone(runner.select_mechanism_checkpoint_rows(rows[:1]))

    def test_N90_censoring(self):
        rows = [
            {"T": 34, "N_hidden": 1, "pass_90": True},
            {"T": 40, "N_hidden": 20, "pass_90": True},
            {"T": 46, "N_hidden": 60, "pass_90": False},
        ]
        boundary = runner.n90_boundary(rows, [1, 20, 60], [34, 40, 46])
        self.assertEqual([row["display"] for row in boundary], ["≤1", "20", ">60"])

    def test_surface_unlock_has_passing_preflight_and_plot_orientation_is_explicit(self):
        self.assertTrue(self.protocol["authorization"]["formal_surface_launch"])
        decision = runner._read_json(runner._preflight_decision_path(self.protocol))
        self.assertTrue(decision["passed"])
        self.assertTrue(decision["formal_surface_authorized_by_results"])
        self.assertFalse(decision["formal_surface_launch_still_requires_yaml_unlock"])
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn('xlabel="Simulation duration T (steps)"', source)
        self.assertIn('ylabel="Total shared hidden neurons N_hid (log scale)"', source)

    def test_one_update_pipeline_writes_runtime_npz_panel_and_ledger(self):
        with tempfile.TemporaryDirectory(dir=runner.BASE) as temporary:
            protocol = deepcopy(self.protocol)
            protocol["execution"]["preflight_root"] = temporary
            protocol["data"]["train_samples"] = 8
            protocol["data"]["validation_samples"] = 8
            protocol["optimization"]["validation_interval_updates"] = 1
            spec = runner._spec(1, 4, 99173, "preflight", 1)
            directory = runner.run_cell(protocol, spec, "cpu")
            for relative in protocol["required_cell_artifacts"]:
                self.assertTrue((directory / relative).exists(), relative)
            with __import__("numpy").load(directory / "validation_predictions.npz") as archive:
                self.assertEqual(archive["declared_delay_targets_steps"].shape, (6,))
                self.assertEqual(archive["arrival_centroid_steps"].shape, (6,))


if __name__ == "__main__":
    unittest.main()
