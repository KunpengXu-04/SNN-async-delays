import unittest

import numpy as np
import torch

from data.boolean_dataset import (
    MarginallyBalancedFixedOperationQueryDataset, compute_label,
)
from scripts.run_mixedop_spatial_temporal_surface_preview import (
    _checkpoint_better, build_config, build_model, grid_specs, load_protocol,
)
from scripts.summarize_mixedop_spatial_temporal_surface_preview import (
    factor_decomposition,
)


class MixedOperationSurfacePreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_frozen_grid_counts_and_latencies(self):
        smoke = grid_specs(self.protocol, "smoke")
        formal = grid_specs(self.protocol, "formal")
        self.assertEqual(len(smoke), 6)
        self.assertEqual(len(formal), 72)
        self.assertEqual(
            sorted({build_config(self.protocol, spec)["T"] for spec in formal if spec["K"] == 5}),
            [30, 40, 50],
        )
        self.assertEqual(
            sorted({build_config(self.protocol, spec)["T"] for spec in formal if spec["K"] == 8}),
            [42, 58, 74],
        )

    def test_resource_fair_width_semantics(self):
        for spec in grid_specs(self.protocol, "formal"):
            cfg = build_config(self.protocol, spec)
            self.assertEqual(cfg["n_hidden_total"], spec["total_hidden"])
            if spec["condition"] == "spatial_independent_d0":
                self.assertEqual(cfg["hidden_per_query"] * spec["K"], spec["total_hidden"])
            else:
                self.assertEqual(cfg["d_max"], (spec["K"] - 1) * spec["output_window_len"])

    def test_oracle_tensor_is_exact_query_schedule(self):
        spec = next(
            item for item in grid_specs(self.protocol, "smoke")
            if item["K"] == 8 and item["condition"] == "shared_temporal_oracle"
        )
        cfg = build_config(self.protocol, spec)
        model = build_model(cfg)
        delays = model.syn_ih.get_delays().detach()
        for query in range(cfg["K"]):
            expected = query * cfg["output_window_len"]
            self.assertTrue(torch.equal(
                delays[4 * query:4 * (query + 1)],
                torch.full_like(delays[4 * query:4 * (query + 1)], expected),
            ))

    def test_dataset_has_exact_query_marginals_and_correct_labels(self):
        ops = ["AND", "OR", "XOR", "XNOR", "NAND"]
        first = MarginallyBalancedFixedOperationQueryDataset(2048, ops, seed=1307)
        second = MarginallyBalancedFixedOperationQueryDataset(2048, ops, seed=1307)
        independent = MarginallyBalancedFixedOperationQueryDataset(2048, ops, seed=2307)
        self.assertTrue(torch.equal(first.A, second.A))
        self.assertFalse(torch.equal(first.A, independent.A))
        for query, op in enumerate(ops):
            pairs = torch.stack((first.A[:, query], first.B[:, query]), dim=1)
            _, counts = torch.unique(pairs, dim=0, return_counts=True)
            self.assertEqual(sorted(counts.tolist()), [512, 512, 512, 512])
            for sample in (0, 17, 301):
                expected = compute_label(op, int(first.A[sample, query]), int(first.B[sample, query]))
                self.assertEqual(int(first.labels[sample, query]), expected)

    def test_checkpoint_selection_keeps_earlier_exact_tie(self):
        baseline = (0.75, 0.25)
        self.assertFalse(_checkpoint_better({
            "worst_query_balanced_accuracy": 0.75,
            "exact_trial_accuracy": 0.25,
        }, baseline))
        self.assertTrue(_checkpoint_better({
            "worst_query_balanced_accuracy": 0.75,
            "exact_trial_accuracy": 0.30,
        }, baseline))
        self.assertTrue(_checkpoint_better({
            "worst_query_balanced_accuracy": 0.76,
            "exact_trial_accuracy": 0.0,
        }, baseline))

    def test_factor_rule_is_exactly_preregistered(self):
        hidden_only = np.asarray([
            [.50, .50, .50], [.65, .65, .65], [.80, .80, .80], [.95, .95, .95]
        ])
        result = factor_decomposition(hidden_only)
        self.assertTrue(result["hidden_more_valuable_language_allowed"])
        interacting = np.asarray([
            [.50, .60, .70], [.60, .70, .80], [.70, .80, .90], [.80, .90, 1.0]
        ])
        result = factor_decomposition(interacting)
        self.assertFalse(result["hidden_more_valuable_language_allowed"])


if __name__ == "__main__":
    unittest.main()
