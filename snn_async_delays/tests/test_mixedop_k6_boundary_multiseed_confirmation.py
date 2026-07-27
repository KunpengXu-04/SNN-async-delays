import unittest

from scripts import run_mixedop_k6_boundary_multiseed_confirmation as runner


class K6BoundaryMultiseedConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.confirmation = runner.load_confirmation()

    def test_frozen_grid_has_200_unique_paths_and_fresh_seeds(self):
        cells = runner.specs(self.confirmation)
        self.assertEqual(len(cells), 200)
        self.assertEqual(len({str(runner.run_dir(self.confirmation, cell)) for cell in cells}), 200)
        seeds = set(self.confirmation["frozen_grid"]["fresh_seeds"])
        self.assertEqual(len(seeds), 5)
        self.assertFalse(seeds & {3899, 3907})

    def test_confirmation_exactly_inherits_parent_recipe(self):
        protocol = runner.materialize_protocol(self.confirmation)
        inherited = self.confirmation["inherited_recipe"]
        self.assertEqual(protocol["optimization"]["optimizer_updates"], inherited["optimizer_updates"])
        self.assertEqual(protocol["optimization"]["validation_interval_updates"], 25)
        self.assertEqual(protocol["optimization"]["weight_learning_rate"], .01)
        self.assertEqual(protocol["model"]["hidden_lif"]["threshold_au"], .2)
        self.assertFalse(protocol["authorization"]["sealed_test"])

    def test_grid_mapping_and_operation_order_are_unchanged(self):
        grid = self.confirmation["frozen_grid"]
        self.assertEqual([10 + 6 * w for w in grid["output_window_lengths"]], grid["total_latency_steps"])
        self.assertEqual(tuple(grid["operations"]), runner.parent.OPS)
        self.assertEqual(grid["total_hidden_neurons"], [1, 2, 3, 4, 5])

    def test_observed_and_consensus_N90(self):
        rows = []
        seeds = [1, 2, 3, 4, 5]
        for seed in seeds:
            for width in [1, 2, 3, 4, 5]:
                rows.append({"seed": seed, "T": 34, "N_hidden": width, "pass_90": width >= (3 if seed < 5 else 4)})
        self.assertEqual(runner.observed_n90(rows, 1, 34), 3)
        self.assertEqual(runner.observed_n90(rows, 5, 34), 4)
        self.assertEqual(runner.consensus_n90(rows, seeds, 34, 4), 3)
        self.assertEqual(runner.consensus_n90(rows, seeds, 34, 5), 4)

    def test_reversal_definition_is_exact(self):
        rows = [
            {"seed": 1, "T": 34, "N_hidden": 3, "pass_90": True},
            {"seed": 1, "T": 34, "N_hidden": 4, "pass_90": False},
            {"seed": 2, "T": 34, "N_hidden": 3, "pass_90": True},
            {"seed": 2, "T": 34, "N_hidden": 4, "pass_90": True},
        ]
        self.assertTrue(runner.reversal_present(rows, 1, 34, 3, 4))
        self.assertFalse(runner.reversal_present(rows, 2, 34, 3, 4))

    def test_nonmonotonicity_does_not_impute_censored_values(self):
        self.assertEqual(runner.observed_increases([3, 3, 2, 4, 3]), (1, 0))
        self.assertEqual(runner.observed_increases([3, None, 4]), (0, 2))

    def test_registered_decisions_and_authorization(self):
        self.assertEqual(self.confirmation["reversal_reproduced_if_fresh_seeds_minimum"], 2)
        self.assertEqual(
            self.confirmation["nonmonotonicity_decision"]["confirmed_if_fresh_seeds_nonmonotonic_minimum"], 3
        )
        self.assertTrue(self.confirmation["authorization"]["confirmation_launch"])
        self.assertFalse(self.confirmation["authorization"]["additional_seeds"])
        self.assertFalse(self.confirmation["authorization"]["sealed_test"])


if __name__ == "__main__":
    unittest.main()
