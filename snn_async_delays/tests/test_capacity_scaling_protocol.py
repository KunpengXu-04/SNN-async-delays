import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.run_spatial_temporal_capacity_slayer import (
    cyclic_rotations, load_protocol, specs,
)
from scripts.summarize_capacity_scaling import (
    aggregate_cells, capacity_rows, worst_query_pass,
)
from utils.capacity_scaling import fit_power_law, raw_capacity


class CapacityScalingProtocolTests(unittest.TestCase):
    def test_registered_budgets_divide_every_k(self):
        protocol = load_protocol()
        for budget in protocol["temporal_scaling"]["output_budgets_B"]:
            for K in protocol["workload"]["K_values"]:
                self.assertEqual(budget % K, 0)

    def test_spatial_specs_hold_total_latency_fixed(self):
        rows = specs(load_protocol(), "spatial")
        self.assertGreater(len(rows), 0)
        self.assertEqual({row["T"] for row in rows}, {58})
        self.assertEqual({row["output_budget_B"] for row in rows}, {48})
        for row in rows:
            self.assertEqual(row["window_width"], 48 // row["K"])
            self.assertEqual(row["input_events_expected"], 8 * row["K"])
            if row["condition"] == "independent_spatial_d0":
                self.assertEqual(row["N_hidden_total"] % row["K"], 0)

    def test_temporal_specs_use_only_the_route_capable_fixed_schedule(self):
        rows = specs(load_protocol(), "temporal")
        self.assertEqual({row["condition"] for row in rows}, {"shared_temporal_fixed"})
        self.assertEqual(len(rows), 72)

    def test_counterbalance_is_a_complete_cyclic_latin_square(self):
        values = ["AND", "OR", "XOR", "XNOR", "NAND", "NOR"]
        rotations = cyclic_rotations(values)
        self.assertEqual(len(rotations), 6)
        for position in range(6):
            self.assertEqual({row[position] for row in rotations}, set(values))

    def test_raw_capacity_does_not_smooth_nonmonotone_rows(self):
        rows = [
            {"N": 2, "K": 1, "pass_reliability": True},
            {"N": 2, "K": 2, "pass_reliability": False},
            {"N": 2, "K": 3, "pass_reliability": True},
        ]
        self.assertEqual(raw_capacity(rows, "N", 2), 3)
        self.assertIsNone(raw_capacity(rows, "N", 3))

    def test_power_fit_refuses_too_few_uncensored_points(self):
        result = fit_power_law([1, 2, 4, 8], [1, None, 3, None])
        self.assertFalse(result["fit_performed"])
        self.assertEqual(result["uncensored_points"], 2)

    def test_slayer_fixed_interface_is_separate_from_task_only(self):
        protocol = load_protocol()
        rows = specs(protocol, "s1_confirmation")
        fixed = [row for row in rows if row["condition"] == "slayer_fixed_interface"]
        task = [row for row in rows if row["condition"] == "task_only_slayer"]
        expected = len(protocol["gates"]["s1"]["confirmation_seeds"])
        self.assertEqual(len(fixed), expected)
        self.assertEqual(len(task), expected)
        self.assertEqual({row["delay_method"] for row in fixed}, {"fixed"})
        self.assertEqual({row["delay_method"] for row in task}, {"task_only_slayer"})

    def test_multiseed_capacity_requires_registered_number_of_passes(self):
        rows = [
            {
                "condition": "fixed", "N_hidden_total": 3, "K": 2,
                "worst_query_balanced_accuracy": 0.95,
                "exact_trial_accuracy": 0.95,
            }
            for _ in range(4)
        ]
        rows.append({
            "condition": "fixed", "N_hidden_total": 3, "K": 2,
            "worst_query_balanced_accuracy": 0.89,
            "exact_trial_accuracy": 0.95,
        })
        cells = aggregate_cells(
            rows, resource_key="N_hidden_total", required_seed_passes=5
        )
        self.assertFalse(cells[0]["pass_reliability"])
        self.assertIsNone(
            capacity_rows(cells, resource_key="N_hidden_total")[0]["capacity"]
        )

    def test_worst_query_only_endpoint_ignores_exact_trial_after_seed_gate(self):
        rows = [
            {
                "condition": "fixed", "N_hidden_total": 3, "K": 2,
                "worst_query_balanced_accuracy": 0.95,
                "exact_trial_accuracy": 0.50,
            }
            for _ in range(2)
        ] + [{
            "condition": "fixed", "N_hidden_total": 3, "K": 2,
            "worst_query_balanced_accuracy": 0.50,
            "exact_trial_accuracy": 1.00,
        }]
        cells = aggregate_cells(
            rows, resource_key="N_hidden_total", required_seed_passes=2,
            expected_seed_count=3, seed_pass=worst_query_pass,
        )
        self.assertTrue(cells[0]["pass_reliability"])

    def test_grid_ceiling_capacity_is_not_used_as_numeric_fit_point(self):
        cells = [
            {
                "condition": "fixed", "N_hidden_total": 4, "K": K,
                "pass_reliability": True,
            }
            for K in [1, 2, 3, 4, 6, 8]
        ]
        row = capacity_rows(cells, resource_key="N_hidden_total")[0]
        self.assertEqual(row["capacity_observed"], 8)
        self.assertIsNone(row["capacity"])
        self.assertEqual(row["censoring"], "above_tested_K")


if __name__ == "__main__":
    unittest.main()
