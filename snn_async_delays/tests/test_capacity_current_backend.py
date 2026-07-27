import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.run_capacity_current_backend import (
    build_current_model, current_fixed_input_schedule,
)
from scripts.run_spatial_temporal_capacity_slayer import load_protocol, specs
from utils.resource_ledger import static_resource_ledger


class CurrentCapacityBackendTests(unittest.TestCase):
    def _spec(self, condition, method, K=2, hidden=4, budget=24):
        return {
            "condition": condition, "delay_method": method, "K": K,
            "N_hidden_total": hidden, "output_budget_B": budget,
            "window_width": budget // K, "T": 10 + budget,
        }

    def test_current_fixed_uses_four_compact_delays_per_query(self):
        model = build_current_model(self._spec("current_fixed", "fixed"))
        self.assertEqual(sum(p.numel() for p in model.delay_params()), 0)
        self.assertTrue(torch.equal(
            model.syn_ih.delay_raw.flatten(), current_fixed_input_schedule(2, 12)
        ))
        ledger = static_resource_ledger(model)
        self.assertEqual(ledger["delay_value_storage_elements"], 8)
        self.assertEqual(ledger["delay_granularity"], "input_axon_per_channel")

    def test_current_task_only_has_exactly_4k_delay_parameters(self):
        model = build_current_model(
            self._spec("current_task_only", "task_only_current", K=3, hidden=5)
        )
        self.assertEqual(sum(p.numel() for p in model.delay_params()), 12)
        self.assertEqual(static_resource_ledger(model)["delay_value_storage_elements"], 12)

    def test_independent_spatial_d0_uses_compact_frozen_axon_delays(self):
        model = build_current_model(
            self._spec("independent_spatial_d0", "d0", K=2, hidden=4)
        )
        ledger = static_resource_ledger(model)
        self.assertEqual(ledger["delay_value_storage_elements"], 8)
        self.assertEqual(ledger["synapses_input_hidden"], 16)

    def test_n_ref_calibration_uses_calibration_seeds_only(self):
        protocol = load_protocol()
        rows = specs(protocol, "spatial_calibration")
        self.assertEqual(len(rows), 50)
        self.assertEqual({row["K"] for row in rows}, {6})
        self.assertEqual({row["output_budget_B"] for row in rows}, {48})
        self.assertEqual(
            {row["seed"] for row in rows},
            set(protocol["gates"]["s1"]["calibration_seeds"]),
        )


if __name__ == "__main__":
    unittest.main()
