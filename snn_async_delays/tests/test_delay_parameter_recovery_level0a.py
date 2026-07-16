import unittest

from scripts.run_delay_parameter_recovery_level0a import (
    expected_cells,
    load_protocol,
    optimize_scalar_delay,
    target_values,
)


class DelayParameterRecoveryLevel0ATests(unittest.TestCase):
    def test_preregistered_grid_has_75_deterministic_cells(self):
        protocol = load_protocol()
        self.assertEqual(expected_cells(protocol), 75)
        self.assertEqual(target_values(protocol), [0.0, 1.0, 5.0, 7.0, 8.0])

    def test_production_sigmoid_initial_delay_matches_formula(self):
        trace = optimize_scalar_delay(
            d_max=8.0, target=5.0, initial_raw=-2.0,
            learning_rate=0.001, optimizer_steps=0,
        )
        self.assertAlmostEqual(float(trace["initial_delay"]), 0.9536234, places=5)

    def test_preregistered_rescue_can_recover_long_interior_target(self):
        trace = optimize_scalar_delay(
            d_max=8.0, target=5.0, initial_raw=-2.0,
            learning_rate=0.05, optimizer_steps=200,
        )
        self.assertTrue(bool(trace["recovered"]))
        self.assertLessEqual(float(trace["final_absolute_error"]), 0.1)

    def test_trace_includes_initial_and_final_states(self):
        trace = optimize_scalar_delay(
            d_max=8.0, target=1.0, initial_raw=0.0,
            learning_rate=0.01, optimizer_steps=12,
        )
        self.assertEqual(len(trace["steps"]), 13)
        self.assertEqual(int(trace["steps"][0]), 0)
        self.assertEqual(int(trace["steps"][-1]), 12)


if __name__ == "__main__":
    unittest.main()
