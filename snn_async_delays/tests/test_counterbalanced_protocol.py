import unittest
from pathlib import Path

import yaml


class CounterbalancedProtocolTests(unittest.TestCase):
    def test_each_operation_occupies_each_position_once(self):
        root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load(
            (root / "configs/simultaneous_temporal_counterbalanced_performance_v1.yaml")
            .read_text(encoding="utf-8")
        )
        orders = list(cfg["operation_orders"].values())
        self.assertEqual(len(orders), 3)
        for position in range(3):
            self.assertEqual(
                {order[position] for order in orders}, {"XOR", "NAND", "NOR"}
            )
        self.assertEqual(
            len(cfg["conditions"]) * len(cfg["held_out_seeds"]) * len(orders),
            cfg["cells"],
        )
        self.assertTrue({0, 1, 42}.isdisjoint(cfg["held_out_seeds"]))


if __name__ == "__main__":
    unittest.main()
