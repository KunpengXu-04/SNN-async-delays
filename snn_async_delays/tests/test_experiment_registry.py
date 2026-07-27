import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_experiment_registry import (
    diagnostic_artifacts_complete,
    infer_status,
    versioned_task_artifacts_complete,
)


class ExperimentRegistryTests(unittest.TestCase):
    def test_complete_diagnostic_cell_is_exploratory_not_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "plots").mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "metrics.json").write_text(
                json.dumps({"complete": True}), encoding="utf-8"
            )
            (root / "final_parameter.pt").write_bytes(b"state")
            (root / "plots" / "diagnostic_data.npz").write_bytes(b"npz")
            (root / "plots" / "diagnostic_panel.png").write_bytes(b"png")
            metrics = {"complete": True}
            self.assertTrue(diagnostic_artifacts_complete(root, metrics))
            status, reason = infer_status(
                "exploratory/unit/cell",
                has_eval=False,
                has_metrics=True,
                diagnostic_complete=True,
            )
            self.assertEqual(status, "exploratory")
            self.assertIn("diagnostic unit", reason)

    def test_partial_diagnostic_cell_remains_incomplete(self):
        status, _ = infer_status(
            "exploratory/unit/cell",
            has_eval=False,
            has_metrics=True,
            diagnostic_complete=False,
        )
        self.assertEqual(status, "incomplete")

    def test_smoke_is_invalid_even_when_artifacts_are_complete(self):
        status, _ = infer_status(
            "smoke/unit/cell",
            has_eval=False,
            has_metrics=True,
            diagnostic_complete=True,
        )
        self.assertEqual(status, "invalid")

    def test_complete_versioned_validation_cell_is_exploratory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "plots").mkdir()
            for relative in (
                "config.json", "validation_results.json", "best_model.pt",
                "last_model.pt", "update_log.csv", "validation_log.csv",
                "validation_predictions.npz", "resource_ledger.json",
                "plots/diagnostic_data.npz", "plots/diagnostic_panel.png",
            ):
                (root / relative).write_bytes(b"artifact")
            (root / "run_complete.json").write_text(
                json.dumps({"completed": True}), encoding="utf-8"
            )
            self.assertTrue(versioned_task_artifacts_complete(root))
            status, reason = infer_status(
                "exploratory/versioned/cell", has_eval=True,
                has_metrics=False, diagnostic_complete=False,
                task_complete=True,
            )
            self.assertEqual(status, "exploratory")
            self.assertIn("versioned validation", reason)

    def test_complete_final_checkpoint_withdrawal_cell_is_exploratory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "plots").mkdir()
            for relative in (
                "config.json", "final_validation_results.json",
                "descriptive_best_model.pt", "last_model.pt", "update_log.csv",
                "validation_log.csv", "final_validation_predictions.npz",
                "resource_ledger.json", "source_checkpoint_provenance.json",
                "initial_and_perturbed_delay_vectors.json",
                "plots/diagnostic_data.npz", "plots/diagnostic_panel.png",
            ):
                (root / relative).write_bytes(b"artifact")
            (root / "run_complete.json").write_text(
                json.dumps({"completed": True}), encoding="utf-8"
            )
            self.assertTrue(versioned_task_artifacts_complete(root))


if __name__ == "__main__":
    unittest.main()
