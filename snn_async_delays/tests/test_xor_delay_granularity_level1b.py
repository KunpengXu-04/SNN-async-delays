"""Structural tests for the preregistered XOR Level 1B protocol."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.run_xor_delay_granularity_level1b import (
    build_model,
    build_optimizer,
    encode_exact_truth,
    expected_stage_a_cells,
    expected_stage_b_control_cells,
    expected_stage_b_learned_cells,
    independent_delay_values,
    load_protocol,
    per_coordinate_base_traces,
    per_parameter_arrival_loss,
    resource_ledger,
    shifted_coordinate_traces,
    stage_a_specs,
    stage_b_control_specs,
    stage_b_learned_specs,
)
from scripts.run_xor_task_bridge_level1a import exact_truth_batch


class XORDelayGranularityLevel1BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol()

    def test_frozen_cell_counts(self) -> None:
        self.assertEqual(expected_stage_a_cells(self.protocol), 60)
        self.assertEqual(expected_stage_b_control_cells(self.protocol), 10)
        self.assertEqual(expected_stage_b_learned_cells(self.protocol), 60)

    def test_stage_a_seed_and_candidate_balance(self) -> None:
        specs = stage_a_specs(self.protocol)
        self.assertEqual(len(specs), 60)
        level1a_seeds = {1, 42, 107, 211, 307}
        self.assertTrue({spec["seed"] for spec in specs}.isdisjoint(level1a_seeds))
        for granularity in ("global", "per_hidden_neuron", "per_synapse"):
            for weight in (0.0, 0.01):
                cells = [
                    spec for spec in specs
                    if spec["granularity"] == granularity
                    and spec["arrival_auxiliary_weight"] == weight
                ]
                self.assertEqual(len(cells), 10)

    def test_event_encoding_is_exact(self) -> None:
        A, B, _, _ = exact_truth_batch("cpu")
        single = encode_exact_truth(
            self.protocol, A, B, encoding="single_event", device="cpu"
        )
        burst = encode_exact_truth(
            self.protocol, A, B, encoding="consecutive_microburst", device="cpu"
        )
        self.assertTrue(torch.all(single.sum(dim=(1, 2)) == 2))
        self.assertEqual(torch.nonzero(single.sum(dim=(0, 2))).reshape(-1).tolist(), [9])
        self.assertTrue(torch.all(burst.sum(dim=(1, 2)) == 4))
        self.assertEqual(torch.nonzero(burst.sum(dim=(0, 2))).reshape(-1).tolist(), [8, 9])

    def test_delay_parameter_shapes_and_broadcast(self) -> None:
        expected = {"global": 1, "per_hidden_neuron": 16, "per_synapse": 64}
        for spec in stage_a_specs(self.protocol):
            if spec["arrival_auxiliary_weight"] != 0.01 or spec["initial_raw"] != -2.0:
                continue
            model = build_model(self.protocol, spec)
            self.assertEqual(model.syn_ih.delay_raw.numel(), expected[spec["granularity"]])
            self.assertEqual(tuple(model.syn_ih.get_delays().shape), (4, 16))
            self.assertTrue(torch.all(model.syn_ho.get_delays() == 0.0))

    def test_resource_ledger_separates_trainable_delays_from_synapses(self) -> None:
        expected = {"global": 1, "per_hidden_neuron": 16, "per_synapse": 64}
        dummy = {
            "input": np.zeros((4, 20, 4), dtype=np.float32),
            "hidden": np.zeros((4, 20, 16), dtype=np.float32),
            "output": np.zeros((4, 20, 2), dtype=np.float32),
        }
        for spec in stage_a_specs(self.protocol):
            if spec["arrival_auxiliary_weight"] != 0.01 or spec["initial_raw"] != -2.0:
                continue
            model = build_model(self.protocol, spec)
            ledger = resource_ledger(model, spec, dummy)
            self.assertEqual(ledger["input_hidden_physical_synapses"], 64)
            self.assertEqual(
                ledger["input_hidden_trainable_delay_parameters"], expected[spec["granularity"]]
            )

    def test_arrival_scaffold_has_per_coordinate_correct_gradients(self) -> None:
        A, B, _, _ = exact_truth_batch("cpu")
        spikes = encode_exact_truth(
            self.protocol, A, B, encoding="single_event", device="cpu"
        )
        for tying, count in (("global", 1), ("post_neuron", 16), ("pair", 64)):
            base = per_coordinate_base_traces(spikes, tying=tying, hidden_neurons=16)
            self.assertEqual(tuple(base.shape), (count, 20))
            for raw_value in (-2.0, 2.0):
                raw = torch.full((count,), raw_value, requires_grad=True)
                delays = 8.0 * torch.sigmoid(raw)
                loss, _, _, _, _ = per_parameter_arrival_loss(
                    base, delays, target_delay_steps=4.0, d_max=8
                )
                gradient = torch.autograd.grad(loss, raw)[0]
                self.assertTrue(torch.all(gradient.abs() > 1e-10))
                self.assertTrue(torch.all(gradient * (delays.detach() - 4.0) > 0.0))

    def test_shift_uses_effective_delay_plus_one(self) -> None:
        base = torch.zeros((1, 20))
        base[0, 9] = 1.0
        shifted = shifted_coordinate_traces(base, torch.tensor([4.0]), d_max=8)
        self.assertEqual(torch.nonzero(shifted[0]).reshape(-1).tolist(), [14])

    def test_stage_locks_are_enforced(self) -> None:
        with self.assertRaises(RuntimeError):
            stage_b_control_specs(self.protocol, {"stage_b_controls_authorized": False})
        controls = stage_b_control_specs(self.protocol, {"stage_b_controls_authorized": True})
        self.assertEqual(len(controls), 10)
        with self.assertRaises(RuntimeError):
            stage_b_learned_specs(self.protocol, {"stage_b_learned_authorized": False})
        learned = stage_b_learned_specs(self.protocol, {"stage_b_learned_authorized": True})
        self.assertEqual(len(learned), 60)

    def test_optimizer_uses_frozen_learning_rates(self) -> None:
        spec = next(
            item for item in stage_a_specs(self.protocol)
            if item["granularity"] == "per_hidden_neuron"
            and item["arrival_auxiliary_weight"] == 0.01
        )
        model = build_model(self.protocol, spec)
        optimizer = build_optimizer(model, self.protocol)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.01, 0.01])

    def test_mean_correct_delay_cannot_hide_failed_coordinate(self) -> None:
        values = torch.tensor([3.8, 4.2])
        errors = (values - 4.0).abs()
        self.assertAlmostEqual(float(values.mean()), 4.0)
        self.assertGreater(float(errors.max()), 0.1)
        self.assertLess(float((errors <= 0.1).float().mean()), 1.0)


if __name__ == "__main__":
    unittest.main()
