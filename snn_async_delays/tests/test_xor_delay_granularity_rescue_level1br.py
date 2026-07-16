"""Structural tests for the Level 1B-R dimension-aware rescue."""

from __future__ import annotations

import unittest

import torch

from scripts import run_xor_delay_granularity_level1b as level1b
from scripts.run_xor_delay_granularity_rescue_level1br import (
    expected_r1_cells,
    load_protocol,
    r1_specs,
    r2_specs,
    r3_specs,
)


class XORDelayGranularityRescueLevel1BRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol()

    def test_r1_has_fifty_frozen_cells(self) -> None:
        self.assertEqual(expected_r1_cells(self.protocol), 50)
        self.assertEqual(len(r1_specs(self.protocol)), 50)

    def test_dimension_matched_lambdas_are_analytic(self) -> None:
        specs = r1_specs(self.protocol)
        expected = {
            "global_anchor": (1, 1.0, 0.01),
            "per_hidden_mean_baseline": (16, 1.0, 0.01),
            "per_hidden_coordinate_matched": (16, 16.0, 0.16),
            "per_synapse_mean_baseline": (64, 1.0, 0.01),
            "per_synapse_coordinate_matched": (64, 64.0, 0.64),
        }
        for condition, values in expected.items():
            selected = [spec for spec in specs if spec["condition"] == condition]
            self.assertEqual(len(selected), 10)
            self.assertEqual(selected[0]["independent_delay_parameters"], values[0])
            self.assertEqual(selected[0]["normalization_factor"], values[1])
            self.assertAlmostEqual(selected[0]["arrival_auxiliary_weight"], values[2])

    def test_seed_sets_are_mutually_disjoint_and_new(self) -> None:
        optimization = self.protocol["optimization"]
        sets = [
            set(optimization["r1_calibration_seeds"]),
            set(optimization["r2_budget_calibration_seeds"]),
            set(optimization["r3_sealed_confirmation_seeds"]),
        ]
        historical = {1, 42, 107, 211, 307, 607, 709, 811, 919, 1021}
        for index, values in enumerate(sets):
            self.assertTrue(values.isdisjoint(historical))
            for other in sets[index + 1:]:
                self.assertTrue(values.isdisjoint(other))

    def test_weighted_arrival_gradient_per_coordinate_is_matched(self) -> None:
        A, B, _, _ = level1b.exact_truth_batch("cpu")
        spikes = level1b.encode_exact_truth(
            self.protocol, A, B, encoding="single_event", device="cpu"
        )
        selected = {
            spec["condition"]: spec for spec in r1_specs(self.protocol)
            if spec["initial_raw"] == -2.0 and spec["seed"] == 1123
        }
        weighted = {}
        for condition in (
            "global_anchor",
            "per_hidden_coordinate_matched",
            "per_synapse_coordinate_matched",
        ):
            spec = selected[condition]
            model = level1b.build_model(self.protocol, spec)
            delays = level1b.independent_delay_values(model, spec)
            base = level1b.per_coordinate_base_traces(
                spikes, tying=spec["delay_tying"], hidden_neurons=16
            )
            loss, _, _, _, _ = level1b.per_parameter_arrival_loss(
                base, delays, target_delay_steps=4.0, d_max=8
            )
            gradient = torch.autograd.grad(loss, model.syn_ih.delay_raw)[0]
            weighted[condition] = (
                float(spec["arrival_auxiliary_weight"])
                * float(gradient.detach().abs().mean().item())
            )
        self.assertAlmostEqual(weighted["global_anchor"], weighted["per_hidden_coordinate_matched"], places=6)
        self.assertAlmostEqual(weighted["global_anchor"], weighted["per_synapse_coordinate_matched"], places=6)

    def test_r2_is_conditional_and_has_eighteen_cells_per_granularity(self) -> None:
        with self.assertRaises(RuntimeError):
            r2_specs(self.protocol, {"r2_authorized": False})
        one = r2_specs(self.protocol, {
            "r2_authorized": True,
            "r2_eligible_granularities": ["per_hidden_neuron"],
        })
        both = r2_specs(self.protocol, {
            "r2_authorized": True,
            "r2_eligible_granularities": ["per_hidden_neuron", "per_synapse"],
        })
        self.assertEqual(len(one), 18)
        self.assertEqual(len(both), 36)
        self.assertEqual(
            {(spec["condition"], spec["delay_learning_rate"], spec["full_batch_updates"]) for spec in one},
            {("lr_only", 0.05, 500), ("budget_only", 0.01, 1000), ("lr_and_budget", 0.05, 1000)},
        )

    def test_r3_keeps_confirmation_sealed_behind_decisions(self) -> None:
        with self.assertRaises(RuntimeError):
            r3_specs(self.protocol, {"global_anchor_pass": False}, None)
        with self.assertRaises(RuntimeError):
            r3_specs(self.protocol, {
                "global_anchor_pass": True,
                "r2_eligible_granularities": ["per_hidden_neuron"],
                "provisional_recipes": [],
            }, None)
        direct = r3_specs(self.protocol, {
            "global_anchor_pass": True,
            "r2_eligible_granularities": [],
            "provisional_recipes": [{
                "granularity": "per_hidden_neuron",
                "source": "r1_dimension_matched",
                "delay_learning_rate": 0.01,
                "full_batch_updates": 500,
            }],
        }, None)
        self.assertEqual(len(direct), 20)
        self.assertEqual({spec["seed"] for spec in direct}, {2003, 2011, 2027, 2039, 2053})

    def test_r3_can_confirm_both_higher_dimensional_recipes(self) -> None:
        r1 = {
            "global_anchor_pass": True,
            "r2_eligible_granularities": ["per_synapse"],
            "provisional_recipes": [{
                "granularity": "per_hidden_neuron",
                "source": "r1_dimension_matched",
                "delay_learning_rate": 0.01,
                "full_batch_updates": 500,
            }],
        }
        r2 = {"provisional_recipes": [{
            "granularity": "per_synapse",
            "source": "r2_lr_only",
            "delay_learning_rate": 0.05,
            "full_batch_updates": 500,
        }]}
        self.assertEqual(len(r3_specs(self.protocol, r1, r2)), 30)

    def test_optimizer_overrides_are_explicit(self) -> None:
        spec = next(
            item for item in r2_specs(self.protocol, {
                "r2_authorized": True,
                "r2_eligible_granularities": ["per_hidden_neuron"],
            })
            if item["condition"] == "lr_only"
        )
        model = level1b.build_model(self.protocol, spec)
        optimizer = level1b.build_optimizer(model, self.protocol, spec)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.01, 0.05])
        self.assertTrue(torch.all(model.syn_ho.get_delays() == 0.0))

    def test_protocol_does_not_materialize_microburst_or_k_greater_than_one(self) -> None:
        self.assertTrue(self.protocol["execution_policy"]["no_microburst_or_K_greater_than_one"])
        self.assertEqual(set(self.protocol["encodings"]), {"single_event"})
        self.assertEqual(self.protocol["task"]["K"], 1)


if __name__ == "__main__":
    unittest.main()
