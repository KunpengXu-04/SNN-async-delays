"""Structural tests for the dimension-aware XOR micro-burst rescue."""

from __future__ import annotations

import unittest

import torch

from scripts import run_xor_delay_granularity_level1b as level1b
from scripts.run_xor_delay_granularity_rescue_microburst import (
    control_specs,
    expected_control_cells,
    expected_learned_cells,
    learned_specs,
    load_protocol,
)


class XORDelayGranularityRescueMicroburstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol()

    def test_frozen_grid_has_forty_five_cells(self) -> None:
        self.assertEqual(expected_control_cells(self.protocol), 5)
        self.assertEqual(expected_learned_cells(self.protocol), 40)
        specs = learned_specs(self.protocol, {"learned_stage_authorized": True})
        self.assertEqual(len(specs), 40)
        for condition in {
            "global_anchor",
            "per_hidden_dimension_matched",
            "per_synapse_dimension_matched",
            "per_hidden_task_only",
        }:
            self.assertEqual(sum(spec["condition"] == condition for spec in specs), 10)

    def test_microburst_encoding_is_exact_and_constant_cost(self) -> None:
        A, B, _, _ = level1b.exact_truth_batch("cpu")
        spikes = level1b.encode_exact_truth(
            self.protocol,
            A,
            B,
            encoding="consecutive_microburst",
            device="cpu",
        )
        self.assertTrue(torch.all(spikes.sum(dim=(1, 2)) == 4))
        self.assertEqual(torch.nonzero(spikes.sum(dim=(0, 2))).reshape(-1).tolist(), [8, 9])

    def test_conditions_use_frozen_dimension_matched_weights(self) -> None:
        specs = learned_specs(self.protocol, {"learned_stage_authorized": True})
        expected = {
            "global_anchor": ("global", 1, 1.0, 0.01),
            "per_hidden_dimension_matched": (
                "per_hidden_neuron",
                16,
                16.0,
                0.16,
            ),
            "per_synapse_dimension_matched": ("per_synapse", 64, 64.0, 0.64),
            "per_hidden_task_only": ("per_hidden_neuron", 16, 0.0, 0.0),
        }
        for condition, values in expected.items():
            selected = [spec for spec in specs if spec["condition"] == condition]
            self.assertEqual(len(selected), 10)
            self.assertEqual(selected[0]["granularity"], values[0])
            self.assertEqual(selected[0]["independent_delay_parameters"], values[1])
            self.assertEqual(selected[0]["normalization_factor"], values[2])
            self.assertAlmostEqual(selected[0]["arrival_auxiliary_weight"], values[3])

    def test_weighted_arrival_gradient_matches_global_under_microburst(self) -> None:
        specs = learned_specs(self.protocol, {"learned_stage_authorized": True})
        selected = {
            spec["condition"]: spec
            for spec in specs
            if spec["initial_raw"] == -2.0 and spec["seed"] == 2333
        }
        A, B, _, _ = level1b.exact_truth_batch("cpu")
        spikes = level1b.encode_exact_truth(
            self.protocol,
            A,
            B,
            encoding="consecutive_microburst",
            device="cpu",
        )
        weighted: dict[str, float] = {}
        for condition in (
            "global_anchor",
            "per_hidden_dimension_matched",
            "per_synapse_dimension_matched",
        ):
            spec = selected[condition]
            model = level1b.build_model(self.protocol, spec)
            delays = level1b.independent_delay_values(model, spec)
            base = level1b.per_coordinate_base_traces(
                spikes,
                tying=spec["delay_tying"],
                hidden_neurons=int(self.protocol["model"]["hidden_neurons"]),
            )
            loss, _, _, _, _ = level1b.per_parameter_arrival_loss(
                base,
                delays,
                target_delay_steps=float(spec["target_delay_steps"]),
                d_max=int(self.protocol["model"]["d_max_steps"]),
            )
            gradient = torch.autograd.grad(loss, model.syn_ih.delay_raw)[0]
            weighted[condition] = float(spec["arrival_auxiliary_weight"]) * float(
                gradient.detach().abs().mean().item()
            )
        self.assertAlmostEqual(
            weighted["global_anchor"],
            weighted["per_hidden_dimension_matched"],
            places=6,
        )
        self.assertAlmostEqual(
            weighted["global_anchor"],
            weighted["per_synapse_dimension_matched"],
            places=6,
        )

    def test_seed_set_is_fresh_and_stage_lock_is_enforced(self) -> None:
        seeds = set(self.protocol["optimization"]["fresh_microburst_seeds"])
        historical = {
            1,
            42,
            107,
            211,
            307,
            607,
            709,
            811,
            919,
            1021,
            1123,
            1229,
            1321,
            1427,
            1523,
            1601,
            1693,
            1789,
            2003,
            2011,
            2027,
            2039,
            2053,
            2309,
        }
        self.assertTrue(seeds.isdisjoint(historical))
        with self.assertRaises(RuntimeError):
            learned_specs(self.protocol, {"learned_stage_authorized": False})
        self.assertEqual(len(control_specs(self.protocol)), 5)

    def test_model_keeps_d0_output_delays_and_expected_parameter_shapes(self) -> None:
        specs = learned_specs(self.protocol, {"learned_stage_authorized": True})
        expected = {"global": 1, "per_hidden_neuron": 16, "per_synapse": 64}
        for granularity, count in expected.items():
            spec = next(
                item
                for item in specs
                if item["granularity"] == granularity
                and item["arrival_auxiliary_weight"] > 0.0
            )
            model = level1b.build_model(self.protocol, spec)
            self.assertEqual(model.syn_ih.delay_raw.numel(), count)
            self.assertTrue(torch.all(model.syn_ho.get_delays() == 0.0))

    def test_protocol_cannot_unlock_k_greater_than_one(self) -> None:
        self.assertEqual(self.protocol["task"]["K"], 1)
        self.assertTrue(self.protocol["execution_policy"]["no_K_greater_than_one"])
        self.assertEqual(set(self.protocol["encodings"]), {"consecutive_microburst"})


if __name__ == "__main__":
    unittest.main()
