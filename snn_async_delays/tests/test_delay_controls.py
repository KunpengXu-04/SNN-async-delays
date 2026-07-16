import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from snn.model import SNNSimultaneousModel
from utils.delay_controls import shuffle_delay_parameters_
from utils.resource_ledger import static_resource_ledger


def make_model(**kwargs):
    defaults = dict(
        n_queries=3, n_hidden=12, win_len=30, read_len=10, d_max=30,
        n_input_channels=2, readout_type="linear", observation_mode="all_time",
    )
    defaults.update(kwargs)
    return SNNSimultaneousModel(**defaults)


class DelayControlTests(unittest.TestCase):
    def test_shared_scalar_is_one_parameter_broadcast_to_all_synapses(self):
        model = make_model(train_mode="weights_and_delays", shared_delay=True)
        self.assertEqual(model.syn_ih.delay_raw.numel(), 1)
        delays = model.get_delays()["ih"]
        self.assertEqual(tuple(delays.shape), (2, 12))
        self.assertTrue(torch.allclose(delays, delays[0, 0].expand_as(delays)))
        delays.sum().backward()
        self.assertIsNotNone(model.syn_ih.delay_raw.grad)

    def test_fixed_heterogeneous_is_reproducible_frozen_and_rng_neutral(self):
        torch.manual_seed(123)
        d0 = make_model(train_mode="weights_only", fixed_delay_value=0.0)
        torch.manual_seed(123)
        fixed_a = make_model(
            train_mode="weights_only", fixed_delay_distribution="uniform",
            fixed_delay_seed=7,
        )
        fixed_b = make_model(
            train_mode="weights_only", fixed_delay_distribution="uniform",
            fixed_delay_seed=7,
        )
        self.assertTrue(torch.equal(d0.syn_ih.weight, fixed_a.syn_ih.weight))
        self.assertTrue(torch.equal(fixed_a.get_delays()["ih"], fixed_b.get_delays()["ih"]))
        self.assertGreater(torch.std(fixed_a.get_delays()["ih"]).item(), 0.0)
        self.assertEqual(len(fixed_a.delay_params()), 0)

    def test_fixed_heterogeneous_respects_matched_delay_support(self):
        fixed = make_model(
            train_mode="weights_only", fixed_delay_distribution="uniform",
            fixed_delay_seed=7, fixed_delay_low=1.0, fixed_delay_high=9.0,
        )
        delays = fixed.get_delays()["ih"]
        self.assertGreaterEqual(float(delays.min()), 1.0)
        self.assertLessEqual(float(delays.max()), 9.0)

    def test_shuffle_preserves_delay_multiset_and_all_non_delay_state(self):
        torch.manual_seed(5)
        model = make_model(train_mode="weights_and_delays")
        with torch.no_grad():
            model.syn_ih.delay_raw.copy_(torch.linspace(-3, 2, model.syn_ih.delay_raw.numel()).reshape_as(model.syn_ih.delay_raw))
        before_delays = model.get_delays()["ih"].detach().clone()
        before_weight = model.syn_ih.weight.detach().clone()
        before_readout = model.readout.weight.detach().clone()
        manifest = shuffle_delay_parameters_(model, seed=11)
        after_delays = model.get_delays()["ih"].detach()
        self.assertTrue(torch.equal(torch.sort(before_delays.flatten()).values,
                                    torch.sort(after_delays.flatten()).values))
        self.assertFalse(torch.equal(before_delays, after_delays))
        self.assertTrue(torch.equal(before_weight, model.syn_ih.weight))
        self.assertTrue(torch.equal(before_readout, model.readout.weight))
        self.assertTrue(manifest["layers"]["syn_ih"]["multiset_preserved"])

    def test_record_includes_membrane_trace(self):
        model = make_model(train_mode="weights_only", fixed_delay_value=0.0)
        x = torch.randint(0, 2, (2, 40, 2)).float()
        _, info = model(x, record=True)
        self.assertEqual(tuple(info["hidden_membrane_train"].shape), (2, 40, 12))

    def test_resource_ledger_charges_actual_delay_storage(self):
        scalar = make_model(train_mode="weights_and_delays", shared_delay=True)
        fixed = make_model(
            train_mode="weights_only", fixed_delay_distribution="uniform",
            fixed_delay_seed=9,
        )
        wad = make_model(train_mode="weights_and_delays")
        self.assertEqual(static_resource_ledger(scalar)["delay_value_storage_elements"], 1)
        self.assertEqual(static_resource_ledger(fixed)["delay_value_storage_elements"], 24)
        self.assertEqual(static_resource_ledger(wad)["delay_value_storage_elements"], 24)


if __name__ == "__main__":
    unittest.main()
