import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.boolean_dataset import MultiQueryDataset
from snn.model import SNNModel, make_slots
from train.eval import evaluate
from data.encoding import encode_trial


class DelayModeTests(unittest.TestCase):
    def test_weights_only_fixed_zero_delay(self):
        model = SNNModel(
            n_input=2,
            n_hidden=8,
            d_max=20,
            train_mode="weights_only",
            delay_param_type="sigmoid",
            fixed_delay_value=0.0,
        )
        delays = model.get_delays()["ih"]
        self.assertTrue(torch.allclose(delays, torch.zeros_like(delays)))
        self.assertEqual(len(model.delay_params()), 0)

    def test_delays_only_freezes_weights(self):
        model = SNNModel(
            n_input=2,
            n_hidden=8,
            d_max=20,
            train_mode="delays_only",
            delay_param_type="sigmoid",
        )
        self.assertEqual(len(model.weight_params()), 0)
        self.assertGreater(len(model.delay_params()), 0)

    def test_continuous_and_quantized_backward(self):
        B, K = 4, 2
        slots = make_slots(K, win_len=8, read_len=4, gap_len=0)
        A = torch.randint(0, 2, (B, K)).float()
        Bv = torch.randint(0, 2, (B, K)).float()
        op = torch.zeros(B, K).long()
        y = torch.randint(0, 2, (B, K)).float()

        for delay_type, step in [("sigmoid", 1.0), ("quantized", 2.0)]:
            model = SNNModel(
                n_input=2,
                n_hidden=10,
                d_max=20,
                train_mode="weights_and_delays",
                delay_param_type=delay_type,
                delay_step=step,
            )
            x = encode_trial(A, Bv, op, slots, n_input=2, r_on=200.0, r_off=10.0, dt=1.0)
            logits, _ = model(x, slots)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), y.reshape(-1))
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    self.assertFalse(torch.isnan(p.grad).any().item())

    def test_throughput_consistency(self):
        cfg = {
            "r_on": 200.0,
            "r_off": 10.0,
            "dt": 1.0,
            "ops_list": ["AND", "OR", "XOR", "XNOR", "NAND", "NOR", "A_IMP_B", "B_IMP_A"],
        }
        K = 2
        slots = make_slots(K, win_len=8, read_len=4, gap_len=0)
        ds = MultiQueryDataset(K=K, n_samples=16, same_op=True, op_name="NAND", seed=1)
        loader = DataLoader(ds, batch_size=8)

        model = SNNModel(
            n_input=2,
            n_hidden=12,
            d_max=20,
            train_mode="weights_only",
            delay_param_type="sigmoid",
            fixed_delay_value=0.0,
        )
        out = evaluate(model, loader, slots, cfg, device="cpu")
        expected = K / out["mean_hidden_spikes"] if out["mean_hidden_spikes"] > 0 else float("nan")
        self.assertAlmostEqual(out["throughput_K_per_spk"], expected, places=7)


if __name__ == "__main__":
    unittest.main()
