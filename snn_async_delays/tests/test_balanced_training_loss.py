import unittest
import tempfile

import torch
import torch.nn.functional as F

from train.trainer import window_class_balanced_bce, window_class_balanced_mean
from train.trainer import SimultaneousTrainer
from snn.model import SNNSimultaneousModel


class BalancedTrainingLossTests(unittest.TestCase):
    def test_macro_average_is_invariant_to_class_duplication(self):
        logits = torch.tensor([[0.2], [-0.4], [-0.4]], requires_grad=True)
        labels = torch.tensor([[1.0], [0.0], [0.0]])
        original = window_class_balanced_bce(logits, labels)

        duplicated_logits = torch.tensor(
            [[0.2], [-0.4], [-0.4], [-0.4], [-0.4]], requires_grad=True
        )
        duplicated_labels = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]])
        duplicated = window_class_balanced_bce(
            duplicated_logits, duplicated_labels
        )
        self.assertAlmostEqual(original.item(), duplicated.item(), places=7)
        original.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_windows_and_classes_receive_equal_macro_weight(self):
        per_sample = torch.tensor([
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 30.0],
            [7.0, 40.0],
        ])
        labels = torch.tensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        # q0 class means: 3 and 7; q1 class means: 10 and 30.
        expected = torch.tensor((3.0 + 7.0 + 10.0 + 30.0) / 4.0)
        self.assertTrue(torch.allclose(
            window_class_balanced_mean(per_sample, labels), expected
        ))

    def test_balanced_loss_differs_from_pooled_on_imbalanced_labels(self):
        logits = torch.zeros(4, 1)
        labels = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
        # Zero logits are symmetric, so both reductions agree at log(2).
        self.assertAlmostEqual(
            window_class_balanced_bce(logits, labels).item(),
            F.binary_cross_entropy_with_logits(logits, labels).item(),
            places=7,
        )
        biased = torch.full((4, 1), 2.0)
        self.assertNotAlmostEqual(
            window_class_balanced_bce(biased, labels).item(),
            F.binary_cross_entropy_with_logits(biased, labels).item(),
            places=5,
        )

    def test_simultaneous_trainer_balanced_membrane_branch_backpropagates(self):
        model = SNNSimultaneousModel(
            n_queries=3, n_hidden=6, win_len=4, read_len=6, d_max=6,
            n_input_channels=6, train_mode="weights_and_delays",
            use_output_spikes=True, n_output_neurons=2,
            opponent_output_mode="shared_windowed",
            observation_mode="windowed_shared", output_window_len=2,
            lif_output_threshold=0.2,
        )
        cfg = {
            "lr_w": 1e-3, "lr_d": 1e-3, "lr_readout": 1e-3,
            "win_len": 4, "read_len": 6, "r_on": 400.0, "r_off": 10.0,
            "dt": 1.0, "n_ops": 0, "output_membrane_warmup_epochs": 2,
            "output_membrane_aux_weight": 0.2,
            "loss_reduction": "window_class_balanced",
            "spike_penalty": 0.0, "delay_penalty": 0.0,
            "homeo_lambda": 0.0,
        }

        def encode(A, B, **kwargs):
            torch.manual_seed(4)
            return torch.randint(0, 2, (A.shape[0], 10, 6)).float()

        A = torch.zeros(8, 3)
        B = torch.zeros(8, 3)
        op_ids = torch.zeros(8, 3, dtype=torch.long)
        labels = torch.tensor([
            [0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1],
            [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0],
        ]).float()
        with tempfile.TemporaryDirectory() as run_dir:
            trainer = SimultaneousTrainer(model, cfg, run_dir, encode_fn=encode)
            trainer.current_epoch = 1
            loss, _, _ = trainer._forward_batch(A, B, op_ids, labels)
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.syn_ho.weight.grad)


if __name__ == "__main__":
    unittest.main()
