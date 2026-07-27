import importlib.util
import os
import sys
import tempfile
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from snn.slayer_backend import btc_to_nct, nct_to_btc


SLAYER_AVAILABLE = (
    importlib.util.find_spec("lava") is not None
    and importlib.util.find_spec("lava.lib.dl") is not None
)


class SlayerBackendBoundaryTests(unittest.TestCase):
    def test_btc_nct_roundtrip_preserves_every_event_and_index(self):
        btc = torch.zeros(2, 11, 7)
        btc[0, 3, 5] = 1.0
        btc[1, 9, 2] = 1.0
        nct = btc_to_nct(btc)
        self.assertEqual(nct.shape, (2, 7, 11))
        self.assertTrue(torch.equal(nct_to_btc(nct), btc))
        self.assertEqual(float(nct.sum()), 2.0)

    @unittest.skipUnless(SLAYER_AVAILABLE, "isolated Lava-DL environment only")
    def test_axonal_count_windows_resources_and_checkpoint(self):
        from snn.slayer_backend import SlayerNativeWindowedModel
        from utils.resource_ledger import static_resource_ledger

        model = SlayerNativeWindowedModel(
            n_queries=2, n_hidden=3, win_len=10, read_len=24, d_max=23,
            delay_method="task_only_slayer", delay_init=4.0,
        )
        self.assertEqual(model.get_delays()["input_axons"].numel(), 8)
        self.assertEqual(model.output_window_len, 12)
        self.assertEqual(sum(p.numel() for p in model.delay_params()), 8)
        spikes = torch.zeros(2, 34, 8)
        logits, info = model(spikes, record=True)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(info["hidden_window_counts"].shape, (2, 2, 3))

        ledger = static_resource_ledger(model)
        self.assertEqual(ledger["delay_value_storage_elements"], 8)
        self.assertEqual(ledger["delayed_synapses_total"], 24)
        self.assertEqual(ledger["delay_granularity"], "input_axon_per_channel")
        self.assertEqual(ledger["delay_buffer_elements_per_sample"], 24 * 8)
        self.assertEqual(ledger["delay_buffer_reads_per_trial"], 34 * 8)
        self.assertEqual(ledger["delay_interpolation_elementwise_ops_per_trial"], 0)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")
            torch.save(model.state_dict(), path)
            restored = SlayerNativeWindowedModel(
                n_queries=2, n_hidden=3, win_len=10, read_len=24, d_max=23,
                delay_method="task_only_slayer", delay_init=0.0,
            )
            restored.load_state_dict(torch.load(path, weights_only=True))
            restored_logits, restored_info = restored(spikes, record=True)
            self.assertTrue(torch.equal(logits, restored_logits))
            self.assertTrue(torch.equal(
                info["hidden_spike_train"], restored_info["hidden_spike_train"]
            ))
            self.assertTrue(torch.equal(
                model.get_delays()["input_axons"],
                restored.get_delays()["input_axons"],
            ))


if __name__ == "__main__":
    unittest.main()
