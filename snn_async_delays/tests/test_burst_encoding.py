import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.encoding import encode_sequential_trial, encode_simultaneous_trial


class BurstEncodingTests(unittest.TestCase):
    def test_jitter_preserves_declared_event_count_sequential(self):
        # Value 1 has two events, value 0 one.  With collision-safe jitter,
        # each A/B channel retains exactly the declared count per sub-window.
        A = torch.tensor([[1., 0.]]).repeat(64, 1)
        B = torch.tensor([[0., 1.]]).repeat(64, 1)
        x = encode_sequential_trial(
            A, B, win_len=20, read_len=4, encoding_mode="burst_jitter",
            burst_n_spikes_on=2, burst_n_spikes_off=1, burst_jitter_ms=1,
        )
        expected = ((2, 1), (1, 2))
        for query, (a_count, b_count) in enumerate(expected):
            start, end = query * 10, (query + 1) * 10
            self.assertTrue(torch.all(x[:, start:end, 0].sum(dim=1) == a_count))
            self.assertTrue(torch.all(x[:, start:end, 1].sum(dim=1) == b_count))

    def test_jitter_preserves_declared_event_count_simultaneous(self):
        A = torch.tensor([[1., 0.]]).repeat(64, 1)
        B = torch.tensor([[0., 1.]]).repeat(64, 1)
        x = encode_simultaneous_trial(
            A, B, win_len=10, read_len=4, encoding_mode="burst_jitter",
            burst_n_spikes_on=2, burst_n_spikes_off=1, burst_jitter_ms=1,
        )
        expected = (2, 1, 1, 2)
        for channel, count in enumerate(expected):
            self.assertTrue(torch.all(x[:, :10, channel].sum(dim=1) == count))


if __name__ == "__main__":
    unittest.main()
