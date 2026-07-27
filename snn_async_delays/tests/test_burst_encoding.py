import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.encoding import encode_sequential_trial, encode_simultaneous_trial


class BurstEncodingTests(unittest.TestCase):
    def test_binary_one_hot_packet_has_exact_eight_events_per_query(self):
        A = torch.tensor([[0., 1., 1.], [1., 0., 1.]])
        B = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        x = encode_simultaneous_trial(
            A, B, win_len=10, read_len=18,
            encoding_mode="binary_one_hot_packet",
            packet_start_step=6, packet_steps=4,
        )

        self.assertEqual(tuple(x.shape), (2, 28, 12))
        self.assertTrue(torch.all(x.sum(dim=(1, 2)) == 24))
        self.assertEqual(float(x[:, :6].sum()), 0.0)
        self.assertEqual(float(x[:, 10:].sum()), 0.0)
        for query in range(3):
            query_events = x[:, 6:10, 4 * query:4 * query + 4]
            self.assertTrue(torch.all(query_events.sum(dim=(1, 2)) == 8))

    def test_binary_one_hot_packet_is_rng_independent(self):
        A = torch.tensor([[0., 1.]])
        B = torch.tensor([[1., 0.]])
        torch.manual_seed(1)
        first = encode_simultaneous_trial(
            A, B, win_len=10, read_len=8,
            encoding_mode="binary_one_hot_packet",
        )
        torch.manual_seed(999)
        second = encode_simultaneous_trial(
            A, B, win_len=10, read_len=8,
            encoding_mode="binary_one_hot_packet",
        )
        self.assertTrue(torch.equal(first, second))

    def test_binary_one_hot_packet_rejects_out_of_window_packet(self):
        A = torch.zeros(1, 1)
        B = torch.ones(1, 1)
        with self.assertRaises(ValueError):
            encode_simultaneous_trial(
                A, B, win_len=8, read_len=4,
                encoding_mode="binary_one_hot_packet",
                packet_start_step=6, packet_steps=4,
            )

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
