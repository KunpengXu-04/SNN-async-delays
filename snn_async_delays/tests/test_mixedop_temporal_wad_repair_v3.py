import unittest

import torch

from scripts.run_mixedop_spatial_temporal_surface_preview import (
    _batch_input, build_config, build_model, encode_fn, loaders,
    routing_alignment_loss,
)
from scripts.run_mixedop_temporal_wad_repair_v3 import base_spec, load_protocol
from utils.seed import set_seed


class MixedOperationTemporalWADRepairV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def _gradient(self, kind: str) -> torch.Tensor:
        spec = base_spec(3507, "gradient_preflight")
        spec["routing_loss_kind"] = kind
        cfg = build_config(self.protocol, spec)
        set_seed(3507)
        model = build_model(cfg)
        train, _ = loaders(cfg)
        batch = next(iter(train))
        _, _, _, _, spikes = _batch_input(batch, cfg, "cpu", encode_fn(cfg))
        loss = routing_alignment_loss(spikes, model, cfg)
        return torch.autograd.grad(loss, model.syn_ih.delay_raw)[0].reshape(-1)

    def test_centroid_objective_moves_all_queries_later_from_delay_three(self):
        gradient = self._gradient("arrival_centroid_huber")
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.all(gradient < 0))

    def test_mass_ce_and_centroid_disagree_for_first_query(self):
        mass_gradient = self._gradient("arrival_mass_ce")
        centroid_gradient = self._gradient("arrival_centroid_huber")
        self.assertGreater(float(mass_gradient[0]), 0.0)
        self.assertLess(float(centroid_gradient[0]), 0.0)

    def test_unknown_routing_objective_is_rejected(self):
        spec = base_spec(3507, "gradient_preflight")
        spec["routing_loss_kind"] = "unknown"
        cfg = build_config(self.protocol, spec)
        model = build_model(cfg)
        train, _ = loaders(cfg)
        batch = next(iter(train))
        _, _, _, _, spikes = _batch_input(batch, cfg, "cpu", encode_fn(cfg))
        with self.assertRaises(ValueError):
            routing_alignment_loss(spikes, model, cfg)


if __name__ == "__main__":
    unittest.main()
