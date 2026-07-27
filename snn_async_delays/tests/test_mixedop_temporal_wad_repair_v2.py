import unittest

import torch

from scripts.run_mixedop_spatial_temporal_surface_preview import (
    _batch_input,
    _delay_diagnostics,
    build_config,
    build_model,
    encode_fn,
    loaders,
    routing_alignment_loss,
)
from scripts.run_mixedop_temporal_wad_repair_v2 import (
    base_spec,
    load_protocol,
)
from train.trainer import window_class_balanced_bce
from utils.resource_ledger import static_resource_ledger
from utils.seed import set_seed


class MixedOperationTemporalWADRepairV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_targets_are_interior_and_initial_delay_is_functionally_three(self):
        cfg = build_config(self.protocol, base_spec(3407, "gradient_preflight"))
        model = build_model(cfg)
        self.assertEqual(cfg["d_max"], 23)
        self.assertEqual(cfg["delay_support_margin_steps"], 4)
        self.assertEqual(cfg["oracle_base_delay_steps"], 3)
        diagnostics = _delay_diagnostics(model, cfg)
        for value in diagnostics["delay_query_mean_steps"]:
            self.assertAlmostEqual(value, 3.0, places=5)
        self.assertEqual(model.syn_ih.delay_raw.numel(), 5)
        self.assertEqual(static_resource_ledger(model)["trainable_delay_parameters"], 5)

    def test_raw_distance_budget_derivation(self):
        dmax = 23.0
        raw_initial = torch.logit(torch.tensor(3.0 / dmax))
        raw_q4 = torch.logit(torch.tensor(19.0 / dmax))
        distance = float(raw_q4 - raw_initial)
        self.assertGreater(distance, 3.4)
        self.assertLess(distance, 3.5)
        self.assertGreater(400 * 0.01, distance)

    def test_route_gradient_points_late_queries_toward_targets(self):
        spec = base_spec(3407, "gradient_preflight")
        cfg = build_config(self.protocol, spec)
        set_seed(3407)
        model = build_model(cfg)
        train, _ = loaders(cfg)
        batch = next(iter(train))
        _, _, _, labels, spikes = _batch_input(batch, cfg, "cpu", encode_fn(cfg))
        logits, _ = model(spikes)
        task = window_class_balanced_bce(logits, labels)
        route = routing_alignment_loss(spikes, model, cfg)
        task_gradient = torch.autograd.grad(
            task, model.syn_ih.delay_raw, retain_graph=True
        )[0]
        route_gradient = torch.autograd.grad(route, model.syn_ih.delay_raw)[0]
        self.assertTrue(torch.isfinite(task_gradient).all())
        self.assertTrue(torch.all(route_gradient.reshape(-1)[1:] < 0))

    def test_decoupled_credit_replaces_task_delay_gradient_only(self):
        spec = base_spec(3407, "gradient_preflight")
        cfg = build_config(self.protocol, spec)
        set_seed(3407)
        model = build_model(cfg)
        train, _ = loaders(cfg)
        batch = next(iter(train))
        _, _, _, labels, spikes = _batch_input(batch, cfg, "cpu", encode_fn(cfg))
        logits, _ = model(spikes)
        task = window_class_balanced_bce(logits, labels)
        route = routing_alignment_loss(spikes, model, cfg)
        expected_route = torch.autograd.grad(
            route, model.syn_ih.delay_raw, retain_graph=True
        )[0].detach().clone()
        task.backward(retain_graph=True)
        weight_gradient = model.syn_ih.weight.grad.detach().clone()
        model.syn_ih.delay_raw.grad = None
        route.backward()
        self.assertTrue(torch.allclose(model.syn_ih.delay_raw.grad, expected_route))
        self.assertTrue(torch.equal(model.syn_ih.weight.grad, weight_gradient))


if __name__ == "__main__":
    unittest.main()
