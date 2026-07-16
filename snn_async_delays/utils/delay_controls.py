"""Causal interventions for trained synaptic-delay models."""

from __future__ import annotations

import torch


def shuffle_delay_parameters_(model, seed: int) -> dict:
    """Deterministically permute learned per-synapse delay parameters in place.

    The intervention preserves every delay value, weight, and decoder parameter.
    Shared scalar delays are rejected because permutation is not an intervention.
    Returns a compact manifest suitable for result JSON.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    manifest = {"intervention": "shuffle_learned_delays", "seed": int(seed), "layers": {}}
    found = False
    for name in ("syn_ih", "syn_h1h2", "syn_ho"):
        layer = getattr(model, name, None)
        if layer is None or not getattr(layer, "train_delays", False):
            continue
        raw = layer.delay_raw
        if raw.numel() <= 1:
            raise ValueError(f"Cannot shuffle shared/scalar delay layer {name}")
        before = raw.detach().clone()
        perm = torch.randperm(raw.numel(), generator=generator, device="cpu").to(raw.device)
        with torch.no_grad():
            raw.copy_(before.reshape(-1)[perm].reshape_as(raw))
        manifest["layers"][name] = {
            "count": raw.numel(),
            "multiset_preserved": bool(torch.equal(
                torch.sort(before.reshape(-1)).values,
                torch.sort(raw.detach().reshape(-1)).values,
            )),
        }
        found = True
    if not found:
        raise ValueError("Model has no trainable per-synapse delays to shuffle")
    return manifest
