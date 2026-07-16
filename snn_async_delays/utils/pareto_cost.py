"""Declared analytic cost proxies for spatial-vs-temporal Pareto studies."""

from __future__ import annotations


def spatial_temporal_ratios(
    *,
    K: int,
    baseline_hidden_per_query: int,
    shared_hidden: int,
    spatial_steps: int,
    temporal_steps: int,
) -> dict[str, float]:
    """Return area, latency and first-order compute ratios B/A.

    The spatial baseline is K independent block-diagonal modules. The temporal
    network has one dense shared hidden layer connected to all K input/output
    groups. ``hidden_update_ratio`` is the n_hidden*T proxy. The synapse ratio
    assumes identical per-query input/output dimensions and therefore exposes
    that K cancels from the dense connection count but not from neuron area.
    """
    values = (K, baseline_hidden_per_query, shared_hidden, spatial_steps, temporal_steps)
    if any(int(v) <= 0 for v in values):
        raise ValueError("all Pareto ratio inputs must be positive")
    area = shared_hidden / (K * baseline_hidden_per_query)
    latency = temporal_steps / spatial_steps
    return {
        "hidden_area_ratio": area,
        "latency_ratio": latency,
        "hidden_update_ratio": area * latency,
        "dense_synapse_compute_ratio": (
            shared_hidden * temporal_steps
            / (baseline_hidden_per_query * spatial_steps)
        ),
    }


def mixed_operation_spatial_hidden_total(per_query_hidden: list[int]) -> int:
    """Matched mixed-operation baseline is a sum, not K times one width."""
    if not per_query_hidden or any(int(value) <= 0 for value in per_query_hidden):
        raise ValueError("per-query hidden widths must be positive")
    return int(sum(per_query_hidden))
