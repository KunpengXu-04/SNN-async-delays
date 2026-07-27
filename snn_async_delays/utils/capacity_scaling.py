"""Capacity-boundary utilities with explicit censoring and no smoothing.

These helpers intentionally operate on raw pass/fail rows.  They never repair
non-monotonic width or time curves, and a missing pass remains ``None`` rather
than being converted into a numeric value.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


RESULT_SCHEMA_VERSION = "capacity_scaling_result_v1"
REQUIRED_RESULT_FIELDS = (
    "model_backend",
    "delay_method",
    "delay_granularity",
    "encoding_mode",
    "output_interface",
    "K",
    "N_hidden_total",
    "output_budget_B",
    "window_width",
    "T",
    "input_events_expected",
    "input_events_measured",
    "delay_parameter_count",
    "delay_storage_count",
    "worst_query_balanced_accuracy",
    "exact_trial_accuracy",
    "per_query_balanced_accuracy",
    "per_window_activity",
    "delay_intervention_metrics",
    "resource_ledger",
)


def make_result_record(**values: Any) -> dict[str, Any]:
    """Create and validate one immutable-schema capacity result record."""
    record = {"schema_version": RESULT_SCHEMA_VERSION, **values}
    validate_result_record(record)
    return record


def validate_result_record(record: Mapping[str, Any]) -> None:
    """Validate the common result surface without accepting NaN values."""
    if record.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("capacity result schema version mismatch")
    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in record]
    if missing:
        raise ValueError(f"capacity result is missing fields: {missing}")
    for key in (
        "K", "N_hidden_total", "output_budget_B", "window_width", "T",
        "input_events_expected", "input_events_measured",
        "delay_parameter_count", "delay_storage_count",
        "worst_query_balanced_accuracy", "exact_trial_accuracy",
    ):
        if not np.isfinite(float(record[key])):
            raise ValueError(f"capacity result field {key} must be finite")


def reliability_pass(
    row: Mapping[str, Any], *, worst_minimum: float = 0.90,
    exact_minimum: float = 0.90,
) -> bool:
    return bool(
        float(row["worst_query_balanced_accuracy"]) >= worst_minimum
        and float(row["exact_trial_accuracy"]) >= exact_minimum
    )


def raw_capacity(
    rows: Iterable[Mapping[str, Any]], resource_key: str, resource_value: int | float,
    *, K_key: str = "K", pass_key: str = "pass_reliability",
) -> int | None:
    """Largest observed passing K at one resource value; no interpolation."""
    passing = [
        int(row[K_key]) for row in rows
        if float(row[resource_key]) == float(resource_value) and bool(row[pass_key])
    ]
    return max(passing) if passing else None


def minimum_passing_resource(
    rows: Iterable[Mapping[str, Any]], K: int, resource_key: str,
    *, K_key: str = "K", pass_key: str = "pass_reliability",
) -> int | float | None:
    passing = [
        row[resource_key] for row in rows
        if int(row[K_key]) == int(K) and bool(row[pass_key])
    ]
    return min(passing) if passing else None


def fit_power_law(
    resources: Iterable[int | float], capacities: Iterable[int | float | None],
    *, minimum_points: int = 4, bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit C=aR^alpha on raw uncensored points and bootstrap the exponent."""
    pairs = [
        (float(resource), float(capacity))
        for resource, capacity in zip(resources, capacities)
        if capacity is not None and float(resource) > 0 and float(capacity) > 0
    ]
    if len(pairs) < minimum_points:
        return {
            "fit_performed": False,
            "reason": "fewer_than_minimum_uncensored_points",
            "uncensored_points": len(pairs),
            "minimum_points": minimum_points,
        }
    x = np.log(np.asarray([pair[0] for pair in pairs], dtype=float))
    y = np.log(np.asarray([pair[1] for pair in pairs], dtype=float))
    alpha, intercept = np.polyfit(x, y, 1)
    predicted = intercept + alpha * x
    ss_res = float(np.square(y - predicted).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    generator = np.random.default_rng(seed)
    boot = []
    for _ in range(int(bootstrap_samples)):
        indices = generator.integers(0, len(pairs), size=len(pairs))
        sample_x = x[indices]
        if np.unique(sample_x).size < 2:
            continue
        boot.append(float(np.polyfit(sample_x, y[indices], 1)[0]))
    if not boot:
        lower = upper = None
    else:
        lower, upper = np.quantile(np.asarray(boot), [0.025, 0.975]).tolist()
    return {
        "fit_performed": True,
        "uncensored_points": len(pairs),
        "a": float(np.exp(intercept)),
        "alpha": float(alpha),
        "alpha_bootstrap_95_interval": [lower, upper],
        "r2_log_space": r2,
        "superlinear": bool(lower is not None and lower > 1.0),
        "smoothing_or_censor_imputation_used": False,
    }
