"""Small common boundary for the historical and SLAYER model backends."""

from __future__ import annotations

from typing import Any

import torch

from utils.resource_ledger import static_resource_ledger


class ModelBackendAdapter:
    """Normalize backend identity and the forward/resource return contract."""

    VALID_BACKENDS = {"current", "slayer_native"}

    def __init__(self, model: Any, model_backend: str | None = None) -> None:
        self.model = model
        self.model_backend = model_backend or getattr(model, "model_backend", "current")
        if self.model_backend not in self.VALID_BACKENDS:
            raise ValueError(f"unknown model_backend: {self.model_backend}")

    def forward(
        self, spike_input: torch.Tensor, **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, int | float | str]]:
        logits, diagnostics = self.model(spike_input, **kwargs)
        diagnostics = dict(diagnostics)
        diagnostics["model_backend"] = self.model_backend
        return logits, diagnostics, static_resource_ledger(self.model)
