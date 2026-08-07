"""Single-owner Spectrum decision boundary around the MiniMax H3 block stack."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .adapter import SpectrumStepAdapter
from .config import SpectrumConfig
from .runtime import SpectrumRuntime, SpectrumStats


def spectrum_sampler_is_safe(sampler: Any) -> bool:
    function = getattr(sampler, "sampler_function", None)
    name = getattr(function, "__name__", "")
    options = getattr(sampler, "extra_options", {})
    return name == "sample_res_multistep" and isinstance(options, dict) and "noise_sampler" not in options


class SpectrumController:
    """Own one exact-or-forecast decision for each worker-local H3 call."""

    def __init__(self, config: SpectrumConfig) -> None:
        self.runtime = SpectrumRuntime(config)
        self.adapter = SpectrumStepAdapter(self.runtime)

    def start_run(self, sigmas: torch.Tensor, *, sampler: Any | None = None) -> None:
        valid = True
        try:
            self.runtime.start_run(sigmas)
        except (RuntimeError, ValueError, TypeError):
            valid = False
        if not self.adapter.sync_all(valid):
            self.runtime.abort_run("Spectrum run initialization failed on one or more ranks")
            raise RuntimeError("Spectrum run initialization failed on one or more ranks")
        if sampler is not None and not self.adapter.sync_all(spectrum_sampler_is_safe(sampler)):
            self.runtime.disable_for_run("sampler is not safe for Spectrum forecasting")

    def execute_h3_stack(
        self,
        *,
        timestep: torch.Tensor | float,
        hidden: torch.Tensor,
        topology: tuple[Any, ...],
        exact: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        try:
            decision = self.adapter.begin_step(timestep, topology=topology)
            if not decision.actual:
                predicted = self.adapter.try_forecast(
                    expected_shape=tuple(hidden.shape),
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                if predicted is not None:
                    self.adapter.finish_step()
                    return predicted

            actual = exact()
            if not isinstance(actual, torch.Tensor) or tuple(actual.shape) != tuple(hidden.shape):
                raise RuntimeError("exact H3 block stack returned an invalid hidden-state tensor")
            self.adapter.observe_actual(actual, topology=topology)
            self.adapter.finish_step()
            return actual
        except Exception as exc:
            self.runtime.abort_run(f"{type(exc).__name__}: {exc}")
            raise

    def end_run(self) -> SpectrumStats:
        stats: SpectrumStats | None = None
        valid = True
        try:
            stats = self.runtime.end_run()
        except (RuntimeError, ValueError, TypeError):
            valid = False
        if not self.adapter.sync_all(valid):
            self.runtime.abort_run("Spectrum run completion failed on one or more ranks")
            raise RuntimeError("Spectrum run completion failed on one or more ranks")
        if stats is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("Spectrum run produced no statistics")
        return stats

    def abort_run(self, reason: str) -> None:
        self.runtime.abort_run(reason)
