"""Single-owner Spectrum decision boundary around the MiniMax H3 block stack."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .adapter import SpectrumStepAdapter
from .config import SpectrumConfig
from .runtime import SpectrumRuntime, SpectrumStats


class SpectrumController:
    """Own one exact-or-forecast decision for each worker-local H3 call."""

    def __init__(self, config: SpectrumConfig) -> None:
        self.runtime = SpectrumRuntime(config)
        self.adapter = SpectrumStepAdapter(self.runtime)

    def start_run(self, sigmas: torch.Tensor) -> None:
        self.runtime.start_run(sigmas)

    def execute_h3_stack(
        self,
        *,
        timestep: torch.Tensor | float,
        hidden: torch.Tensor,
        topology: tuple[Any, ...],
        exact: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        try:
            decision = self.adapter.begin_step(timestep)
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
        return self.runtime.end_run()

    def abort_run(self, reason: str) -> None:
        self.runtime.abort_run(reason)
