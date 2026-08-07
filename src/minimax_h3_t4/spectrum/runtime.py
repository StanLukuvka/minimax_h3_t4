"""Native per-worker Spectrum policy for MiniMax H3.

The policy is derived from ComfyUI-Spectrum-MiniMax-H3 commit
85ec1da66277e893079ecd46e32cc865c56cfe53 (GPL-3.0-or-later), but owns
H3 calls directly instead of installing ComfyUI sampler/model wrappers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .config import SpectrumConfig
from .forecast import HistoryWeightForecaster
from .sampling import SpectrumSamplerPolicy


@dataclass(frozen=True, slots=True)
class StepDecision:
    step_id: int
    coordinate: float
    actual: bool
    reason: str


@dataclass(slots=True)
class SpectrumStats:
    total_steps: int = 0
    actual_steps: int = 0
    forecast_steps: int = 0
    disabled: bool = False
    disable_reason: str | None = None


class SpectrumRuntime:
    def __init__(self, config: SpectrumConfig) -> None:
        self.config = config.validate()
        self.forecaster = HistoryWeightForecaster(
            degree=config.degree,
            ridge_lambda=config.ridge_lambda,
            max_history=config.max_history,
            history_storage=config.history_storage,
        )
        self.stats = SpectrumStats()
        self._active = False
        self._step: StepDecision | None = None
        self._next_step = 0
        self._sigma_min = 0.0
        self._sigma_max = 1.0
        self._current_window = float(config.window_size)
        self._consecutive_forecasts = 0
        self._tail_actual_steps = int(config.tail_actual_steps)
        self._maximum_consecutive_forecasts = 1
        self._topology: tuple[Any, ...] | None = None

    @property
    def current_step(self) -> StepDecision | None:
        return self._step

    def start_run(self, sigmas: torch.Tensor) -> None:
        values = torch.as_tensor(sigmas, device="cpu", dtype=torch.float64).reshape(-1)
        if values.numel() < 2 or not bool(torch.isfinite(values).all().item()):
            raise ValueError("Spectrum requires a finite sigma schedule with at least two entries")
        self.forecaster.reset()
        self.stats = SpectrumStats(total_steps=int(values.numel() - 1))
        self._active = True
        self._step = None
        self._next_step = 0
        self._sigma_min = float(values.min().item())
        self._sigma_max = float(values.max().item())
        self._current_window = float(self.config.window_size)
        self._consecutive_forecasts = 0
        self._tail_actual_steps = int(self.config.tail_actual_steps)
        self._maximum_consecutive_forecasts = 1
        self._topology = None

    def apply_sampler_policy(self, policy: SpectrumSamplerPolicy) -> None:
        if not self._active or self._step is not None:
            raise RuntimeError("sampler policy must be applied before the first Spectrum step")
        self._tail_actual_steps = max(int(self.config.tail_actual_steps), policy.minimum_tail_actual_steps)
        self._maximum_consecutive_forecasts = int(policy.maximum_consecutive_forecasts)

    def end_run(self) -> SpectrumStats:
        if self._step is not None:
            raise RuntimeError("cannot end Spectrum run with an unfinished step")
        self._active = False
        self.forecaster.reset()
        self._topology = None
        return self.stats

    def abort_run(self, reason: str) -> None:
        self._active = False
        self._step = None
        self.stats.disabled = True
        self.stats.disable_reason = str(reason)
        self.forecaster.reset()
        self._topology = None

    def disable_for_run(self, reason: str) -> None:
        self.stats.disabled = True
        self.stats.disable_reason = str(reason)
        self.forecaster.reset()
        self._topology = None

    def check_topology(self, topology: tuple[Any, ...]) -> bool:
        normalized = tuple(topology)
        if self._topology is None:
            return True
        if normalized == self._topology:
            return True
        self.disable_for_run("packed H3 topology changed")
        return False

    def fail_closed_step(self, timestep: torch.Tensor | float, reason: str) -> StepDecision:
        if not self._active:
            raise RuntimeError("Spectrum runtime is outside a sampling run")
        if self._step is not None:
            self.force_actual(reason)
            return self._step
        if self._next_step >= self.stats.total_steps:
            raise RuntimeError("H3 call count exceeded the supplied sigma schedule")
        try:
            coordinate = self._coordinate(timestep)
        except (RuntimeError, ValueError, TypeError):
            coordinate = 0.0
        self.disable_for_run(reason)
        self._step = StepDecision(self._next_step, coordinate, True, str(reason))
        self._next_step += 1
        return self._step

    def _coordinate(self, timestep: torch.Tensor | float) -> float:
        value = torch.as_tensor(timestep, device="cpu", dtype=torch.float64).reshape(-1)
        if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
            raise ValueError("Spectrum timestep must be finite")
        if not bool(torch.allclose(value, value[0].expand_as(value))):
            raise ValueError("Spectrum step contains multiple timesteps")
        span = self._sigma_max - self._sigma_min
        if not math.isfinite(span) or span <= 0.0:
            return 0.0
        coordinate = 2.0 * (float(value[0].item()) - self._sigma_min) / span - 1.0
        return max(-1.0, min(1.0, coordinate))

    def begin_step(self, timestep: torch.Tensor | float) -> StepDecision:
        if not self._active:
            raise RuntimeError("Spectrum runtime is outside a sampling run")
        if self._step is not None:
            raise RuntimeError("previous Spectrum step was not finished")
        if self._next_step >= self.stats.total_steps:
            raise RuntimeError("H3 call count exceeded the supplied sigma schedule")
        step_id = self._next_step
        tail_start = max(0, self.stats.total_steps - self._tail_actual_steps)
        if not self.config.enabled:
            actual, reason = True, "disabled"
        elif self.stats.disabled:
            actual, reason = True, self.stats.disable_reason or "disabled after fallback"
        elif step_id < self.config.warmup_steps:
            actual, reason = True, "warmup"
        elif step_id >= tail_start:
            actual, reason = True, "final actual tail"
        elif not self.forecaster.ready(self.config.min_fit_points):
            actual, reason = True, "insufficient actual history"
        else:
            interval = max(1, math.floor(self._current_window))
            actual = self._consecutive_forecasts >= self._maximum_consecutive_forecasts
            if not actual:
                actual = ((self._consecutive_forecasts + 1) % interval) == 0
            reason = "adaptive recompute" if actual else "adaptive forecast"
        self._step = StepDecision(step_id, self._coordinate(timestep), actual, reason)
        self._next_step += 1
        return self._step

    def force_actual(self, reason: str) -> None:
        if self._step is None:
            raise RuntimeError("no active Spectrum step")
        self.stats.disabled = True
        self.stats.disable_reason = str(reason)
        self.forecaster.reset()
        self._topology = None
        self._step = StepDecision(self._step.step_id, self._step.coordinate, True, str(reason))

    def observe_actual(self, feature: torch.Tensor, *, topology: tuple[Any, ...]) -> None:
        if self._step is None or not self._step.actual:
            raise RuntimeError("actual feature observed outside an actual Spectrum step")
        normalized = tuple(topology)
        if self._topology is None:
            self._topology = normalized
        elif normalized != self._topology:
            self.force_actual("packed H3 topology changed")
            self._topology = normalized
        self.forecaster.update(self._step.coordinate, feature)

    def predict(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._step is None or self._step.actual:
            raise RuntimeError("forecast requested outside a forecast Spectrum step")
        return self.forecaster.predict(
            self._step.coordinate,
            self.config.blend_weight,
            device=device,
            dtype=dtype,
        )

    def finish_step(self) -> None:
        if self._step is None:
            raise RuntimeError("no active Spectrum step")
        if self._step.actual:
            self.stats.actual_steps += 1
            self._consecutive_forecasts = 0
            if self._step.reason == "adaptive recompute":
                ceiling = max(float(self.config.window_size), float(self.config.max_history))
                self._current_window = min(self._current_window + self.config.flex_window, ceiling)
        else:
            self.stats.forecast_steps += 1
            self._consecutive_forecasts += 1
        self._step = None
