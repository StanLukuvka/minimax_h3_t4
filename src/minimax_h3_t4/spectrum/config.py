from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpectrumConfig:
    enabled: bool = True
    blend_weight: float = 0.50
    degree: int = 4
    ridge_lambda: float = 0.10
    window_size: float = 2.0
    flex_window: float = 0.75
    warmup_steps: int = 5
    tail_actual_steps: int = 1
    max_history: int = 8
    history_storage: str = "system_ram"
    debug: bool = False

    @property
    def min_fit_points(self) -> int:
        return max(2, self.degree + 1)

    def validate(self) -> SpectrumConfig:
        if not isinstance(self.enabled, bool) or not isinstance(self.debug, bool):
            raise TypeError("enabled and debug must be booleans")
        if not math.isfinite(self.blend_weight) or not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("blend_weight must be finite and in [0, 1]")
        if isinstance(self.degree, bool) or not isinstance(self.degree, int) or self.degree < 1:
            raise ValueError("degree must be an integer >= 1")
        if not math.isfinite(self.ridge_lambda) or self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be finite and >= 0")
        if not math.isfinite(self.window_size) or self.window_size < 1.0:
            raise ValueError("window_size must be finite and >= 1")
        if not math.isfinite(self.flex_window) or self.flex_window < 0.0:
            raise ValueError("flex_window must be finite and >= 0")
        for name, value in (("warmup_steps", self.warmup_steps), ("tail_actual_steps", self.tail_actual_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        if isinstance(self.max_history, bool) or not isinstance(self.max_history, int) or self.max_history < self.min_fit_points:
            raise ValueError(f"max_history must be an integer >= {self.min_fit_points}")
        if self.history_storage != "system_ram":
            raise ValueError("MiniMax H3 T4 Spectrum history_storage must be 'system_ram'")
        return self
