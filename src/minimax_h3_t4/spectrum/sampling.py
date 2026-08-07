"""Sampler admission policy for worker-local Spectrum forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SpectrumSamplerPolicy:
    name: str
    minimum_tail_actual_steps: int
    maximum_consecutive_forecasts: int = 1


_SUPPORTED = {
    "sample_euler": SpectrumSamplerPolicy("sample_euler", minimum_tail_actual_steps=0),
    "sample_res_multistep": SpectrumSamplerPolicy("sample_res_multistep", minimum_tail_actual_steps=3),
    "sample_res_multistep_cfg_pp": SpectrumSamplerPolicy(
        "sample_res_multistep_cfg_pp",
        minimum_tail_actual_steps=3,
    ),
}


def spectrum_sampler_policy(sampler: Any) -> SpectrumSamplerPolicy | None:
    """Return the frozen safe policy, or ``None`` for exact-only execution."""
    function = getattr(sampler, "sampler_function", None)
    name = getattr(function, "__name__", "")
    policy = _SUPPORTED.get(name)
    if policy is None:
        return None

    options = getattr(sampler, "extra_options", {})
    if not isinstance(options, dict):
        return None
    if name == "sample_euler":
        try:
            churn = float(options.get("s_churn", 0.0))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(churn) or churn > 0.0:
            return None
    return policy
