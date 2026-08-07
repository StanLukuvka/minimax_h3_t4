from __future__ import annotations

import torch

from src.minimax_h3_t4.spectrum.config import SpectrumConfig
from src.minimax_h3_t4.spectrum.adapter import SpectrumStepAdapter, sanitize_prediction_bounded
from src.minimax_h3_t4.spectrum.runtime import SpectrumRuntime


def _forecast_ready_runtime() -> SpectrumRuntime:
    runtime = SpectrumRuntime(
        SpectrumConfig(
            degree=1,
            max_history=4,
            warmup_steps=2,
            tail_actual_steps=0,
            window_size=8.0,
            flex_window=0.0,
        )
    )
    runtime.start_run(torch.linspace(1.0, 0.0, 5))
    for timestep, value in ((1.0, 1.0), (0.75, 2.0)):
        assert runtime.begin_step(timestep).actual
        runtime.observe_actual(torch.full((1, 2, 3), value), topology=(2, 3))
        runtime.finish_step()
    return runtime


def test_adapter_fails_closed_to_exact_when_any_rank_rejects_forecast() -> None:
    runtime = _forecast_ready_runtime()
    sync_results = iter((True, True, False))
    adapter = SpectrumStepAdapter(
        runtime,
        sync_mode=lambda _actual: False,
        sync_all=lambda _valid: next(sync_results),
    )

    decision = adapter.begin_step(0.5, topology=(2, 3))
    assert not decision.actual
    assert adapter.try_forecast(expected_shape=(1, 2, 3), device=torch.device("cpu"), dtype=torch.float32) is None
    assert runtime.current_step is not None
    assert runtime.current_step.actual
    assert runtime.stats.disabled


def test_sanitizer_rejects_nonfinite_forecasts_without_full_fp32_clone() -> None:
    forecast = torch.zeros((1, 1024, 1024), dtype=torch.bfloat16)
    forecast[0, -1, -1] = float("nan")

    result = sanitize_prediction_bounded(
        forecast,
        expected_shape=tuple(forecast.shape),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        chunk_bytes=4 * 1024 * 1024,
    )

    assert result is None


def test_adapter_returns_valid_prediction_when_all_ranks_agree() -> None:
    runtime = _forecast_ready_runtime()
    adapter = SpectrumStepAdapter(
        runtime,
        sync_mode=lambda _actual: False,
        sync_all=lambda _valid: True,
    )

    assert not adapter.begin_step(0.5, topology=(2, 3)).actual
    predicted = adapter.try_forecast(expected_shape=(1, 2, 3), device=torch.device("cpu"), dtype=torch.bfloat16)

    assert predicted is not None
    assert predicted.shape == (1, 2, 3)
    adapter.finish_step()
    assert runtime.stats.forecast_steps == 1


def test_adapter_promotes_forecast_to_exact_when_another_rank_requires_actual() -> None:
    runtime = _forecast_ready_runtime()
    adapter = SpectrumStepAdapter(
        runtime,
        sync_mode=lambda _local_actual: True,
        sync_all=lambda value: value,
    )

    decision = adapter.begin_step(0.5, topology=(2, 3))

    assert decision.actual
    assert runtime.stats.disabled
    assert adapter.try_forecast(expected_shape=(1, 2, 3), device=torch.device("cpu"), dtype=torch.float32) is None
