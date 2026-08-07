from __future__ import annotations

import pytest
import torch

from src.minimax_h3_t4.spectrum.config import SpectrumConfig
from src.minimax_h3_t4.spectrum.forecast import HistoryWeightForecaster
from src.minimax_h3_t4.spectrum.runtime import SpectrumRuntime


def test_config_rejects_history_smaller_than_polynomial_fit() -> None:
    with pytest.raises(ValueError, match="max_history"):
        SpectrumConfig(degree=4, max_history=4).validate()


def test_forecaster_predicts_linear_feature_without_large_coefficients() -> None:
    forecaster = HistoryWeightForecaster(
        degree=2,
        ridge_lambda=0.0,
        max_history=4,
        chunk_bytes=4096,
        history_storage="system_ram",
    )
    for coordinate in (-1.0, -0.5, 0.0):
        feature = torch.full((1, 2, 3), 2.0 * coordinate + 3.0, dtype=torch.bfloat16)
        forecaster.update(coordinate, feature)

    predicted = forecaster.predict(0.5, blend_weight=1.0, dtype=torch.bfloat16)

    assert predicted.shape == (1, 2, 3)
    assert torch.allclose(predicted.float(), torch.full((1, 2, 3), 4.0), atol=0.03)
    assert forecaster.last_prediction_max_fp32_elements <= 1024


def test_runtime_uses_warmup_forecast_recompute_and_actual_tail() -> None:
    config = SpectrumConfig(
        degree=1,
        max_history=4,
        warmup_steps=2,
        tail_actual_steps=1,
        window_size=2.0,
        flex_window=0.0,
    ).validate()
    runtime = SpectrumRuntime(config)
    runtime.start_run(torch.linspace(1.0, 0.0, 7))

    modes = []
    for step, timestep in enumerate(torch.linspace(1.0, 1.0 / 6.0, 6)):
        decision = runtime.begin_step(timestep)
        modes.append(decision.actual)
        if decision.actual:
            feature = torch.full((1, 2, 2), float(step), dtype=torch.float32)
            runtime.observe_actual(feature, topology=(2, 2))
        else:
            prediction = runtime.predict(device=torch.device("cpu"), dtype=torch.float32)
            assert prediction.shape == (1, 2, 2)
        runtime.finish_step()

    assert modes == [True, True, False, True, False, True]
    assert runtime.stats.actual_steps == 4
    assert runtime.stats.forecast_steps == 2
