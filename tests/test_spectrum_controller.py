from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.minimax_h3_t4.spectrum.config import SpectrumConfig
from src.minimax_h3_t4.spectrum.controller import SpectrumController


def _controller() -> SpectrumController:
    controller = SpectrumController(
        SpectrumConfig(
            degree=1,
            max_history=4,
            warmup_steps=2,
            tail_actual_steps=0,
            window_size=8.0,
            flex_window=0.0,
        )
    )
    controller.start_run(torch.linspace(1.0, 0.0, 5))
    return controller


def test_controller_owns_exact_or_forecast_for_each_h3_call() -> None:
    controller = _controller()
    exact_calls = 0

    def run(timestep: float, value: float) -> torch.Tensor:
        nonlocal exact_calls

        def exact() -> torch.Tensor:
            nonlocal exact_calls
            exact_calls += 1
            return torch.full((2, 3), value)

        return controller.execute_h3_stack(
            timestep=timestep,
            hidden=torch.zeros((2, 3)),
            topology=(2, 3, "packed-h3"),
            exact=exact,
        )

    run(1.0, 1.0)
    run(0.75, 2.0)
    forecast = run(0.5, 99.0)

    assert exact_calls == 2
    assert forecast.shape == (2, 3)
    assert controller.runtime.stats.forecast_steps == 1


def test_controller_aborts_and_clears_history_when_exact_h3_fails() -> None:
    controller = _controller()

    with pytest.raises(RuntimeError, match="block failure"):
        controller.execute_h3_stack(
            timestep=1.0,
            hidden=torch.zeros((2, 3)),
            topology=(2, 3),
            exact=lambda: (_ for _ in ()).throw(RuntimeError("block failure")),
        )

    assert controller.runtime.stats.disabled
    assert controller.runtime.forecaster.history_length == 0


def test_h3_forward_wraps_only_the_worker_local_block_stack() -> None:
    source = (Path(__file__).parents[1] / "src" / "minimax_h3_t4" / "runtime" / "h3_forward.py").read_text()

    decision = source.index("spectrum.execute_h3_stack")
    gather = source.index("get_sp_group().all_gather", decision)
    final_layer = source.index("self.final_layer", gather)
    assert decision < gather < final_layer
