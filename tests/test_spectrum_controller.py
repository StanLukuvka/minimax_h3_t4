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

    def sample_euler():
        return None

    sampler = type("Sampler", (), {"sampler_function": sample_euler, "extra_options": {}})()
    controller.start_run(torch.linspace(1.0, 0.0, 5), sampler=sampler)
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


def test_topology_change_is_detected_before_a_forecast_decision() -> None:
    controller = _controller()
    exact_calls = 0

    def execute(step: float, topology: tuple[int, ...]) -> None:
        nonlocal exact_calls

        def exact() -> torch.Tensor:
            nonlocal exact_calls
            exact_calls += 1
            return torch.zeros((2, 3))

        controller.execute_h3_stack(
            timestep=step,
            hidden=torch.zeros((2, 3)),
            topology=topology,
            exact=exact,
        )

    execute(1.0, (2, 3))
    execute(0.75, (2, 3))
    execute(0.5, (99, 3))

    assert exact_calls == 3
    assert controller.runtime.stats.disabled


def test_noise_or_ancestry_sampler_fails_closed_to_exact_execution() -> None:
    controller = SpectrumController(SpectrumConfig(degree=1, max_history=4, warmup_steps=0, tail_actual_steps=0))

    def sample_res_multistep_ancestral():
        return None

    sampler = type("Sampler", (), {"sampler_function": sample_res_multistep_ancestral, "extra_options": {}})()
    controller.start_run(torch.linspace(1.0, 0.0, 3), sampler=sampler)
    controller.adapter.sync_all = lambda _value: (_ for _ in ()).throw(AssertionError("unexpected Spectrum sync"))
    controller.adapter.sync_mode = lambda _value: (_ for _ in ()).throw(AssertionError("unexpected Spectrum sync"))
    calls = 0

    def exact() -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.zeros((2, 3))

    controller.execute_h3_stack(
        timestep=1.0,
        hidden=torch.zeros((2, 3)),
        topology=(2, 3),
        exact=exact,
    )

    assert calls == 1
    assert controller.runtime.stats.disabled
    assert controller.runtime.stats.disable_reason == "sampler is not safe for Spectrum forecasting"


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
