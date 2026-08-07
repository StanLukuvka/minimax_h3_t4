from __future__ import annotations

import math

import pytest
import torch

from src.minimax_h3_t4.spectrum.config import SpectrumConfig
from src.minimax_h3_t4.spectrum.runtime import SpectrumRuntime
from src.minimax_h3_t4.spectrum.sampling import spectrum_sampler_policy


def _sampler(name: str, options: object | None = None):
    def sampler_function():
        return None

    sampler_function.__name__ = name
    return type(
        "Sampler",
        (),
        {"sampler_function": sampler_function, "extra_options": {} if options is None else options},
    )()


@pytest.mark.parametrize(
    "name,minimum_tail",
    [
        ("sample_euler", 0),
        ("sample_res_multistep", 3),
        ("sample_res_multistep_cfg_pp", 3),
    ],
)
def test_exact_frozen_sampler_allowlist(name: str, minimum_tail: int) -> None:
    policy = spectrum_sampler_policy(_sampler(name))
    assert policy is not None
    assert policy.minimum_tail_actual_steps == minimum_tail
    assert policy.maximum_consecutive_forecasts == 1


@pytest.mark.parametrize(
    "name",
    [
        "sample_euler_ancestral",
        "sample_euler_ancestral_RF",
        "sample_euler_ancestral_cfg_pp",
        "sample_res_multistep_ancestral",
        "sample_res_multistep_ancestral_cfg_pp",
        "sample_heun",
        "sample_res_multistep_alias",
        "",
    ],
)
def test_ancestral_unknown_and_alias_samplers_bypass(name: str) -> None:
    assert spectrum_sampler_policy(_sampler(name)) is None


@pytest.mark.parametrize("churn", [0.1, "invalid", math.inf, math.nan])
def test_euler_noise_or_invalid_churn_bypasses(churn: object) -> None:
    assert spectrum_sampler_policy(_sampler("sample_euler", {"s_churn": churn})) is None


@pytest.mark.parametrize("churn", [None, 0.0, -0.1])
def test_euler_without_positive_churn_is_admitted(churn: float | None) -> None:
    options = {} if churn is None else {"s_churn": churn}
    assert spectrum_sampler_policy(_sampler("sample_euler", options)) is not None


def _schedule(name: str, *, configured_tail: int) -> list[bool]:
    runtime = SpectrumRuntime(
        SpectrumConfig(
            degree=1,
            max_history=4,
            warmup_steps=0,
            tail_actual_steps=configured_tail,
            window_size=8.0,
            flex_window=0.0,
        )
    )
    runtime.start_run(torch.linspace(1.0, 0.0, 9))
    policy = spectrum_sampler_policy(_sampler(name))
    assert policy is not None
    runtime.apply_sampler_policy(policy)

    decisions: list[bool] = []
    for timestep in torch.linspace(1.0, 0.125, 8):
        decision = runtime.begin_step(timestep)
        decisions.append(decision.actual)
        if decision.actual:
            runtime.observe_actual(torch.ones((2, 3)), topology=(2, 3, "packed"))
        runtime.finish_step()
    return decisions


def test_res_policy_has_no_adjacent_forecasts_and_three_actual_tail_steps() -> None:
    decisions = _schedule("sample_res_multistep", configured_tail=0)
    assert decisions[-3:] == [True, True, True]
    assert all(current or following for current, following in zip(decisions, decisions[1:], strict=False))


def test_euler_policy_preserves_user_configured_tail() -> None:
    decisions = _schedule("sample_euler", configured_tail=1)
    assert decisions[-1]
    assert not decisions[-2]
