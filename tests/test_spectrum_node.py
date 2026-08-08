from __future__ import annotations

import pytest

from src.minimax_h3_t4.runtime.lifecycle import H3T4ActorGroup
from src.minimax_h3_t4.runtime.topology import ExactH3T4Topology
from src.minimax_h3_t4.spectrum.config import SpectrumConfig
from src.minimax_h3_t4.spectrum_node import H3T4Spectrum


class RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class FakeRay:
    @staticmethod
    def get(value):
        return value


class FakeWorker:
    def __init__(self, rank: int, configured: list[tuple[int, dict]]) -> None:
        self.configure_spectrum = RemoteMethod(lambda config: configured.append((rank, config)))


def test_spectrum_node_configures_both_workers_with_system_ram_history() -> None:
    configured: list[tuple[int, dict]] = []
    group = H3T4ActorGroup(
        workers=[FakeWorker(0, configured), FakeWorker(1, configured)],
        ray_module=FakeRay(),
        topology=ExactH3T4Topology(),
    )

    (result,) = H3T4Spectrum().configure(
        group,
        enabled=True,
        blend_weight=0.5,
        degree=4,
        ridge_lambda=0.1,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=5,
        tail_actual_steps=1,
        max_history=8,
        debug=False,
    )

    assert result is group
    assert [rank for rank, _config in configured] == [0, 1]
    assert all(config["history_storage"] == "system_ram" for _rank, config in configured)


def test_disabled_spectrum_is_a_true_node_bypass_before_validation_or_rpc() -> None:
    configured: list[tuple[int, dict]] = []
    group = H3T4ActorGroup(
        workers=[FakeWorker(0, configured), FakeWorker(1, configured)],
        ray_module=FakeRay(),
        topology=ExactH3T4Topology(),
    )

    (result,) = H3T4Spectrum().configure(
        group,
        enabled=False,
        blend_weight=float("nan"),
        degree=0,
        ridge_lambda=-1.0,
        window_size=0.0,
        flex_window=-1.0,
        warmup_steps=-1,
        tail_actual_steps=-1,
        max_history=0,
        debug=False,
    )

    assert result is group
    assert configured == []


def test_spectrum_configuration_polls_parent_interrupt_checker() -> None:
    configured: list[tuple[int, dict]] = []
    group = H3T4ActorGroup(
        workers=[FakeWorker(0, configured), FakeWorker(1, configured)],
        ray_module=FakeRay(),
        topology=ExactH3T4Topology(),
    )
    checks = 0

    class Interrupted(RuntimeError):
        pass

    def check_interrupt() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise Interrupted("cancelled during Spectrum setup")

    with pytest.raises(Interrupted, match="cancelled during Spectrum setup"):
        H3T4Spectrum().configure(
            group,
            enabled=True,
            blend_weight=0.5,
            degree=4,
            ridge_lambda=0.1,
            window_size=2.0,
            flex_window=0.75,
            warmup_steps=5,
            tail_actual_steps=3,
            max_history=8,
            debug=False,
            interrupt_checker=check_interrupt,
        )


def test_spectrum_config_rejects_vram_history() -> None:
    try:
        SpectrumConfig(history_storage="vram").validate()
    except ValueError as exc:
        assert "system_ram" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("VRAM history must not be accepted on two T4s")
