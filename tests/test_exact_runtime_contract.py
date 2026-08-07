from __future__ import annotations

import types

import pytest

from src.minimax_h3_t4.runtime.int8 import (
    _bounded_eager_int8_linear,
    call_cuda_int8_with_oom_retry,
    require_tesla_t4,
    run_with_cuda_int8_backend,
)
from src.minimax_h3_t4.runtime.lifecycle import H3T4ActorGroup
from src.minimax_h3_t4.runtime.topology import ExactH3T4Topology
from src.minimax_h3_t4.runtime.ulysses import inject_minimax_h3_ulysses


def test_topology_is_the_exact_two_t4_slice() -> None:
    topology = ExactH3T4Topology()

    assert topology.as_worker_config() == {
        "world_size": 2,
        "ulysses_degree": 2,
        "ring_degree": 1,
        "cfg_degree": 1,
        "dp_degree": 1,
        "fsdp": True,
        "fsdp_cpu_offload": False,
        "attention": "TORCH_EFFICIENT",
    }


class _Cuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _Torch:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self) -> None:
        self.cuda = _Cuda()


def test_cuda_int8_retries_exactly_once_after_emptying_cache() -> None:
    torch = _Torch()
    attempts = 0

    def operation(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise torch.OutOfMemoryError("fragmented")
        return kwargs["x"]

    result = call_cuda_int8_with_oom_retry(operation, torch_module=torch, x="result")

    assert result == "result"
    assert attempts == 2
    assert torch.cuda.empty_cache_calls == 1


def test_cuda_int8_does_not_retry_a_second_oom() -> None:
    torch = _Torch()
    attempts = 0

    def operation(**kwargs):
        nonlocal attempts
        attempts += 1
        raise torch.OutOfMemoryError("still full")

    with pytest.raises(torch.OutOfMemoryError, match="still full"):
        call_cuda_int8_with_oom_retry(operation, torch_module=torch, x=None)

    assert attempts == 2
    assert torch.cuda.empty_cache_calls == 1


def test_int8_gateway_rejects_non_cuda_backend(monkeypatch) -> None:
    monkeypatch.setenv("H3_T4_INT8_BACKEND", "eager")
    with pytest.raises(ValueError, match="CUDA-only"):
        _bounded_eager_int8_linear(None, None, None)


def test_cuda_int8_inference_is_scoped_through_eager_dispatch_gateway() -> None:
    events = []

    class Backend:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    kitchen = types.SimpleNamespace(use_backend=lambda name: (events.append(("backend", name)), Backend())[1])
    result = run_with_cuda_int8_backend(lambda: events.append("sample") or "result", kitchen_module=kitchen)

    assert result == "result"
    assert events == [("backend", "eager"), "enter", "sample", "exit"]


def test_worker_requires_real_tesla_t4_capability() -> None:
    good = types.SimpleNamespace(name="Tesla T4", major=7, minor=5)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_properties=lambda _device: good))
    require_tesla_t4(fake_torch, 0)

    bad = types.SimpleNamespace(name="NVIDIA A10", major=8, minor=6)
    fake_torch.cuda.get_device_properties = lambda _device: bad
    with pytest.raises(RuntimeError, match="Tesla T4"):
        require_tesla_t4(fake_torch, 0)


def test_h3_only_ulysses_injection_patches_attention_and_forward() -> None:
    native_forward = lambda *_args, **_kwargs: "native"  # noqa: E731
    block = types.SimpleNamespace(attn=types.SimpleNamespace(forward=native_forward))
    diffusion = types.SimpleNamespace(blocks=[block], _forward=native_forward)
    base_model = types.SimpleNamespace(diffusion_model=diffusion)
    patcher = types.SimpleNamespace(model=base_model)

    inject_minimax_h3_ulysses(
        patcher,
        minimax_h3_class=type(base_model),
        attention_forward=lambda *_args, **_kwargs: "attention",
        dit_forward=lambda *_args, **_kwargs: ["video", "audio"],
    )

    assert block.attn.forward() == "attention"
    assert diffusion._forward() == ["video", "audio"]


def test_h3_only_ulysses_injection_rejects_other_models() -> None:
    with pytest.raises(TypeError, match="MiniMaxH3"):
        inject_minimax_h3_ulysses(
            types.SimpleNamespace(model=object()),
            minimax_h3_class=types.SimpleNamespace,
            attention_forward=lambda: None,
            dit_forward=lambda: None,
        )


def test_actor_group_refuses_success_when_worker_death_is_unconfirmed() -> None:
    class Method:
        def __init__(self, value):
            self.value = value

        def remote(self):
            return self.value

    worker = types.SimpleNamespace(shutdown=Method("shutdown-ref"), __ray_ready__=Method("ready-ref"))

    class Ray:
        class exceptions:
            class RayActorError(RuntimeError):
                pass

        @staticmethod
        def wait(refs, *, num_returns, timeout):
            return refs, []

        @staticmethod
        def get(ref):
            return True

        @staticmethod
        def kill(_worker, *, no_restart):
            return None

        @staticmethod
        def shutdown():
            raise AssertionError("Ray shutdown must not run before actor death is confirmed")

    group = H3T4ActorGroup([worker], Ray(), ExactH3T4Topology())
    with pytest.raises(RuntimeError, match="could not be confirmed"):
        group.close(timeout_seconds=0.05)
    assert group.alive is True


def test_confirmed_force_kill_still_surfaces_graceful_shutdown_failure() -> None:
    events: list[str] = []

    class Ray:
        class exceptions:
            class RayActorError(RuntimeError):
                pass

        @staticmethod
        def wait(refs, *, num_returns, timeout):
            return refs, []

        @staticmethod
        def get(ref):
            if ref == "ready":
                raise Ray.exceptions.RayActorError("actor exited")
            return True

        @staticmethod
        def kill(_worker, *, no_restart):
            events.append("kill")

        @staticmethod
        def shutdown():
            events.append("shutdown")

    class Method:
        def __init__(self, ref: str) -> None:
            self.ref = ref

        def remote(self):
            return self.ref

    worker = types.SimpleNamespace(shutdown=Method("graceful"), __ray_ready__=Method("ready"))
    group = H3T4ActorGroup([worker], Ray(), ExactH3T4Topology())

    with pytest.raises(RuntimeError, match="returned without terminating"):
        group.close(timeout_seconds=3.0)

    assert group.alive is False
    assert events == ["kill", "shutdown"]


def test_actor_group_releases_workers_and_ray_resources_boundedly() -> None:
    events: list[object] = []

    class Ray:
        class exceptions:
            class RayActorError(RuntimeError):
                pass

        @staticmethod
        def wait(refs, *, num_returns, timeout):
            events.append(("wait", tuple(refs), num_returns, timeout))
            return refs, []

        @staticmethod
        def get(ref):
            events.append(("get", ref))
            raise Ray.exceptions.RayActorError("actor exited")

        @staticmethod
        def kill(worker, *, no_restart):
            events.append(("kill", worker, no_restart))

        @staticmethod
        def shutdown():
            events.append("shutdown")

    class Method:
        def __init__(self, rank):
            self.rank = rank

        def remote(self):
            events.append(("shutdown.remote", self.rank))
            return f"done-{self.rank}"

    workers = [types.SimpleNamespace(shutdown=Method(rank), __ray_ready__=Method(f"ready-{rank}")) for rank in range(2)]
    group = H3T4ActorGroup(workers=workers, ray_module=Ray(), topology=ExactH3T4Topology())

    group.close(timeout_seconds=3.0)

    assert group.alive is False
    assert events[0:2] == [("shutdown.remote", 0), ("shutdown.remote", 1)]
    assert events[2][0:3] == ("wait", ("done-0", "done-1"), 2)
    assert ("get", "done-0") in events
    assert ("get", "done-1") in events
    assert ("shutdown.remote", "ready-0") in events
    assert ("shutdown.remote", "ready-1") in events
    assert not any(isinstance(event, tuple) and event[0] == "kill" for event in events)
    assert events[-1] == "shutdown"
