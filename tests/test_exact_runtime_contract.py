from __future__ import annotations

import types

import pytest

from src.minimax_h3_t4.runtime.int8 import call_cuda_int8_with_oom_retry
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


def test_actor_group_releases_workers_and_ray_resources_boundedly() -> None:
    events: list[object] = []

    class Ray:
        @staticmethod
        def wait(refs, *, num_returns, timeout):
            events.append(("wait", tuple(refs), num_returns, timeout))
            return refs, []

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

    workers = [types.SimpleNamespace(shutdown=Method(0)), types.SimpleNamespace(shutdown=Method(1))]
    group = H3T4ActorGroup(workers=workers, ray_module=Ray(), topology=ExactH3T4Topology())

    group.close(timeout_seconds=3.0)

    assert group.alive is False
    assert events == [
        ("shutdown.remote", 0),
        ("shutdown.remote", 1),
        ("wait", ("done-0", "done-1"), 2, 3.0),
        ("kill", workers[0], True),
        ("kill", workers[1], True),
        "shutdown",
    ]
