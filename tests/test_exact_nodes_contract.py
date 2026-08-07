from __future__ import annotations

import types

import pytest

from src.minimax_h3_t4.nodes import H3T4Initializer, H3T4UNETLoader
from src.minimax_h3_t4.runtime.lifecycle import H3T4ActorGroup
from src.minimax_h3_t4.runtime.topology import ExactH3T4Topology
from src.minimax_h3_t4.sampler_nodes import H3T4BasicGuider, H3T4BasicScheduler, H3T4SamplerAdvanced


class RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class FakeRay:
    def __init__(self) -> None:
        self.init_kwargs = None
        self.shutdown_calls = 0

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    @staticmethod
    def get(value):
        return value

    @staticmethod
    def wait(refs, *, num_returns, timeout):
        return refs, []

    @staticmethod
    def kill(_worker, *, no_restart):
        assert no_restart is True

    def shutdown(self):
        self.shutdown_calls += 1


class FakeWorker:
    def __init__(self, rank: int, events: list[object]) -> None:
        self.rank = rank
        self.events = events
        self.load_unet = RemoteMethod(self._load_unet)
        self.get_sigmas = RemoteMethod(lambda scheduler, steps, denoise: [scheduler, steps, denoise])
        self.sample_advanced = RemoteMethod(self._sample)
        self.shutdown = RemoteMethod(lambda: self.events.append(("shutdown", self.rank)))

    def _load_unet(self, path, weight_dtype):
        self.events.append(("load", self.rank, path, weight_dtype))
        return True

    def _sample(self, add_noise, noise, guider, sampler, sigmas, latent):
        self.events.append(("sample", self.rank, add_noise, guider["type"]))
        result = dict(latent)
        result["samples"] = ["sampled-video", "sampled-audio"]
        denoised = dict(latent)
        denoised["samples"] = ["denoised-video", "denoised-audio"]
        return result, denoised


def _group(events: list[object], ray: FakeRay | None = None) -> H3T4ActorGroup:
    ray = ray or FakeRay()
    return H3T4ActorGroup(
        workers=[FakeWorker(0, events), FakeWorker(1, events)],
        ray_module=ray,
        topology=ExactH3T4Topology(),
    )


def test_initializer_contract_is_fixed_and_object_store_is_bounded() -> None:
    required = H3T4Initializer.INPUT_TYPES()["required"]

    assert "GPU" not in required
    assert "ulysses_degree" not in required
    assert "FSDP" not in required
    assert required["ray_object_store_gb"][1]["max"] == 1.0
    assert H3T4Initializer.RETURN_TYPES == ("H3T4_ACTOR_GROUP",)


def test_initializer_releases_conditioning_before_ray_init(monkeypatch) -> None:
    events: list[object] = []
    fake_ray = FakeRay()
    fake_comfy = types.SimpleNamespace(
        unload_all_models=lambda: events.append("unload"),
        soft_empty_cache=lambda: events.append("empty"),
    )

    initializer = H3T4Initializer(
        ray_module=fake_ray,
        model_management=fake_comfy,
        worker_factory=lambda rank, _config: FakeWorker(rank, events),
    )
    monkeypatch.setattr(fake_ray, "init", lambda **kwargs: (events.append("ray.init"), setattr(fake_ray, "init_kwargs", kwargs)))

    (group,) = initializer.initialize(load_after=object(), ray_object_store_gb=0.5)

    assert events[:3] == ["unload", "empty", "ray.init"]
    assert len(group.workers) == 2
    assert fake_ray.init_kwargs["object_store_memory"] == 512 * 1024**2


def test_int8_unet_load_is_strictly_sequential_after_conditioning_release() -> None:
    events: list[object] = []
    group = _group(events)
    model_management = types.SimpleNamespace(
        unload_all_models=lambda: events.append("unload"),
        soft_empty_cache=lambda: events.append("empty"),
    )
    loader = H3T4UNETLoader(
        path_resolver=lambda name: f"/models/{name}",
        model_management=model_management,
    )

    (loaded_group,) = loader.load(group, "minimax_h3_int8.safetensors", load_after=object())

    assert loaded_group is group
    assert events == [
        "unload",
        "empty",
        ("load", 0, "/models/minimax_h3_int8.safetensors", "int8"),
        ("load", 1, "/models/minimax_h3_int8.safetensors", "int8"),
    ]


def test_unet_loader_requires_conditioning_dependency() -> None:
    with pytest.raises(ValueError, match="load_after"):
        H3T4UNETLoader(path_resolver=lambda name: name).load(_group([]), "h3.safetensors", load_after=None)


def test_scheduler_and_basic_guider_keep_stock_comfy_contracts() -> None:
    events: list[object] = []
    group = _group(events)

    assert H3T4BasicScheduler().get_sigmas(group, "simple", 20, 1.0) == (["simple", 20, 1.0],)
    conditioning = [["text", {"pooled_output": "pooled"}]]
    guider = H3T4BasicGuider().get_guider(group, conditioning)[0]
    assert guider == {"group": group, "type": "basic", "positive": conditioning}


def test_advanced_sampler_preserves_av_and_shuts_down_workers() -> None:
    events: list[object] = []
    ray = FakeRay()
    group = _group(events, ray)
    guider = H3T4BasicGuider().get_guider(group, "conditioning")[0]

    output, denoised = H3T4SamplerAdvanced().sample(
        True,
        "noise",
        guider,
        "sampler",
        [1.0, 0.0],
        {"samples": ["video", "audio"]},
    )

    assert output["samples"] == ["sampled-video", "sampled-audio"]
    assert denoised["samples"] == ["denoised-video", "denoised-audio"]
    assert [event for event in events if isinstance(event, tuple) and event[0] == "sample"] == [
        ("sample", 0, True, "basic"),
        ("sample", 1, True, "basic"),
    ]
    assert group.alive is False
    assert ray.shutdown_calls == 1


def test_advanced_sampler_shuts_down_after_worker_failure() -> None:
    events: list[object] = []
    ray = FakeRay()
    group = _group(events, ray)
    group.workers[1].sample_advanced = RemoteMethod(lambda *_args: (_ for _ in ()).throw(RuntimeError("worker failed")))
    guider = H3T4BasicGuider().get_guider(group, "conditioning")[0]

    with pytest.raises(RuntimeError, match="worker failed"):
        H3T4SamplerAdvanced().sample(True, "noise", guider, "sampler", [1.0], {"samples": ["v", "a"]})

    assert group.alive is False
    assert ray.shutdown_calls == 1
