from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.minimax_h3_t4 import nodes


def test_public_surface_is_exactly_loader_and_sampler() -> None:
    assert set(nodes.NODE_CLASS_MAPPINGS) == {"H3T4Loader", "H3T4Sampler"}


def test_loader_hides_distributed_settings() -> None:
    loader = nodes.NODE_CLASS_MAPPINGS["H3T4Loader"]
    required = loader.INPUT_TYPES()["required"]

    assert set(required) == {"unet_name", "acceleration", "load_after"}
    assert required["acceleration"][0] == ("Exact", "Spectrum")


def test_sampler_owns_standard_sampling_controls() -> None:
    sampler = nodes.NODE_CLASS_MAPPINGS["H3T4Sampler"]
    required = sampler.INPUT_TYPES()["required"]

    assert set(required) == {
        "model",
        "conditioning",
        "latent_image",
        "seed",
        "sampler_name",
        "scheduler",
        "steps",
        "denoise",
        "add_noise",
    }
    assert sampler.RETURN_TYPES == ("LATENT", "LATENT")


@dataclass
class FakeGroup:
    alive: bool = True
    closed: int = 0

    def close(self, timeout_seconds: float = 30.0) -> None:
        self.closed += 1
        self.alive = False


class FakeInitializer:
    def __init__(self, group: FakeGroup, calls: list[tuple]) -> None:
        self.group = group
        self.calls = calls

    def initialize(self, *, load_after):
        self.calls.append(("initialize", load_after))
        return (self.group,)


class FakeUNETLoader:
    def __init__(self, calls: list[tuple], failure: Exception | None = None) -> None:
        self.calls = calls
        self.failure = failure

    def load(self, group, unet_name, load_after):
        self.calls.append(("load", group, unet_name, load_after))
        if self.failure is not None:
            raise self.failure
        return (group,)


class FakeSpectrum:
    def __init__(self, calls: list[tuple], failure: Exception | None = None) -> None:
        self.calls = calls
        self.failure = failure

    def configure(self, model, **settings):
        self.calls.append(("spectrum", model, settings))
        if self.failure is not None:
            raise self.failure
        return (model,)


def test_loader_executes_hidden_pipeline_with_frozen_spectrum_profile() -> None:
    calls: list[tuple] = []
    group = FakeGroup()
    loader = nodes.H3T4Loader(
        initializer=FakeInitializer(group, calls),
        unet_loader=FakeUNETLoader(calls),
        spectrum=FakeSpectrum(calls),
    )
    conditioning = object()

    assert loader.load("h3.safetensors", "Spectrum", conditioning) == (group,)
    assert calls[:2] == [
        ("initialize", conditioning),
        ("load", group, "h3.safetensors", conditioning),
    ]
    settings = calls[2][2]
    assert settings == {
        "enabled": True,
        "blend_weight": 0.5,
        "degree": 4,
        "ridge_lambda": 0.1,
        "window_size": 2.0,
        "flex_window": 0.75,
        "warmup_steps": 5,
        "tail_actual_steps": 3,
        "max_history": 8,
        "debug": False,
    }


def test_loader_exact_mode_does_not_install_spectrum() -> None:
    calls: list[tuple] = []
    group = FakeGroup()
    loader = nodes.H3T4Loader(
        initializer=FakeInitializer(group, calls),
        unet_loader=FakeUNETLoader(calls),
        spectrum=FakeSpectrum(calls),
    )

    assert loader.load("h3.safetensors", "Exact", object()) == (group,)
    assert [call[0] for call in calls] == ["initialize", "load"]


def test_loader_closes_owned_group_when_acceleration_setup_fails() -> None:
    calls: list[tuple] = []
    group = FakeGroup()
    loader = nodes.H3T4Loader(
        initializer=FakeInitializer(group, calls),
        unet_loader=FakeUNETLoader(calls),
        spectrum=FakeSpectrum(calls, RuntimeError("forecast setup failed")),
    )

    with pytest.raises(RuntimeError, match="forecast setup failed"):
        loader.load("h3.safetensors", "Spectrum", object())

    assert group.closed == 1


class FakeScheduler:
    def __init__(self, calls: list[tuple], failure: Exception | None = None) -> None:
        self.calls = calls
        self.failure = failure

    def get_sigmas(self, model, scheduler, steps, denoise):
        self.calls.append(("schedule", model, scheduler, steps, denoise))
        if self.failure is not None:
            raise self.failure
        return ("sigmas",)


class FakeGuider:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    def get_guider(self, model, conditioning):
        self.calls.append(("guide", model, conditioning))
        return ("guider",)


class FakeAdvancedSampler:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    def sample(self, add_noise, noise, guider, sampler, sigmas, latent_image):
        self.calls.append(("sample", add_noise, noise, guider, sampler, sigmas, latent_image))
        return "output", "denoised"


def test_sampler_hides_schedule_guidance_noise_and_sampler_construction() -> None:
    calls: list[tuple] = []
    group = FakeGroup()
    conditioning = object()
    latent = object()
    sampler = nodes.H3T4Sampler(
        scheduler_node=FakeScheduler(calls),
        guider_node=FakeGuider(calls),
        sampler_node=FakeAdvancedSampler(calls),
        noise_factory=lambda seed: ("noise", seed),
        sampler_factory=lambda name: ("sampler", name),
    )

    assert sampler.sample(group, conditioning, latent, 42, "res_multistep", "simple", 20, 1.0, True) == (
        "output",
        "denoised",
    )
    assert calls == [
        ("schedule", group, "simple", 20, 1.0),
        ("guide", group, conditioning),
        ("sample", True, ("noise", 42), "guider", ("sampler", "res_multistep"), "sigmas", latent),
    ]


def test_sampler_closes_workers_when_setup_fails_before_advanced_sampler() -> None:
    group = FakeGroup()
    sampler = nodes.H3T4Sampler(
        scheduler_node=FakeScheduler([], RuntimeError("schedule failed")),
        guider_node=FakeGuider([]),
        sampler_node=FakeAdvancedSampler([]),
        noise_factory=lambda seed: ("noise", seed),
        sampler_factory=lambda name: ("sampler", name),
    )

    with pytest.raises(RuntimeError, match="schedule failed"):
        sampler.sample(group, object(), object(), 0, "res_multistep", "simple", 20, 1.0, True)

    assert group.closed == 1
