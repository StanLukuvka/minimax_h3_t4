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
    calls: list[tuple | str] | None = None
    forces: list[bool] | None = None

    def close(self, timeout_seconds: float = 30.0, *, force: bool = False) -> None:
        self.closed += 1
        self.alive = False
        if self.forces is not None:
            self.forces.append(force)
        if self.calls is not None:
            self.calls.append("close")


class FakeInitializer:
    def __init__(self, group: FakeGroup, calls: list[tuple]) -> None:
        self.group = group
        self.calls = calls

    def initialize(
        self,
        *,
        load_after,
        manage_parent_memory=True,
        interrupt_checker=None,
        force_on_failure=False,
    ):
        self.calls.append(("initialize", load_after, manage_parent_memory))
        return (self.group,)


class FakeUNETLoader:
    def __init__(self, calls: list[tuple], failure: Exception | None = None) -> None:
        self.calls = calls
        self.failure = failure

    def load(
        self,
        group,
        unet_name,
        load_after,
        manage_parent_memory=True,
        interrupt_checker=None,
        force_on_failure=False,
    ):
        self.calls.append(("load", group, unet_name, load_after, manage_parent_memory))
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


class BombRuntime:
    def __getattr__(self, name):
        raise AssertionError(f"lazy loader touched runtime component {name}")


def test_loader_returns_inert_descriptor_without_touching_runtime() -> None:
    loader = nodes.H3T4Loader(
        initializer=BombRuntime(),
        unet_loader=BombRuntime(),
        spectrum=BombRuntime(),
    )
    conditioning = object()

    (model,) = loader.load("h3.safetensors", "Spectrum", conditioning)

    assert model.unet_name == "h3.safetensors"
    assert model.acceleration == "Spectrum"
    assert not hasattr(model, "workers")


def test_loader_descriptor_does_not_retain_conditioning() -> None:
    conditioning = object()

    (model,) = nodes.H3T4Loader().load("h3.safetensors", "Exact", conditioning)

    assert conditioning not in vars(model).values()


class FakeScheduler:
    def __init__(self, calls: list[tuple], failure: Exception | None = None) -> None:
        self.calls = calls
        self.failure = failure

    def get_sigmas(self, model, scheduler, steps, denoise, interrupt_checker=None):
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

    def sample(
        self,
        add_noise,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        *,
        close_group=True,
        interrupt_checker=None,
    ):
        self.calls.append(
            (
                "sample",
                add_noise,
                noise,
                guider,
                sampler,
                sigmas,
                latent_image,
                close_group,
                interrupt_checker,
            )
        )
        return "output", "denoised"


class FakeModelManagement:
    def __init__(self, calls):
        self.calls = calls

    def unload_all_models(self):
        self.calls.append("unload")

    def soft_empty_cache(self):
        self.calls.append("empty")


def test_sampler_owns_runtime_from_parent_release_through_confirmed_close() -> None:
    calls: list[tuple | str] = []
    group = FakeGroup(calls=calls)
    conditioning = object()
    latent = object()

    def check_interrupt():
        calls.append("interrupt")

    sampler = nodes.H3T4Sampler(
        initializer=FakeInitializer(group, calls),
        unet_loader=FakeUNETLoader(calls),
        spectrum=FakeSpectrum(calls),
        model_management=FakeModelManagement(calls),
        gc_collect=lambda: calls.append("gc"),
        interrupt_checker=check_interrupt,
        scheduler_node=FakeScheduler(calls),
        guider_node=FakeGuider(calls),
        sampler_node=FakeAdvancedSampler(calls),
        noise_factory=lambda seed: ("noise", seed),
        sampler_factory=lambda name: ("sampler", name),
    )
    model = nodes.H3T4Loader().load("h3.safetensors", "Spectrum", conditioning)[0]

    assert sampler.sample(model, conditioning, latent, 42, "res_multistep", "simple", 20, 1.0, True) == (
        "output",
        "denoised",
    )
    assert calls[:7] == [
        "interrupt",
        "unload",
        "empty",
        "gc",
        "interrupt",
        ("initialize", conditioning, False),
        ("load", group, "h3.safetensors", conditioning, False),
    ]
    assert calls[7][0] == "spectrum"
    assert calls[8:10] == [
        ("schedule", group, "simple", 20, 1.0),
        ("guide", group, conditioning),
    ]
    assert calls[10][0:7] == (
        "sample",
        True,
        ("noise", 42),
        "guider",
        ("sampler", "res_multistep"),
        "sigmas",
        latent,
    )
    assert calls[10][7:] == (False, check_interrupt)
    assert calls[11] == "close"


def test_sampler_closes_workers_when_setup_fails_before_advanced_sampler() -> None:
    calls: list[tuple | str] = []
    group = FakeGroup(calls=calls)
    conditioning = object()
    sampler = nodes.H3T4Sampler(
        initializer=FakeInitializer(group, calls),
        unet_loader=FakeUNETLoader(calls),
        spectrum=FakeSpectrum(calls),
        model_management=FakeModelManagement(calls),
        gc_collect=lambda: calls.append("gc"),
        interrupt_checker=lambda: None,
        scheduler_node=FakeScheduler(calls, RuntimeError("schedule failed")),
        guider_node=FakeGuider(calls),
        sampler_node=FakeAdvancedSampler(calls),
        noise_factory=lambda seed: ("noise", seed),
        sampler_factory=lambda name: ("sampler", name),
    )
    model = nodes.H3T4Loader().load("h3.safetensors", "Exact", conditioning)[0]

    with pytest.raises(RuntimeError, match="schedule failed"):
        sampler.sample(model, conditioning, object(), 0, "res_multistep", "simple", 20, 1.0, True)

    assert group.closed == 1


class InterruptingAdvancedSampler:
    def sample(self, *_args, interrupt_checker, **_kwargs):
        interrupt_checker()
        raise AssertionError("interrupt checker should have raised")


def test_sampler_force_cleans_workers_when_parent_interrupts_sampling() -> None:
    calls: list[tuple | str] = []
    forces: list[bool] = []
    group = FakeGroup(calls=calls, forces=forces)
    conditioning = object()
    checks = 0

    class Interrupted(RuntimeError):
        pass

    def check_interrupt():
        nonlocal checks
        checks += 1
        if checks == 3:
            raise Interrupted("cancelled")

    sampler = nodes.H3T4Sampler(
        initializer=FakeInitializer(group, calls),
        unet_loader=FakeUNETLoader(calls),
        spectrum=FakeSpectrum(calls),
        model_management=FakeModelManagement(calls),
        gc_collect=lambda: calls.append("gc"),
        interrupt_checker=check_interrupt,
        scheduler_node=FakeScheduler(calls),
        guider_node=FakeGuider(calls),
        sampler_node=InterruptingAdvancedSampler(),
        noise_factory=lambda seed: ("noise", seed),
        sampler_factory=lambda name: ("sampler", name),
    )
    model = nodes.H3T4Loader().load("h3.safetensors", "Exact", conditioning)[0]

    with pytest.raises(Interrupted, match="cancelled"):
        sampler.sample(model, conditioning, object(), 0, "res_multistep", "simple", 20, 1.0, True)

    assert forces == [True]
