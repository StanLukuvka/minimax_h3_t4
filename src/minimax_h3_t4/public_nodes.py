from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class H3T4ModelSpec:
    """Inert configuration passed from the loader to the lifecycle-owning sampler."""

    unet_name: str
    acceleration: str


class H3T4Loader:
    """Describe the fixed two-T4 runtime without creating workers or loading weights."""

    def __init__(self, **_legacy_parts: Any) -> None:
        # Dependency arguments from the pre-lazy implementation are accepted so saved
        # tests/integrations fail soft, but the loader deliberately touches no runtime.
        pass

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        try:
            import folder_paths

            names = folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("checkpoints")
        except ImportError:
            names = []
        return {
            "required": {
                "unet_name": (names,),
                "acceleration": (("Exact", "Spectrum"), {"default": "Spectrum"}),
                "load_after": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("H3T4_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MiniMax H3 T4"

    def load(self, unet_name: str, acceleration: str, load_after: Any) -> tuple[H3T4ModelSpec]:
        if load_after is None:
            raise ValueError("MiniMax-H3 loader requires conditioning connected to load_after")
        if acceleration not in {"Exact", "Spectrum"}:
            raise ValueError("acceleration must be Exact or Spectrum")
        return (H3T4ModelSpec(unet_name=unet_name, acceleration=acceleration),)


class H3T4Sampler:
    """Sample on the opaque two-T4 runtime and return ordinary AV latents."""

    def __init__(
        self,
        *,
        initializer: Any | None = None,
        unet_loader: Any | None = None,
        spectrum: Any | None = None,
        model_management: Any | None = None,
        gc_collect: Any | None = None,
        interrupt_checker: Any | None = None,
        scheduler_node: Any | None = None,
        guider_node: Any | None = None,
        sampler_node: Any | None = None,
        noise_factory: Any | None = None,
        sampler_factory: Any | None = None,
    ) -> None:
        self._initializer = initializer
        self._unet_loader = unet_loader
        self._spectrum = spectrum
        self._model_management = model_management
        self._gc_collect = gc_collect
        self._interrupt_checker = interrupt_checker
        self._scheduler_node = scheduler_node
        self._guider_node = guider_node
        self._sampler_node = sampler_node
        self._noise_factory = noise_factory
        self._sampler_factory = sampler_factory

    def _runtime_parts(self) -> tuple[Any, Any, Any, Any, Any, Any]:
        if self._initializer is None or self._unet_loader is None:
            from .loader_nodes import H3T4Initializer, H3T4UNETLoader

            self._initializer = self._initializer or H3T4Initializer()
            self._unet_loader = self._unet_loader or H3T4UNETLoader()
        if self._spectrum is None:
            from .spectrum_node import H3T4Spectrum

            self._spectrum = H3T4Spectrum()
        if self._model_management is None:
            import comfy.model_management

            self._model_management = comfy.model_management
        if self._gc_collect is None:
            import gc

            self._gc_collect = gc.collect
        if self._interrupt_checker is None:
            self._interrupt_checker = self._model_management.throw_exception_if_processing_interrupted
        return (
            self._initializer,
            self._unet_loader,
            self._spectrum,
            self._model_management,
            self._gc_collect,
            self._interrupt_checker,
        )

    def _parts(self) -> tuple[Any, Any, Any, Any, Any]:
        if self._scheduler_node is None or self._guider_node is None or self._sampler_node is None:
            from .sampler_nodes import H3T4BasicGuider, H3T4BasicScheduler, H3T4SamplerAdvanced

            self._scheduler_node = self._scheduler_node or H3T4BasicScheduler()
            self._guider_node = self._guider_node or H3T4BasicGuider()
            self._sampler_node = self._sampler_node or H3T4SamplerAdvanced()
        if self._noise_factory is None:
            from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

            self._noise_factory = Noise_RandomNoise
        if self._sampler_factory is None:
            import comfy.samplers

            self._sampler_factory = comfy.samplers.sampler_object
        return (
            self._scheduler_node,
            self._guider_node,
            self._sampler_node,
            self._noise_factory,
            self._sampler_factory,
        )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        try:
            import comfy.samplers

            samplers = comfy.samplers.SAMPLER_NAMES
            schedulers = comfy.samplers.SCHEDULER_NAMES
        except ImportError:
            samplers = ["res_multistep"]
            schedulers = ["simple"]
        return {
            "required": {
                "model": ("H3T4_MODEL",),
                "conditioning": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "add_noise": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "MiniMax H3 T4"

    def sample(
        self,
        model: Any,
        conditioning: Any,
        latent_image: Any,
        seed: int,
        sampler_name: str,
        scheduler: str,
        steps: int,
        denoise: float,
        add_noise: bool,
    ) -> tuple[Any, Any]:
        if not isinstance(model, H3T4ModelSpec):
            raise TypeError("H3T4Sampler requires the inert model descriptor from H3T4Loader")
        initializer, unet_loader, spectrum_node, model_management, gc_collect, check_interrupt = self._runtime_parts()
        scheduler_node, guider_node, sampler_node, noise_factory, sampler_factory = self._parts()
        group: Any | None = None
        primary_error: BaseException | None = None
        try:
            check_interrupt()
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            gc_collect()
            check_interrupt()
            group = initializer.initialize(
                load_after=conditioning,
                manage_parent_memory=False,
                interrupt_checker=check_interrupt,
                force_on_failure=True,
            )[0]
            group = unet_loader.load(
                group,
                model.unet_name,
                conditioning,
                manage_parent_memory=False,
                interrupt_checker=check_interrupt,
                force_on_failure=True,
            )[0]
            if model.acceleration == "Spectrum":
                group = spectrum_node.configure(
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
                )[0]
            sigmas = scheduler_node.get_sigmas(
                group,
                scheduler,
                steps,
                denoise,
                interrupt_checker=check_interrupt,
            )[0]
            guider = guider_node.get_guider(group, conditioning)[0]
            noise = noise_factory(seed)
            sampler = sampler_factory(sampler_name)
            return sampler_node.sample(
                add_noise,
                noise,
                guider,
                sampler,
                sigmas,
                latent_image,
                close_group=False,
                interrupt_checker=check_interrupt,
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if group is not None and getattr(group, "alive", False):
                try:
                    group.close(timeout_seconds=30.0, force=primary_error is not None)
                except BaseException as close_exc:
                    if primary_error is None:
                        raise
                    add_note = getattr(primary_error, "add_note", None)
                    if callable(add_note):
                        add_note(f"Secondary worker teardown failure: {type(close_exc).__name__}: {close_exc}")
