from __future__ import annotations

from typing import Any

from .runtime.lifecycle import H3T4ActorGroup, remote, resolve


class H3T4BasicScheduler:
    """Build the H3 sigma schedule on a model-owning worker."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        try:
            import comfy.samplers

            schedulers = comfy.samplers.SCHEDULER_NAMES
        except ImportError:
            schedulers = ["simple"]
        return {
            "required": {
                "model": ("H3T4_MODEL",),
                "scheduler": (schedulers,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "MiniMax H3 T4/sampling"

    def get_sigmas(
        self,
        model: H3T4ActorGroup,
        scheduler: str,
        steps: int,
        denoise: float,
    ) -> tuple[Any]:
        model.require_alive()
        result = resolve(model.ray_module, remote(model.workers[0], "get_sigmas", scheduler, steps, denoise))
        return (result,)


class H3T4BasicGuider:
    """Bind positive conditioning to the only supported H3 guidance mode."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        return {"required": {"model": ("H3T4_MODEL",), "conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("H3T4_GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "MiniMax H3 T4/sampling"

    def get_guider(self, model: H3T4ActorGroup, conditioning: Any) -> tuple[dict[str, Any]]:
        model.require_alive()
        return ({"group": model, "type": "basic", "positive": conditioning},)


class H3T4SamplerAdvanced:
    """Run one custom advanced sample and always release the two workers."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        return {
            "required": {
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise": ("NOISE",),
                "guider": ("H3T4_GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "MiniMax H3 T4/sampling"

    def sample(
        self,
        add_noise: bool,
        noise: Any,
        guider: dict[str, Any],
        sampler: Any,
        sigmas: Any,
        latent_image: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(guider, dict) or guider.get("type") != "basic":
            raise ValueError("H3T4SamplerAdvanced requires H3T4BasicGuider")
        group = guider.get("group")
        if not isinstance(group, H3T4ActorGroup):
            raise ValueError("H3T4 guider has no valid actor group")
        group.require_alive()
        refs: list[Any] = []
        worker_guider = {"type": "basic", "positive": guider["positive"]}
        try:
            for worker in group.workers:
                refs.append(
                    remote(
                        worker,
                        "sample_advanced",
                        add_noise,
                        noise,
                        worker_guider,
                        sampler,
                        sigmas,
                        latent_image,
                    )
                )
            results = resolve(group.ray_module, refs)
            if not isinstance(results, list):
                results = [results]
            result = next((item for item in results if item is not None), None)
            if result is None:
                raise RuntimeError("MiniMax-H3 workers returned no sampling result")
            output, denoised = result
            self._validate_av_latent(output)
            self._validate_av_latent(denoised)
            return output, denoised
        finally:
            group.close(timeout_seconds=30.0)

    @staticmethod
    def _validate_av_latent(latent: Any) -> None:
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if not isinstance(samples, (list, tuple)) or len(samples) != 2:
            raise RuntimeError("MiniMax-H3 worker must preserve samples as [video, audio]")
