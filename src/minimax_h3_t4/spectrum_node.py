"""ComfyUI configuration node for worker-local MiniMax H3 Spectrum acceleration."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .runtime.lifecycle import H3T4ActorGroup, remote, resolve
from .spectrum.config import SpectrumConfig


class H3T4Spectrum:
    """Configure the native Spectrum forecaster on both H3 model workers."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        return {
            "required": {
                "model": ("H3T4_MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "blend_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "degree": ("INT", {"default": 4, "min": 1, "max": 8}),
                "ridge_lambda": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 0.25}),
                "flex_window": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 10.0, "step": 0.25}),
                "warmup_steps": ("INT", {"default": 5, "min": 0, "max": 100}),
                "tail_actual_steps": ("INT", {"default": 1, "min": 0, "max": 100}),
                "max_history": ("INT", {"default": 8, "min": 2, "max": 32}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("H3T4_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "configure"
    CATEGORY = "MiniMax H3 T4/acceleration"

    def configure(
        self,
        model: H3T4ActorGroup,
        enabled: bool,
        blend_weight: float,
        degree: int,
        ridge_lambda: float,
        window_size: float,
        flex_window: float,
        warmup_steps: int,
        tail_actual_steps: int,
        max_history: int,
        debug: bool,
    ) -> tuple[H3T4ActorGroup]:
        model.require_alive()
        config = SpectrumConfig(
            enabled=enabled,
            blend_weight=blend_weight,
            degree=degree,
            ridge_lambda=ridge_lambda,
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            tail_actual_steps=tail_actual_steps,
            max_history=max_history,
            history_storage="system_ram",
            debug=debug,
        ).validate()
        refs = [remote(worker, "configure_spectrum", asdict(config)) for worker in model.workers]
        resolve(model.ray_module, refs)
        return (model,)
