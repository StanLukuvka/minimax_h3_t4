"""Minimal ComfyUI ModelPatcher policy for an already-sharded H3 model.

Apache-2.0 donor provenance is recorded in NOTICE.md.
"""

from __future__ import annotations

from typing import Any


def h3_fsdp_patcher_class(base_class: type[Any]) -> type[Any]:
    """Return a narrow patcher that never moves FSDP DTensors between devices."""

    class H3T4FSDPModelPatcher(base_class):
        def __init__(self, *args: Any, cpu_offload: bool = False, **kwargs: Any) -> None:
            self._h3_cpu_offload = cpu_offload
            super().__init__(*args, **kwargs)

        def is_dynamic(self) -> bool:
            return False

        def model_size(self) -> int:
            return int(self.size)

        def _reject_weight_patches(self) -> None:
            if self.patches or self.object_patches or self.weight_wrapper_patches:
                raise ValueError("MiniMax H3 T4 does not support weight or model patches")

        def _mark_loaded(self, device_to: Any = None) -> int:
            self._reject_weight_patches()
            previous = int(getattr(self.model, "model_loaded_weight_memory", 0))
            self.model.model_lowvram = False
            self.model.lowvram_patch_counter = 0
            # When FSDP CPU offloading is active, parameters live on host RAM.
            # Report zero GPU resident weight memory so ComfyUI's VRAM tracker
            # doesn't assume the model occupies GPU space during inference.
            gpu_weight = 0 if self._h3_cpu_offload else self.model_size()
            self.model.model_loaded_weight_memory = gpu_weight
            self.model.model_offload_buffer_memory = 0
            self.model.device = self.load_device if device_to is None else device_to
            self.model.current_weight_patches_uuid = self.patches_uuid
            self.model.current_patcher = self
            self.inject_model()
            return max(0, gpu_weight - previous)

        def load(
            self,
            device_to: Any = None,
            lowvram_model_memory: int = 0,
            force_patch_weights: bool = False,
            full_load: bool = False,
        ) -> None:
            del lowvram_model_memory, full_load
            if force_patch_weights:
                raise ValueError("MiniMax H3 T4 does not support forced weight patches")
            self._mark_loaded(device_to)

        def partially_load(
            self,
            device_to: Any,
            extra_memory: int = 0,
            force_patch_weights: bool = False,
        ) -> int:
            del extra_memory
            if force_patch_weights:
                raise ValueError("MiniMax H3 T4 does not support forced weight patches")
            return self._mark_loaded(device_to)

        def partially_unload(
            self,
            device_to: Any,
            memory_to_free: int = 0,
            force_patch_weights: bool = False,
        ) -> int:
            del device_to, memory_to_free, force_patch_weights
            return 0

        def unpatch_model(self, device_to: Any = None, unpatch_weights: bool = True) -> None:
            del device_to
            self.eject_model()
            if unpatch_weights:
                self.unpatch_hooks()
                self.backup.clear()
                self.object_patches_backup.clear()
                self.model.current_weight_patches_uuid = None
            # Never call model.to(offload_device): that materializes full DTensors.
            self.model.current_patcher = None
            self.model.model_loaded_weight_memory = 0
            self.model.model_offload_buffer_memory = 0

    return H3T4FSDPModelPatcher
