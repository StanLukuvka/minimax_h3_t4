from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

from .runtime.lifecycle import H3T4ActorGroup, remote, resolve
from .runtime.topology import ExactH3T4Topology


def _ray_module() -> Any:
    import ray

    return ray


def _model_management() -> Any:
    import comfy.model_management

    return comfy.model_management


def _resolve_unet(name: str) -> str:
    import folder_paths

    path = folder_paths.get_full_path("diffusion_models", name)
    if path is None:
        path = folder_paths.get_full_path_or_raise("checkpoints", name)
    return path


def _release_conditioning(model_management: Any) -> None:
    model_management.unload_all_models()
    model_management.soft_empty_cache()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class H3T4Initializer:
    """Create the fixed local two-worker MiniMax-H3 execution group."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        return {
            "required": {
                "ray_cluster_namespace": ("STRING", {"default": "minimax-h3-t4"}),
                "GPU_SELECT": ("STRING", {"default": "0,1"}),
                "ray_object_store_gb": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1},
                ),
            },
            "optional": {"load_after": ("CONDITIONING",)},
        }

    RETURN_TYPES = ("H3T4_ACTOR_GROUP",)
    RETURN_NAMES = ("actor_group",)
    FUNCTION = "initialize"
    CATEGORY = "MiniMax H3 T4"

    def __init__(
        self,
        *,
        ray_module: Any | None = None,
        model_management: Any | None = None,
        worker_factory: Callable[[int, dict[str, object]], Any] | None = None,
    ) -> None:
        self._ray = ray_module
        self._model_management = model_management
        self._worker_factory = worker_factory

    def _create_worker(self, rank: int, config: dict[str, object]) -> Any:
        if self._worker_factory is not None:
            return self._worker_factory(rank, config)
        from .runtime.worker import H3T4Worker

        actor_class = self._ray.remote(num_gpus=1)(H3T4Worker)
        return actor_class.options(name=f"H3T4Worker:{rank}").remote(rank, config)

    def initialize(
        self,
        ray_cluster_namespace: str = "minimax-h3-t4",
        GPU_SELECT: str = "0,1",
        ray_object_store_gb: float = 0.5,
        load_after: Any | None = None,
    ) -> tuple[H3T4ActorGroup]:
        if not 0.1 <= float(ray_object_store_gb) <= 1.0:
            raise ValueError("ray_object_store_gb must be between 0.1 and 1.0 GiB")
        selected = [part.strip() for part in GPU_SELECT.split(",") if part.strip()]
        if selected != ["0", "1"]:
            raise ValueError("The exact MiniMax-H3 T4 runtime requires GPU_SELECT='0,1'")
        ray_module = self._ray or _ray_module()
        self._ray = ray_module
        model_management = self._model_management or _model_management()
        if load_after is not None:
            load_after = None
            _release_conditioning(model_management)
        ray_module.shutdown()
        ray_module.init(
            address="local",
            namespace=ray_cluster_namespace,
            object_store_memory=int(float(ray_object_store_gb) * 1024**3),
            include_dashboard=False,
            runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"}},
        )
        topology = ExactH3T4Topology()
        config = topology.as_worker_config()
        workers = [self._create_worker(rank, config) for rank in range(topology.world_size)]
        return (H3T4ActorGroup(workers, ray_module, topology),)


class H3T4UNETLoader:
    """Sequentially load an INT8 MiniMax-H3 checkpoint on both FSDP workers."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        try:
            import folder_paths

            names = folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("checkpoints")
        except ImportError:
            names = []
        return {
            "required": {
                "actor_group": ("H3T4_ACTOR_GROUP",),
                "unet_name": (names,),
                "load_after": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("H3T4_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MiniMax H3 T4"

    def __init__(
        self,
        *,
        path_resolver: Callable[[str], str] | None = None,
        model_management: Any | None = None,
    ) -> None:
        self._path_resolver = path_resolver or _resolve_unet
        self._model_management = model_management

    def load(
        self,
        actor_group: H3T4ActorGroup,
        unet_name: str,
        load_after: Any | None,
    ) -> tuple[H3T4ActorGroup]:
        actor_group.require_alive()
        if load_after is None:
            raise ValueError("MiniMax-H3 INT8 loading requires conditioning connected to load_after")
        load_after = None
        _release_conditioning(self._model_management or _model_management())
        path = self._path_resolver(unet_name)
        for worker in actor_group.workers:
            resolve(actor_group.ray_module, remote(worker, "load_unet", path, "int8"))
        actor_group.checkpoint = path
        return (actor_group,)
