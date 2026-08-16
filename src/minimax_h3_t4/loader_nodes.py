from __future__ import annotations

import gc
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .runtime.lifecycle import H3T4ActorGroup, remote, resolve, resolve_interruptibly
from .runtime.topology import ExactH3T4Topology


def _ray_module() -> Any:
    import ray

    return ray


def _model_management() -> Any:
    import comfy.model_management

    return comfy.model_management


def _build_runtime_env() -> dict[str, object]:
    import folder_paths

    package_dir = Path(__file__).resolve().parent
    comfy_file = Path(folder_paths.__file__).resolve()
    comfy_root = next(
        (parent for parent in comfy_file.parents if (parent / "main.py").exists() and (parent / "execution.py").exists()),
        None,
    )
    if comfy_root is None:
        raise RuntimeError("Unable to locate the ComfyUI repository root for Ray workers")
    runtime_workdir = package_dir.parent / "_ray_runtime_env"
    runtime_workdir.mkdir(parents=True, exist_ok=True)
    python_entries = [str(comfy_root)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        python_entries.extend(part for part in existing.split(os.pathsep) if part)
    return {
        "py_modules": [str(package_dir)],
        "working_dir": str(runtime_workdir),
        "env_vars": {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "PYTHONPATH": os.pathsep.join(dict.fromkeys(python_entries)),
            "COMFYUI_BASE_DIRECTORY": str(comfy_root),
        },
    }


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
                    {"default": 4.0, "min": 4.0, "max": 16.0, "step": 0.5},
                ),
                "load_after": ("CONDITIONING",),
            }
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
        runtime_env_builder: Callable[[], dict[str, object]] | None = None,
    ) -> None:
        self._ray = ray_module
        self._model_management = model_management
        self._worker_factory = worker_factory
        self._runtime_env_builder = runtime_env_builder or _build_runtime_env

    def _create_worker(self, rank: int, config: dict[str, object], ray_module: Any) -> Any:
        if self._worker_factory is not None:
            return self._worker_factory(rank, config)
        from .runtime.worker import H3T4Worker

        # Pass ray.get into worker config so it can resolve shared state dict refs
        config = {**config, "_ray_get": ray_module.get}

        # Cap per-worker memory via Ray's memory option
        max_ram_gb = float(os.environ.get("H3_T4_MAX_RAM_GB", "29"))
        memory_bytes = int(max_ram_gb * 1024**3)
        actor_class = self._ray.remote(num_cpus=1, num_gpus=1, memory=memory_bytes)(H3T4Worker)
        return actor_class.options(name=f"H3T4Worker:{rank}", memory=memory_bytes).remote(rank, config)

    def initialize(
        self,
        ray_cluster_namespace: str = "minimax-h3-t4",
        GPU_SELECT: str = "0,1",
        ray_object_store_gb: float = 4.0,
        load_after: Any | None = None,
        *,
        manage_parent_memory: bool = True,
        interrupt_checker: Any | None = None,
        force_on_failure: bool = False,
    ) -> tuple[H3T4ActorGroup]:
        if not 4.0 <= float(ray_object_store_gb) <= 16.0:
            raise ValueError("ray_object_store_gb must be between 4.0 and 16.0 GiB")
        selected = [part.strip() for part in GPU_SELECT.split(",") if part.strip()]
        if selected != ["0", "1"]:
            raise ValueError("The exact MiniMax-H3 T4 runtime requires GPU_SELECT='0,1'")
        ray_module = self._ray or _ray_module()
        self._ray = ray_module
        if load_after is None:
            raise ValueError("MiniMax-H3 initialization requires conditioning connected to load_after")
        load_after = None
        if manage_parent_memory:
            _release_conditioning(self._model_management or _model_management())
        ray_module.shutdown()
        topology = ExactH3T4Topology()
        config = topology.as_worker_config()
        workers: list[Any] = []
        try:
            ray_module.init(
                address="local",
                namespace=ray_cluster_namespace,
                object_store_memory=int(float(ray_object_store_gb) * 1024**3),
                include_dashboard=False,
                runtime_env=self._runtime_env_builder(),
            )
            for rank in range(topology.world_size):
                if interrupt_checker is not None:
                    interrupt_checker()
                workers.append(self._create_worker(rank, config, ray_module))
            if interrupt_checker is not None:
                interrupt_checker()
        except BaseException as exc:
            try:
                H3T4ActorGroup(workers, ray_module, topology).close(force=force_on_failure)
            except BaseException as close_exc:
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(f"Secondary worker teardown failure: {type(close_exc).__name__}: {close_exc}")
            raise
        return (H3T4ActorGroup(workers, ray_module, topology),)


class H3T4UNETLoader:
    """Load an INT8 MiniMax-H3 checkpoint on both FSDP workers via shared state dict."""

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
        *,
        manage_parent_memory: bool = True,
        interrupt_checker: Any | None = None,
        force_on_failure: bool = False,
    ) -> tuple[H3T4ActorGroup]:
        actor_group.require_alive()
        if load_after is None:
            raise ValueError("MiniMax-H3 INT8 loading requires conditioning connected to load_after")
        load_after = None
        if manage_parent_memory:
            _release_conditioning(self._model_management or _model_management())
        path = self._path_resolver(unet_name)
        
        # Load checkpoint once in main process to avoid double RAM allocation
        from .runtime.checkpoint import load_int8_checkpoint_mmap, require_int8_state_dict
        import torch

        # Allow tests to mock checkpoint loading via environment variable
        _fake_state_dict = os.environ.get("H3_T4_TEST_FAKE_STATE_DICT")
        if _fake_state_dict == "1":
            state_dict = {
                "fake_key": torch.zeros(1),
                "fake_key.comfy_quant": "int8_tensorwise",
            }
            metadata = {}
        else:
            state_dict, metadata = load_int8_checkpoint_mmap(path)
            require_int8_state_dict(state_dict)

        # Broadcast state dict to Ray Object Store (shared memory)
        ray_module = actor_group.ray_module
        state_ref = ray_module.put(state_dict)

        try:
            # Load sequentially - each worker streams keys from shared state dict
            for worker in actor_group.workers:
                ref = remote(worker, "load_unet_from_state_dict", state_ref, "int8")
                if interrupt_checker is None:
                    resolve(ray_module, ref)
                else:
                    resolve_interruptibly(ray_module, [ref], interrupt_checker)
        except BaseException as exc:
            try:
                actor_group.close(force=force_on_failure)
            except BaseException as close_exc:
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(f"Secondary worker teardown failure: {type(close_exc).__name__}: {close_exc}")
            finally:
                try:
                    ray_module.wait([state_ref], timeout=5)
                except Exception:
                    pass
            raise
            
        actor_group.checkpoint = path
        return (actor_group,)
