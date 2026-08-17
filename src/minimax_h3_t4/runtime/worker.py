from __future__ import annotations

import gc
import os
from datetime import timedelta
from typing import Any

from .checkpoint import load_int8_checkpoint_mmap, normalize_state_dict_prefix, require_int8_state_dict
from .topology import ExactH3T4Topology
from .diag_logger import log_memory, log_fsdp_shard_state, log_host, log_tensor


def validate_worker_config(config: dict[str, object]) -> None:
    expected = ExactH3T4Topology().as_worker_config()
    if config != expected:
        raise ValueError(f"MiniMax-H3 worker requires the exact two-T4 topology: {expected!r}")


def zero_noise_like(samples: Any, torch_module: Any) -> Any:
    """Create zero noise while preserving ComfyUI's NestedTensor AV container."""
    if getattr(samples, "is_nested", False):
        return type(samples)(tuple(torch_module.zeros_like(tensor) for tensor in samples.unbind()))
    return torch_module.zeros_like(samples)


def reconstruct_nested_x0(
    samples: Any,
    x0: Any,
    *,
    nested_type: Any,
    unpack_latents: Any,
) -> Any:
    if getattr(samples, "is_nested", False) and not getattr(x0, "is_nested", False):
        latent_shapes = [tensor.shape for tensor in samples.unbind()]
        return nested_type(unpack_latents(x0, latent_shapes))
    return x0


# ---------------------------------------------------------------------------
# Monkey-patch psutil so ComfyUI's MAX_PINNED_MEMORY uses conservative RAM.
# Kaggle reports 32 GiB but only ~30 GiB is usable; pinning ~12.8 GiB
# competes with Ray-worker checkpoint loading and triggers OOM kills.
# ---------------------------------------------------------------------------
def _patch_psutil() -> None:
    """Patch ``psutil.virtual_memory`` to report 30 GiB total RAM.

    ComfyUI derives ``MAX_PINNED_MEMORY`` from ``psutil.virtual_memory().total``
    as ``max(ram * 0.40, min(ram * 0.90, ram - 4GiB))``. Kaggle advertises
    32 GiB, which pins ~12.8 GiB and starves the Ray workers that need host
    RAM for INT8 checkpoint loading. Reporting the ~30 GiB that is genuinely
    usable keeps the pinned-memory budget conservative.

    ``available`` is deliberately reported as equal to ``total``: a low
    ``available`` makes ComfyUI skip pinned memory entirely, which is worse
    than a slightly optimistic figure.

    ``psutil`` is imported lazily so this module stays importable without it.
    """
    import types

    import psutil

    total = 30 * 1024**3
    used = 2 * 1024**3

    snapshot = types.SimpleNamespace(
        total=total,
        available=total,  # equal to total on purpose; see docstring
        used=used,
        free=total - used,
        percent=round(used / total * 100, 1),
        active=used,
        inactive=0,
        buffers=0,
        cached=0,
        shared=0,
        slab=0,
    )

    def virtual_memory() -> types.SimpleNamespace:
        return snapshot

    psutil.virtual_memory = virtual_memory


class H3T4Worker:
    """One rank of the fixed two-T4 FSDP + Ulysses MiniMax-H3 runtime."""

    def __init__(self, rank: int, config: dict[str, object]) -> None:
        validate_worker_config(config)
        self.rank = int(rank)
        self.config = dict(config)
        self.model = None
        self.device_mesh = None
        self.spectrum_controller = None
        self._initialize_distributed()

    def _initialize_distributed(self) -> None:
        import torch
        import torch.distributed as dist
        from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

        from .int8 import require_tesla_t4

        # Kaggle reports 32GB but only has ~30GB usable. Patch psutil so
        # ComfyUI computes conservative memory budgets and stops trying to
        # pin into phantom RAM that triggers the Ray OOM killer.
        _patch_psutil()

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        require_tesla_t4(torch, self.device)
        if not dist.is_initialized():
            dist.init_process_group(
                "nccl",
                rank=self.rank,
                world_size=2,
                timeout=timedelta(minutes=2),
                init_method=(f"tcp://{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MASTER_PORT', '29500')}"),
            )
        self.device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(2,))
        init_distributed_environment(rank=self.rank, world_size=2, local_rank=0, backend="nccl")
        initialize_model_parallel(
            data_parallel_degree=1,
            sequence_parallel_degree=2,
            classifier_free_guidance_degree=1,
            ring_degree=1,
            ulysses_degree=2,
            pipeline_parallel_degree=1,
        )

    def load_unet(self, path: str, weight_dtype: str) -> bool:
        if weight_dtype != "int8":
            raise ValueError("MiniMax-H3 two-T4 worker is INT8-only")
        import comfy.model_detection
        import comfy.model_management
        import comfy.model_patcher
        import comfy.utils
        import torch

        from .fsdp import fully_shard_bottom_up, load_from_full_model_state_dict
        from .h3_forward import h3_ulysses_attention, h3_ulysses_forward
        from .int8 import install_int8_cuda_oom_retry
        from .ulysses import inject_minimax_h3_ulysses

        install_int8_cuda_oom_retry(torch)
        cpu_offload = bool(self.config.get("fsdp_cpu_offload", False))
        # Enable mmap-backed checkpoint loading to avoid a full per-worker
        # RAM peak on constrained Kaggle hosts.
        import comfy.utils as _cu
        _cu.MMAP_TORCH_FILES = True
        log_memory("load_unet_start", extra={"cpu_offload": cpu_offload})
        log_host("load_unet_start")
        state_dict, metadata = load_int8_checkpoint_mmap(path)
        require_int8_state_dict(state_dict)
        log_memory("after_checkpoint_load", extra={"num_tensors": len(state_dict)})
        log_host("after_checkpoint_load")
        prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
        state_dict = normalize_state_dict_prefix(state_dict, prefix)
        state_dict, metadata = comfy.utils.convert_old_quants(state_dict, "", metadata=metadata)
        config = comfy.model_detection.model_config_from_unet(state_dict, "", metadata=metadata)
        if config is None:
            raise ValueError("Checkpoint is not a native ComfyUI MiniMax-H3 model")
        model = config.get_model(state_dict, "")
        from comfy.model_base import MiniMaxH3

        if not isinstance(model, MiniMaxH3):
            raise TypeError(f"Expected MiniMaxH3 checkpoint, detected {type(model).__name__}")
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        model.load_model_weights(state_dict, "", assign=True)
        # Drop the checkpoint now that the model owns its weights.
        # This avoids a second ~12GB allocation when we build `full_state`.
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        log_memory("after_model_init", extra={"model_params": sum(p.numel() for p in model.parameters())})
        log_host("after_model_init")
        local_model_size = max(1, comfy.model_management.module_size(model) // 2)
        from .patcher import h3_fsdp_patcher_class

        patcher_class = h3_fsdp_patcher_class(comfy.model_patcher.ModelPatcher)
        patcher = patcher_class(model, load_device, offload_device, size=local_model_size, cpu_offload=cpu_offload)
        full_state = normalize_state_dict_prefix(
            patcher.model_state_dict(filter_prefix="diffusion_model."),
            "diffusion_model.",
        )
        log_memory("before_fsdp_shard", extra={"full_state_keys": len(full_state)})
        log_host("before_fsdp_shard")
        diffusion = model.diffusion_model
        diffusion.to("meta")
        fsdp_kwargs: dict[str, Any] = {"mesh": self.device_mesh, "reshard_after_forward": True}
        if cpu_offload:
            from torch.distributed.fsdp import CPUOffloadPolicy

            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
        fully_shard_bottom_up(
            diffusion,
            fsdp_kwargs=fsdp_kwargs,
            native_ignore_scale=False,
        )
        log_fsdp_shard_state(diffusion, "post_shard")
        load_from_full_model_state_dict(
            diffusion,
            full_state,
            self.device,
            strict=True,
            cpu_offload=cpu_offload,
            release_sd=True,
        )
        log_memory("after_fsdp_materialize", extra={"cpu_offload": cpu_offload})
        log_host("after_fsdp_materialize")
        inject_minimax_h3_ulysses(
            patcher,
            minimax_h3_class=MiniMaxH3,
            attention_forward=h3_ulysses_attention,
            dit_forward=h3_ulysses_forward,
        )
        gc.collect()
        torch.cuda.empty_cache()
        log_memory("load_unet_complete")
        log_host("load_unet_complete")
        self.model = patcher
        return True

    def load_unet_from_state_dict(self, state_dict_ref: Any, weight_dtype: str) -> bool:
        """Load model from an already-loaded state dict (broadcast from main process).

        This avoids the double-RAM peak when multiple workers each load the full checkpoint.
        Each worker resolves its own reference from the shared state dict via Ray Object Store.
        """
        if weight_dtype != "int8":
            raise ValueError("MiniMax-H3 two-T4 worker is INT8-only")
        import torch
        import comfy.model_detection
        import comfy.model_management
        import comfy.model_patcher
        import comfy.utils

        from .fsdp import fully_shard_bottom_up, load_from_full_model_state_dict
        from .h3_forward import h3_ulysses_attention, h3_ulysses_forward
        from .int8 import install_int8_cuda_oom_retry
        from .ulysses import inject_minimax_h3_ulysses

        # Resolve the Ray object reference to get the actual state dict
        ray_get = self.config.get("_ray_get")
        if ray_get is not None:
            state_dict = ray_get(state_dict_ref)  # type: ignore[misc]
        else:
            # Fallback for test environments that don't set up Ray
            state_dict = state_dict_ref

        if state_dict is None:
            raise RuntimeError("No shared state dict available for this worker")

        install_int8_cuda_oom_retry(torch)
        cpu_offload = bool(self.config.get("fsdp_cpu_offload", False))
        
        # Use the shared state dict directly - no disk load needed
        require_int8_state_dict(state_dict)
        prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
        state_dict = normalize_state_dict_prefix(state_dict, prefix)
        state_dict, metadata = comfy.utils.convert_old_quants(state_dict, "", metadata={})
        config = comfy.model_detection.model_config_from_unet(state_dict, "", metadata=metadata)
        if config is None:
            raise ValueError("Checkpoint is not a native ComfyUI MiniMax-H3 model")
        model = config.get_model(state_dict, "")
        from comfy.model_base import MiniMaxH3

        if not isinstance(model, MiniMaxH3):
            raise TypeError(f"Expected MiniMax-H3 checkpoint, detected {type(model).__name__}")
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        model.load_model_weights(state_dict, "", assign=True)
        # Drop the checkpoint now that the model owns its weights
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        local_model_size = max(1, comfy.model_management.module_size(model) // 2)
        from .patcher import h3_fsdp_patcher_class

        patcher_class = h3_fsdp_patcher_class(comfy.model_patcher.ModelPatcher)
        patcher = patcher_class(model, load_device, offload_device, size=local_model_size, cpu_offload=cpu_offload)
        full_state = normalize_state_dict_prefix(
            patcher.model_state_dict(filter_prefix="diffusion_model."),
            "diffusion_model.",
        )
        diffusion = model.diffusion_model
        diffusion.to("meta")
        fsdp_kwargs: dict[str, Any] = {"mesh": self.device_mesh, "reshard_after_forward": True}
        if cpu_offload:
            from torch.distributed.fsdp import CPUOffloadPolicy

            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
        fully_shard_bottom_up(
            diffusion,
            fsdp_kwargs=fsdp_kwargs,
            native_ignore_scale=False,
        )
        load_from_full_model_state_dict(
            diffusion,
            full_state,
            self.device,
            strict=True,
            cpu_offload=cpu_offload,
            release_sd=True,
        )
        inject_minimax_h3_ulysses(
            patcher,
            minimax_h3_class=MiniMaxH3,
            attention_forward=h3_ulysses_attention,
            dit_forward=h3_ulysses_forward,
        )
        gc.collect()
        torch.cuda.empty_cache()
        self.model = patcher
        return True

    def configure_spectrum(self, config: dict[str, Any]) -> bool:
        if self.model is None:
            raise RuntimeError("load the MiniMax-H3 model before configuring Spectrum")
        from ..spectrum.config import SpectrumConfig
        from ..spectrum.controller import SpectrumController

        spectrum_config = SpectrumConfig(**config).validate()
        base_model = getattr(self.model, "model", self.model)
        diffusion_model = getattr(base_model, "diffusion_model", None)
        if diffusion_model is None:
            raise RuntimeError("loaded model has no MiniMax-H3 diffusion model")
        if not spectrum_config.enabled:
            if hasattr(diffusion_model, "_h3_t4_spectrum_controller"):
                delattr(diffusion_model, "_h3_t4_spectrum_controller")
            self.spectrum_controller = None
            return True
        controller = SpectrumController(spectrum_config)
        diffusion_model._h3_t4_spectrum_controller = controller
        self.spectrum_controller = controller
        return True

    def get_sigmas(self, scheduler: str, steps: int, denoise: float):
        import comfy.samplers
        import torch

        if self.model is None:
            raise RuntimeError("MiniMax-H3 model is not loaded")
        if denoise <= 0.0:
            return torch.FloatTensor([])
        total_steps = int(steps / denoise) if denoise < 1.0 else steps
        sampling = self.model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, total_steps).cpu()
        return sigmas[-(steps + 1) :]

    def sample_advanced(self, add_noise, noise_source, guider_spec, sampler, sigmas, latent):
        import comfy.model_management
        import comfy.sample
        import comfy.samplers
        import comfy.utils
        import torch

        if self.model is None:
            raise RuntimeError("MiniMax-H3 model is not loaded")
        log_memory("sample_start", extra={
            "latent_shape": list(latent.get("samples", {}).shape if hasattr(latent.get("samples"), "shape") else []),
            "sigmas_len": len(sigmas) if sigmas is not None else 0,
        })
        if guider_spec.get("type") != "basic":
            raise ValueError("Only the MiniMax-H3 basic guider is supported")

        class BasicGuider(comfy.samplers.CFGGuider):
            def set_conds(self, positive):
                self.inner_set_conds({"positive": positive})

        input_latent = dict(latent)
        latent_samples = comfy.sample.fix_empty_latent_channels(
            self.model,
            input_latent["samples"],
            input_latent.get("downscale_ratio_spacial"),
            input_latent.get("downscale_ratio_temporal"),
        )
        input_latent["samples"] = latent_samples
        if add_noise:
            generated_noise = noise_source.generate_noise(input_latent)
            seed = getattr(noise_source, "seed", None)
        else:
            generated_noise = zero_noise_like(latent_samples, torch)
            seed = None
        guider = BasicGuider(self.model)
        guider.set_conds(guider_spec["positive"])
        x0_output: dict[str, Any] = {}
        if self.spectrum_controller is not None:
            self.spectrum_controller.start_run(sigmas, sampler=sampler)
        try:
            from .int8 import run_with_cuda_int8_backend

            samples = run_with_cuda_int8_backend(
                lambda: guider.sample(
                    generated_noise,
                    latent_samples,
                    sampler,
                    sigmas,
                    denoise_mask=input_latent.get("noise_mask"),
                    callback=lambda _step, x0, _x, _total: x0_output.update(x0=x0),
                    disable_pbar=self.rank != 0 or not comfy.utils.PROGRESS_BAR_ENABLED,
                    seed=seed,
                )
            )
        except Exception as exc:
            if self.spectrum_controller is not None:
                self.spectrum_controller.abort_run(f"sampling failed: {type(exc).__name__}: {exc}")
            raise
        else:
            if self.spectrum_controller is not None:
                self.spectrum_controller.end_run()
        # Both ranks execute collectives; only rank zero sends AV latents back.
        if self.rank != 0:
            return None
        samples = samples.to(comfy.model_management.intermediate_device())
        output = dict(input_latent)
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        output["samples"] = samples
        denoised = dict(output)
        if "x0" in x0_output:
            import comfy.nested_tensor

            x0 = reconstruct_nested_x0(
                samples,
                x0_output["x0"],
                nested_type=comfy.nested_tensor.NestedTensor,
                unpack_latents=comfy.utils.unpack_latents,
            )
            denoised["samples"] = self.model.model.process_latent_out(x0.cpu())
        return output, denoised

    def shutdown(self) -> None:
        import ray
        import torch
        import torch.distributed as dist

        release_error: BaseException | None = None
        destroy_error: BaseException | None = None
        patcher = self.model
        self.model = None
        self.spectrum_controller = None
        try:
            if patcher is not None:
                base_model = getattr(patcher, "model", None)
                if base_model is not None and hasattr(base_model, "current_patcher"):
                    base_model.current_patcher = None
                patcher.cleanup()
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as exc:
            release_error = exc
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except BaseException as exc:
            destroy_error = exc
        if release_error is not None:
            if destroy_error is not None:
                raise release_error from destroy_error
            raise release_error
        if destroy_error is not None:
            raise destroy_error
        ray.actor.exit_actor()
