from __future__ import annotations

import gc
import os
from datetime import timedelta
from typing import Any

from .checkpoint import require_int8_state_dict
from .topology import ExactH3T4Topology


def validate_worker_config(config: dict[str, object]) -> None:
    expected = ExactH3T4Topology().as_worker_config()
    if config != expected:
        raise ValueError(f"MiniMax-H3 worker requires the exact two-T4 topology: {expected!r}")


class H3T4Worker:
    """One rank of the fixed two-T4 FSDP + Ulysses MiniMax-H3 runtime."""

    def __init__(self, rank: int, config: dict[str, object]) -> None:
        validate_worker_config(config)
        self.rank = int(rank)
        self.config = dict(config)
        self.model = None
        self.device_mesh = None
        self._initialize_distributed()

    def _initialize_distributed(self) -> None:
        import torch
        import torch.distributed as dist
        from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        if not dist.is_initialized():
            dist.init_process_group(
                "nccl",
                rank=self.rank,
                world_size=2,
                timeout=timedelta(minutes=2),
                init_method=(f"tcp://{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MASTER_PORT', '29500')}"),
            )
        self.device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(2,))
        init_distributed_environment(rank=self.rank, world_size=2)
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
        state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        require_int8_state_dict(state_dict)
        prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
        normalized = comfy.utils.state_dict_prefix_replace(state_dict, {prefix: ""}, filter_keys=True)
        if normalized:
            state_dict = normalized
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
        patcher = comfy.model_patcher.ModelPatcher(model, load_device, offload_device)
        model.load_model_weights(state_dict, "", assign=True)
        full_state = patcher.model_state_dict(filter_prefix="diffusion_model.")
        diffusion = model.diffusion_model
        diffusion.to("meta")
        fully_shard_bottom_up(
            diffusion,
            fsdp_kwargs={"mesh": self.device_mesh, "reshard_after_forward": True},
            native_ignore_scale=False,
        )
        load_from_full_model_state_dict(
            diffusion,
            full_state,
            self.device,
            strict=True,
            cpu_offload=False,
            release_sd=True,
        )
        inject_minimax_h3_ulysses(
            patcher,
            minimax_h3_class=MiniMaxH3,
            attention_forward=h3_ulysses_attention,
            dit_forward=h3_ulysses_forward,
        )
        state_dict.clear()
        full_state.clear()
        gc.collect()
        torch.cuda.empty_cache()
        self.model = patcher
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
            generated_noise = torch.zeros_like(latent_samples)
            seed = None
        guider = BasicGuider(self.model)
        guider.set_conds(guider_spec["positive"])
        x0_output: dict[str, Any] = {}
        samples = guider.sample(
            generated_noise,
            latent_samples,
            sampler,
            sigmas,
            denoise_mask=input_latent.get("noise_mask"),
            callback=lambda _step, x0, _x, _total: x0_output.update(x0=x0),
            disable_pbar=self.rank != 0 or not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=seed,
        )
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
            denoised["samples"] = self.model.model.process_latent_out(x0_output["x0"].cpu())
        return output, denoised

    def shutdown(self) -> bool:
        import torch
        import torch.distributed as dist

        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()
        return True
