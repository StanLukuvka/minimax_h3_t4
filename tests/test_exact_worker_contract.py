from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.minimax_h3_t4.runtime.checkpoint import require_int8_state_dict
from src.minimax_h3_t4.runtime.worker import H3T4Worker, validate_worker_config, zero_noise_like


ROOT = Path(__file__).parents[1]


def test_worker_rejects_any_topology_drift() -> None:
    exact = {
        "world_size": 2,
        "ulysses_degree": 2,
        "ring_degree": 1,
        "cfg_degree": 1,
        "dp_degree": 1,
        "fsdp": True,
        "fsdp_cpu_offload": False,
        "attention": "TORCH_EFFICIENT",
    }
    validate_worker_config(exact)

    for key, bad_value in {
        "world_size": 3,
        "ulysses_degree": 1,
        "ring_degree": 2,
        "cfg_degree": 2,
        "dp_degree": 2,
        "fsdp": False,
        "fsdp_cpu_offload": True,
        "attention": "TORCH_FLASH",
    }.items():
        drifted = dict(exact)
        drifted[key] = bad_value
        with pytest.raises(ValueError, match="exact two-T4"):
            validate_worker_config(drifted)


def test_checkpoint_accepts_only_comfykitchen_tensorwise_int8() -> None:
    int8 = {
        "block.weight": object(),
        "block.weight_scale": object(),
        "block.comfy_quant": b'{"format":"int8_tensorwise","convrot":true}',
    }
    require_int8_state_dict(int8)

    fp8 = {
        "block.weight": object(),
        "block.weight_scale": object(),
        "block.comfy_quant": b'{"format":"float8_e4m3fn"}',
    }
    with pytest.raises(ValueError, match="INT8-only"):
        require_int8_state_dict(fp8)

    with pytest.raises(ValueError, match="INT8-only"):
        require_int8_state_dict({"block.weight": object()})


def test_zero_noise_preserves_comfy_nested_av_container() -> None:
    class Nested:
        is_nested = True

        def __init__(self, tensors):
            self.tensors = list(tensors)

        def unbind(self):
            return self.tensors

    class FakeTorch:
        @staticmethod
        def zeros_like(value):
            return f"zero:{value}"

    result = zero_noise_like(Nested(("video", "audio")), FakeTorch)

    assert isinstance(result, Nested)
    assert result.unbind() == ["zero:video", "zero:audio"]


def test_worker_attaches_native_spectrum_to_owned_h3_model() -> None:
    class Diffusion:
        pass

    class Base:
        diffusion_model = Diffusion()

    class Patcher:
        model = Base()

    worker = object.__new__(H3T4Worker)
    worker.model = Patcher()
    worker.spectrum_controller = None
    configured = worker.configure_spectrum(
        {
            "enabled": True,
            "blend_weight": 0.5,
            "degree": 2,
            "ridge_lambda": 0.1,
            "window_size": 2.0,
            "flex_window": 0.0,
            "warmup_steps": 2,
            "tail_actual_steps": 1,
            "max_history": 4,
            "history_storage": "system_ram",
            "debug": False,
        }
    )

    assert configured is True
    assert worker.spectrum_controller is getattr(Base.diffusion_model, "_h3_t4_spectrum_controller")


def test_runtime_closure_has_no_donor_or_excluded_model_imports() -> None:
    forbidden = (
        "raylight",
        "comfyui_spectrum_h3",
        "ControlNet",
        "PipeFusion",
        "GGUF",
        "NVFP4",
        "EasyCache",
        "LoRA",
    )
    violations = []
    for path in sorted((ROOT / "src" / "minimax_h3_t4" / "runtime").rglob("*.py")):
        source = path.read_text()
        for marker in forbidden:
            if marker in source:
                violations.append((path.name, marker))
    assert violations == []


def test_h3_forward_implementation_preserves_av_return_shape() -> None:
    path = ROOT / "src" / "minimax_h3_t4" / "runtime" / "h3_forward.py"
    tree = ast.parse(path.read_text())
    forward = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "h3_ulysses_forward")
    returns = [node for node in ast.walk(forward) if isinstance(node, ast.Return)]
    assert any(isinstance(node.value, ast.List) and len(node.value.elts) == 2 for node in returns)
