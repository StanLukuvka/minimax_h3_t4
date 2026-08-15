from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

from src.minimax_h3_t4.runtime.checkpoint import (
    load_int8_checkpoint_mmap,
    normalize_state_dict_prefix,
    require_int8_state_dict,
)
from src.minimax_h3_t4.runtime.ulysses import (
    inject_minimax_h3_ulysses,
    passthrough_preprocess_text_embeds,
)
from src.minimax_h3_t4.runtime.worker import (
    H3T4Worker,
    reconstruct_nested_x0,
    validate_worker_config,
    zero_noise_like,
)


ROOT = Path(__file__).parents[1]


def test_worker_rejects_any_topology_drift() -> None:
    exact = {
        "world_size": 2,
        "ulysses_degree": 2,
        "ring_degree": 1,
        "cfg_degree": 1,
        "dp_degree": 1,
        "fsdp": True,
        "fsdp_cpu_offload": True,
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
        "fsdp_cpu_offload": False,
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

    no_convrot = {
        "block.weight": object(),
        "block.weight_scale": object(),
        "block.comfy_quant": b'{"format":"int8_tensorwise","convrot":false}',
    }
    with pytest.raises(ValueError, match="ConvRot"):
        require_int8_state_dict(no_convrot)


def test_checkpoint_loader_requires_safetensors_and_uses_cpu_mmap_loader() -> None:
    calls = []

    def fake_load(path, *, device):
        calls.append((path, device))
        return {"weight": "mapped"}

    loaded = load_int8_checkpoint_mmap(
        "/models/h3.safetensors",
        load_file=fake_load,
        metadata_loader=lambda path: {"path": path},
    )
    assert loaded == ({"weight": "mapped"}, {"path": "/models/h3.safetensors"})
    assert calls == [("/models/h3.safetensors", "cpu")]
    with pytest.raises(ValueError, match="safetensors"):
        load_int8_checkpoint_mmap("/models/h3.ckpt", load_file=fake_load)


def test_checkpoint_prefix_normalization_strips_matching_namespace() -> None:
    table = object()
    weight = object()

    normalized = normalize_state_dict_prefix(
        {
            "diffusion_model.adaln_t_table": table,
            "diffusion_model.blocks.0.weight": weight,
            "unrelated.weight": object(),
        },
        "diffusion_model.",
    )

    assert normalized == {
        "adaln_t_table": table,
        "blocks.0.weight": weight,
    }


def test_checkpoint_prefix_normalization_keeps_bare_keys_when_prefix_unused() -> None:
    fallback_prefix = "model."
    # ComfyUI returns "model." as a bare fallback even when no key carries it.
    bare = {"adaln_t_table": object(), "blocks.0.weight": object()}

    normalized = normalize_state_dict_prefix(bare, fallback_prefix)

    assert normalized is bare  # unchanged (bare), as the old contract did


def test_ulysses_injection_defers_text_projection_to_root_forward() -> None:
    """FSDP forbids forwarding a sharded leaf (condition_proj) before the root.

    ``MiniMaxH3.extra_conds`` calls ``preprocess_text_embeds``, which would run
    ``condition_proj``/``token_refiner`` as the first forward on the sharded tree.
    The runtime must replace that with an identity pass-through so projection is
    deferred into the root forward (``h3_ulysses_forward``), preserving FSDP's
    "enter through root first" invariant.
    """
    calls = []

    class Attn:
        pass

    class Block:
        def __init__(self):
            self.attn = Attn()

    class MiniMaxH3:
        pass

    class Diffusion(MiniMaxH3):
        def __init__(self):
            self.blocks = [Block(), Block()]
            self._forward = "original"
            self.preprocess_text_embeds = lambda self, text: (calls.append(("orig", text)), object())[1]

    class Base(MiniMaxH3):
        diffusion_model = Diffusion()

    class Patcher:
        model = Base()

    patcher = inject_minimax_h3_ulysses(
        Patcher(),
        minimax_h3_class=MiniMaxH3,
        attention_forward=lambda self, *a, **k: "attn",
        dit_forward=lambda *a, **k: "dit",
    )
    diffusion = patcher.model.diffusion_model

    # preprocess_text_embeds must be an identity pass-through (no sharded leaf).
    token = object()
    out = diffusion.preprocess_text_embeds(token)
    assert out is token
    assert calls == []  # original eager projection never invoked
    # _forward + attention still injected.
    assert diffusion._forward != "original"  # injected (callable bound method)
    assert callable(diffusion._forward)
    assert diffusion.blocks[0].attn.forward() == "attn"


def test_passthrough_preprocess_returns_input_unchanged() -> None:
    token = object()
    assert passthrough_preprocess_text_embeds(object(), token) is token


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


def test_flat_callback_x0_is_reconstructed_to_nested_av_shape() -> None:
    class Nested:
        is_nested = True

        def __init__(self, tensors):
            self.tensors = tuple(tensors)

        def unbind(self):
            return self.tensors

    video = types.SimpleNamespace(shape=(1, 4, 16, 32, 32))
    audio = types.SimpleNamespace(shape=(1, 8, 64))
    samples = Nested((video, audio))
    flat_x0 = types.SimpleNamespace(is_nested=False)
    calls = []

    rebuilt = reconstruct_nested_x0(
        samples,
        flat_x0,
        nested_type=Nested,
        unpack_latents=lambda value, shapes: (calls.append((value, shapes)), ("video-x0", "audio-x0"))[1],
    )

    assert rebuilt.unbind() == ("video-x0", "audio-x0")
    assert calls == [(flat_x0, [video.shape, audio.shape])]


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


def test_disabled_spectrum_config_removes_controller_without_sampling_hooks() -> None:
    diffusion = types.SimpleNamespace(_h3_t4_spectrum_controller=object())
    worker = H3T4Worker.__new__(H3T4Worker)
    worker.model = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=diffusion))
    worker.spectrum_controller = None

    assert worker.configure_spectrum({"enabled": False})
    assert worker.spectrum_controller is None
    assert not hasattr(diffusion, "_h3_t4_spectrum_controller")


def test_runtime_closure_has_no_donor_or_excluded_model_imports() -> None:
    forbidden = (
        "raylight",
        "comfyui_spectrum_h3",
        "controlnet",
        "pipefusion",
        "gguf",
        "nvfp4",
        "fp8",
        "easycache",
        "bob_triton",
        "lora",
    )
    violations = []
    for path in sorted((ROOT / "src" / "minimax_h3_t4" / "runtime").rglob("*.py")):
        source = path.read_text().lower()
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
