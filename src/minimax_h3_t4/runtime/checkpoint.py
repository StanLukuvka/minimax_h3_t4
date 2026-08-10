from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _decode(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        return json.loads(value.decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    if hasattr(value, "detach"):
        raw = value.detach().cpu().numpy().tobytes()
        return json.loads(raw.decode("utf-8"))
    raise TypeError(f"Unsupported comfy_quant metadata: {type(value).__name__}")


def load_int8_checkpoint_mmap(
    path: str,
    *,
    load_file: Any | None = None,
    metadata_loader: Any | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    if Path(path).suffix.lower() not in {".safetensors", ".sft"}:
        raise ValueError("The exact MiniMax-H3 runtime requires a safetensors checkpoint for bounded mmap loading")
    loader = load_file
    if loader is None:
        from safetensors.torch import load_file as loader
    state_dict = loader(path, device="cpu")
    if metadata_loader is not None:
        return state_dict, metadata_loader(path)
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return state_dict, handle.metadata() or {}


def normalize_state_dict_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Strip ``prefix`` from every matching key, mirroring ComfyUI loading.

    ``unet_prefix_from_state_dict`` may return ``"model."`` as a bare fallback even
    when no key carries that namespace. Mirror the prior working contract:
    only rewrite keys that actually start with ``prefix``; keys that do not are
    left unchanged (bare), matching ``state_dict_prefix_replace(..., filter_keys=True)``
    when it yields no rewrite.
    """
    if any(key.startswith(prefix) for key in state_dict):
        normalized = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
        if not normalized:
            raise ValueError(f"Checkpoint has no model state under prefix {prefix!r}")
        return normalized
    return state_dict


def require_int8_state_dict(state_dict: dict[str, Any]) -> None:
    """Fail closed unless every quantized payload is ComfyKitchen tensorwise INT8."""
    configs = [_decode(value) for key, value in state_dict.items() if key.endswith(".comfy_quant")]
    if not configs or any(config is None or config.get("format") != "int8_tensorwise" for config in configs):
        raise ValueError("The standalone MiniMax-H3 runtime is INT8-only (int8_tensorwise required)")
    if any(config.get("convrot") is not True for config in configs if config is not None):
        raise ValueError("The standalone MiniMax-H3 runtime requires ConvRot on every INT8 payload")
