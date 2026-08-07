from __future__ import annotations

import json
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


def require_int8_state_dict(state_dict: dict[str, Any]) -> None:
    """Fail closed unless every quantized payload is ComfyKitchen tensorwise INT8."""
    configs = [_decode(value) for key, value in state_dict.items() if key.endswith(".comfy_quant")]
    if not configs or any(config is None or config.get("format") != "int8_tensorwise" for config in configs):
        raise ValueError("The standalone MiniMax-H3 runtime is INT8-only (int8_tensorwise required)")
