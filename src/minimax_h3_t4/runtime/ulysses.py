"""MiniMax-H3-only Ulysses patch injection."""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any


def inject_minimax_h3_ulysses(
    model_patcher: Any,
    *,
    minimax_h3_class: type,
    attention_forward: Callable[..., Any],
    dit_forward: Callable[..., Any],
) -> Any:
    base_model = model_patcher.model
    if not isinstance(base_model, minimax_h3_class):
        raise TypeError(f"Expected ComfyUI MiniMaxH3 model, got {type(base_model).__name__}")
    model = base_model.diffusion_model
    for block in model.blocks:
        block.attn.forward = types.MethodType(attention_forward, block.attn)
    model._forward = types.MethodType(dit_forward, model)
    return model_patcher
