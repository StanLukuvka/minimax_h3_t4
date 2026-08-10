"""MiniMax-H3-only Ulysses patch injection.

Apache-2.0 donor provenance is recorded in NOTICE.md.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any


def passthrough_preprocess_text_embeds(self, text_states):
    """Return text unchanged so FSDP never forwards a sharded leaf first.

    The stock ComfyUI implementation runs ``condition_proj``/``token_refiner``
    eagerly (when text_dim != hidden_size). Under ``fully_shard`` those are
    separately-wrapped leaf modules, and forwarding one before the root
    ``MiniMaxH3Model`` forward violates FSDP's "enter through the root first"
    invariant. The injected ``h3_ulysses_forward`` performs this projection
    inside the root forward, so the pre-pass here is an identity instead.
    """
    del self
    return text_states


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
    model.preprocess_text_embeds = types.MethodType(passthrough_preprocess_text_embeds, model)
    return model_patcher
