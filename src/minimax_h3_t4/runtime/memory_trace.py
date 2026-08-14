"""Memory diagnostics for the MiniMax-H3 two-T4 runtime.

Gated by H3_T4_MEMORY_TRACE=1. When enabled, prints a per-rank, per-phase
snapshot of CUDA allocation state (allocated, reserved, peak, free) at each
named checkpoint so an OOM can be attributed to the exact stage that pushed
the card past its ceiling.

No behavior change when the env var is unset: every call becomes a no-op so
the instrumentation cannot perturb the run it is diagnosing.
"""
from __future__ import annotations

import os

import torch

_TRACE = os.environ.get("H3_T4_MEMORY_TRACE", "0") == "1"


def memory_trace_enabled() -> bool:
    return _TRACE


def _fmt_gib(n: float) -> str:
    return f"{n / (1024 ** 3):.2f}GiB"


def memory_snapshot(tag: str, *, device: torch.device | None = None, extra: dict[str, object] | None = None) -> None:
    """Log one allocation-state checkpoint. No-op unless H3_T4_MEMORY_TRACE=1."""
    if not _TRACE:
        return
    dev = device if device is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    try:
        peak = torch.cuda.max_memory_allocated(dev)
    except Exception:  # pragma: no cover - torch version variance
        peak = allocated
    try:
        free_total = torch.cuda.mem_get_info(dev)[0]
    except Exception:  # pragma: no cover
        free_total = 0
    total = torch.cuda.get_device_properties(dev).total_memory
    rank = os.environ.get("H3_T4_RANK", "-1")
    parts = [
        f"[h3-mem] rank={rank}",
        f"phase={tag}",
        f"alloc={_fmt_gib(allocated)}",
        f"reserved={_fmt_gib(reserved)}",
        f"peak={_fmt_gib(peak)}",
        f"free={_fmt_gib(free_total)}",
        f"total={_fmt_gib(total)}",
    ]
    if extra:
        for key, value in extra.items():
            parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)


def reset_peak(device: torch.device | None = None) -> None:
    """Reset the peak-memory counter for a fresh region (call before a block)."""
    if not _TRACE:
        return
    torch.cuda.reset_peak_memory_stats(device if device is not None else torch.cuda.current_device())
