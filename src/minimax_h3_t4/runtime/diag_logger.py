"""Comprehensive H3-T4 diagnostic logger.

Logs CUDA memory, tensor shapes, and FSDP shard state at every critical
checkpoint so we can pinpoint the exact allocation causing OOM.
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


_DIAG_ENABLED = os.environ.get("H3_T4_DIAG", "1") == "1"
_DIAG_DIR = os.environ.get("H3_T4_DIAG_DIR", "/tmp/h3-t4-diag")


def _ensure_dir() -> None:
    os.makedirs(_DIAG_DIR, exist_ok=True)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "-1"))


def log_memory(tag: str, extra: dict[str, Any] | None = None) -> None:
    """Log CUDA memory state at the current point."""
    if not _DIAG_ENABLED:
        return
    dev = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    free, total = torch.cuda.mem_get_info(dev)
    
    lines = [
        f"[{_ts()}] rank={_rank()} tag={tag}",
        f"  allocated={allocated / 1024**3:.2f}GiB",
        f"  reserved={reserved / 1024**3:.2f}GiB",
        f"  free={free / 1024**3:.2f}GiB",
        f"  total={total / 1024**3:.2f}GiB",
        f"  utilization={allocated / total * 100:.1f}%",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"  {k}={v}")
    print("\n".join(lines), flush=True)


def log_tensor(name: str, tensor: torch.Tensor, tag: str = "") -> None:
    """Log tensor shape and memory footprint."""
    if not _DIAG_ENABLED or tensor is None:
        return
    numel = tensor.numel()
    elem_size = tensor.element_size()
    bytes = numel * elem_size
    lines = [
        f"[{_ts()}] TENSOR {name} {tag}",
        f"  shape={tuple(tensor.shape)}",
        f"  dtype={tensor.dtype}",
        f"  numel={numel}",
        f"  bytes={bytes / 1024**2:.2f}MiB",
    ]
    print("\n".join(lines), flush=True)


def log_fsdp_shard_state(model: torch.nn.Module, tag: str = "") -> None:
    """Log FSDP shard info for all parameters."""
    if not _DIAG_ENABLED:
        return
    lines = [f"[{_ts()}] FSDP SHARD STATE {tag}"]
    total_params = 0
    total_bytes = 0
    for name, param in model.named_parameters():
        if hasattr(param, "_fsdp_wrapped_module"):
            # FSDP wrapped
            shard_info = getattr(param, "_shard_metadata", {})
            lines.append(f"  {name}: FSDP-sharded, local_shape={tuple(param.shape)}")
        else:
            lines.append(f"  {name}: local_shape={tuple(param.shape)}")
        total_params += param.numel()
        total_bytes += param.numel() * param.element_size()
    lines.append(f"  TOTAL: {total_params} params, {total_bytes / 1024**3:.2f}GiB (full)")
    print("\n".join(lines), flush=True)


def log_attention_memory(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tag: str = "") -> None:
    """Log attention input tensors and predicted output size."""
    if not _DIAG_ENABLED:
        return
    bs_q, seq_q, heads_q, dim_q = q.shape
    bs_k, seq_k, heads_k, dim_k = k.shape
    bs_v, seq_v, heads_v, dim_v = v.shape
    
    # Scaled dot-product attention memory estimate
    attn_scores = seq_q * seq_k  # Q @ K^T
    attn_mem = attn_scores * 2 * bs_q  # float16/bfloat16 = 2 bytes
    out_mem = seq_q * heads_q * dim_q * 2  # output
    
    lines = [
        f"[{_ts()}] ATTENTION MEMORY {tag}",
        f"  Q: shape={q.shape}, bytes={q.numel() * q.element_size() / 1024**2:.2f}MiB",
        f"  K: shape={k.shape}, bytes={k.numel() * k.element_size() / 1024**2:.2f}MiB",
        f"  V: shape={v.shape}, bytes={v.numel() * v.element_size() / 1024**2:.2f}MiB",
        f"  attn_scores: {attn_scores} elements, ~{attn_mem / 1024**2:.2f}MiB",
        f"  output: ~{out_mem / 1024**2:.2f}MiB",
        f"  TOTAL ESTIMATE: ~{(attn_mem + out_mem + q.numel() * q.element_size() + k.numel() * k.element_size() + v.numel() * v.element_size()) / 1024**2:.2f}MiB",
    ]
    print("\n".join(lines), flush=True)


def log_ulysses_alltoall(input_shape: tuple, output_shape: tuple, tag: str = "") -> None:
    """Log all-to-all communication memory."""
    if not _DIAG_ENABLED:
        return
    in_bytes = 1  # dummy, actual computed from shape
    out_bytes = 1
    # Estimate: all-to-all needs input + output buffers
    est_bytes = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * 2  # bfloat16
    lines = [
        f"[{_ts()}] ALL-TO-ALL {tag}",
        f"  input_shape={input_shape}",
        f"  output_shape={output_shape}",
        f"  estimated_communication_bytes={est_bytes / 1024**2:.2f}MiB",
    ]
    print("\n".join(lines), flush=True)


def profile_start() -> float:
    """Start a timing probe."""
    return time.perf_counter()


def profile_end(label: str, start: float) -> float:
    """End a timing probe and log."""
    elapsed = time.perf_counter() - start
    print(f"[{_ts()}] PROFILE {label}: {elapsed * 1000:.1f}ms", flush=True)
    return elapsed


def _get_host_rss_bytes() -> int:
    """Return the current process RSS in bytes.

    Uses ``psutil`` when available (preferred) and falls back to
    ``/proc/self/status`` on Linux so the module stays importable without
    dependencies when psutil is absent.
    """
    try:
        import psutil

        pid = os.getpid()
        process = psutil.Process(pid)
        return process.memory_info().rss
    except ImportError:
        pass
    proc_status = Path("/proc/self/status")
    if proc_status.is_file():
        text = proc_status.read_text()
        for line in text.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) == 3 and parts[1].lower() == "kb":
                    try:
                        return int(parts[2]) * 1024
                    except ValueError:
                        pass
    return -1


def log_host(tag: str, extra: dict[str, Any] | None = None) -> None:
    """Log host-process RSS memory at the current point.

    No-op when ``H3_T4_DIAG`` is disabled.  Uses psutil when available and
    falls back to ``/proc/self/status`` on Linux.  Never imports psutil at
    module load time so the diagnostic submodule remains importable in
    environments that lack psutil.
    """
    if not _DIAG_ENABLED:
        return
    rss_bytes = _get_host_rss_bytes()
    parts = [
        f"[{_ts()}] rank={_rank()} HOST tag={tag}",
        f"  rss={rss_bytes / 1024**3:.2f}GiB" if rss_bytes >= 0 else "  rss=unavailable",
    ]
    if extra:
        for k, v in extra.items():
            parts.append(f"  {k}={v}")
    print(" \n".join(parts), flush=True)


def write_summary(filepath: str, data: dict[str, Any]) -> None:
    """Write a summary JSON file."""
    import json
    _ensure_dir()
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[{_ts()}] SUMMARY written to {filepath}", flush=True)
