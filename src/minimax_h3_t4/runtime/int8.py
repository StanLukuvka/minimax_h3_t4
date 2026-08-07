# INT8 tensorwise patches for standalone MiniMax-H3 FSDP operations.
from __future__ import annotations

import os
from typing import Any, cast

import torch


def h3_cuda_phase_report(*args, **kwargs):
    return None


def h3_memory_snapshot(*args, **kwargs):
    return None


def h3_memory_trace_enabled():
    return False


def h3_phase_profile_active():
    return False


def h3_phase_profile_enabled():
    return False


_PATCHED = False
_MISSING = object()
_ORIG_LAYOUT_PRE = _MISSING
_ORIG_LAYOUT_POST = _MISSING
_PATCHED_LAYOUT = None
_ORIG_QT_PRE = _MISSING
_ORIG_QT_POST = _MISSING
_ORIG_EAGER_INT8_LINEAR = _MISSING
_ORIG_CUDA_INT8_LINEAR = _MISSING
_INT8_TRACE_CALLS = 0
_INT8_PHASE_CALLS = 0
_INT8_PHASE_EVENTS = []
_INT8_PHASE_NAMES = {
    (5376, 21504): "qkv_int8",
    (7168, 5376): "attention_output_int8",
    (5376, 28672): "fc1_int8",
    (14336, 5376): "fc2_int8",
}


def call_cuda_int8_with_oom_retry(operation, *, torch_module=torch, **kwargs):
    """Retry one CUDA INT8 operation exactly once after a genuine CUDA OOM."""
    try:
        return operation(**kwargs)
    except torch_module.OutOfMemoryError:
        torch_module.cuda.empty_cache()
        return operation(**kwargs)


def _int8_accumulator_mib() -> int:
    raw = os.environ.get("H3_T4_INT8_ACCUMULATOR_MIB", "128")
    try:
        accumulator_mib = int(raw)
    except ValueError as exc:
        raise ValueError(f"H3_T4_INT8_ACCUMULATOR_MIB must be an integer, got {raw!r}") from exc
    if not 16 <= accumulator_mib <= 256:
        raise ValueError(f"H3_T4_INT8_ACCUMULATOR_MIB must be between 16 and 256, got {accumulator_mib}")
    return accumulator_mib


def _bounded_eager_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
) -> torch.Tensor:
    """Run eager INT8 linear without a full int32 accumulator plus BF16 part list.

    Comfy Kitchen's generic eager fallback computes the complete int32 GEMM, builds
    every scaled BF16 chunk, then concatenates those chunks.  H3 on a 15 GiB T4
    reaches 13.22 GiB before that final 629 MiB concatenation.  Compute GEMM rows
    in bounded pieces directly into one output allocation instead.
    """
    global _INT8_PHASE_CALLS, _INT8_TRACE_CALLS

    backend = os.environ.get("H3_T4_INT8_BACKEND", "cuda").strip().lower()
    if backend == "cuda":
        return _profiled_cuda_int8_linear(
            x=x,
            weight=weight,
            weight_scale=weight_scale,
            bias=bias,
            out_dtype=out_dtype,
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
            input_act=input_act,
        )
    if backend != "eager":
        raise ValueError(f"Unsupported H3 INT8 backend: {backend!r}")

    from comfy_kitchen.backends.eager import quantization as eager_quantization
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

    trace_call = _INT8_TRACE_CALLS
    phase_call = _INT8_PHASE_CALLS
    profile_this_linear = h3_phase_profile_active() and phase_call < 4
    if h3_phase_profile_active():
        _INT8_PHASE_CALLS += 1
    phase_start = None
    if profile_this_linear:
        phase_start = torch.cuda.Event(enable_timing=True)
        phase_start.record()
    x = eager_quantization._apply_input_act(x, input_act)
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}")

    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)} "
            f"for weight shape {tuple(weight.shape)}"
        )
    if convrot:
        if x.shape[-1] % convrot_groupsize != 0:
            raise ValueError(f"ConvRot group size {convrot_groupsize} does not divide input features {x.shape[-1]}")
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, h, convrot_groupsize)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    x_8, x_scale = eager_quantization.quantize_int8_rowwise(x_2d)
    m, n = x_8.shape[0], weight.shape[0]
    accumulator_mib = _int8_accumulator_mib()
    # Allocator snapshots can synchronize the device and contaminate CUDA-event
    # timings. Keep detailed INT8 memory tracing and phase timing as exclusive
    # diagnostic modes; attention-boundary snapshots remain available in the
    # phase-profile run.
    trace_this_linear = h3_memory_trace_enabled() and not h3_phase_profile_enabled() and _INT8_TRACE_CALLS < 12
    _INT8_TRACE_CALLS += 1
    if trace_this_linear:
        h3_memory_snapshot(
            "int8_linear_start",
            call=trace_call,
            m=m,
            k=x_8.shape[1],
            n=n,
            accumulator_mib=accumulator_mib,
            convrot=convrot,
        )
    output = torch.empty((m, n), dtype=out_dtype, device=x.device)
    if trace_this_linear:
        h3_memory_snapshot(
            "int8_output_allocated",
            call=trace_call,
            output_bytes=output.numel() * output.element_size(),
        )
    weight_t = weight.T.contiguous()
    weight_scale = weight_scale.reshape(1, -1)
    bias_out = None if bias is None else bias.to(device=x.device, dtype=out_dtype).reshape(1, -1)

    # Default to the measured 128 MiB baseline. Larger canvases can lower this
    # proactively without changing ConvRot, DTensor, or FSDP semantics.
    chunk_size = max(1, min(m, accumulator_mib * 1024 * 1024 // (n * 4)))
    for i in range(0, m, chunk_size):
        end_i = min(i + chunk_size, m)
        accumulator = eager_quantization._int8_matmul_accumulate(x_8[i:end_i], weight_t)
        scaled = accumulator.float()
        scales = x_scale[i:end_i].to(device=x.device, dtype=torch.float32) * weight_scale
        scaled.mul_(scales)
        output[i:end_i].copy_(scaled.to(out_dtype))
        if bias_out is not None:
            output[i:end_i].add_(bias_out)

    phase_end = None
    if profile_this_linear:
        phase_end = torch.cuda.Event(enable_timing=True)
        phase_end.record()
    if trace_this_linear:
        h3_memory_snapshot("int8_linear_end", call=trace_call, chunk_rows=chunk_size)
    if profile_this_linear:
        phase_name = _INT8_PHASE_NAMES.get((x_8.shape[1], n), f"int8_call_{phase_call}")
        _INT8_PHASE_EVENTS.append((phase_name, phase_start, phase_end))
        if phase_call == 3:
            h3_cuda_phase_report(
                _INT8_PHASE_EVENTS,
                group="h3_first_block_packed_int8",
            )
            _INT8_PHASE_EVENTS.clear()
    return output.reshape(*orig_shape[:-1], weight.shape[0])


def _call_cuda_int8_with_oom_retry(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    convrot: bool,
    convrot_groupsize: int,
    input_act: str | None,
) -> torch.Tensor:
    """Retry one fragmented large-output allocation after releasing CUDA cache."""
    kwargs = {
        "x": x,
        "weight": weight,
        "weight_scale": weight_scale,
        "bias": bias,
        "out_dtype": out_dtype,
        "convrot": convrot,
        "convrot_groupsize": convrot_groupsize,
        "input_act": input_act,
    }
    try:
        return cast(Any, _ORIG_CUDA_INT8_LINEAR)(**kwargs)
    except torch.OutOfMemoryError:
        free_before, _ = torch.cuda.mem_get_info(x.device)
        torch.cuda.empty_cache()
        free_after, _ = torch.cuda.mem_get_info(x.device)
        output_rows = x.numel() // x.shape[-1]
        output_bytes = output_rows * weight.shape[0] * torch.tensor([], dtype=out_dtype).element_size()
        print(
            "[h3-t4-int8-oom-retry] "
            f"rank={os.environ.get('H3_T4_RANK', '-1')} "
            f"output_mib={output_bytes / (1024 * 1024):.1f} "
            f"free_before_mib={free_before / (1024 * 1024):.1f} "
            f"free_after_mib={free_after / (1024 * 1024):.1f}"
        )
        return cast(Any, _ORIG_CUDA_INT8_LINEAR)(**kwargs)


def _profiled_cuda_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
) -> torch.Tensor:
    """Time the four first-block CUDA INT8 calls without per-op synchronization."""
    global _INT8_PHASE_CALLS

    if os.environ.get("H3_T4_INT8_BACKEND", "eager").strip().lower() != "cuda":
        raise RuntimeError("Eager INT8 requested but Comfy Kitchen selected CUDA backend")
    phase_call = _INT8_PHASE_CALLS
    profile_this_linear = h3_phase_profile_active() and phase_call < 4
    if h3_phase_profile_active():
        _INT8_PHASE_CALLS += 1
    phase_start = None
    if profile_this_linear:
        phase_start = torch.cuda.Event(enable_timing=True)
        phase_start.record()

    result = _call_cuda_int8_with_oom_retry(
        x=x,
        weight=weight,
        weight_scale=weight_scale,
        bias=bias,
        out_dtype=out_dtype,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
        input_act=input_act,
    )

    if profile_this_linear:
        phase_end = torch.cuda.Event(enable_timing=True)
        phase_end.record()
        activated_k = x.shape[-1] // (2 if input_act == "swiglu" else 1)
        phase_name = _INT8_PHASE_NAMES.get((activated_k, weight.shape[0]), f"int8_call_{phase_call}")
        _INT8_PHASE_EVENTS.append((phase_name, phase_start, phase_end))
        if phase_call == 3:
            h3_cuda_phase_report(
                _INT8_PHASE_EVENTS,
                group="h3_first_block_cuda_int8",
                backend="cuda",
                rank=int(os.environ.get("H3_T4_RANK", "-1")),
            )
            _INT8_PHASE_EVENTS.clear()
    return result


def _get_op(path: str) -> Any:
    cur = torch
    for part in path.split(".")[1:]:
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def install_int8_patches() -> None:
    global _PATCHED
    global _ORIG_LAYOUT_PRE, _ORIG_LAYOUT_POST, _PATCHED_LAYOUT, _ORIG_QT_PRE, _ORIG_QT_POST
    global _ORIG_EAGER_INT8_LINEAR, _ORIG_CUDA_INT8_LINEAR
    if _PATCHED:
        return

    from comfy_kitchen.tensor.base import QuantizedTensor, register_layout_op
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as KitchenTensorWiseINT8Layout
    from comfy_kitchen.backends import cuda as cuda_backend
    from comfy_kitchen.backends import eager as eager_backend

    if _ORIG_EAGER_INT8_LINEAR is _MISSING:
        _ORIG_EAGER_INT8_LINEAR = eager_backend.int8_linear
    eager_backend.int8_linear = _bounded_eager_int8_linear
    if _ORIG_CUDA_INT8_LINEAR is _MISSING:
        _ORIG_CUDA_INT8_LINEAR = cuda_backend.int8_linear

    try:
        from comfy.quant_ops import get_layout_class as comfy_get_layout_class

        TensorWiseINT8Layout = comfy_get_layout_class("TensorWiseINT8Layout") or KitchenTensorWiseINT8Layout
    except Exception:  # pragma: no cover
        TensorWiseINT8Layout = KitchenTensorWiseINT8Layout
    _PATCHED_LAYOUT = TensorWiseINT8Layout

    def maybe_register(op):
        def deco(fn):
            if op is not None:
                register_layout_op(op, TensorWiseINT8Layout)(fn)
            return fn

        return deco

    def _is_row_scale(scale: Any, qdata: torch.Tensor) -> bool:
        return (
            isinstance(scale, torch.Tensor)
            and scale.dim() > 0
            and scale.numel() > 1
            and qdata.dim() > 0
            and int(scale.shape[0]) == int(qdata.shape[0])
        )

    def _wrap_int8_tensor(qtensor, qdata: torch.Tensor, *, scale=None, orig_shape=None, transposed=None):
        old = qtensor._params
        params = TensorWiseINT8Layout.Params(
            scale=old.scale if scale is None else scale,
            orig_dtype=old.orig_dtype,
            orig_shape=tuple(qdata.shape) if orig_shape is None else tuple(orig_shape),
            is_weight=getattr(old, "is_weight", True),
            convrot=getattr(old, "convrot", False),
            convrot_groupsize=getattr(old, "convrot_groupsize", 256),
            transposed=getattr(old, "transposed", False) if transposed is None else transposed,
        )
        return QuantizedTensor(qdata, qtensor._layout_cls, params)

    def _slice_row_scale(input_tensor, dim, start, end, step):
        scale = input_tensor._params.scale
        if dim != 0 or not _is_row_scale(scale, input_tensor._qdata):
            return scale
        row_slice = [slice(None)] * scale.dim()
        row_slice[0] = slice(start, end, step)
        return scale[tuple(row_slice)]

    def _is_flatten_to_1d(args, kwargs) -> bool:
        shape = None
        if len(args) > 1:
            shape = args[1:] if len(args) > 2 else args[1]
        else:
            shape = kwargs.get("shape", kwargs.get("size", None))
        if isinstance(shape, int):
            return shape == -1
        if isinstance(shape, torch.Size):
            shape = tuple(shape)
        if isinstance(shape, (list, tuple)) and len(shape) == 1:
            try:
                return int(shape[0]) == -1
            except TypeError:
                return False
        return False

    def pre_all_gather(qtensor: QuantizedTensor, mesh):
        qdata = qtensor._qdata.contiguous()
        scale = qtensor._params.scale
        scale_is_sharded = _is_row_scale(scale, qtensor._qdata)
        metadata = {
            "scale_is_sharded": scale_is_sharded,
            "orig_dtype": qtensor._params.orig_dtype,
            "orig_shape": qtensor._params.orig_shape,
            "is_weight": getattr(qtensor._params, "is_weight", True),
            "convrot": getattr(qtensor._params, "convrot", False),
            "convrot_groupsize": getattr(qtensor._params, "convrot_groupsize", 256),
            "transposed": getattr(qtensor._params, "transposed", False),
        }
        if scale_is_sharded:
            return (qdata, scale.contiguous()), metadata
        if isinstance(scale, torch.Tensor):
            scale = scale.to(device=qdata.device)
        metadata["scale"] = scale
        return (qdata,), metadata

    def post_all_gather(qtensor: QuantizedTensor, all_gather_outputs, metadata: Any, param_dtype: torch.dtype, *, out=None):
        gathered_qdata = all_gather_outputs[0] if isinstance(all_gather_outputs, tuple) else all_gather_outputs
        if metadata.get("scale_is_sharded", False):
            scale = all_gather_outputs[1]
        else:
            scale = metadata["scale"]
        if isinstance(scale, torch.Tensor):
            scale = scale.to(device=gathered_qdata.device)
        orig_shape = tuple(gathered_qdata.shape)
        if metadata.get("transposed", False):
            orig_shape = tuple(metadata.get("orig_shape", orig_shape))
        params = TensorWiseINT8Layout.Params(
            scale=scale,
            orig_dtype=metadata.get("orig_dtype", param_dtype),
            orig_shape=orig_shape,
            is_weight=metadata.get("is_weight", True),
            convrot=metadata.get("convrot", False),
            convrot_groupsize=metadata.get("convrot_groupsize", 256),
            transposed=metadata.get("transposed", False),
        )
        if out is not None:
            if not isinstance(out, QuantizedTensor):
                raise TypeError(f"Expected QuantizedTensor out, got {type(out)}")
            out._qdata = gathered_qdata
            out._params = params
            return None
        tensors = (gathered_qdata, scale) if metadata.get("scale_is_sharded", False) else (gathered_qdata,)
        return QuantizedTensor(gathered_qdata, qtensor._layout_cls, params), tensors

    if _ORIG_LAYOUT_PRE is _MISSING:
        _ORIG_LAYOUT_PRE = getattr(TensorWiseINT8Layout, "pre_all_gather", _MISSING)
    if _ORIG_LAYOUT_POST is _MISSING:
        _ORIG_LAYOUT_POST = getattr(TensorWiseINT8Layout, "post_all_gather", _MISSING)
    setattr(TensorWiseINT8Layout, "pre_all_gather", pre_all_gather)
    setattr(TensorWiseINT8Layout, "post_all_gather", post_all_gather)

    def fsdp_pre_all_gather(self, mesh):
        return self.layout_cls.pre_all_gather(self, mesh)

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        return self.layout_cls.post_all_gather(self, all_gather_outputs, metadata, param_dtype, out=out)

    if _ORIG_QT_PRE is _MISSING:
        _ORIG_QT_PRE = getattr(QuantizedTensor, "fsdp_pre_all_gather", _MISSING)
    if _ORIG_QT_POST is _MISSING:
        _ORIG_QT_POST = getattr(QuantizedTensor, "fsdp_post_all_gather", _MISSING)
    setattr(QuantizedTensor, "fsdp_pre_all_gather", fsdp_pre_all_gather)
    setattr(QuantizedTensor, "fsdp_post_all_gather", fsdp_post_all_gather)

    op_all_gather_ops = tuple(
        op
        for op in (
            _get_op("torch.ops.c10d_functional.all_gather_into_tensor.default"),
            _get_op("torch.ops._c10d_functional.all_gather_into_tensor.default"),
        )
        if op is not None
    )
    op_wait_tensor_ops = tuple(
        op
        for op in (
            _get_op("torch.ops.c10d_functional.wait_tensor.default"),
            _get_op("torch.ops._c10d_functional.wait_tensor.default"),
        )
        if op is not None
    )
    op_alias = _get_op("torch.ops.aten.alias.default")
    op_view = _get_op("torch.ops.aten.view.default")
    op_reshape = _get_op("torch.ops.aten.reshape.default")
    op_slice = _get_op("torch.ops.aten.slice.Tensor")
    op_split = _get_op("torch.ops.aten.split.Tensor")
    op_split_with_sizes = _get_op("torch.ops.aten.split_with_sizes.default")
    op_cat = _get_op("torch.ops.aten.cat.default")
    op_new_zeros = _get_op("torch.ops.aten.new_zeros.default")
    op_as_strided = _get_op("torch.ops.aten.as_strided.default")

    def _wait_tensor_if_available(tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor) or len(op_wait_tensor_ops) == 0:
            return tensor
        return op_wait_tensor_ops[0](tensor)

    def _handle_all_gather_impl(op_all_gather, args, kwargs):
        input_tensor = None
        input_idx = None
        for idx, arg in enumerate(args):
            if isinstance(arg, QuantizedTensor):
                input_tensor = arg
                input_idx = idx
                break
        if input_tensor is None:
            return op_all_gather(*args, **kwargs)
        input_idx_i = cast(int, input_idx)

        q_args = list(args)
        q_args[input_idx_i] = input_tensor._qdata.contiguous().view(torch.uint8)
        q_ret = op_all_gather(*q_args, **kwargs)
        q_bytes = q_ret if isinstance(q_ret, torch.Tensor) else q_args[input_idx_i]
        q_bytes = _wait_tensor_if_available(q_bytes)
        gathered_qdata = q_bytes.view(input_tensor._qdata.dtype)

        scale = input_tensor._params.scale
        if _is_row_scale(scale, input_tensor._qdata):
            s_args = list(args)
            s_args[input_idx_i] = scale.contiguous().view(torch.uint8)
            s_ret = op_all_gather(*s_args, **kwargs)
            s_bytes = s_ret if isinstance(s_ret, torch.Tensor) else s_args[input_idx_i]
            s_bytes = _wait_tensor_if_available(s_bytes)
            scale = s_bytes.view(scale.dtype)

        return _wrap_int8_tensor(input_tensor, gathered_qdata, scale=scale, orig_shape=tuple(gathered_qdata.shape))

    for _op_all_gather in op_all_gather_ops:

        @maybe_register(_op_all_gather)
        def handle_all_gather(qt, args, kwargs, _op=_op_all_gather):
            return _handle_all_gather_impl(_op, args, kwargs)

    def _handle_wait_tensor_impl(op_wait_tensor, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_wait_tensor(*args, **kwargs)
        waited_q_bytes = op_wait_tensor(input_tensor._qdata.view(torch.uint8), *args[1:], **kwargs)
        waited_qdata = waited_q_bytes.view(input_tensor._qdata.dtype)
        scale = input_tensor._params.scale
        if _is_row_scale(scale, input_tensor._qdata):
            waited_s_bytes = op_wait_tensor(scale.contiguous().view(torch.uint8), *args[1:], **kwargs)
            scale = waited_s_bytes.view(scale.dtype)
        return _wrap_int8_tensor(input_tensor, waited_qdata, scale=scale, orig_shape=tuple(waited_qdata.shape))

    for _op_wait_tensor in op_wait_tensor_ops:

        @maybe_register(_op_wait_tensor)
        def handle_wait_tensor(qt, args, kwargs, _op=_op_wait_tensor):
            return _handle_wait_tensor_impl(_op, args, kwargs)

    @maybe_register(op_alias)
    def handle_alias(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_alias(*args, **kwargs)
        return _wrap_int8_tensor(input_tensor, op_alias(input_tensor._qdata), orig_shape=input_tensor._params.orig_shape)

    def _handle_shape_op(op, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op(*args, **kwargs)
        new_qdata = op(input_tensor._qdata, *args[1:], **kwargs)
        scale = input_tensor._params.scale
        if _is_row_scale(scale, input_tensor._qdata) and (new_qdata.dim() == 0 or int(new_qdata.shape[0]) != int(scale.shape[0])):
            if _is_flatten_to_1d(args, kwargs):
                return _wrap_int8_tensor(input_tensor, new_qdata, scale=scale)
            raise RuntimeError(
                "INT8 rowwise/ConvRot QuantizedTensor cannot be reshaped without invalidating scale metadata. "
                f"qdata_shape={tuple(input_tensor._qdata.shape)}, new_qdata_shape={tuple(new_qdata.shape)}, "
                f"scale_shape={tuple(scale.shape)}"
            )
        return _wrap_int8_tensor(input_tensor, new_qdata)

    @maybe_register(op_view)
    def handle_view(qt, args, kwargs):
        return _handle_shape_op(op_view, args, kwargs)

    @maybe_register(op_reshape)
    def handle_reshape(qt, args, kwargs):
        return _handle_shape_op(op_reshape, args, kwargs)

    @maybe_register(op_as_strided)
    def handle_as_strided(qt, args, kwargs):
        return _handle_shape_op(op_as_strided, args, kwargs)

    @maybe_register(op_slice)
    def handle_slice(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_slice(*args, **kwargs)
        sliced_qdata = op_slice(input_tensor._qdata, *args[1:], **kwargs)
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        dim = dim if dim >= 0 else dim + input_tensor._qdata.dim()
        start = args[2] if len(args) > 2 else kwargs.get("start", None)
        end = args[3] if len(args) > 3 else kwargs.get("end", None)
        step = args[4] if len(args) > 4 else kwargs.get("step", None)
        scale = _slice_row_scale(input_tensor, dim, start, end, step)
        return _wrap_int8_tensor(input_tensor, sliced_qdata, scale=scale)

    @maybe_register(op_split)
    def handle_split(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_split(*args, **kwargs)
        chunks = op_split(input_tensor._qdata, *args[1:], **kwargs)
        dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
        dim = dim if dim >= 0 else dim + input_tensor._qdata.dim()
        scale = input_tensor._params.scale
        if dim == 0 and _is_row_scale(scale, input_tensor._qdata):
            scale_chunks = op_split(scale, args[1], 0)
            return tuple(_wrap_int8_tensor(input_tensor, chunk, scale=scale_chunk) for chunk, scale_chunk in zip(chunks, scale_chunks))
        return tuple(_wrap_int8_tensor(input_tensor, chunk) for chunk in chunks)

    @maybe_register(op_split_with_sizes)
    def handle_split_with_sizes(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_split_with_sizes(*args, **kwargs)
        chunks = op_split_with_sizes(input_tensor._qdata, *args[1:], **kwargs)
        dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
        dim = dim if dim >= 0 else dim + input_tensor._qdata.dim()
        scale = input_tensor._params.scale
        if dim == 0 and _is_row_scale(scale, input_tensor._qdata):
            scale_chunks = op_split_with_sizes(scale, args[1], 0)
            return tuple(_wrap_int8_tensor(input_tensor, chunk, scale=scale_chunk) for chunk, scale_chunk in zip(chunks, scale_chunks))
        return tuple(_wrap_int8_tensor(input_tensor, chunk) for chunk in chunks)

    @maybe_register(op_cat)
    def handle_cat(qt, args, kwargs):
        tensors = args[0]
        if not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
            return op_cat(*args, **kwargs)
        for tensor in tensors:
            if not isinstance(tensor, QuantizedTensor):
                return op_cat(*args, **kwargs)
        dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
        concatenated = op_cat([tensor._qdata for tensor in tensors], *args[1:], **kwargs)
        first = tensors[0]
        scale = first._params.scale
        if dim == 0 and all(_is_row_scale(tensor._params.scale, tensor._qdata) for tensor in tensors):
            scale = op_cat([tensor._params.scale for tensor in tensors], 0)
        return _wrap_int8_tensor(first, concatenated, scale=scale, orig_shape=tuple(concatenated.shape))

    @maybe_register(op_new_zeros)
    def handle_new_zeros(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_new_zeros(*args, **kwargs)
        new_qdata = op_new_zeros(input_tensor._qdata, *args[1:], **kwargs)
        scale = input_tensor._params.scale
        if _is_row_scale(scale, input_tensor._qdata) and new_qdata.dim() > 0 and int(new_qdata.shape[0]) != int(scale.shape[0]):
            scale = scale.new_zeros((int(new_qdata.shape[0]), *tuple(scale.shape[1:])), device=new_qdata.device)
        return _wrap_int8_tensor(input_tensor, new_qdata, scale=scale)

    _PATCHED = True


def install_int8_cuda_oom_retry(torch_module=torch) -> None:
    """Install the INT8-only FSDP hooks and select ComfyKitchen's CUDA backend."""
    if torch_module is not torch:
        raise TypeError("worker INT8 patches must use the imported torch module")
    os.environ.setdefault("H3_T4_INT8_BACKEND", "cuda")
    install_int8_patches()


def restore_int8_patches() -> None:
    global _PATCHED
    global _ORIG_LAYOUT_PRE, _ORIG_LAYOUT_POST, _PATCHED_LAYOUT, _ORIG_QT_PRE, _ORIG_QT_POST
    global _ORIG_EAGER_INT8_LINEAR, _ORIG_CUDA_INT8_LINEAR

    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as KitchenTensorWiseINT8Layout
    from comfy_kitchen.backends import eager as eager_backend

    if _ORIG_EAGER_INT8_LINEAR is not _MISSING:
        eager_backend.int8_linear = _ORIG_EAGER_INT8_LINEAR

    TensorWiseINT8Layout = _PATCHED_LAYOUT or KitchenTensorWiseINT8Layout

    if _ORIG_LAYOUT_PRE is _MISSING:
        if hasattr(TensorWiseINT8Layout, "pre_all_gather"):
            delattr(TensorWiseINT8Layout, "pre_all_gather")
    else:
        setattr(TensorWiseINT8Layout, "pre_all_gather", _ORIG_LAYOUT_PRE)

    if _ORIG_LAYOUT_POST is _MISSING:
        if hasattr(TensorWiseINT8Layout, "post_all_gather"):
            delattr(TensorWiseINT8Layout, "post_all_gather")
    else:
        setattr(TensorWiseINT8Layout, "post_all_gather", _ORIG_LAYOUT_POST)

    if _ORIG_QT_PRE is _MISSING:
        if hasattr(QuantizedTensor, "fsdp_pre_all_gather"):
            delattr(QuantizedTensor, "fsdp_pre_all_gather")
    else:
        setattr(QuantizedTensor, "fsdp_pre_all_gather", _ORIG_QT_PRE)

    if _ORIG_QT_POST is _MISSING:
        if hasattr(QuantizedTensor, "fsdp_post_all_gather"):
            delattr(QuantizedTensor, "fsdp_post_all_gather")
    else:
        setattr(QuantizedTensor, "fsdp_post_all_gather", _ORIG_QT_POST)

    _ORIG_LAYOUT_PRE = _MISSING
    _ORIG_LAYOUT_POST = _MISSING
    _PATCHED_LAYOUT = None
    _ORIG_QT_PRE = _MISSING
    _ORIG_QT_POST = _MISSING
    _ORIG_EAGER_INT8_LINEAR = _MISSING
    _ORIG_CUDA_INT8_LINEAR = _MISSING
    _PATCHED = False
