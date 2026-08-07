"""Direct TensorWise INT8-to-FSDP shard mapping.

Apache-2.0 donor provenance is recorded in NOTICE.md.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

try:
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class
except Exception:  # pragma: no cover
    from comfy_kitchen.tensor import QuantizedTensor, get_layout_class  # type: ignore

    QUANT_ALGOS = {}


"""
 This stuff is for systematically fully_shard from bottom up,
 with "*-1 parents are sharded, then continue up to root*", the def collect_bottom_up_shard_order is the function
 the tree looks like this:

model                          [FSDP]
├── block0                     [FSDP]
│   ├── qkv                    [FSDP, ignored_params={q.scale,k.scale,v.scale}]
│   │   ├── q.weight           [SHARDED]
│   │   ├── q.scale            [IGNORED]
│   │   ├── k.weight           [SHARDED]
│   │   ├── k.scale            [IGNORED]
│   │   ├── v.weight           [SHARDED]
│   │   └── v.scale            [IGNORED]
│   ├── ffn                    [FSDP, ignored_params={scale}]
│   │   ├── weight             [SHARDED]
│   │   └── scale              [IGNORED]
│   └── conv                   [FSDP, ignored_params={} ]
│       ├── weight             [SHARDED]
│       └── bias               [SHARDED]
│
├── block1                     [FSDP]
│   ├── qkv                    [FSDP, ignored_params={q.scale,k.scale,v.scale}]
│   ├── ffn                    [FSDP, ignored_params={scale}]
│   └── conv                   [FSDP]
│
└── block2                     [FSDP]
    ├── qkv                    [FSDP, ignored_params={q.scale,k.scale,v.scale}]
    ├── ffn                    [FSDP, ignored_params={scale}]
    └── conv                   [FSDP]
"""


def freeze_and_detect_qt(model: torch.nn.Module) -> bool:
    has_qt = False
    for param in model.parameters():
        param.requires_grad = False
        local = getattr(param, "_local_tensor", None)
        if isinstance(param, QuantizedTensor) or isinstance(local, QuantizedTensor):
            has_qt = True
    return has_qt


def _mod_name(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _module_has_subtree_params(module: torch.nn.Module) -> bool:
    return any(True for _ in module.named_parameters(recurse=True))


def _module_has_direct_params(module: torch.nn.Module) -> bool:
    return any(True for _ in module.named_parameters(recurse=False))


def _children_with_params(name: str, module: torch.nn.Module, named_map: dict[str, torch.nn.Module]) -> list[str]:
    out: list[str] = []
    for child_name, _child in module.named_children():
        full = _mod_name(name, child_name)
        if _module_has_subtree_params(named_map[full]):
            out.append(full)
    return out


def _is_descendant(path: str, ancestor: str) -> bool:
    if ancestor == "":
        return path != ""
    return path == ancestor or path.startswith(ancestor + ".")


def _collect_leaf_parent_targets(model: torch.nn.Module) -> set[str]:
    named = dict(model.named_modules())
    all_names = [name for name in named.keys() if name != ""]

    structural_groups: set[str] = set()
    for name in all_names:
        children = _children_with_params(name, named[name], named)
        if len(children) >= 2:
            structural_groups.add(name)

    targets: set[str] = set(structural_groups)
    for name in all_names:
        if _module_has_direct_params(named[name]):
            if not any(_is_descendant(name, group_name) for group_name in structural_groups):
                targets.add(name)

    return targets


def _add_ancestors_to_root(targets: set[str]) -> set[str]:
    out = set(targets)
    for target in list(targets):
        cur = target
        while "." in cur:
            cur = cur.rsplit(".", 1)[0]
            out.add(cur)
    out.add("")
    return out


def _depth(name: str) -> int:
    return 0 if name == "" else name.count(".") + 1


def _supports_fully_shard(module: torch.nn.Module) -> bool:
    return type(module).forward is not torch.nn.Module.forward


def collect_bottom_up_shard_order(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    named = dict(model.named_modules())
    leaf_parents = _collect_leaf_parent_targets(model)
    all_targets = _add_ancestors_to_root(leaf_parents)
    ordered_names = sorted(all_targets, key=_depth, reverse=True)

    out: list[tuple[str, torch.nn.Module]] = []
    for name in ordered_names:
        module = model if name == "" else named[name]
        if _supports_fully_shard(module):
            out.append((name, module))
    return out


def collect_scale_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for param_name, param in module.named_parameters(recurse=True):
        if "scale" in param_name:
            ignored.add(param)
    return ignored


def collect_input_scale_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for param_name, param in module.named_parameters(recurse=True):
        if "input_scale" in param_name:
            ignored.add(param)
    return ignored


def collect_scalar_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for _param_name, param in module.named_parameters(recurse=True):
        if param.ndim == 0:
            ignored.add(param)
    return ignored


def _has_odd_shard_dim(param: torch.Tensor) -> bool:
    return param.ndim > 0 and (int(param.shape[0]) % 2) == 1


def collect_odd_dim0_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for _param_name, param in module.named_parameters(recurse=True):
        if _has_odd_shard_dim(param):
            ignored.add(param)
    return ignored


def _should_materialize_unsharded_param(
    param_name: str,
    param: torch.Tensor,
    full_sd: dict[str, Any] | None = None,
) -> bool:
    if param_name.endswith("input_scale") or param.ndim == 0:
        return True
    if not _has_odd_shard_dim(param):
        return False
    if full_sd is not None and _is_quant_param(param_name, full_sd, param):
        return False
    return True


def _get_parent_module_and_name(model: torch.nn.Module, param_name: str) -> tuple[torch.nn.Module, str]:
    if "." not in param_name:
        return model, param_name
    parent_name, leaf_name = param_name.rsplit(".", 1)
    return model.get_submodule(parent_name), leaf_name


def _maybe_collapse_replicated_leading_dim(full_tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    expected_shape = tuple(target_shape)
    if tuple(full_tensor.shape) == expected_shape:
        return full_tensor
    if full_tensor.ndim == 0 or full_tensor.ndim != len(expected_shape):
        return full_tensor
    if tuple(full_tensor.shape[1:]) != expected_shape[1:]:
        return full_tensor

    actual_leading = full_tensor.shape[0]
    expected_leading = expected_shape[0]
    if expected_leading <= 0 or actual_leading < expected_leading or actual_leading % expected_leading != 0:
        return full_tensor

    replicas = actual_leading // expected_leading
    if replicas <= 1:
        return full_tensor

    collapsed = full_tensor.reshape(expected_leading, replicas, *full_tensor.shape[1:])
    canonical = collapsed[:, 0, ...]
    if torch.equal(collapsed, canonical.unsqueeze(1).expand_as(collapsed)):
        return canonical
    return full_tensor


def _materialize_unsharded_param(
    model: torch.nn.Module,
    param_name: str,
    meta_param: torch.Tensor,
    full_tensor: torch.Tensor,
    device: torch.device,
    cpu_offload: bool,
) -> None:
    full_tensor = _maybe_collapse_replicated_leading_dim(full_tensor, meta_param.shape)
    full_tensor = full_tensor.to(dtype=meta_param.dtype, device=device)
    if cpu_offload:
        full_tensor = full_tensor.cpu()
    parent_module, leaf_name = _get_parent_module_and_name(model, param_name)
    parent_module.register_parameter(
        leaf_name,
        torch.nn.Parameter(full_tensor, requires_grad=meta_param.requires_grad),
    )


def _materialize_missing_ignored_params(
    model: torch.nn.Module,
    full_sd: dict[str, Any],
    device: torch.device,
    strict: bool,
    cpu_offload: bool,
    release_sd: bool,
) -> None:
    for param_name, param in list(model.named_parameters()):
        if not getattr(param, "is_meta", False):
            continue
        if not _should_materialize_unsharded_param(param_name, param, full_sd):
            continue
        full_tensor = full_sd.get(param_name)
        if full_tensor is None:
            if strict:
                raise ValueError(f"Missing parameter {param_name} in state_dict")
            continue
        _materialize_unsharded_param(model, param_name, param, full_tensor, device, cpu_offload)
        if release_sd:
            full_sd[param_name] = None


def _collect_subtree_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    return set(module.parameters())


def fully_shard_bottom_up(
    model: torch.nn.Module,
    fsdp_kwargs: dict[str, Any],
    native_ignore_scale: bool,
    ignored_modules: set[torch.nn.Module] | None = None,
) -> int:
    excluded_params: set[torch.nn.Parameter] = set()
    ignored_module_ids: set[int] = set()
    if ignored_modules:
        for mod in ignored_modules:
            excluded_params |= _collect_subtree_params(mod)
            ignored_module_ids.update(id(child) for child in mod.modules())

    shard_order = collect_bottom_up_shard_order(model)
    if ignored_modules:
        shard_order = [(n, m) for n, m in shard_order if id(m) not in ignored_module_ids]

    num_layers_sharded = 0
    for _name, module in shard_order:
        kwargs = dict(fsdp_kwargs)
        ignored_params: set[torch.nn.Parameter] = set()
        if native_ignore_scale:
            ignored_params |= collect_scale_ignored_params(module)

        ignored_params |= collect_input_scale_ignored_params(module)
        ignored_params |= collect_scalar_ignored_params(module)
        ignored_params |= collect_odd_dim0_ignored_params(module)

        subtree_params = set(module.parameters())
        ignored_params |= excluded_params & subtree_params

        if ignored_params:
            kwargs["ignored_params"] = ignored_params

        fully_shard(module, **kwargs)
        num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError("No layer modules were sharded. Please check if shard conditions are working as expected.")
    return num_layers_sharded


def materialize_excluded_params(
    model: torch.nn.Module,
    excluded_modules: set[torch.nn.Module],
    full_sd: dict[str, Any],
    device: torch.device,
    cpu_offload: bool = False,
) -> int:
    """Load parameters of modules excluded from FSDP wrapping from a full state dict.

    After FSDP wrapping, excluded-module parameters remain on meta device.
    This function materializes them onto *device* (or CPU when cpu_offload).
    Returns the number of parameters materialized.
    """
    module_to_prefix: dict[int, str] = {}
    for name, mod in model.named_modules():
        module_to_prefix[id(mod)] = name

    count = 0
    for module in excluded_modules:
        prefix = module_to_prefix.get(id(module), "")
        for param_name, param in module.named_parameters(recurse=True):
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if not getattr(param, "is_meta", False):
                continue
            full_tensor = full_sd.get(full_name)
            if full_tensor is None:
                continue
            _materialize_unsharded_param(model, full_name, param, full_tensor, device, cpu_offload)
            count += 1
    return count


def _decode_comfy_quant(conf: Any) -> dict[str, Any] | None:
    if conf is None:
        return None
    if isinstance(conf, dict):
        return conf
    if isinstance(conf, (bytes, bytearray)):
        return json.loads(conf.decode("utf-8"))
    if isinstance(conf, torch.Tensor):
        raw = conf.detach().cpu().numpy().tobytes()
        if conf.dtype == torch.uint8:
            return json.loads(raw)
        return json.loads(raw.decode("utf-8"))
    if isinstance(conf, str):
        return json.loads(conf)
    raise TypeError(f"Unsupported comfy_quant type: {type(conf)}")


def _quant_payload_debug_info(param_name: str, full_sd: dict[str, Any]) -> str:
    prefix = param_name[: -len("weight")] if param_name.endswith("weight") else param_name
    debug_bits = {
        "weight": param_name in full_sd,
        "comfy_quant": f"{prefix}comfy_quant" in full_sd,
        "weight_scale": f"{prefix}weight_scale" in full_sd,
    }
    prefix_keys = sorted(
        key
        for key in full_sd.keys()
        if key.startswith(prefix) and (key == param_name or key.endswith("comfy_quant") or key.endswith("weight_scale"))
    )
    return f"payload={debug_bits}, prefix_keys={prefix_keys}"


def _shard_tensor(
    full_tensor: torch.Tensor,
    sharded_meta_param: Any,
    device: torch.device,
    *,
    pad_to_local_meta: bool = True,
) -> torch.Tensor:
    if not hasattr(sharded_meta_param, "device_mesh"):
        return full_tensor.to(device=device)

    mesh = sharded_meta_param.device_mesh
    if mesh.ndim > 1:
        raise NotImplementedError(f"only support 1D FSDP but got {mesh.ndim}")

    shard_mesh_dim = 0
    shard_world_size = mesh.size(shard_mesh_dim)
    shard_rank = cast(torch.distributed.ProcessGroup, mesh.get_group(shard_mesh_dim)).rank()

    chunk = torch.tensor_split(full_tensor, shard_world_size, dim=0)[shard_rank].to(device=device)

    local_meta = getattr(sharded_meta_param, "_local_tensor", None)
    if not pad_to_local_meta or not isinstance(local_meta, torch.Tensor):
        return chunk

    local_shape = tuple(local_meta.shape)
    if tuple(chunk.shape) == local_shape:
        return chunk
    if len(local_shape) != chunk.ndim:
        return chunk
    if any(local_dim < chunk_dim for local_dim, chunk_dim in zip(local_shape, chunk.shape)):
        return chunk

    sharded_param = full_tensor.new_zeros(local_shape, device=device)
    if chunk.numel() > 0:
        sharded_param[tuple(slice(0, dim) for dim in chunk.shape)].copy_(chunk)
    return sharded_param


def _is_quant_param(param_name: str, full_sd: dict[str, Any], sharded_meta_param: Any) -> bool:
    if isinstance(full_sd.get(param_name), QuantizedTensor):
        return True

    prefix = param_name[: -len("weight")] if param_name.endswith("weight") else None
    if prefix is not None and f"{prefix}comfy_quant" in full_sd:
        return True

    if isinstance(sharded_meta_param, QuantizedTensor):
        return True
    if hasattr(sharded_meta_param, "_local_tensor") and isinstance(sharded_meta_param._local_tensor, QuantizedTensor):
        return True

    return False


def _build_quantized_tensor(
    param_name: str,
    full_sd: dict[str, Any],
    sharded_meta_param: Any,
    device: torch.device,
):
    def _local_orig_shape(local_qdata: torch.Tensor, logical_orig_shape: tuple[int, ...] | None) -> tuple[int, ...]:
        if logical_orig_shape is None:
            return tuple(local_qdata.shape)
        return tuple(local_qdata.shape)

    if not param_name.endswith("weight"):
        return None

    full_q = full_sd.get(param_name)
    if isinstance(full_q, QuantizedTensor):
        qt = cast(Any, full_q)
        if qt._layout_cls != "TensorWiseINT8Layout":
            raise NotImplementedError(f"Standalone H3 FSDP accepts only TensorWiseINT8Layout, got {qt._layout_cls} for {param_name}.")
        local_qdata = _shard_tensor(qt._qdata, sharded_meta_param, device, pad_to_local_meta=False)
        local_params = replace(
            qt._params,
            orig_shape=_local_orig_shape(local_qdata, getattr(qt._params, "orig_shape", None)),
        )
        return QuantizedTensor(local_qdata, qt._layout_cls, local_params)

    prefix = param_name[: -len("weight")]
    conf = _decode_comfy_quant(full_sd.get(f"{prefix}comfy_quant"))
    if conf is None:
        return None

    quant_format = conf.get("format", None)
    if quant_format is None or quant_format not in QUANT_ALGOS:
        raise ValueError(f"Unknown quantization format for {param_name}: {quant_format}")
    if quant_format != "int8_tensorwise":
        raise ValueError(f"Standalone MiniMax-H3 FSDP is INT8-only; got {quant_format!r} for {param_name}")
    qconfig = QUANT_ALGOS[quant_format]
    layout_name = qconfig["comfy_tensor_layout"]
    layout_cls = get_layout_class(layout_name)
    if layout_cls is None:
        raise ValueError(f"Missing layout class for {layout_name}")

    full_qdata = full_sd.get(param_name)
    if full_qdata is None:
        raise ValueError(f"Missing quantized weight for {param_name}")

    qdata = full_qdata.to(dtype=qconfig["storage_t"])
    qdata = _shard_tensor(qdata, sharded_meta_param, device, pad_to_local_meta=False)

    params_kwargs: dict[str, Any] = {"orig_shape": tuple(qdata.shape)}

    local_meta = None
    if isinstance(sharded_meta_param, QuantizedTensor):
        local_meta = sharded_meta_param
    elif hasattr(sharded_meta_param, "_local_tensor") and isinstance(sharded_meta_param._local_tensor, QuantizedTensor):
        local_meta = sharded_meta_param._local_tensor
    orig_dtype = None
    if local_meta is not None and hasattr(local_meta, "_params"):
        orig_dtype = getattr(local_meta._params, "orig_dtype", None)
    if orig_dtype is None:
        orig_dtype = getattr(sharded_meta_param, "dtype", None)
    if orig_dtype is not None:
        params_kwargs["orig_dtype"] = orig_dtype

    logical_orig_shape = getattr(getattr(local_meta, "_params", None), "orig_shape", None)
    params_kwargs["orig_shape"] = _local_orig_shape(qdata, logical_orig_shape)

    scale = full_sd.get(f"{prefix}weight_scale")
    if scale is None:
        raise ValueError(f"Missing INT8 weight scale for {param_name}")
    if isinstance(scale, torch.Tensor) and scale.dim() > 0 and scale.numel() > 1:
        scale = _shard_tensor(scale, sharded_meta_param, device, pad_to_local_meta=False)
    elif isinstance(scale, torch.Tensor):
        scale = scale.to(device=device)
    params_kwargs["scale"] = scale
    params_conf = conf.get("params", {})
    if not isinstance(params_conf, dict):
        params_conf = {}
    if conf.get("convrot", params_conf.get("convrot", False)):
        params_kwargs["convrot"] = True
        params_kwargs["convrot_groupsize"] = int(conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)))

    params = layout_cls.Params(**params_kwargs)
    return QuantizedTensor(qdata, layout_name, params)


def _release_quant_keys(full_sd: dict[str, Any], param_name: str) -> None:
    prefix = param_name[: -len("weight")]
    for key in (
        param_name,
        f"{prefix}weight_scale",
        f"{prefix}comfy_quant",
    ):
        if key in full_sd:
            full_sd[key] = None


def _wrap_quantized_local_as_dtensor(
    quant_tensor: torch.Tensor,
    sharded_meta_param: DTensor,
) -> DTensor:
    """Wrap a local quantized shard without invoking tensor view dispatch.

    ``DTensor.from_local`` creates an autograd view with
    ``input.view_as(input)``. Comfy Kitchen handles that view by dequantizing
    the wrapper subclass, transiently expanding an INT8 ConvRot shard to a
    full floating tensor and exhausting a T4. Inference parameters do not
    require the differentiable ``from_local`` boundary, so use DTensor's
    internal constructor with the FSDP-created metadata spec unchanged.
    """
    return DTensor(
        quant_tensor,
        sharded_meta_param._spec,
        requires_grad=quant_tensor.requires_grad,
    )


# Heavily modified from
# https://github.com/meta-pytorch/torchtune/blob/d0f63bb33d00b8bd3905a010b71d8c6324c2e980/torchtune/training/_distributed.py#L336
# Need to be done since dcp loader cause wrong dtype among rank when broadcasting.
def load_from_full_model_state_dict(
    model,
    full_sd,
    device,
    strict=False,
    cpu_offload=False,
    release_sd=True,
):
    meta_sharded_sd = model.state_dict()
    sharded_sd: dict[str, torch.Tensor] = {}
    for param_name, sharded_meta_param in meta_sharded_sd.items():
        parent_module, leaf_name = _get_parent_module_and_name(model, param_name)
        is_buffer = leaf_name in parent_module._buffers
        if not is_buffer and _should_materialize_unsharded_param(param_name, sharded_meta_param, full_sd):
            full_tensor = full_sd.get(param_name)
            if full_tensor is None:
                if strict:
                    raise ValueError(f"Missing parameter {param_name} in state_dict")
                continue
            _materialize_unsharded_param(model, param_name, sharded_meta_param, full_tensor, device, cpu_offload)
            if release_sd:
                full_sd[param_name] = None
            continue

        if not is_buffer and _is_quant_param(param_name, full_sd, sharded_meta_param):
            quant_tensor = _build_quantized_tensor(param_name, full_sd, sharded_meta_param, device)
            if quant_tensor is None:
                debug_info = _quant_payload_debug_info(param_name, full_sd)
                raise ValueError(f"Expected quantized tensor for {param_name}, but could not build it ({debug_info})")
            if hasattr(sharded_meta_param, "device_mesh"):
                sharded_tensor = _wrap_quantized_local_as_dtensor(
                    quant_tensor,
                    sharded_meta_param,
                )
            else:
                sharded_tensor = quant_tensor
            if cpu_offload:
                sharded_tensor = sharded_tensor.cpu()
            sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
            if release_sd:
                _release_quant_keys(full_sd, param_name)
            continue
        full_tensor = full_sd.get(param_name)
        if full_tensor is None:
            if strict:
                raise ValueError(f"Missing parameter {param_name} in state_dict")
            continue
        if not hasattr(sharded_meta_param, "device_mesh"):
            full_tensor = _maybe_collapse_replicated_leading_dim(full_tensor, sharded_meta_param.shape)
            full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)
            sharded_tensor = full_tensor
        else:
            full_tensor = full_tensor.to(sharded_meta_param.dtype)
            local_dense = _shard_tensor(full_tensor, sharded_meta_param, device)
            sharded_tensor = DTensor.from_local(
                local_dense,
                device_mesh=sharded_meta_param.device_mesh,
                placements=sharded_meta_param.placements,
            )
        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()
        sharded_sd[param_name] = sharded_tensor if is_buffer else torch.nn.Parameter(sharded_tensor)
        if release_sd:
            full_sd[param_name] = None
    out = model.load_state_dict(sharded_sd, strict=strict, assign=True)
    _materialize_missing_ignored_params(model, full_sd, device, strict, cpu_offload, release_sd)
    return out
