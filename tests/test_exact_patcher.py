from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.minimax_h3_t4.runtime.patcher import h3_fsdp_patcher_class


class FakeBasePatcher:
    def __init__(self, model, load_device, offload_device, size=0):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.size = size
        self.patches = {}
        self.object_patches = {}
        self.weight_wrapper_patches = {}
        self.object_patches_backup = {}
        self.backup = {}
        self.patches_uuid = "exact"
        self.ejected = False
        self.hooks_unpatched = False
        self.injected = False

    def inject_model(self):
        self.injected = True

    def eject_model(self):
        self.ejected = True

    def unpatch_hooks(self):
        self.hooks_unpatched = True


class Model:
    def __init__(self):
        self.model_loaded_weight_memory = 0
        self.model_offload_buffer_memory = 0
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.device = "cpu"
        self.current_weight_patches_uuid = None
        self.current_patcher = None
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


def make_patcher(cpu_offload: bool = False):
    patcher_type = h3_fsdp_patcher_class(FakeBasePatcher)
    model = Model()
    return patcher_type(model, "cuda:0", "cpu", size=10_000, cpu_offload=cpu_offload), model


def test_h3_fsdp_patcher_marks_preloaded_shards_without_moving_model() -> None:
    patcher, model = make_patcher()

    loaded = patcher.partially_load("cuda:0", extra_memory=10_000)

    assert loaded == 10_000
    assert model.model_loaded_weight_memory == 10_000
    assert model.current_patcher is patcher
    assert patcher.injected
    assert model.to_calls == []


def test_h3_fsdp_patcher_reports_zero_gpu_memory_when_cpu_offloaded() -> None:
    patcher, model = make_patcher(cpu_offload=True)

    loaded = patcher.partially_load("cuda:0", extra_memory=10_000)

    assert loaded == 0
    assert model.model_loaded_weight_memory == 0
    assert model.current_patcher is patcher
    assert patcher.injected


def test_h3_fsdp_patcher_never_offloads_dtensor_model() -> None:
    patcher, model = make_patcher()
    patcher.partially_load("cuda:0")

    patcher.unpatch_model("cpu")

    assert model.to_calls == []
    assert model.model_loaded_weight_memory == 0
    assert patcher.ejected


def test_h3_fsdp_patcher_rejects_weight_patches() -> None:
    patcher, _model = make_patcher()
    patcher.patches = {"diffusion_model.block.weight": SimpleNamespace()}

    with pytest.raises(ValueError, match="does not support weight"):
        patcher.partially_load("cuda:0")
