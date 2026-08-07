"""Registration tests for the MiniMax H3 T4 extension package."""

from pathlib import Path
import importlib.util
import sys
import types

from src import minimax_h3_t4
from src.minimax_h3_t4.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


EXPECTED_NODES = {
    "H3T4Initializer",
    "H3T4UNETLoader",
    "H3T4Spectrum",
    "H3T4BasicScheduler",
    "H3T4BasicGuider",
    "H3T4SamplerAdvanced",
}


def test_manifest_includes_git_clone_comfy_entry_point() -> None:
    manifest = (Path(__file__).parents[1] / "MANIFEST.in").read_text()
    assert "include __init__.py" in manifest.splitlines()


def test_git_clone_entry_point_uses_ray_importable_worker_module() -> None:
    root = Path(__file__).parents[1]
    parent = types.ModuleType("custom_nodes")
    parent.__path__ = [str(root.parent)]
    sys.modules["custom_nodes"] = parent
    spec = importlib.util.spec_from_file_location(
        "custom_nodes.minimax_h3_t4_test",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        worker_module = module.NODE_CLASS_MAPPINGS["H3T4Initializer"].__module__
        assert worker_module.startswith("minimax_h3_t4.")
    finally:
        sys.modules.pop(spec.name, None)
        sys.modules.pop("custom_nodes", None)


def test_source_package_is_a_comfy_entry_point() -> None:
    assert minimax_h3_t4.NODE_CLASS_MAPPINGS is NODE_CLASS_MAPPINGS
    assert minimax_h3_t4.NODE_DISPLAY_NAME_MAPPINGS is NODE_DISPLAY_NAME_MAPPINGS
    assert minimax_h3_t4.WEB_DIRECTORY == "./web"


def test_extension_registers_only_the_supported_h3_nodes() -> None:
    assert set(NODE_CLASS_MAPPINGS) == EXPECTED_NODES
    assert set(NODE_DISPLAY_NAME_MAPPINGS) == EXPECTED_NODES


def test_registered_nodes_have_complete_comfy_contracts() -> None:
    for node_class in NODE_CLASS_MAPPINGS.values():
        assert callable(node_class.INPUT_TYPES)
        assert isinstance(node_class.RETURN_TYPES, tuple)
        assert isinstance(node_class.FUNCTION, str)
        assert isinstance(node_class.CATEGORY, str)
        assert hasattr(node_class, node_class.FUNCTION)
