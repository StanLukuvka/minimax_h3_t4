from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
EXPECTED_NODES = {
    "H3T4Initializer",
    "H3T4UNETLoader",
    "H3T4BasicScheduler",
    "H3T4BasicGuider",
    "H3T4Spectrum",
    "H3T4SamplerAdvanced",
}
FORBIDDEN_RUNTIME_IMPORTS = ("from raylight", "import raylight", "comfyui_spectrum_h3")


def _literal_mapping(path: Path, name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Dict)
    return {
        ast.literal_eval(key): value
        for key, value in zip(assignment.value.keys, assignment.value.values, strict=True)
    }


def test_custom_node_registers_only_the_h3_surface() -> None:
    mapping = _literal_mapping(ROOT / "__init__.py", "NODE_CLASS_MAPPINGS")
    assert set(mapping) == EXPECTED_NODES


def test_runtime_source_has_no_external_custom_node_imports() -> None:
    source_files = sorted((ROOT / "src" / "minimax_h3_t4").rglob("*.py"))
    assert source_files
    violations: list[str] = []
    for path in source_files:
        source = path.read_text()
        for marker in FORBIDDEN_RUNTIME_IMPORTS:
            if marker in source:
                violations.append(f"{path.relative_to(ROOT)}: {marker}")
    assert violations == []
