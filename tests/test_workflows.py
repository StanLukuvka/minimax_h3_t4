from __future__ import annotations

import json
from pathlib import Path

import pytest


WORKFLOWS = Path(__file__).parents[1] / "workflows"
CUSTOM_TYPES = {"H3T4Loader", "H3T4Sampler"}
RETIRED_TYPES = {
    "H3T4Initializer",
    "H3T4UNETLoader",
    "H3T4Spectrum",
    "H3T4BasicScheduler",
    "H3T4BasicGuider",
    "H3T4SamplerAdvanced",
    "RandomNoise",
    "KSamplerSelect",
}
FORBIDDEN_TYPES = {
    "RayInitializer",
    "RayUNETLoader",
    "RaySpectrumApplyMiniMaxH3",
    "RayBasicScheduler",
    "RayBasicGuider",
    "XFuserSamplerCustomAdvanced",
}


def find_sampling_graph(value):
    if isinstance(value, dict):
        nodes = value.get("nodes", [])
        if any(isinstance(node, dict) and node.get("type") == "H3T4Sampler" for node in nodes):
            return value
        for child in value.values():
            found = find_sampling_graph(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_sampling_graph(child)
            if found is not None:
                return found
    return None


def find_node(value, node_type):
    if isinstance(value, dict):
        for node in value.get("nodes", []):
            if isinstance(node, dict) and node.get("type") == node_type:
                return node
        for child in value.values():
            found = find_node(child, node_type)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_node(child, node_type)
            if found is not None:
                return found
    return None


def validate_links(graph) -> None:
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link["id"]: link for link in graph["links"]}
    assert len(links) == len(graph["links"])

    for node in nodes.values():
        for slot, input_spec in enumerate(node.get("inputs", [])):
            link_id = input_spec.get("link")
            if link_id is None:
                continue
            link = links[link_id]
            assert link["target_id"] == node["id"]
            assert link["target_slot"] == slot
        for slot, output_spec in enumerate(node.get("outputs", [])):
            for link_id in output_spec.get("links") or []:
                link = links[link_id]
                assert link["origin_id"] == node["id"]
                assert link["origin_slot"] == slot


def test_packaged_workflows_match_git_clone_workflows() -> None:
    root = Path(__file__).parents[1]
    sources = sorted(WORKFLOWS.glob("*.json"))
    assert len(sources) == 4
    for source in sources:
        packaged = root / "src" / "minimax_h3_t4" / "workflows" / source.name
        assert packaged.read_bytes() == source.read_bytes()


@pytest.mark.parametrize(
    ("filename", "acceleration", "length", "width", "height"),
    (
        ("minimax_h3_t4_exact.json", "Exact", 124, 736, 416),
        ("minimax_h3_t4_spectrum.json", "Spectrum", 124, 736, 416),
        ("minimax_h3_t4_exact_10s.json", "Exact", 243, 512, 288),
        ("minimax_h3_t4_spectrum_10s.json", "Spectrum", 243, 512, 288),
    ),
)
def test_workflow_exposes_only_two_h3_nodes(filename: str, acceleration: str, length: int, width: int, height: int) -> None:
    data = json.loads((WORKFLOWS / filename).read_text())
    graph = find_sampling_graph(data)
    assert graph is not None
    node_types = [node["type"] for node in graph["nodes"]]

    assert not set(node_types).intersection(FORBIDDEN_TYPES | RETIRED_TYPES)
    assert [node_type for node_type in node_types if node_type.startswith("H3T4")] == ["H3T4Loader", "H3T4Sampler"]

    loader = next(node for node in graph["nodes"] if node["type"] == "H3T4Loader")
    sampler = next(node for node in graph["nodes"] if node["type"] == "H3T4Sampler")
    conditioning = next(node for node in graph["nodes"] if node["type"] == "MiniMaxH3ImageToVideo")
    workflow_node = find_node(data, graph["id"])

    assert workflow_node is not None
    assert workflow_node["widgets_values"][1:4] == [width, height, 10 if length == 243 else 5]
    assert loader["widgets_values"][1] == acceleration
    assert conditioning["widgets_values"][3] == length
    assert [spec["name"] for spec in loader["inputs"]] == ["unet_name", "acceleration", "load_after"]
    assert [spec["name"] for spec in sampler["inputs"]] == ["model", "conditioning", "latent_image", "seed"]
    validate_links(graph)
