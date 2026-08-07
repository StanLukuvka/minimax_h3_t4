from __future__ import annotations

import json
from pathlib import Path

import pytest


WORKFLOWS = Path(__file__).parents[1] / "workflows"
CUSTOM_TYPES = {
    "H3T4Initializer",
    "H3T4UNETLoader",
    "H3T4Spectrum",
    "H3T4BasicScheduler",
    "H3T4BasicGuider",
    "H3T4SamplerAdvanced",
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
        if any(isinstance(node, dict) and node.get("type") == "H3T4SamplerAdvanced" for node in nodes):
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


@pytest.mark.parametrize(
    ("filename", "spectrum"),
    (("minimax_h3_t4_exact.json", False), ("minimax_h3_t4_spectrum.json", True)),
)
def test_standalone_workflow_uses_only_registered_h3_t4_nodes(filename: str, spectrum: bool) -> None:
    data = json.loads((WORKFLOWS / filename).read_text())
    graph = find_sampling_graph(data)
    assert graph is not None
    node_types = {node["type"] for node in graph["nodes"]}

    assert not node_types.intersection(FORBIDDEN_TYPES)
    assert CUSTOM_TYPES - ({"H3T4Spectrum"} if not spectrum else set()) <= node_types
    assert ("H3T4Spectrum" in node_types) is spectrum
    assert "RandomNoise" in node_types
    validate_links(graph)
