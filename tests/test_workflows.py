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
EXPECTED_INPUTS = {
    "H3T4Initializer": ["load_after"],
    "H3T4UNETLoader": ["actor_group", "unet_name", "load_after"],
    "H3T4Spectrum": ["model"],
    "H3T4BasicScheduler": ["model"],
    "H3T4BasicGuider": ["model", "conditioning"],
    "H3T4SamplerAdvanced": ["add_noise", "noise", "guider", "sampler", "sigmas", "latent_image"],
}
EXPECTED_WIDGET_COUNTS = {
    "H3T4Initializer": 3,
    "H3T4UNETLoader": 1,
    "H3T4Spectrum": 10,
    "H3T4BasicScheduler": 3,
    "H3T4BasicGuider": 0,
    "H3T4SamplerAdvanced": 1,
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


def test_packaged_workflows_match_git_clone_workflows() -> None:
    root = Path(__file__).parents[1]
    for source in WORKFLOWS.glob("*.json"):
        packaged = root / "src" / "minimax_h3_t4" / "workflows" / source.name
        assert packaged.read_bytes() == source.read_bytes()


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
    for node in graph["nodes"]:
        node_type = node["type"]
        if node_type in EXPECTED_INPUTS:
            assert [spec["name"] for spec in node.get("inputs", [])] == EXPECTED_INPUTS[node_type]
            assert len(node.get("widgets_values", [])) == EXPECTED_WIDGET_COUNTS[node_type]
    validate_links(graph)
