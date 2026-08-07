"""Registration tests for the MiniMax H3 T4 extension package."""

from src.minimax_h3_t4.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


EXPECTED_NODES = {
    "H3T4Initializer",
    "H3T4UNETLoader",
    "H3T4Spectrum",
    "H3T4BasicScheduler",
    "H3T4BasicGuider",
    "H3T4SamplerAdvanced",
}


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
