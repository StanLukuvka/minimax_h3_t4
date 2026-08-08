"""ComfyUI node registration surface for the MiniMax H3 T4 extension."""

from .public_nodes import H3T4Loader, H3T4Sampler

NODE_CLASS_MAPPINGS = {
    "H3T4Loader": H3T4Loader,
    "H3T4Sampler": H3T4Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3T4Loader": "MiniMax H3 T4 Loader",
    "H3T4Sampler": "MiniMax H3 T4 Sampler",
}

__all__ = [
    "H3T4Loader",
    "H3T4Sampler",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
