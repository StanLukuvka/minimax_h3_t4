"""ComfyUI node registration surface for the MiniMax H3 T4 extension."""

from .loader_nodes import H3T4Initializer, H3T4UNETLoader
from .sampler_nodes import H3T4BasicGuider, H3T4BasicScheduler, H3T4SamplerAdvanced
from .spectrum_node import H3T4Spectrum

NODE_CLASS_MAPPINGS = {
    "H3T4Initializer": H3T4Initializer,
    "H3T4UNETLoader": H3T4UNETLoader,
    "H3T4Spectrum": H3T4Spectrum,
    "H3T4BasicScheduler": H3T4BasicScheduler,
    "H3T4BasicGuider": H3T4BasicGuider,
    "H3T4SamplerAdvanced": H3T4SamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3T4Initializer": "MiniMax H3 T4 Initializer",
    "H3T4UNETLoader": "MiniMax H3 T4 Loader",
    "H3T4Spectrum": "MiniMax H3 T4 Spectrum",
    "H3T4BasicScheduler": "MiniMax H3 T4 Scheduler",
    "H3T4BasicGuider": "MiniMax H3 T4 Guider",
    "H3T4SamplerAdvanced": "MiniMax H3 T4 Sampler",
}

__all__ = [
    "H3T4Initializer",
    "H3T4UNETLoader",
    "H3T4Spectrum",
    "H3T4BasicScheduler",
    "H3T4BasicGuider",
    "H3T4SamplerAdvanced",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
