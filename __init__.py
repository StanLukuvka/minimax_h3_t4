from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from minimax_h3_t4.nodes import H3T4Initializer, H3T4UNETLoader
from minimax_h3_t4.sampler_nodes import H3T4BasicGuider, H3T4BasicScheduler, H3T4SamplerAdvanced
from minimax_h3_t4.spectrum_node import H3T4Spectrum

NODE_CLASS_MAPPINGS = {
    "H3T4Initializer": H3T4Initializer,
    "H3T4UNETLoader": H3T4UNETLoader,
    "H3T4BasicScheduler": H3T4BasicScheduler,
    "H3T4BasicGuider": H3T4BasicGuider,
    "H3T4Spectrum": H3T4Spectrum,
    "H3T4SamplerAdvanced": H3T4SamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3T4Initializer": "MiniMax H3 T4 Initializer",
    "H3T4UNETLoader": "MiniMax H3 T4 Loader",
    "H3T4BasicScheduler": "MiniMax H3 T4 Scheduler",
    "H3T4BasicGuider": "MiniMax H3 T4 Guider",
    "H3T4Spectrum": "MiniMax H3 Spectrum",
    "H3T4SamplerAdvanced": "MiniMax H3 T4 Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
