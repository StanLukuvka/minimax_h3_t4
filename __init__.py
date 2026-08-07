"""ComfyUI entry point for the MiniMax H3 two-T4 extension."""

from pathlib import Path
import sys

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Stan Lukuvka"""
__email__ = "stanluku@gmail.com"
__version__ = "0.1.0"

_SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if __name__ == "minimax_h3_t4":
    # Pytest imports the repository root as a package because this file exists.
    # The installed wheel resolves directly to src/minimax_h3_t4 instead.
    from .src.minimax_h3_t4 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY
else:
    if str(_SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(_SOURCE_ROOT))
    from minimax_h3_t4 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY
