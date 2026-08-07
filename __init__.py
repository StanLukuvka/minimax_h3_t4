"""Top-level package for minimax_h3_t4."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Stan Lukuvka"""
__email__ = "stanluku@gmail.com"
__version__ = "0.1.0"

from .src.minimax_h3_t4.nodes import NODE_CLASS_MAPPINGS
from .src.minimax_h3_t4.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
