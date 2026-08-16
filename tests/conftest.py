from pathlib import Path
import os
import sys

# Exercise the installed package layout without requiring an editable install.
PROJECT_ROOT = Path(__file__).parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

# Two test_loader_nodes tests use fake checkpoint paths and rely on the loader
# short-circuiting through H3_T4_TEST_FAKE_STATE_DICT=1 instead of mmap-loading
# real safetensors files from disk. See loader_nodes._fake_state_dict branch.
os.environ.setdefault("H3_T4_TEST_FAKE_STATE_DICT", "1")
