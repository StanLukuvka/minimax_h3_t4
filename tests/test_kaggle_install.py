from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "notebooks" / "kaggle_install.py"


def load_script():
    spec = importlib.util.spec_from_file_location("kaggle_h3_t4_install_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kaggle_installer_requires_exactly_two_t4s(monkeypatch) -> None:
    module = load_script()

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="Tesla T4\nTesla T4\n"),
    )
    module.require_two_t4s()

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="Tesla T4\n"),
    )
    with pytest.raises(RuntimeError, match="exactly two Tesla T4"):
        module.require_two_t4s()


def test_kaggle_installer_has_no_donor_plugin_installation() -> None:
    source = SCRIPT.read_text()
    lowered = source.lower()
    assert "raylight" not in lowered
    assert "comfyui-spectrum" not in lowered
    assert "easycache" not in lowered
