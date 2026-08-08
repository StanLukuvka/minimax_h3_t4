from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import hashlib
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


def test_kaggle_installer_requires_immutable_extension_commit() -> None:
    module = load_script()
    with pytest.raises(ValueError, match="full 40-character commit"):
        module.require_full_commit("main", variable="H3_T4_EXTENSION_REF")
    module.require_full_commit("a" * 40, variable="H3_T4_EXTENSION_REF")


def test_kaggle_installer_defaults_to_the_published_accepted_extension(monkeypatch) -> None:
    monkeypatch.delenv("H3_T4_EXTENSION_REPO_URL", raising=False)
    monkeypatch.delenv("H3_T4_EXTENSION_REF", raising=False)
    module = load_script()
    assert module.EXTENSION_REPO == "https://github.com/StanLukuvka/minimax_h3_t4.git"
    assert module.EXTENSION_REF == "7e9e992b9d40a7b7852a6196579c5330908a474d"


def test_kaggle_installer_allows_explicit_immutable_extension_override(monkeypatch) -> None:
    monkeypatch.setenv("H3_T4_EXTENSION_REPO_URL", "https://example.invalid/override.git")
    monkeypatch.setenv("H3_T4_EXTENSION_REF", "a" * 40)
    module = load_script()
    assert module.EXTENSION_REPO == "https://example.invalid/override.git"
    assert module.EXTENSION_REF == "a" * 40


def test_kaggle_runtime_dependencies_are_exactly_pinned() -> None:
    module = load_script()
    assert "ray==2.56.1" in module.PINNED_RUNTIME
    assert all(">=" not in dependency and "<=" not in dependency for dependency in module.PINNED_RUNTIME)


def test_cloudflared_download_is_checksum_pinned(monkeypatch, tmp_path: Path) -> None:
    module = load_script()
    payload = b"pinned cloudflared test binary"
    target = tmp_path / "cloudflared"
    monkeypatch.setattr(module, "CLOUDFLARED", target)
    monkeypatch.setattr(module, "CLOUDFLARED_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: SimpleNamespace(read=lambda: payload),
    )

    assert module.ensure_cloudflared() == target
    assert target.read_bytes() == payload


def test_kaggle_notebook_executes_the_versioned_installer() -> None:
    notebook_path = SCRIPT.with_name("minimax_h3_t4_kaggle.ipynb")
    notebook = json.loads(notebook_path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert len(code_cells) == 1
    assert "".join(code_cells[0]["source"]).rstrip() == SCRIPT.read_text().rstrip()


def test_packaged_kaggle_installer_matches_git_clone_entry() -> None:
    packaged = SCRIPT.parents[1] / "src" / "minimax_h3_t4" / "notebooks" / "kaggle_install.py"
    assert packaged.read_bytes() == SCRIPT.read_bytes()


def test_packaged_kaggle_notebook_matches_git_clone_entry() -> None:
    notebook = SCRIPT.with_name("minimax_h3_t4_kaggle.ipynb")
    packaged = SCRIPT.parents[1] / "src" / "minimax_h3_t4" / "notebooks" / notebook.name
    assert packaged.read_bytes() == notebook.read_bytes()


def test_kaggle_installer_has_no_donor_plugin_installation() -> None:
    source = SCRIPT.read_text()
    lowered = source.lower()
    assert "raylight" not in lowered
    assert "comfyui-spectrum" not in lowered
    assert "easycache" not in lowered
