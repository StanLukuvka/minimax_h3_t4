from __future__ import annotations

import importlib.util
import json
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


def test_kaggle_installer_requires_attached_model_input(monkeypatch, tmp_path: Path) -> None:
    module = load_script()
    monkeypatch.setattr(module, "REQUIRED_MODELS", {"models": ("model.safetensors",)})
    input_root = tmp_path / "input"
    input_root.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing attached Kaggle model inputs: model.safetensors"):
        module.find_models(input_root=input_root)


def test_kaggle_installer_resolves_each_required_model_once(tmp_path: Path) -> None:
    module = load_script()
    expected = {filename for names in module.REQUIRED_MODELS.values() for filename in names}
    for filename in expected:
        path = tmp_path / "weights" / filename
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"weights")

    assert set(module.find_models(input_root=tmp_path)) == expected


def test_kaggle_installer_rejects_ambiguous_model_inputs(monkeypatch, tmp_path: Path) -> None:
    module = load_script()
    monkeypatch.setattr(module, "REQUIRED_MODELS", {"models": ("model.safetensors",)})
    for dataset in ("first", "second"):
        path = tmp_path / dataset / "model.safetensors"
        path.parent.mkdir()
        path.write_bytes(b"weights")

    with pytest.raises(RuntimeError, match="Multiple attached Kaggle inputs provide model.safetensors"):
        module.find_models(input_root=tmp_path)


def test_kaggle_installer_preflights_models_before_reset(monkeypatch, tmp_path: Path) -> None:
    module = load_script()
    app_root = tmp_path / "app"
    app_root.mkdir()
    marker = app_root / "keep"
    marker.write_text("present")
    monkeypatch.setattr(module, "APP_ROOT", app_root)
    monkeypatch.setattr(module, "RESET_INSTALL", True)
    monkeypatch.setattr(module, "require_two_t4s", lambda: None)

    def missing_models() -> None:
        raise FileNotFoundError("missing models")

    monkeypatch.setattr(module, "find_models", missing_models, raising=False)

    with pytest.raises(FileNotFoundError, match="missing models"):
        module.install()
    assert marker.is_file()


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
    assert module.EXTENSION_REF == "2c7e66a4c661447606fa82f71b9b07fc6d6b77c9"


def test_kaggle_installer_bootstraps_dedicated_app_python(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("H3_T4_PYTHON", str(tmp_path / "kernel-python"))
    module = load_script()
    assert module.PYTHON == module.VENV_DIR / "bin" / "python"
    venv_dir = tmp_path / "venv"
    bootstrap_dir = tmp_path / "bootstrap"
    python = venv_dir / "bin" / "python"
    monkeypatch.setattr(module, "VENV_DIR", venv_dir)
    monkeypatch.setattr(module, "BOOTSTRAP_DIR", bootstrap_dir)
    monkeypatch.setattr(module, "PYTHON", python)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_run(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        if "virtualenv" in args:
            python.parent.mkdir(parents=True)
            python.touch()

    monkeypatch.setattr(module, "run", record_run)
    module.ensure_app_python()

    assert module.PYTHON == module.VENV_DIR / "bin" / "python"
    assert calls[0][0][-3:] == ("--target", bootstrap_dir, module.VIRTUALENV_PIN)
    assert calls[1][0][-3:] == ("virtualenv", "--system-site-packages", venv_dir)
    environment = calls[1][1]["env"]
    assert isinstance(environment, dict)
    assert environment["PYTHONPATH"] == str(bootstrap_dir)


def test_kaggle_installer_preserves_extension_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("H3_T4_EXTENSION_REPO_URL", "https://example.invalid/override.git")
    monkeypatch.setenv("H3_T4_EXTENSION_REF", "a" * 40)
    module = load_script()
    assert module.EXTENSION_REPO == "https://example.invalid/override.git"
    assert module.EXTENSION_REF == "a" * 40


def test_kaggle_runtime_dependencies_are_exactly_pinned() -> None:
    module = load_script()
    assert "ray==2.56.1" in module.PINNED_RUNTIME
    assert all(">=" not in dependency and "<=" not in dependency for dependency in module.PINNED_RUNTIME)


def test_git_stored_kaggle_notebook_enables_cloudflare() -> None:
    notebook_path = SCRIPT.with_name("minimax_h3_t4_kaggle.ipynb")
    notebook_text = notebook_path.read_text()
    notebook = json.loads(notebook_text)
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert len(code_cells) == 1
    code = "".join(code_cells[0]["source"])
    compile(code, str(notebook_path), "exec")
    assert "aa60e03" in code
    assert "95dff37d26466812be9473cc577d72ca85999404a9250979c45bb4ade3e415c0" in code
    # Cloudflare tunnel should be enabled by default
    assert "H3_T4_ENABLE_CLOUDFLARE" in code


def test_git_stored_kaggle_installer_has_cloudflare_configuration() -> None:
    source = SCRIPT.read_text().lower()
    assert "cloudflare" in source
    assert "comfy.lukuvka.com" in source


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
