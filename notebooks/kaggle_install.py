"""Kaggle entry script for the standalone MiniMax H3 two-T4 ComfyUI extension.

Run this file in a Kaggle GPU notebook with exactly two Tesla T4 GPUs and the
four required model files attached as Kaggle input datasets.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

APP_ROOT = Path(os.environ.get("H3_T4_APP_ROOT", "/kaggle/working/minimax-h3-t4"))
COMFY_DIR = APP_ROOT / "ComfyUI"
EXTENSION_DIR = COMFY_DIR / "custom_nodes" / "minimax_h3_t4"
VENV_DIR = APP_ROOT / "venv"
BOOTSTRAP_DIR = APP_ROOT / "bootstrap"
PYTHON = VENV_DIR / "bin" / "python"
VIRTUALENV_PIN = "virtualenv==20.34.0"
COMFY_REPO = os.environ.get("COMFY_REPO_URL", "https://github.com/Comfy-Org/ComfyUI.git")
COMFY_REF = os.environ.get("COMFY_COMMIT", "9a9fdb10ed144ce760d9682cb247526ea23cc525")
EXTENSION_REPO = os.environ.get(
    "H3_T4_EXTENSION_REPO_URL",
    "https://github.com/StanLukuvka/minimax_h3_t4.git",
)
EXTENSION_REF = os.environ.get(
    "H3_T4_EXTENSION_REF",
    "e35d2cff8cbbe99c1e486cc0b047a76906f0354c",
)
RESET_INSTALL = os.environ.get("H3_T4_RESET_INSTALL", "0") == "1"
PORT = int(os.environ.get("H3_T4_PORT", "8188"))
ENABLE_CLOUDFLARE = os.environ.get("H3_T4_ENABLE_CLOUDFLARE", "0") == "1"
CLOUDFLARE_PUBLIC_URL = "https://comfy.lukuvka.com"
CLOUDFLARE_SOURCE_DIR = Path(
    os.environ.get(
        "H3_T4_CLOUDFLARE_SOURCE_DIR",
        "/kaggle/input/datasets/stanlukuvka/cloudflare-files",
    )
)
CLOUDFLARE_WORK_DIR = Path(os.environ.get("H3_T4_CLOUDFLARE_WORK_DIR", "/kaggle/working/cloudflare"))
CLOUDFLARE_CONFIG_FILE = CLOUDFLARE_WORK_DIR / "config.yml"
CLOUDFLARED = Path(os.environ.get("H3_T4_CLOUDFLARED", "/kaggle/working/cloudflared"))
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/download/2026.7.3/cloudflared-linux-amd64"
CLOUDFLARED_SHA256 = "9d71c677db00134c1bd4144b7783486b654ad281b1ea62b4972098d19f770f17"

PINNED_RUNTIME = (
    "transformers==5.0.0",
    "diffusers==0.37.1",
    "kernels==0.14.0",
    "xfuser==0.4.5",
    "yunchang==0.6.4",
    "ray==2.56.1",
    "safetensors==0.8.0",
)
REQUIRED_MODELS = {
    "diffusion_models": ("minimax_h3_fl2va_pruned_int8_convrot.safetensors",),
    "text_encoders": ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",),
    "vae": (
        "minimax_h3_video_vae_fp16.safetensors",
        "minimax_h3_audio_vae_fp32.safetensors",
    ),
}


def run(
    *args: object,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    command = [str(arg) for arg in args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, env=env)


def ensure_app_python() -> None:
    if PYTHON.exists():
        return
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        BOOTSTRAP_DIR,
        VIRTUALENV_PIN,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(BOOTSTRAP_DIR)
    run(
        sys.executable,
        "-m",
        "virtualenv",
        "--system-site-packages",
        VENV_DIR,
        env=environment,
    )
    if not PYTHON.exists():
        raise RuntimeError(f"virtualenv did not create {PYTHON}")


def require_full_commit(ref: str, *, variable: str) -> None:
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref) is None:
        raise ValueError(f"{variable} must be an immutable full 40-character commit SHA")


def checkout(url: str, destination: Path, ref: str) -> None:
    require_full_commit(ref, variable="checkout ref")
    if not destination.exists():
        run("git", "clone", "--filter=blob:none", url, destination)
    elif not (destination / ".git").is_dir():
        raise RuntimeError(f"Existing checkout is not a Git repository: {destination}")
    run("git", "fetch", "--depth", "1", "origin", ref, cwd=destination)
    run("git", "reset", "--hard", "FETCH_HEAD", cwd=destination)
    run("git", "clean", "-fdx", cwd=destination)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=destination)


def require_two_t4s() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"MiniMax H3 T4 requires exactly two Tesla T4 GPUs; detected {names!r}")
    print("GPU topology:", names)


def find_models(*, input_root: Path = Path("/kaggle/input")) -> dict[str, Path]:
    required = {filename for names in REQUIRED_MODELS.values() for filename in names}
    matches: dict[str, Path] = {}
    for path in input_root.rglob("*"):
        if path.name not in required or not path.is_file():
            continue
        if path.name in matches:
            raise RuntimeError(f"Multiple attached Kaggle inputs provide {path.name}")
        matches[path.name] = path
    missing = sorted(required.difference(matches))
    if missing:
        raise FileNotFoundError(f"Missing attached Kaggle model inputs: {', '.join(missing)}")
    return matches


def require_cloudflare_input(source_dir: Path = CLOUDFLARE_SOURCE_DIR) -> Path:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing attached Kaggle input stanlukuvka/cloudflare-files: expected {source_dir}")
    configs = sorted(source_dir.rglob("config.yml")) + sorted(source_dir.rglob("config.yaml"))
    if not configs:
        raise FileNotFoundError(f"No Cloudflare tunnel config found under {source_dir}")
    return source_dir


def link_models(sources: dict[str, Path]) -> None:
    for folder, names in REQUIRED_MODELS.items():
        target_dir = COMFY_DIR / "models" / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in names:
            source = sources[filename]
            target = target_dir / filename
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(source)
            print(f"model: {target} -> {source}")


def install() -> None:
    if not EXTENSION_REPO:
        raise ValueError("H3_T4_EXTENSION_REPO_URL must name the published extension repository")
    require_full_commit(COMFY_REF, variable="COMFY_COMMIT")
    require_full_commit(EXTENSION_REF, variable="H3_T4_EXTENSION_REF")
    require_two_t4s()
    model_sources = find_models()
    if ENABLE_CLOUDFLARE:
        require_cloudflare_input()
    if RESET_INSTALL and APP_ROOT.exists():
        shutil.rmtree(APP_ROOT)
    APP_ROOT.mkdir(parents=True, exist_ok=True)
    checkout(COMFY_REPO, COMFY_DIR, COMFY_REF)
    checkout(EXTENSION_REPO, EXTENSION_DIR, EXTENSION_REF)
    ensure_app_python()
    run(PYTHON, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    run(PYTHON, "-m", "pip", "install", "-r", COMFY_DIR / "requirements.txt")
    run(PYTHON, "-m", "pip", "install", "--upgrade", *PINNED_RUNTIME)
    run(PYTHON, "-m", "pip", "install", "-e", EXTENSION_DIR)
    link_models(model_sources)
    workflow_dir = COMFY_DIR / "user" / "default" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    for workflow in (EXTENSION_DIR / "workflows").glob("*.json"):
        shutil.copy2(workflow, workflow_dir / workflow.name)


def wait_for_port(port: int, timeout: float = 900.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"ComfyUI did not open port {port} within {timeout:.0f}s")


def validate_readiness(port: int) -> None:
    probe = "import minimax_h3_t4, ray, safetensors, xfuser, yunchang; assert set(minimax_h3_t4.NODE_CLASS_MAPPINGS) == {'H3T4Loader', 'H3T4Sampler'}"
    run(PYTHON, "-c", probe, cwd=COMFY_DIR)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=30) as response:
        system_stats = json.load(response)
    if not isinstance(system_stats, dict):
        raise RuntimeError("ComfyUI /system_stats did not return an object")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/object_info", timeout=30) as response:
        object_info = json.load(response)
    required_nodes = {"H3T4Loader", "H3T4Sampler"}
    missing_nodes = sorted(required_nodes.difference(object_info))
    if missing_nodes:
        raise RuntimeError(f"ComfyUI did not register required MiniMax-H3 nodes: {missing_nodes}")
    workflow_dir = COMFY_DIR / "user" / "default" / "workflows"
    required_workflows = {
        "minimax_h3_t4_exact.json",
        "minimax_h3_t4_spectrum.json",
        "minimax_h3_t4_exact_10s.json",
        "minimax_h3_t4_spectrum_10s.json",
    }
    missing_workflows = sorted(name for name in required_workflows if not (workflow_dir / name).is_file())
    if missing_workflows:
        raise RuntimeError(f"Kaggle workflows were not installed: {missing_workflows}")


def is_linux_elf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def ensure_cloudflared(
    *,
    source_dir: Path = CLOUDFLARE_SOURCE_DIR,
    work_dir: Path = CLOUDFLARE_WORK_DIR,
    config_file: Path = CLOUDFLARE_CONFIG_FILE,
) -> tuple[Path, Path]:
    require_cloudflare_input(source_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(source_dir, work_dir, symlinks=False)

    configs = sorted(work_dir.rglob("config.yml")) + sorted(work_dir.rglob("config.yaml"))
    if not config_file.exists():
        config_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(configs[0], config_file)

    if not is_linux_elf(CLOUDFLARED):
        CLOUDFLARED.unlink(missing_ok=True)
        preferred = (
            work_dir / "cloudflared",
            work_dir / "cloudflared-linux-amd64",
            source_dir / "cloudflared",
            source_dir / "cloudflared-linux-amd64",
        )
        discovered = tuple(sorted(work_dir.rglob("cloudflared*"))) + tuple(sorted(source_dir.rglob("cloudflared*")))
        for candidate in (*preferred, *discovered):
            if is_linux_elf(candidate):
                CLOUDFLARED.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, CLOUDFLARED)
                break
    if CLOUDFLARED.exists():
        digest = hashlib.sha256(CLOUDFLARED.read_bytes()).hexdigest()
        if digest == CLOUDFLARED_SHA256:
            CLOUDFLARED.chmod(0o755)
            for sensitive in (config_file, config_file.parent / "tunnel.json"):
                if sensitive.exists():
                    sensitive.chmod(0o600)
            return CLOUDFLARED, config_file
        CLOUDFLARED.unlink()
    data = urllib.request.urlopen(CLOUDFLARED_URL, timeout=120).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != CLOUDFLARED_SHA256:
        raise RuntimeError(f"cloudflared checksum mismatch: {digest}")
    CLOUDFLARED.write_bytes(data)
    CLOUDFLARED.chmod(0o755)
    for sensitive in (config_file, config_file.parent / "tunnel.json"):
        if sensitive.exists():
            sensitive.chmod(0o600)
    return CLOUDFLARED, config_file


def start_cloudflare(port: int) -> tuple[subprocess.Popen[bytes], str]:
    del port
    binary, config = ensure_cloudflared()
    log_path = APP_ROOT / "cloudflared.log"
    log_handle = log_path.open("wb")
    process = subprocess.Popen(
        [str(binary), "tunnel", "--config", str(config), "--no-autoupdate", "run"],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=config.parent,
    )
    time.sleep(2.0)
    if process.poll() is not None:
        log_handle.close()
        raise RuntimeError(f"cloudflared exited early; see {log_path}")
    log_handle.close()
    return process, CLOUDFLARE_PUBLIC_URL


def start() -> subprocess.Popen[bytes]:
    command = [
        str(PYTHON),
        "main.py",
        "--listen",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--cache-none",
        "--preview-method",
        "none",
    ]
    print("+", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=COMFY_DIR)
    wait_for_port(PORT)
    validate_readiness(PORT)
    print(f"ComfyUI is ready on port {PORT}; load minimax_h3_t4_spectrum.json or minimax_h3_t4_exact.json")
    return process


if __name__ == "__main__":
    install()
    COMFY_PROCESS = start()
    if ENABLE_CLOUDFLARE:
        CLOUDFLARE_PROCESS, COMFY_URL = start_cloudflare(PORT)
        print("Named Cloudflare ComfyUI URL:", COMFY_URL)
