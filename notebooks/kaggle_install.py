"""Kaggle entry script for the standalone MiniMax H3 two-T4 ComfyUI extension.

Run this file in a Kaggle GPU notebook with exactly two Tesla T4 GPUs and the
four required model files attached as Kaggle input datasets.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

APP_ROOT = Path(os.environ.get("H3_T4_APP_ROOT", "/kaggle/working/minimax-h3-t4"))
COMFY_DIR = APP_ROOT / "ComfyUI"
EXTENSION_DIR = COMFY_DIR / "custom_nodes" / "minimax_h3_t4"
VENV_DIR = APP_ROOT / "venv"
PYTHON = VENV_DIR / "bin" / "python"
COMFY_REPO = os.environ.get("COMFY_REPO_URL", "https://github.com/Comfy-Org/ComfyUI.git")
COMFY_REF = os.environ.get("COMFY_COMMIT", "9a9fdb10ed144ce760d9682cb247526ea23cc525")
EXTENSION_REPO = os.environ.get(
    "H3_T4_EXTENSION_REPO_URL",
    "https://github.com/StanLukuvka/minimax_h3_t4.git",
)
EXTENSION_REF = os.environ.get("H3_T4_EXTENSION_REF", "main")
RESET_INSTALL = os.environ.get("H3_T4_RESET_INSTALL", "0") == "1"
PORT = int(os.environ.get("H3_T4_PORT", "8188"))

PINNED_RUNTIME = (
    "transformers==5.0.0",
    "diffusers==0.37.1",
    "kernels==0.14.0",
    "xfuser==0.4.5",
    "yunchang==0.6.4",
    "ray>=2.48.0",
)
REQUIRED_MODELS = {
    "diffusion_models": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoders": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae": (
        "minimax_h3_video_vae_fp16.safetensors",
        "minimax_h3_audio_vae_fp32.safetensors",
    ),
}


def run(*args: object, cwd: Path | None = None) -> None:
    command = [str(arg) for arg in args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def checkout(url: str, destination: Path, ref: str) -> None:
    if not destination.exists():
        run("git", "clone", "--filter=blob:none", url, destination)
    run("git", "fetch", "--depth", "1", "origin", ref, cwd=destination)
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


def find_model(filename: str) -> Path:
    matches = sorted(Path("/kaggle/input").rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Attach a Kaggle dataset containing {filename}")
    return matches[0]


def link_models() -> None:
    for folder, names in REQUIRED_MODELS.items():
        if isinstance(names, str):
            names = (names,)
        target_dir = COMFY_DIR / "models" / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in names:
            source = find_model(filename)
            target = target_dir / filename
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(source)
            print(f"model: {target} -> {source}")


def install() -> None:
    require_two_t4s()
    if RESET_INSTALL and APP_ROOT.exists():
        shutil.rmtree(APP_ROOT)
    APP_ROOT.mkdir(parents=True, exist_ok=True)
    checkout(COMFY_REPO, COMFY_DIR, COMFY_REF)
    checkout(EXTENSION_REPO, EXTENSION_DIR, EXTENSION_REF)
    if not PYTHON.exists():
        run(sys.executable, "-m", "venv", "--system-site-packages", VENV_DIR)
    run(PYTHON, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    run(PYTHON, "-m", "pip", "install", "-r", COMFY_DIR / "requirements.txt")
    run(PYTHON, "-m", "pip", "install", "--upgrade", *PINNED_RUNTIME)
    run(PYTHON, "-m", "pip", "install", "-e", EXTENSION_DIR)
    run(PYTHON, "-m", "pip", "check")
    link_models()
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
    print(f"ComfyUI is ready on port {PORT}; load minimax_h3_t4_spectrum.json or minimax_h3_t4_exact.json")
    return process


if __name__ == "__main__":
    install()
    COMFY_PROCESS = start()
