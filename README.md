# MiniMax H3 T4 for ComfyUI

A standalone ComfyUI extension for running MiniMax H3 on exactly two NVIDIA Tesla T4 GPUs. It contains its own MiniMax-H3-specific Ray/FSDP/Ulysses execution path and a native Spectrum forecasting implementation.

This repository does **not** install or import Raylight, ComfyUI-Spectrum-MiniMax-H3, EasyCache, or another community custom-node package at runtime.

## Status

Development is in progress. The exact two-worker runtime and native Spectrum core have executable contract tests, but this standalone package has not yet completed its clean Kaggle two-T4 acceptance run. The proven source runs used the pinned implementations listed in [NOTICE.md](NOTICE.md).

## Supported configuration

- Exactly 2 × NVIDIA Tesla T4, 16 GiB each
- MiniMax H3 INT8 ConvRot checkpoint
- Ulysses sequence parallelism: 2
- FSDP: enabled
- FSDP CPU offload: disabled
- xFuser attention: `TORCH_EFFICIENT`
- Worker-local ComfyKitchen CUDA INT8
- VAE decoding in the ComfyUI parent process, after worker teardown
- Exact execution or native Spectrum acceleration

The initial release intentionally excludes EasyCache, TeaCache, combined cache modes, Bob Triton, LoRA, GGUF, FP8/NVFP4, ControlNet, PipeFusion, distributed VAE, and unrelated diffusion models.

## Nodes

The extension registers six focused nodes:

- **MiniMax H3 T4 Initializer** — starts the fixed local two-worker Ray group after conditioning memory is released.
- **MiniMax H3 T4 Loader** — sequentially maps the INT8 checkpoint onto rank 0 and then rank 1.
- **MiniMax H3 T4 Spectrum** — configures the worker-local Spectrum forecaster. Bypass this node for exact execution.
- **MiniMax H3 T4 Scheduler** — calculates the sigma schedule using the worker-owned model.
- **MiniMax H3 T4 Guider** — binds MiniMax H3 conditioning to the distributed model.
- **MiniMax H3 T4 Sampler** — runs both ranks, performs bounded teardown, and returns the video/audio latent pair.

Stock ComfyUI nodes remain responsible for CLIP loading, `MiniMaxH3ImageToVideo`, sampler selection, VAE loading and decoding, and video output.

## Installation

Clone the repository into ComfyUI's `custom_nodes` directory and install its Python dependencies with the same Python environment used by ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/StanLukuvka/minimax_h3_t4.git
cd minimax_h3_t4
python -m pip install -e .
```

Restart ComfyUI after installation.

Runtime dependencies declared by the extension are limited to:

- `ray>=2.48.0`
- `xfuser>=0.4.4`

PyTorch, ComfyKitchen, and the MiniMax H3 model implementation are supplied by the host ComfyUI installation.

## Workflow ordering

Connect the conditioning output to both `load_after` inputs:

```text
MiniMaxH3ImageToVideo ──┬──> MiniMax H3 T4 Initializer
                        └──> MiniMax H3 T4 Loader

Initializer -> Loader -> [optional Spectrum] -> Scheduler / Guider -> Sampler
Sampler -> parent-process VAE decode -> video output
```

This dependency ordering ensures that the large conditioning model and conditioning-time VAE finish before the two denoiser workers begin sequential checkpoint loading.

## Included workflows and Kaggle entry

- `workflows/minimax_h3_t4_exact.json` — immutable exact control path.
- `workflows/minimax_h3_t4_spectrum.json` — native Spectrum path with the conservative defaults below.
- `notebooks/kaggle_install.py` — checks for exactly two T4 GPUs, installs pinned ComfyUI/runtime dependencies, links attached model datasets, installs both workflows, and starts ComfyUI.

The Kaggle entry defaults to the proven ComfyUI commit. Before publishing a reproducible notebook, set `H3_T4_EXTENSION_REF` to a full release commit rather than `main`.

## Spectrum defaults

The defaults reproduce the proven conservative configuration:

```text
blend_weight=0.5
degree=4
ridge_lambda=0.1
window_size=2.0
flex_window=0.75
warmup_steps=5
tail_actual_steps=1
max_history=8
history_storage=system_ram
```

Histories remain bounded and rank-local in system RAM. Forecast validity and exact/forecast mode are synchronized over the Ulysses sequence-parallel group. Invalid or nonfinite forecasts fall back to exact H3 execution on every rank.

## Development

The project was generated from the official ComfyUI extension cookiecutter. See [SCAFFOLD.md](SCAFFOLD.md) for the exact template commit.

Run the local verification gates with:

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m compileall -q src __init__.py
uv build
```

## Licensing and attribution

The package is GPL-3.0-or-later because it incorporates and modifies GPL-covered Spectrum forecasting code. Distributed execution code adapted from Raylight remains attributed to its Apache-2.0 source. Exact source commits and component notices are recorded in [NOTICE.md](NOTICE.md).
