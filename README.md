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

The extension registers exactly two nodes:

- **MiniMax H3 T4 Loader** — releases parent ComfyUI model memory, starts the fixed two-worker runtime, sequentially maps the INT8 checkpoint onto rank 0 then rank 1, and applies either the Exact or frozen Spectrum profile.
- **MiniMax H3 T4 Sampler** — owns noise, sampler selection, scheduling, guidance, distributed execution, and bounded confirmed worker teardown before returning ordinary video/audio latents.

Ray, NCCL, FSDP, Ulysses, GPU assignment, object-store sizing, and Spectrum tuning are implementation details rather than workflow settings. Stock ComfyUI remains responsible for CLIP loading, `MiniMaxH3ImageToVideo`, parent-process VAE loading and decoding, and video output.

Spectrum forecasting is admitted only for the frozen, reviewed `sample_euler`, `sample_res_multistep`, and `sample_res_multistep_cfg_pp` functions. Ancestral samplers, Euler with positive or invalid `s_churn`, aliases, and custom samplers execute exactly. Forecasts never occur consecutively; RES samplers enforce at least three final exact steps.

## Installation

Clone the repository into ComfyUI's `custom_nodes` directory and install its Python dependencies with the same Python environment used by ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone <published-repository-url> minimax_h3_t4
cd minimax_h3_t4
python -m pip install -e .
```

The repository has not been published yet; substitute its immutable release URL after publication.

Restart ComfyUI after installation.

Runtime dependencies declared by the extension are limited to:

- `ray>=2.48.0`
- `safetensors>=0.4.0`
- `xfuser>=0.4.4`
- `yunchang>=0.6.4`

PyTorch, ComfyKitchen, and the MiniMax H3 model implementation are supplied by the host ComfyUI installation.

## Workflow ordering

Connect the stock conditioning output to both public nodes:

```text
MiniMaxH3ImageToVideo ──┬──> MiniMax H3 T4 Loader ──> MiniMax H3 T4 Sampler
                        └────────────────────────────> MiniMax H3 T4 Sampler

MiniMax H3 T4 Sampler -> confirmed worker death -> parent VAE decode -> video output
```

This dependency ordering lets ComfyUI unload parent models before the two denoiser workers begin sequential checkpoint loading. FSDP shards remain worker-local and are never handed to stock per-layer CPU offloading; the safe distributed unload operation is confirmed actor teardown.

## Included workflows and Kaggle entry

- `workflows/minimax_h3_t4_exact.json` — 124-frame exact control workflow.
- `workflows/minimax_h3_t4_spectrum.json` — 124-frame native Spectrum workflow.
- `workflows/minimax_h3_t4_exact_10s.json` — experimental `512×288×243` (10.125-second) exact workflow.
- `workflows/minimax_h3_t4_spectrum_10s.json` — experimental `512×288×243` Spectrum workflow.
- `notebooks/minimax_h3_t4_kaggle.ipynb` — directly runnable Kaggle notebook containing the versioned installer.
- `notebooks/kaggle_install.py` — authoritative installer source used by the notebook.

The 243-frame workflows are structurally valid and fall inside MiniMax H3's documented trained frame range, but they have not yet passed standalone two-T4 hardware acceptance. They are experiments, not accepted performance claims.

The Kaggle entry defaults to the proven ComfyUI commit. It intentionally refuses mutable or missing extension coordinates: set `H3_T4_EXTENSION_REPO_URL` to the published repository and `H3_T4_EXTENSION_REF` to a full release commit.

Kaggle does not expose notebook ports directly. Setting `H3_T4_ENABLE_CLOUDFLARE=1` enables the script's checksum-pinned Cloudflare quick tunnel, but that URL is unauthenticated and should be treated as temporary public access. The tunnel is disabled by default.

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
