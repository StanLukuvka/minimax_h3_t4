# MiniMax-H3-T4 ComfyUI Standalone Deployment

## Project Overview

**Goal:** Deploy MiniMax-H3 (an INT8 video+audio diffusion model) on exactly two NVIDIA Tesla T4 GPUs using a standalone ComfyUI instance, exposed via Cloudflare tunnel.

**Constraints:**
- Private Kaggle notebook: `stanlukuvka/minimax-h3-t4-standalone-comfyui` (kernel_id `130081857`)
- Cloudflare tunnel: `comfy.lukuvka.com`
- T4 = Turing sm_75, 16GB nominal (14.56 GiB device limit), no FP8/FP4 tensor cores
- Host RAM is only 32GB (30.00GB usable) — **binding constraint**
- Exact two-T4, fail-closed model discovery, positive worker-death proof, Spectrum validation
- Exact public nodes `H3T4Loader`/`H3T4Sampler` — these names are required
- INT8-only quantization (int8_tensorwise format with ConvRot)
- Self-managed custom raylight mod exists but is "a mess" — can't directly reuse

## Current State

**Head commit:** `d04f5541fce452d1c8b4f60afb1cc5004c0ead3b`
**Branch:** `main`
**Repo:** `https://github.com/StanLukuvka/minimax_h3_t4`
**Working dir:** `/agent/projects/minimax-h3 project/minimax_h3_t4`

**Pinned installer SHA256:** `5f35a87254d215cff7c28fb6e0b3cc938ebe74904f801758c99c6177b55182e5`

**Tests:** 110/110 passing

## Blocker

**Host-RAM OOM during `load_unet`:**
- Ray 2.56 OOM killer fires at 95% of 30GB host RAM (28.56GB threshold)
- Worker 0 peaks at 24.46GB during checkpoint load
- Both workers killed simultaneously

**Root cause:** Each worker independently loads the full ~12GB INT8 checkpoint into host RAM, then builds a `full_state` copy for FSDP sharding. Two workers = ~24GB peak.

## Recent Fixes (committed & pushed)

| Commit | Description |
|--------|-------------|
| `d04f554` | Cap Ray worker memory via `H3_T4_MAX_RAM_GB=29` env var |
| `284b2ee` | Single-load checkpoint broadcast — load once, share via Ray Object Store |
| `4063ac7` | Free checkpoint after model loads (`del state_dict` to halve CPU RAM peak) |
| `2a53db5` | FSDP CPU-direct offload (no GPU transient spike) |
| `6754ddc` | Memory trace diagnostic instrumentation |

**Architectural changes:**
- `loader_nodes.py`: Loads checkpoint once in main process, puts in Ray Object Store, passes ref to workers
- `worker.py`: Added `load_unet_from_state_dict` method that streams keys from shared state ref
- `worker.py`: Set `MMAP_TORCH_FILES = True` to use mmap-backed loading
- `worker.py`: `del state_dict; gc.collect(); torch.cuda.empty_cache()` after model loads weights
- `fsdp.py`: `_materialize_unsharded_param` loads directly to CPU when `cpu_offload=True`

## Next Steps

1. **User must attach datasets manually** to Kaggle notebook:
   - `stanlukuvka/minimax-h3-comfyui-weights` (4 safetensors files, ~42.5GB)
   - `stanlukuvka/cloudflare-files` (tunnel config)

2. **Run notebook v33** with attached datasets

3. **If still OOM, consider:**
   - Monkey-patching `psutil.virtual_memory()` to make ComfyUI see less RAM (lowering `MAX_PINNED_MEMORY`)
   - Further reducing `H3_T4_MAX_RAM_GB` (e.g., 24)
   - Disabling pinned memory entirely in ComfyUI startup args

## Required Model Files

In `stanlukuvka/minimax-h3-comfyui-weights`:
- `minimax_h3_audio_vae_fp32.safetensors`
- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `minimax_h3_video_vae_fp16.safetensors`
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`

## Environment

- BYOD image `gcr.io/kaggle-private-byod/python@sha256:37c64f7d…`
- venv python3.12
- Pinned: `transformers==5.0.0`, `kernels==0.14.0`, `ray==2.56.1`, `xfuser`, `yunchang`

## Key Technical Details

- **Ray OOM-kill policy change (2.56):** kills MULTIPLE workers, selected by time-since-task-start
- **Legacy single-worker policy:** `RAY_worker_killing_policy_by_group=true`
- **Other Ray knobs:** `RAY_memory_usage_threshold`, `RAY_memory_monitor_refresh_ms=0`, `RAY_idle_worker_killing_memory_threshold_bytes`
- **ComfyUI pinned memory** (line 1570 in `model_management.py`):
  `MAX_PINNED_MEMORY = max(ram * 0.40, min(ram * 0.90, ram - 4GB))`
- **SageAttention MANDATORY** (user override) — planned but not implemented
- **Pinning strategy:** never amend/force-push after push; fix release commit, commit-on-top

## Active Task

**Immediate:** Get one successful 10s/0.4MP generation end-to-end.

**Currently debugging:** Whether V32/V33's single-load broadcast + 29GB cap is enough to avoid the OOM, or whether we need to add the `psutil` monkey-patch to lower ComfyUI's reported RAM and `MAX_PINNED_MEMORY`.

## Memory Budget (per rank, 10s/0.4MP fp16)

- Activations: ~9.1GB
- FSDP gathered INT8 + prefetch: ~12GB
- All-to-all: ~0.6GB before ceiling
- Full packed seq: ~31GB (binding ceiling)
- Every rank materializes full `h` at h3_forward.py:251 before Ulysses split
- Host RAM 30.00GB usable = binding constraint

## Architecture Notes

- FSDP sharding: 2 ranks, world_size=2, FSDP + Ulysses sequence parallel
- Sequence parallel degree: 2, Ulysses degree: 2
- Each worker has its own GPU (0, 1)
- Checkpoint format: INT8 tensorwise with ConvRot
- INT8 + ConvRot is required (fail-closed validation)
