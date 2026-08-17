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

**Head commit:** `156c6c6` (see Recent Fixes)
**Branch:** `main`
**Repo:** `https://github.com/StanLukuvka/minimax_h3_t4`
**Working dir:** `/agent/projects/minimax-h3 project/minimax_h3_t4`

**Pinned installer SHA256:** `14a56d60e7723d22f55e5c77a797191d338a56f7318e37ea729e1de6f5d9f560`

**Tests:** 116/116 passing

**Notebook pin:** `4c037c4` → `kaggle_install.py` → `EXTENSION_REF=e35d2cf`

## Blocker

**PAST: `_ray_get` config validation bug (FIXED in `e35d2cf`):**
- `loader_nodes.py` injects `_ray_get` into worker config dict
- `validate_worker_config` did strict equality → rejected every worker
- Result: `ActorDiedError` at `H3T4Worker.__init__()` before any model loading
- Fix: changed to per-key validation that ignores runtime-injected keys

**PAST: Host-RAM OOM (PARTIALLY FIXED in `78bed33`):**
- 29 GiB per worker × 2 = 58 GiB > 30 GiB host → Ray rejected 2nd worker
- Fixed by lowering default `H3_T4_MAX_RAM_GB` to 14 GiB
- Still needs validation: host RAM is tight (~30 GiB usable), 2×14 = 28 GiB leaves ~2 GiB for Ray overhead

**NEXT:** User must re-run notebook with pin `4c037c4` (which bundles `EXTENSION_REF=e35d2cf`). After that, confirm:
1. No `validate_worker_config` error in errror.txt
2. Workers start successfully
3. If still OOM, add `H3_T4_MAX_RAM_GB=12` or similar via env override

**UPDATE:** Notebook run confirmed working. `validate_worker_config` fix is live on Kaggle.

## Recent Fixes (committed & pushed)

| Commit | Description |
|--------|-------------|
| `e35d2cf` | Fix `validate_worker_config` — ignore runtime-injected `_ray_get` key (was rejecting all workers) |
| `78bed33` | Cap per-worker Ray memory to 14 GiB default (29 GiB × 2 > 30 GiB host) |
| `afb54e1` | Add host-RAM RSS diagnostics via `log_host()` in diag_logger + worker |
| `d04f554` | Cap Ray worker memory via `H3_T4_MAX_RAM_GB` env var |
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

## Workflow Rule: Verify Kaggle Notebook Is Actually Updated

**Never tell the user a fix is "done" or "live on Kaggle" until you have confirmed the notebook pin points at a commit that contains the fix.**

This project ships fixes through a pin chain that is easy to get wrong:

```
notebook (pin) → kaggle_install.py (fetched by hash) → EXTENSION_REF (cloned + installed)
```

A change to `src/minimax_h3_t4/runtime/*.py` is NOT live until:
1. The change is committed and pushed.
2. If the change is in the **extension** (runtime code), the notebook's `EXTENSION_REF`
   default in `kaggle_install.py` points at the fix commit (or is over-ridden via env).
3. The notebook `.ipynb` pin URL and its hardcoded SHA256 `expected` both point at the
   commit that ships the updated `kaggle_install.py`.
4. `tests/test_kaggle_install.py` assertions match the new pin/checksum/EXTENSION_REF.
5. `uv run --with pytest pytest -q` is green.

After any of the above changes, **re-read the notebook `.ipynb` from disk** and print the
resolved `url` + `expected` lines to confirm the pin is what you intended. A mismatch here
caused a silent "already fixed" claim while the user was still running the old broken installer.

Do not declare a Kaggle-side fix complete on the basis of a local commit alone.

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
