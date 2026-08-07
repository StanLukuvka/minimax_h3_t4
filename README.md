# ComfyUI MiniMax H3 T4

Standalone ComfyUI custom-node package for running MiniMax H3 across two NVIDIA T4 GPUs with FSDP, Ulysses sequence parallelism, worker-local CUDA INT8, and optional Spectrum forecasting.

This repository is being extracted from the proven dual-T4 Raylight experiment. It does not require the Raylight, ComfyUI Spectrum MiniMax H3, or EasyCache custom-node packages at runtime.

## Target profile

- 2 × NVIDIA Tesla T4
- Quantized ConvRot MiniMax H3 checkpoint
- Two-rank FSDP and Ulysses
- CUDA INT8 backend
- Spectrum history in system RAM

The exact-attention control and Spectrum mode are the only supported execution modes in the initial standalone package.
