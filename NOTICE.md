# Third-party provenance

This project is a derivative integration built from the following research and custom-node work.

- Raylight, by Micko Lesmana and contributors, Apache-2.0. Distributed ComfyUI execution, Ray actor ownership, FSDP loading, Ulysses integration, and sampler/lifecycle structure were adapted from commit `908b1857e69721d83240bd6e7213eed6abe8f06d` plus later MiniMax-H3 experiment commits through `95fc45713f6bcd1d4e55f480588a0bb8ef884db8`.
- ComfyUI-Spectrum-MiniMax-H3, copyright 2026 xmarre, GPL-3.0-or-later. Spectrum configuration, forecasting, solver policy, and MiniMax-H3 feature transaction semantics are adapted from commit `85ec1da66277e893079ecd46e32cc865c56cfe53`.
- ComfyUI and its MiniMax-H3 implementation provide the host model and workflow contracts.
- xFuser provides Ulysses sequence-parallel collectives.
- ComfyKitchen provides quantized linear execution.

The standalone package is licensed GPL-3.0-or-later because it incorporates and modifies GPL-covered Spectrum source. Modified derivative files retain provenance in their module docstrings or headers.
