# Plan: Mandatory SageAttention (sm75) for the two-T4 MiniMax-H3 runtime

Status: PLANNING ONLY — no implementation.
Scope: integrate the netrunner-exe SageAttention2 Colab wheel (sm75/Turing) as a
**mandatory** component of the exact two-T4 H3 runtime, alongside the existing
Ulysses sequence-parallel attention.

---

## 1. Ground truth I established (read the code, not guesses)

### 1.1 H3 attention does NOT go through ComfyUI's `optimized_attention`
The H3 forward (`h3_forward.py::h3_ulysses_attention`) calls
`xfuser_optimized_attention(q, k, v, self.heads, skip_reshape=True)` directly.
That is `make_ulysses_attention("TORCH_EFFICIENT", False)` →
`xFuserLongContextAttention` from `xfuser.core.long_ctx_attention`, which does a
**Ulysses all-to-all** (sequence/head redistribution across the 2 ranks) and then
an inner attention kernel.

Consequence: **`--use-sage-attention` / `optimized_attention = attention_sage`
will do NOTHING for H3 today.** The H3 path bypasses ComfyUI's global attention
dispatch entirely. This is the single biggest planning constraint.

### 1.2 SageAttention's dtype contract (from the wheel README)
- Inputs must be fp16 **or** bf16, else `sageattn` raises and ComfyUI falls back
  to PyTorch attention (`optimized_attention` line 580).
- The wheel is **cp312, linux_x86_64, built for SM75** (T4) — matches our
  Python 3.12 venv and GPU.
- Builds provided: torch 2.11.0+cu130 (rec), torch 2.11.0+cu128 (legacy),
  torch 2.13.0+cu130 (latest).

### 1.3 The Kaggle stack is torch-pinned and NOT on 2.11
The installer (`kaggle_install.py`) pins `transformers==5.0.0`, `diffusers`,
`kernels`, `ray==2.56.1`, `xfuser`, `yunchang`, all matched to ComfyUI
`9a9fdb1`. Confirmed torch/cuda in the BYOD image is whatever `9a9fdb1`
requires — NOT necessarily 2.11/cu128. Installing the wheel against a
mismatched torch is the real risk.

### 1.4 The H3 forward already profile-tracks memory
`h3_memory_trace.py` (committed in the diagnostic build) snapshots allocation
state at every attention phase, so we can empirically verify Sage gives a
measured win (and doesn't regress) after wiring it.

---

## 2. The design decision: where does SageAttention plug in?

There are two viable insertion points. The Ulysses coupling decides which one.

### Option A — Replace xfuser's INNER kernel (preferred, keeps Ulysses)
The xfuser Ulysses operator does: all-to-all (head redistribution) → local
attention on gathered heads. If the local attention kernel is configurable, we
swap the inner QK^T·V kernel to `sageattn` **while keeping the all-to-all
communication**. This preserves exact 2-rank Ulysses semantics and only
accelerates the per-rank local attention.

Need to confirm at implementation time: does
`xFuserLongContextAttention(attn_type=...)` accept a kernel override, or is the
inner kernel hardcoded? If hardcoded to TORCH_EFFICIENT, we cannot reach
SageAttention without forking/replacing the operator.

### Option B — Bypass xfuser entirely, do Ulysses manually + SageAttention
Replace `make_ulysses_attention` with a hand-rolled Ulysses exchange
(`all_to_all_4D` from yunchang, already a dependency) followed by a direct
`optimized_attention`-style call routed to SageAttention. More control, but more
code and we re-derive what xfuser gives us.

**Decision at implementation:** attempt Option A first (minimal, keeps xfuser's
proven all-to-all). If the inner kernel is not swappable, fall to Option B.

---

## 3. The torch-matching problem (must solve before wiring)

The wheel is built against torch 2.11/cu128 (or 2.13/cu130). `sageattn` is a
C++/CUDA extension loaded into the process; it links against the installed
torch's ABI. Installing an sm75 wheel built for torch 2.11 into a process
running torch 2.9 (or whatever triton/xfuser pins) can segfault.

**Mandatory precondition, in order:**
1. Check the installed torch.cuda / torch.__version__ on the Kaggle venv.
2. If it matches a wheel build (2.11+cu128, 2.11+cu130, 2.13+cu130) → use that wheel.
3. If NOT, pick the closest wheel and treat a version-mismatched C++ ABI as a
   hard blocker until confirmed. The safd decision (rewheel for our torch, or
   bump torch to 2.11) is a separate change to the pinned stack.

This is not optional polish — it is the thing most likely to crash.

---

## 4. dtype check on the H3 QKV path

The wheel needs fp16/bf16 inputs. The H3 attention gets q/k/v after
`qkv_proj` — need to confirm the compute dtype at that point. The INT8 weights
(nvfp4/int8_convrot) affect the *linear*s, not necessarily the attention
tensors: after qkv_proj, q/k/v are typically the compute dtype (fp16/bf16).
If they're fp32, we must cast for the attention call and cast back (or keep fp32
path — but then SageAttention won't engage).

---

## 5. Implementation steps (ordered, each independently testable)

1. **Probe the operator:** read `comfy/ldm/minimax/model.py` + xfuser
   `long_ctx_attention/hybrid/attn_layer.py` to confirm whether the inner
   attention kernel is overridable. Decide A vs B.
2. **Baseline memory profile** (already have the tool): `H3_T4_MEMORY_TRACE=1`
   run at the failing 10s/0.4MP config → capture per-phase allocation. This is
   the before-number.
3. **Install the sm75 wheel** in a scratch venv (NOT the deployed one), confirm:
   `import sageattention; sageattn` loads without ABI error at the right torch.
4. **Wire SageAttention into the Ulysses inner kernel** (Option A) or the manual
   Ulysses path (Option B). Gate behind `H3_T4_ENABLE_SAGE=1` initially (contrary
   to the "mandatory" end-state, the first integration is gated to A/B test).
5. **Run the identical 10s/0.4MP workload** with `H3_T4_MEMORY_TRACE=1` and
   Sage on → capture after-number. Compare: peak VRAM, per-step time, and
   whether the OOM is gone.
6. **If it works**: flip to mandatory. Update the installer to install the wheel
   (pinned, checksum-checked) and remove the env gate. Update tests to assert
   SageAttention is present at runtime.
7. **If it does NOT fit the 10s/0.4MP** even with Sage (the all-to-all + 6GB
   activations may still exceed 14.56GiB): the plan extends to activation
   offload — but that is a *separate* change, kept out of this plan.

---

## 6. Risks / unknowns (explicit, not hand-waved)

- **torch ABI mismatch** (most severe). Unc: which torch version the Kaggle image
  + pinned stack resolves to. Blocks step 3 until known.
- **Inner kernel not overridable** in xfuser → forces Option B (more code).
- **dtype mismatch**: if H3 attention q/k/v are fp32 (not fp16/bf16), the wheel
  silently no-ops and falls back — must verify by reading the actual tensor
  dtype at the attention call, or the "2.3×" is imaginary.
- **The all-to-all is not accelerated by Sage.** Sage only speeds the local
  matmul; the NCCL head-redistribution cost is unchanged. Net win is smaller
  than the README's 2.3× which is single-GPU (no Ulysses all-to-all).
- **The wheel is community/unofficial** — vendoring it means trusting
  netrunner-exe's build. Mitigation: verify its sha256, pin the exact URL, and
  treat the binary as untrusted until it matches the expected build provenance.

---

## 7. Deliverables / acceptance

- SageAttention sm75 wheel installed in the Kaggle venv, pinned + checksummed.
- H3 attention routes through the SageAttention inner kernel (Option A preferred).
- Identical 10s/0.4MP run completes without the Ulysses OOM (or with peak VRAM
  clearly below the ceiling at the attention phase).
- Measured: peak-allocated and per-step time, before vs after (from
  `H3_T4_MEMORY_TRACE`).
- If the memory win is confirmed, Sage is mandatory in the installer (no gate);
  tests assert the wheel loads and is used.
