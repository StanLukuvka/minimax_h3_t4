"""MiniMax-H3 Ulysses forward implementation.

Apache-2.0 donor provenance is recorded in NOTICE.md.
"""

import os

import torch

import comfy
from comfy.ldm.minimax.model import (
    pack_audio,
    patchify_video,
    rope_rotation_table,
    time_shift_sigma,
    time_shift_slope,
    unpack_audio,
    unpatchify_video,
    PackedLayout,
    VISUAL_COND_TIMESTEP,
    AUDIO_COND_TIMESTEP,
)
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

from .attention import make_ulysses_attention
from .memory_trace import memory_snapshot as _mem_snapshot, memory_trace_enabled as _trace_enabled
from .utils import pad_to_world_size
from .diag_logger import (
    log_memory,
    log_tensor,
    log_attention_memory,
    log_ulysses_alltoall,
    profile_start,
    profile_end,
    log_fsdp_shard_state,
)


xfuser_optimized_attention = make_ulysses_attention("TORCH_EFFICIENT", False)
_H3_TRACE_FORWARD_DONE = False
_H3_ATTN_PROFILE_DONE = False
_H3_FORWARD_COUNT = 0


def h3_memory_snapshot(tag, **extra):
    _mem_snapshot(tag, device=torch.cuda.current_device() if torch.cuda.is_available() else None, extra=extra)


def h3_memory_trace_enabled():
    return _trace_enabled()


def h3_cuda_phase_report(*args, **kwargs):
    return None


def h3_phase_profile_active():
    return False


def h3_phase_profile_enabled():
    return False


def h3_stop_after_first_forward():
    return _trace_enabled() and os.environ.get("H3_T4_STOP_AFTER_FIRST_FORWARD", "1") == "1"


def set_h3_phase_profile_active(_active):
    return None


def _split_packed_sequence(h, rope_freqs, mod_segments):
    world_size = get_sequence_parallel_world_size()
    local_size = h.shape[0] // world_size
    start = get_sequence_parallel_rank() * local_size
    end = start + local_size
    local_segments = []
    for segment_start, segment_end, row in mod_segments:
        segment_start = max(segment_start, start)
        segment_end = min(segment_end, end)
        if segment_start < segment_end:
            local_segments.append((segment_start - start, segment_end - start, row))
    # Materialize rank-local storage. Returning views would keep the complete
    # pre-split hidden state and RoPE table alive through every DiT block.
    local_h = h[start:end].clone()
    local_rope = rope_freqs[:, start:end].clone()
    return local_h, local_rope, local_segments


def h3_ulysses_attention(self, x, rope_freqs=None, transformer_options={}):
    global _H3_ATTN_PROFILE_DONE
    profile_this_attention = h3_phase_profile_active() and not _H3_ATTN_PROFILE_DONE
    trace_this_attention = h3_memory_trace_enabled()
    phase_events = []

    def mark_phase():
        if not profile_this_attention:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    sequence_length = x.shape[0]
    phase_start = mark_phase()
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    phase_end = mark_phase()
    if trace_this_attention:
        h3_memory_snapshot("h3_attention_after_qkv", local_rows=sequence_length, heads=self.heads, head_dim=self.head_dim)
        log_memory("after_qkv_proj", extra={"q_bytes": q.numel() * q.element_size(), "k_bytes": k.numel() * k.element_size(), "v_bytes": v.numel() * v.element_size()})
    if profile_this_attention:
        phase_events.append(("qkv_projection", phase_start, phase_end))
    phase_start = mark_phase()
    v = v.view(sequence_length, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, sequence_length, self.heads, self.head_dim)
        k = k.view(1, sequence_length, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(sequence_length, self.heads, self.head_dim))
        k = self.k_norm(k.view(sequence_length, self.heads, self.head_dim))
    if trace_this_attention:
        h3_memory_snapshot("h3_attention_after_rmsnorm_rope", local_rows=sequence_length)
    phase_end = mark_phase()
    if profile_this_attention:
        phase_events.append(("rmsnorm_rope", phase_start, phase_end))
        h3_memory_snapshot("h3_attention_after_rmsnorm_rope", local_rows=sequence_length)
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    phase_start = mark_phase()
    if trace_this_attention:
        h3_memory_snapshot("h3_attention_before_ulysses", local_rows=sequence_length)
        log_memory("before_ulysses", extra={
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "q_bytes": q.numel() * q.element_size(),
            "k_bytes": k.numel() * k.element_size(),
            "v_bytes": v.numel() * v.element_size(),
        })
        log_attention_memory(q, k, v, "pre_alltoall")
    # All-to-all communication for Ulysses
    log_ulysses_alltoall(tuple(q.shape), tuple(q.shape), "pre_attention")
    out = xfuser_optimized_attention(q, k, v, self.heads, skip_reshape=True)
    phase_end = mark_phase()
    if trace_this_attention:
        h3_memory_snapshot("h3_attention_after_ulysses", local_rows=sequence_length)
        log_memory("after_ulysses", extra={"out_bytes": out.numel() * out.element_size()})
    if profile_this_attention:
        phase_events.append(("ulysses_attention", phase_start, phase_end))
        h3_memory_snapshot("h3_attention_after_ulysses", local_rows=sequence_length)
    phase_start = mark_phase()
    result = self.out_proj(out.squeeze(0))
    phase_end = mark_phase()
    if profile_this_attention:
        phase_events.append(("attention_output_projection", phase_start, phase_end))
        h3_memory_snapshot("h3_attention_after_output_projection", local_rows=sequence_length)
        _H3_ATTN_PROFILE_DONE = True
        h3_cuda_phase_report(
            phase_events,
            local_rows=sequence_length,
            heads=self.heads,
            head_dim=self.head_dim,
        )
    return result


def h3_ulysses_forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
    global _H3_FORWARD_COUNT, _H3_TRACE_FORWARD_DONE
    phase_profile_enabled = h3_phase_profile_enabled()
    profile_this_forward = phase_profile_enabled and _H3_FORWARD_COUNT == 1
    set_h3_phase_profile_active(False)
    trace_this_forward = h3_memory_trace_enabled() and not _H3_TRACE_FORWARD_DONE and (not phase_profile_enabled or profile_this_forward)
    stop_this_forward = h3_stop_after_first_forward() and (not phase_profile_enabled or profile_this_forward)
    if trace_this_forward:
        torch.cuda.reset_peak_memory_stats()
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype  # compute dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    if trace_this_forward:
        h3_memory_snapshot(
            "h3_forward_start",
            video_shape=list(video_x.shape),
            audio_shape=list(audio_x.shape),
            text_len=text_len,
        )
    # extra_conds prebuilds the layout once per sampling run
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = PackedLayout(
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
            frame_count=payload.get("frame_count"),
        )

    # model_base passes model_sampling.timestep(sigma) = sigma * 1000
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

    # distinct timesteps are known analytically: text/pad follow video, cond rows pin near 1
    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
    has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
    seg_t = {
        "text": t_v,
        "video": t_v,
        "audio": t_a,
        "cond": max(t_v, vis_aug),
        "ref_img": max(t_v, vis_aug),
        "ref_audio": max(t_a, aud_aug),
    }
    unique_t = sorted({t_v, t_a} | ({seg_t["cond"]} if has_vis_cond else set()) | ({seg_t["ref_audio"]} if has_aud_cond else set()))
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            # the presentation text span mixes tags (vision pads carry the video modality) split into tag runs
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                    run_start = i
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))

    # embed
    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = pack_audio(audio_x.to(torch.float32))
    cond_video_rows = self._cond_video_rows(payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(self.condition_proj(text_states), transformer_options=transformer_options)

    # segments are contiguous: assemble by slices, embed rows follow segment order
    h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    voff = aoff = 0
    for a, b, kind in layout.segments:
        n = b - a
        if kind == "text":
            h[a:b] = text_states
        elif kind in ("cond", "ref_img", "video"):
            h[a:b] = video_embed[voff : voff + n]
            voff += n
        else:  # ref_audio / audio
            h[a:b] = audio_embed[aoff : aoff + n]
            aoff += n
    if trace_this_forward:
        h3_memory_snapshot("h3_after_embed", seq_len=layout.seq_len)

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        # adaln projections consume interpolated coordinates of the time-embedding curve
        table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)  # t in [0,1] -> fractional grid index, out-of-range t clamps to the curve ends
        i0 = (
            pos.floor().long().clamp(max=table.shape[0] - 2)
        )  # lower grid row, max-clamp keeps t=1.0 on the last interval instead of reading past the table
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))  # blend the two rows by the fractional part
    else:
        t_emb = self.time_embedder(t_vals).to(dtype)

    # rotation table computed once per forward, consumed by the kitchen split-half rope
    rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)
    # ===================== SP SPLIT ====================== #
    if trace_this_forward:
        h3_memory_snapshot("h3_before_split", seq_len=layout.seq_len)
    h, h_orig_size = pad_to_world_size(h, dim=0)
    rope_freqs, _ = pad_to_world_size(rope_freqs, dim=1)
    h, rope_freqs, mod_segments = _split_packed_sequence(h, rope_freqs, mod_segments)

    # These full-sequence embedding intermediates are no longer consumed once
    # rank-local hidden state has been materialized above.
    del video_embed, audio_embed
    del all_video_rows, all_audio_rows, video_rows, audio_rows
    del cond_video_rows, cond_audio_rows, text_states, context
    del img_update, audio_update, payload, minimax_payload
    if trace_this_forward:
        h3_memory_snapshot("h3_after_split_release", local_rows=h.shape[0])
    log_memory("after_embed_release", extra={"local_seq_len": h.shape[0]})

    # One owner chooses exact block execution or a synchronized Spectrum forecast.
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})

    def run_exact_blocks(hidden):
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
        for i, block in enumerate(self.blocks):
            if trace_this_forward:
                torch.cuda.reset_peak_memory_stats()
                h3_memory_snapshot("h3_block_start", block=i, local_rows=hidden.shape[0])
                log_memory(f"block_{i}_start", extra={"local_seq_len": hidden.shape[0]})
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
            if trace_this_forward:
                h3_memory_snapshot("h3_block_after_prefetch", block=i, local_rows=hidden.shape[0])
            if profile_this_forward and i == 0:
                set_h3_phase_profile_active(True)
            if ("double_block", i) in blocks_replace:

                def block_wrap(args):
                    return {
                        "img": block(
                            args["img"],
                            args["t_emb"],
                            args["mod_segments"],
                            args["rope_freqs"],
                            transformer_options=args["transformer_options"],
                        )
                    }

                hidden = blocks_replace[("double_block", i)](
                    {
                        "img": hidden,
                        "t_emb": t_emb,
                        "mod_segments": mod_segments,
                        "rope_freqs": rope_freqs,
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )["img"]
            else:
                hidden = block(hidden, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
            if profile_this_forward and i == 0:
                set_h3_phase_profile_active(False)
            if trace_this_forward:
                h3_memory_snapshot("h3_block_end", block=i, local_rows=hidden.shape[0])
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
        return hidden

    spectrum = getattr(self, "_h3_t4_spectrum_controller", None)
    if spectrum is None:
        h = run_exact_blocks(h)
    else:
        topology = (
            tuple(h.shape),
            int(h_orig_size),
            tuple(tuple(segment) for segment in mod_segments),
        )
        h = spectrum.execute_h3_stack(
            timestep=sigma_v,
            hidden=h,
            topology=topology,
            exact=lambda: run_exact_blocks(h),
        )

    # ===================== SP GATHER ===================== #
    if trace_this_forward:
        h3_memory_snapshot("h3_before_final_gather", local_rows=h.shape[0])
    h = get_sp_group().all_gather(h.contiguous(), dim=0)
    h = h[:h_orig_size]

    video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
    audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
    v, a = self.final_layer(h, t_emb, video_seg, audio_seg)

    video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = unpack_audio(a)

    # The sampler integrates the flat ODE dX/dsigma_v = (X - denoised)/sigma_v.
    # Scaling the audio velocity by d(sigma_a)/d(sigma_v) makes that ODE equal
    # to the audio stream's true ODE on its own shifted schedule.
    slope_a = time_shift_slope(sigma_v, shift_v, shift_a).to(audio_out.dtype)
    if trace_this_forward:
        h3_memory_snapshot("h3_forward_end", gathered_rows=h.shape[0])
    if trace_this_forward or stop_this_forward:
        _H3_TRACE_FORWARD_DONE = True
    _H3_FORWARD_COUNT += 1
    set_h3_phase_profile_active(False)
    if stop_this_forward:
        raise RuntimeError("MiniMax-H3 diagnostic intentionally stopped after the first denoiser forward")
    return [-video_out.to(video_x.dtype), (-slope_a) * audio_out.to(audio_x.dtype)]
