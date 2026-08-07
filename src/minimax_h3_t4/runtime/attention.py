from __future__ import annotations


def make_ulysses_attention(attention: str, sync_ulysses: bool):
    """Create the fixed xFuser TORCH_EFFICIENT long-context attention."""
    if attention != "TORCH_EFFICIENT":
        raise ValueError("The exact two-T4 path requires TORCH_EFFICIENT attention")
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
    from yunchang.kernels import AttnType

    operator = xFuserLongContextAttention(
        use_sync=sync_ulysses,
        attn_type=AttnType[attention],
    )

    def attention_op(q, k, v, heads, *, skip_reshape=False, **kwargs):
        if skip_reshape:
            batch, _, _, dim_head = q.shape
        else:
            batch, _, packed = q.shape
            dim_head = packed // heads
            q, k, v = (item.view(batch, -1, heads, dim_head).transpose(1, 2) for item in (q, k, v))
        output = operator(
            None,
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            softmax_scale=kwargs.get("scale"),
        ).transpose(1, 2)
        return output.transpose(1, 2).reshape(batch, -1, heads * dim_head)

    return attention_op
