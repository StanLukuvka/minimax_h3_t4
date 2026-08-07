from __future__ import annotations


def pad_to_world_size(tensor, dim=0):
    """Pad one dimension so Ulysses can split it evenly."""
    from xfuser.core.distributed import get_sequence_parallel_world_size

    import torch

    original = tensor.shape[dim]
    world_size = get_sequence_parallel_world_size()
    padding = (-original) % world_size
    if padding == 0:
        return tensor, original
    shape = list(tensor.shape)
    shape[dim] = padding
    return torch.cat((tensor, tensor.new_zeros(shape)), dim=dim), original
