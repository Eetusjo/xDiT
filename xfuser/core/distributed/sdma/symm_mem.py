"""Staged Ulysses exchanges through PyTorch's ROCm ND collective."""

import functools
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as sm


@functools.lru_cache(None)
def _initialize(group):
    world = dist.get_world_size()
    ranks = dist.get_process_group_ranks(group)
    if world != int(os.environ.get("LOCAL_WORLD_SIZE", "0")) or not 2 <= world <= 8:
        raise ValueError("symm_mem Ulysses requires a single torchrun node with 2-8 GPUs")
    if ranks != list(range(world)) and not (
        world == 8 and ranks in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6], [1, 3, 5, 7])
    ):
        raise ValueError(f"Unsupported symm_mem Ulysses group: {ranks}")
    if torch.version.hip is None or not hasattr(torch.ops.symm_mem, "rocshmem_all_to_all_nd"):
        raise RuntimeError("symm_mem Ulysses requires PyTorch with rocSHMEM all_to_all_nd")
    sm.set_backend("NVSHMEM")


@functools.lru_cache(None)
def _buffer(group, slot, numel, device, dtype):
    # Equal-sized CFG teams must enter world heap allocations in the same order.
    stage = sm.empty(numel, dtype=dtype, device=device)
    sm.rendezvous(stage, group=group)
    return stage, torch.cuda.Event()


@functools.lru_cache(None)
def stream_for(device):
    return torch.cuda.Stream(device=device)


def exchange(src, group, slot, scatter_dim, rows):
    """Stage on the producer stream. Return ordinary output and its readiness event."""
    return _exchange(src, src.numel(), group, slot, scatter_dim, rows, lambda inp: inp.copy_(src.view_as(inp)))


def exchange_planes(pieces, group, slot):
    """Pack equal Q/K/V planes directly into symmetric storage on the producer stream."""
    first = pieces[0]
    shape = (group.size(), len(pieces), *first.shape[1:])

    def pack(inp):
        torch.stack(pieces, dim=1, out=inp.view(shape))

    return _exchange(first, len(pieces) * first.numel(), group, slot, 0, len(pieces), pack)


def _exchange(src, numel, group, slot, scatter_dim, rows, fill):
    _initialize(group)
    p = group.size()
    stage, retired = _buffer(group, slot, numel, src.device, src.dtype)
    shape = (p, rows, -1) if scatter_dim == 0 else (rows, p, -1)
    inp = stage.view(shape)
    out = torch.empty_like(inp.transpose(0, 1), memory_format=torch.contiguous_format)
    producer = torch.cuda.current_stream(src.device)
    producer.wait_event(retired)
    fill(inp)
    comm = stream_for(src.device)
    comm.wait_stream(producer)
    with torch.cuda.stream(comm):
        sm.all_to_all_nd(inp, out, scatter_dim, 1 - scatter_dim, group=group.group_name)
        retired.record()
    return out, retired
