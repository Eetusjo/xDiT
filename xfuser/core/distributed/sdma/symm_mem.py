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
    """Return ordinary output and a readiness event. Keep src alive through the wait."""
    _initialize(group)
    p = group.size()
    stage, retired = _buffer(group, slot, src.numel(), src.device, src.dtype)
    shape = (p, rows, -1) if scatter_dim == 0 else (rows, p, -1)
    inp = stage.view(shape)
    out = torch.empty_like(inp.transpose(0, 1), memory_format=torch.contiguous_format)
    producer = torch.cuda.current_stream(src.device)
    comm = stream_for(src.device)
    comm.wait_stream(producer)
    try:
        with torch.cuda.stream(comm):
            # This stream owns every reuse. Stage from the original strided layout
            # while the producer projects or attends the next head group.
            stage.view(src.shape).copy_(src)
            sm.all_to_all_nd(inp, out, scatter_dim, 1 - scatter_dim, group=group.group_name)
            retired.record()
    except BaseException:
        producer.wait_stream(comm)
        raise
    return out, retired
