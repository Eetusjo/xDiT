"""Single-node, inference-only Ulysses exchange through PyTorch symmetric memory."""

import functools
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as sm


@functools.lru_cache(None)
def stream_for(device):
    return torch.cuda.Stream(device=device)


@functools.lru_cache(None)
def _partition(group):
    world = dist.get_world_size()
    ranks = dist.get_process_group_ranks(group)
    if world != int(os.environ.get("LOCAL_WORLD_SIZE", "0")) or not 2 <= world <= 8:
        raise ValueError("rocSHMEM Ulysses needs one torchrun node with 2-8 GPUs")
    if ranks == list(range(world)):
        return 0, 1, world
    if world == 8 and ranks in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6], [1, 3, 5, 7]):
        return ranks[0], ranks[1] - ranks[0], 4
    raise ValueError(f"Unsupported rocSHMEM Ulysses group: {ranks}")


@functools.lru_cache(None)
def _initialize():
    if library := os.environ.get("XDIT_ROCSHMEM_LIBRARY"):
        torch.ops.load_library(library)
    if not hasattr(torch.ops.symm_mem, "rocshmem_exchange_planes"):
        raise RuntimeError(
            "Missing rocSHMEM plane exchange: build diffusion-models-inference/docker/Dockerfile.ci with this xDiT revision"
        )
    sm.set_backend("NVSHMEM")


@functools.lru_cache(None)
def _workspace(group, slot, numel, dtype, device):
    # Heap allocations are world collective, even when payloads stay within CFG teams.
    out = sm.empty(numel, dtype=dtype, device=device)
    handle = sm.rendezvous(out, group=dist.group.WORLD)
    return out, handle, torch.cuda.current_stream(device)


def exchange(src, group, slot, planes=False):
    partition = _partition(group)
    _initialize()
    out, _, stream = _workspace(group, slot, src.numel(), src.dtype, src.device)
    if torch.cuda.current_stream(src.device) != stream:
        raise RuntimeError("Cannot reuse a symmetric inbox on a different stream")
    torch.ops.symm_mem.rocshmem_exchange_planes(src, out, *partition, 3 if planes else 1)
    return out.view(src.shape)


def use_planes(ex):
    return ex.batch == 1 and ex.tensors == 3 and len(set(ex.rows)) == len(set(ex.heads)) == 1


def all_gather_chunk(src, group, slot):
    """Fan out a contiguous shard; receive one shard per sender in rank order."""
    partition = _partition(group)
    _initialize()
    if not hasattr(torch.ops.symm_mem, "rocshmem_all_gather_out"):
        raise RuntimeError("AsyncTP needs the PyTorch rocSHMEM bridge built with this xDiT kernel revision")
    out, _, stream = _workspace(group, slot, src.numel() * group.size(), src.dtype, src.device)
    if torch.cuda.current_stream(src.device) != stream:
        raise RuntimeError("Cannot reuse a symmetric inbox on a different stream")
    torch.ops.symm_mem.rocshmem_all_gather_out(src, out, *partition)
    return out.view(group.size(), *src.shape)
