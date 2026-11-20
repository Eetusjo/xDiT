"""Wan's block-cyclic AsyncTP schedule over PyTorch symmetric memory (inference)."""

from typing import Optional

import torch
import torch.distributed as dist

from xfuser.core.distributed.sdma import shmem


def _setup(x, group_name, chunks, factor):
    if torch.version.hip is None or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("rocSHMEM AsyncTP requires ROCm without graph capture")
    group = dist.distributed_c10d._resolve_process_group(group_name)
    if chunks < 1 or x.shape[1] % (chunks * factor):
        raise ValueError(f"{x.shape[1]} rows do not split evenly into {chunks * factor} pieces")
    compute, comm = torch.cuda.current_stream(x.device), shmem.stream_for(x.device)
    comm.wait_stream(compute)
    return group, compute, comm


@torch.library.custom_op("xfuser::tp_rocshmem_all_gather_linear", mutates_args=())
def all_gather_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], group_name: str, chunks: int
) -> torch.Tensor:
    group, compute, comm = _setup(x, group_name, chunks, 1)
    batch, rows, features = x.shape
    block, world = rows // chunks, group.size()
    gathered, ready = [], []
    with torch.cuda.stream(comm):
        for c in range(chunks):
            shard = x.narrow(1, c * block, block).contiguous()
            gathered.append(shmem.all_gather_chunk(shard, group, ("tp-ag", c)))
            ready.append(comm.record_event())
    x.record_stream(comm)
    out = x.new_empty(chunks, batch, world * block, weight.shape[0])
    for c in range(chunks):
        compute.wait_event(ready[c])
        a = gathered[c].transpose(0, 1).reshape(-1, features)
        dest = out[c].view(-1, weight.shape[0])
        if bias is None:
            torch.mm(a, weight.t(), out=dest)
        else:
            torch.addmm(bias, a, weight.t(), out=dest)
    return out.transpose(0, 1).reshape(batch, world * rows, weight.shape[0])


@all_gather_linear.register_fake
def _(x, weight, bias, group_name, chunks):
    world = dist.distributed_c10d._resolve_process_group(group_name).size()
    return x.new_empty(x.shape[0], x.shape[1] * world, weight.shape[0])


@torch.library.custom_op("xfuser::tp_rocshmem_linear_reduce_scatter", mutates_args=())
def linear_reduce_scatter(
    x: torch.Tensor, weight: torch.Tensor, group_name: str, tp_size: int, chunks: int
) -> torch.Tensor:
    group, compute, comm = _setup(x, group_name, chunks, tp_size)
    if tp_size != group.size():
        raise ValueError("TP size must match group size")
    batch, rows, _ = x.shape
    span, block = rows // chunks, rows // (chunks * tp_size)
    out = x.new_empty(chunks, batch, block, weight.shape[0])
    received, ready = [], []

    def reduce(c):
        compute.wait_event(ready[c])
        out[c].copy_(received[c].sum(dim=0, dtype=torch.float32))

    for c in range(chunks):
        partial = torch.matmul(x.narrow(1, c * span, span), weight.t())
        tiles = partial.view(batch, tp_size, block, weight.shape[0]).transpose(0, 1).contiguous()
        comm.wait_stream(compute)
        with torch.cuda.stream(comm):
            received.append(shmem.exchange(tiles, group, ("tp-rs", c)))
            ready.append(comm.record_event())
        tiles.record_stream(comm)
        if c:
            reduce(c - 1)
    reduce(chunks - 1)
    return out.transpose(0, 1).reshape(batch, rows // tp_size, weight.shape[0])


@linear_reduce_scatter.register_fake
def _(x, weight, group_name, tp_size, chunks):
    return x.new_empty(x.shape[0], x.shape[1] // tp_size, weight.shape[0])
