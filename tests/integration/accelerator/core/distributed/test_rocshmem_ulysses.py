import os

import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]


def _check_queued_layers(rank, group, device, pressure_path):
    import torch
    import torch.distributed as dist

    from xfuser.core.distributed.sdma import ulysses

    peers, rows, dim = group.size(), 256, 64
    ex = ulysses.Exchange.make(peers, 1, dim, torch.float32, [rows] * 3, [2] * 3, [(0, 0)] * 3, 2)
    heads = torch.arange(2, device=device).view(1, 1, 1, 2, 1)

    def layer(epoch):
        ulysses.begin(group, ex, rocshmem=True)
        for gi in range(2):
            value = 100 * epoch + 10 * rank + gi
            for tensors in ((0, 1), (2,)):
                part = torch.full((peers, len(tensors), rows, dim), float(value), device=device)
                ulysses.send(group, gi, part, tensors)
        out = ulysses.finish(group, lambda q, k, v: q + k + v)
        return (out.view(1, rows, peers, 2, dim) == 3 * (100 * epoch + 10 * rank + heads)).all()

    assert layer(0).item()
    host = dist.new_group(backend="gloo")
    # hipMalloc can synchronize the device. Prime the allocator before delaying peers so
    # even a regression can enqueue all layers and reach the assertion.
    if rank == 0:
        reserve = [torch.empty(3 * peers * rows * dim, device=device) for _ in range(64)]
        del reserve
    # Hold back rank 0's peers: its CPU can enqueue layers, but its first SDMA read cannot finish.
    if rank != 0:
        dist.barrier(group=host)
    checks = [layer(epoch) for epoch in range(1, 9)]
    if rank == 0:
        stats = torch.cuda.memory_stats(device)
        awaiting_free = stats["active_bytes.all.current"] - stats["allocated_bytes.all.current"]
        dist.barrier(group=host)
        layer_bytes = ex.groups * (ex.tensors + 1) * peers * rows * dim * ex.dtype.itemsize
        pressure_path.write_text(str(awaiting_free / layer_bytes))
    assert torch.stack(checks).all().item()


def _exchange(rank, world, init_method, pressure_path):
    import torch
    import torch.distributed as dist

    from xfuser.core.distributed.sdma.shmem import exchange

    os.environ["LOCAL_WORLD_SIZE"] = str(world)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world, device_id=device)
    groups = [dist.group.WORLD]
    for members in (list(range(4)), list(range(4, 8))):
        pg = dist.new_group(members)
        if rank in members:
            groups.append(pg)
    for group in groups:
        peers = dist.get_process_group_ranks(group)
        for planes in (1, 3):
            offsets = torch.arange(64 * planes, device=device).view(planes, 64)
            for epoch in range(4):
                src = torch.stack([offsets + (100 * epoch + 8 * rank + p) * 256 for p in peers])
                got = exchange(src, group, planes, planes=planes == 3).view(planes, len(peers), 64)
                expected = torch.stack([offsets + (100 * epoch + 8 * p + rank) * 256 for p in peers], dim=1)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
    _check_queued_layers(rank, groups[-1], device, pressure_path)
    dist.destroy_process_group()


def test_world_and_cfg_inbox_reuse(accelerator_ranks, tmp_path):
    import torch

    if not torch.cuda.is_available() or (
        not os.environ.get("XDIT_ROCSHMEM_LIBRARY") and not hasattr(torch.ops.symm_mem, "rocshmem_exchange_planes")
    ):
        pytest.skip("requires the PyTorch rocSHMEM plane exchange")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    pressure = tmp_path / "pending_source_layers"
    accelerator_ranks(_exchange, world_size=8, args=(pressure,))
    # Allow one layer of temporaries, not one per outstanding layer.
    assert float(pressure.read_text()) <= 1
