import os

import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]


def _exchange(rank, world, init_method):
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
    dist.destroy_process_group()


def test_world_and_cfg_inbox_reuse(accelerator_ranks):
    import torch

    if not torch.cuda.is_available() or (
        not os.environ.get("XDIT_ROCSHMEM_LIBRARY") and not hasattr(torch.ops.symm_mem, "rocshmem_exchange_planes")
    ):
        pytest.skip("requires the PyTorch rocSHMEM plane exchange")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    accelerator_ranks(_exchange, world_size=8)
