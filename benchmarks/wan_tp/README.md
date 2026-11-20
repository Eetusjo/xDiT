# Wan2.2 tensor parallelism: profiling and the comm/GEMM overlap study

Wan2.2-T2V-A14B runs Megatron-style TP with a sequence-sharded residual stream: each rank keeps
1/tp of the tokens between layers. Each column-parallel GEMM (Q/K/V, cross-attention Q, FFN up)
all-gathers rows into it; each row-parallel GEMM (attention out, FFN down) reduce-scatters rows out.
`--tensor_parallel_degree N` combines with `--ulysses_degree` / `--ring_degree`.

| Environment | Values | Effect |
|---|---|---|
| `XDIT_WAN_TP_CROSS_ATTENTION` | `sharded` (default), `replicated` | Split cross-attention like self-attention (3 AG + 3 RS per block), or keep it whole on every rank's own rows (2 + 2). |
| `XDIT_TP_COMM_OVERLAP` | `none` (default), `chunked` | Pipeline each collective against its GEMM: RCCL/NCCL on a side stream, GEMM of chunk c overlapping the transfer of chunk c+1. |
| `XDIT_TP_OVERLAP_CHUNKS` | integer >= 2, default 4 | Chunks for `chunked`. |
| `XDIT_TP_COMM_OVERLAP` | `symm_mem` | torch's fused `symm_mem.fused_all_gather_matmul` / `fused_matmul_reduce_scatter` for every pair (peer copies pipelined against per-shard GEMMs). NVLink only: segfaults on torch 2.13 / ROCm 7.16. |
| `XDIT_TP_COMM_OVERLAP` | `rocshmem` | Both AG and RS use the PyTorch rocSHMEM bridge, block-cyclic rows, and `XDIT_TP_OVERLAP_CHUNKS`. No xDiT ROCr TP library is loaded. |

TP refuses FP8/FP4/FP6 GEMMs, FP8 comms, step caching, FSDP, PipeFusion and CPU offload.

### PyTorch rocSHMEM transport (experimental)

Build the SDMA-enabled PyTorch bridge with this branch's
`xfuser/core/distributed/sdma/rocshmem_exchange.cu`. The previous Ulysses
image already supports the one-plane exchange for RS, but lacks the
`symm_mem::rocshmem_all_gather_out` fan-out operator. For AG, it reads one
ordinary contiguous source shard and puts it into the sender's slot on every
peer. For RS the destination-specific tiles are exchanged, then summed locally
in FP32. PyTorch events order each chunk's transfer against its GEMM, including
inbox reuse. Both directions use public rocSHMEM APIs, not HSA engine IDs.

Start with a single gfx950 node and TP8. Allow enough symmetric heap for the
cached chunk inboxes (`ROCSHMEM_HEAP_SIZE=6442450944` in the 720p test).
The generic exchange supports only the base branch's regular team partitions
and requires world-symmetric workspace allocation order/sizes. This prototype
does not implement the ROCr transport's bounded peer waits or CUDA-graph replay.

Run `benchmarks/tp_comm_gemm_overlap.py --rocshmem --chunks 4` under eight
torchrun ranks; run `--sdma` in a **fresh process** as the ROCr control,
since the symmetric allocator is process-global. For the full Wan model use
`XDIT_TP_COMM_OVERLAP=rocshmem XDIT_TP_OVERLAP_CHUNKS=4` with
`--tensor_parallel_degree 8`. Set `XDIT_WAN_TP_CROSS_ATTENTION=replicated`
to match the PR's replicated-cross-attention benchmark.

For PyTorch upstream, the intended primitive is a group-scoped, stream-ordered
all-gather into a symmetric output (and equal-tile all-to-all for RS), not a
Wan-specific fused op. The experimental `first/stride/size` arguments should
be resolved by c10d from a process group. Source lifetime, output aliasing,
subgroup creation and completion need defined contracts. The xDiT wrappers
keep block-cyclic row order; they are not drop-in implementations of torch's
natural-order fused AsyncTP APIs.

## Tools

- `benchmarks/tp_comm_gemm_overlap.py`: microbenchmark of each AG+GEMM / GEMM+RS site of a block at
  Wan shapes. Variants are `baseline`, `chunked_k{2,4,8}` and, with `--symm-mem`, torch's
  `symm_mem.fused_all_gather_matmul` / `fused_matmul_reduce_scatter`. Writes `tp<P>.md`.
- The original [AsyncTP comparison branch](https://github.com/jjuvonen-amd/xDiT/tree/5d107504490f8f3a8f88e2e221619dcc2be6ee40)
  contains the profiling scripts and the optional `patches/pytorch` prototype used by `--symm-mem-sdma`.
