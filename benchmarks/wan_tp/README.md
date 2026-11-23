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

TP refuses FP8/FP4/FP6 GEMMs, FP8 comms, step caching, FSDP, PipeFusion and CPU offload.

## Tools

- `benchmarks/tp_comm_gemm_overlap.py`: microbenchmark of each AG+GEMM / GEMM+RS site of a block at
  Wan shapes. Variants are `baseline`, `chunked_k{2,4,8}` and, with `--symm-mem`, torch's
  `symm_mem.fused_all_gather_matmul` / `fused_matmul_reduce_scatter`. Writes `tp<P>.md`.
- The original [AsyncTP comparison branch](https://github.com/jjuvonen-amd/xDiT/tree/5d107504490f8f3a8f88e2e221619dcc2be6ee40)
  contains the profiling scripts and the optional `patches/pytorch` prototype used by `--symm-mem-sdma`.
