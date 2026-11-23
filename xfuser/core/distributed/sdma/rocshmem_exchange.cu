// xDiT direct-placement op for PyTorch symmetric memory, single-node IPC only.
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <rocshmem/rocshmem.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/library.h>

namespace {
using namespace rocshmem;

// All PEs must create both children in the same order, including nonmembers.
// Keep this restriction local to this op rather than changing TeamManager for
// unrelated PyTorch collectives. One torchrun process owns one device.
rocshmem_team_t exchange_team(int first, int stride, int size) {
  int world = rocshmem_n_pes(), me = rocshmem_my_pe();
  if (size == world && first == 0 && stride == 1) return ROCSHMEM_TEAM_WORLD;
  TORCH_CHECK(world == 8 && size == 4 && (stride == 1 || stride == 2) &&
              first == (stride == 1 ? 4 * (me / 4) : me % 2), "unsupported rocSHMEM partition");
  static rocshmem_team_t local = ROCSHMEM_TEAM_INVALID;
  static int cached_stride = 0;
  if (local == ROCSHMEM_TEAM_INVALID) {
    for (int i = 0; i < 2; ++i) {
      int start = stride == 1 ? i * 4 : i;
      rocshmem_team_t child = ROCSHMEM_TEAM_INVALID;
      TORCH_CHECK(rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, start, stride, size, nullptr, 0, &child) == 0);
      if (first == start) local = child;
    }
    cached_stride = stride;
  }
  TORCH_CHECK(cached_stride == stride && local != ROCSHMEM_TEAM_INVALID, "cannot change CFG partition");
  return local;
}

__global__ void exchange(const char* src, char* dst, size_t bytes, int planes, rocshmem_team_t team) {
  const int peers = rocshmem_team_n_pes(team), sender = rocshmem_team_my_pe(team);
  rocshmem_ctx_sync_wg(ROCSHMEM_CTX_DEFAULT, team);  // previous inbox readers have finished
  for (int j = threadIdx.x; j < peers; j += blockDim.x) {
    const int pe = rocshmem_team_translate_pe(team, j, ROCSHMEM_TEAM_WORLD);
    for (int p = 0; p < planes; ++p) {
      rocshmem_putmem_nbi(dst + (p * peers + sender) * bytes, src + (j * planes + p) * bytes, bytes, pe);
    }
    rocshmem_pe_quiet(&pe, 1);
  }
  __syncthreads();
  rocshmem_ctx_sync_wg(ROCSHMEM_CTX_DEFAULT, team);
}

void exchange_planes(const at::Tensor& src, at::Tensor& out, int64_t first, int64_t stride, int64_t size,
                     int64_t planes) {
  TORCH_CHECK(src.is_cuda() && src.device() == out.device() && src.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(src.scalar_type() == out.scalar_type() && src.numel() == out.numel() && !src.is_alias_of(out));
  TORCH_CHECK(size >= 2 && size <= 8 && (planes == 1 || planes == 3) && src.numel() > 0 &&
              src.numel() % (size * planes) == 0);
  c10::cuda::CUDAGuard guard(src.device());
  auto handle = c10d::symmetric_memory::rendezvous(out, "0");
  TORCH_CHECK(handle && handle->world_within_direct_access(), "destination must be a symmetric IPC tensor");
  auto team = exchange_team(first, stride, size);
  hipLaunchKernelGGL(exchange, dim3(1), dim3(64), 0, at::cuda::getCurrentCUDAStream(),
                     static_cast<const char*>(src.const_data_ptr()), static_cast<char*>(out.mutable_data_ptr()),
                     src.nbytes() / (size * planes), static_cast<int>(planes), team);
  C10_HIP_CHECK(hipGetLastError());
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def("rocshmem_exchange_planes(Tensor src, Tensor(a!) out, int first, int stride, int size, int planes) -> ()");
}
TORCH_LIBRARY_IMPL(symm_mem, CUDA, m) {
  m.impl("rocshmem_exchange_planes", exchange_planes);
}
