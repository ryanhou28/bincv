// The launch floor's kernel, and the only reason this translation unit exists:
// a `__global__` needs nvcc, and the benchmark translation units that print the
// floor are host C++ compiled by the host compiler.
//
// The kernel body is empty on purpose and must STAY empty. What it measures is
// the cost of enqueueing and retiring a launch with nothing inside it -- the
// number every kernel-resident figure in this backend sits on top of. On WSL2
// that cost is inflated relative to native Linux, which is exactly why it has
// to be printed rather than assumed small: several of the backend's ops are
// individually cheap enough that a microbenchmark ratio between two of them
// would otherwise be a ratio of launch overheads.

#include <cuda_runtime.h>

#include "cuda_bench_util.hpp"

namespace cudabench {
namespace {

/// @brief Nothing. Deliberately: a kernel that touched memory would measure the
/// memory, and one that took a parameter it used would measure the read.
__global__ void nullKernel() {}

} // namespace

void launchNullKernel(dim3 grid, dim3 block, cudaStream_t stream) {
    nullKernel<<<grid, block, 0, stream>>>();
}

} // namespace cudabench
