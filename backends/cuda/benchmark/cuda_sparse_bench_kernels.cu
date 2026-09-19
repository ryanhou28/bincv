// The device-side ratio pass that completes cv::cuda's BFMatcher arm, and
// nothing else.
//
// WHY IT EXISTS AT ALL, AND WHY IT IS COMMITTED RATHER THAN ASSUMED
//
// binCV's matcher returns the best match, the second best and the ratio
// verdict from ONE launch. `cv::cuda::DescriptorMatcher`'s like-for-like route
// is `knnMatchAsync(k = 2)` followed by a pass that applies Lowe's ratio to the
// two distances, and OpenCV's own way of getting at that pass --
// `knnMatchConvert` -- is a HOST download into
// `std::vector<std::vector<DMatch>>`.
//
// Timing a download inside a KERNEL-RESIDENT window would compare binCV's
// resident kernel against an OpenCV round trip and hand binCV a win it did not
// earn -- the mirror image of measuring against a fallback nobody would use. So
// the OpenCV arm's ratio pass is this kernel, enqueued on the same stream, and
// `knnMatchConvert` is outside the event window on every arm.
//
// THE LAYOUT IS OpenCV'S OWN, read out of cudafeatures2d's
// brute_force_matcher.cpp: for `k == 2` the output is a two-row `CV_32SC2`
// matrix, row 0 holding one `int2` of train indices per query and row 1 one
// `float2` of distances. Nothing is reinterpreted beyond that.
//
// The record it writes is the SHAPE binCV's matcher writes -- four 32-bit
// fields, 16 bytes -- so the two arms produce the same bytes per query and the
// comparison is not quietly between different amounts of output.

#include <cstdint>

#include <cuda_runtime.h>

namespace {

__global__ void bfRatioKernel(const int2* trainIdx, const float2* distance, int nQuery,
                              unsigned maxRatio, uint4* out) {
    const int q = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (q >= nQuery) return;
    const int2 idx = trainIdx[q];
    const float2 d = distance[q];

    // OpenCV marks "no match" with a train index of -1 (`trainIdx.setTo(-1)`
    // before the kernel runs), which is the same information binCV's sentinel
    // carries. A query with no second match cannot pass a ratio test, exactly
    // as in binCV's own `valid`.
    const unsigned best = idx.x >= 0 ? static_cast<unsigned>(d.x) : 0xFFFFFFFFu;
    const unsigned second = idx.y >= 0 ? static_cast<unsigned>(d.y) : 0xFFFFFFFFu;
    const bool ok = idx.x >= 0 && idx.y >= 0 &&
                    static_cast<unsigned long long>(best) * 100ull <=
                        static_cast<unsigned long long>(second) *
                            static_cast<unsigned long long>(maxRatio);
    uint4 r;
    r.x = idx.x >= 0 ? static_cast<unsigned>(idx.x) : 0u;
    r.y = best;
    r.z = second;
    r.w = ok ? 1u : 0u;
    out[q] = r;
}

} // namespace

void launchBfRatioPass(const void* trainIdxRow, const void* distanceRow, int nQuery,
                       unsigned maxRatio, void* dOut, cudaStream_t stream) {
    if (nQuery <= 0) return;
    const unsigned block = 128;
    const unsigned grid = (static_cast<unsigned>(nQuery) + block - 1) / block;
    bfRatioKernel<<<grid, block, 0, stream>>>(static_cast<const int2*>(trainIdxRow),
                                              static_cast<const float2*>(distanceRow),
                                              nQuery, maxRatio, static_cast<uint4*>(dOut));
}
