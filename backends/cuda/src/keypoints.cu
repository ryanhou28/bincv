// The detector-to-keypoint-set link. One kernel, one template, two element
// types -- see cuda/keypoints.hpp for why the op exists at all.
//
// THERE IS NO SECOND ARM AND NO SWITCH, AND THAT IS A DECISION RATHER THAN AN
// OMISSION. A switch exists so a fast arm can be proven to be running; this
// kernel has no fast arm to prove. It issues one load and two stores per
// keypoint over at most a few hundred keypoints, which is under a microsecond
// of traffic against this platform's launch floor, so every shape of it is the
// same measurement. What DOES need proving -- that keeping the conversion on
// the device is worth a launch -- is a property of the PIPELINE, not of this
// kernel, and the resident frontend example carries both arms of it: the
// device conversion against the download-convert-upload round trip it replaces.

#include "bincv/cuda/keypoints.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

constexpr unsigned kBlock = 128;

/// One thread owns one SLOT of the destination, not one corner: the slots past
/// the count have to be written too, and writing them is what lets the
/// consuming launches be sized by capacity.
template <typename CornerT>
__global__ void keypointsFromCornersKernel(const CornerT* __restrict__ corners,
                                           const uint32_t* __restrict__ dCount,
                                           float* __restrict__ xy, uint32_t capacity) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    // Every thread reads the same word. That is a broadcast out of L1 after the
    // first warp, not `capacity` loads of DRAM, and it is what keeps the count
    // on the device where the atomic left it.
    const uint32_t count = *dCount;
    const uint32_t live = count < capacity ? count : capacity;
    float x = 0.0f;
    float y = 0.0f;
    if (i < live) {
        const CornerT c = corners[i];
        x = static_cast<float>(c.x);
        y = static_cast<float>(c.y);
    }
    xy[2 * static_cast<size_t>(i)] = x;
    xy[2 * static_cast<size_t>(i) + 1] = y;
}

template <typename CornerT>
cudaError_t launch(const CornerT* corners, const uint32_t* dCount, float* xy,
                   uint32_t capacity, cudaStream_t stream) {
    if (capacity == 0) return cudaSuccess;
    BINCV_ASSERT(corners != nullptr && dCount != nullptr && xy != nullptr,
                 "keypointsFromCorners: a non-empty call needs non-null pointers");
    if (corners == nullptr || dCount == nullptr || xy == nullptr)
        return cudaErrorInvalidValue;
    const unsigned blocks = (capacity + kBlock - 1) / kBlock;
    keypointsFromCornersKernel<CornerT><<<blocks, kBlock, 0, stream>>>(corners, dCount, xy,
                                                                      capacity);
    return cudaGetLastError();
}

} // namespace

cudaError_t keypointsFromCorners(const DeviceCorner* corners, const uint32_t* dCount,
                                 float* xy, uint32_t capacity, cudaStream_t stream) {
    return launch(corners, dCount, xy, capacity, stream);
}

cudaError_t keypointsFromCorners(const DeviceFastCorner* corners, const uint32_t* dCount,
                                 float* xy, uint32_t capacity, cudaStream_t stream) {
    return launch(corners, dCount, xy, capacity, stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
