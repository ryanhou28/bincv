// Instantiates the tracker at the depths and word type the shipped ladder uses, so a
// syntax-only compile exercises the aarch64 NEON paths that x86 never sees.
#include "bincv-cpp/ops/opticalFlow.hpp"
template void bincv::calcOpticalFlowPyrLK<uint32_t, 1, 2, 2, 2>(
    const bincv::LKLevels<uint32_t, 1, 2, 2, 2>&, const bincv::Point2f*, bincv::Point2f*,
    uint8_t*, float*, size_t, const bincv::LKParams&);
