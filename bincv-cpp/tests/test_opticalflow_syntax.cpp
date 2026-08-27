// The aarch64 SYNTAX surface for scripts/check_arm_syntax.sh.
//
// Roughly a third of this library is invisible to every x86 build -- D-33's tap
// batching, X-40's accumulator, ops/fast.hpp's NEON mask, ops/pack.hpp's bit-weight
// fold -- because it lives behind `#if BINCV_HAVE_NEON && __aarch64__`. An edit there
// can be structurally broken and still pass all four verify.sh configurations.
//
// THIS FILE MUST INCLUDE AND INSTANTIATE EVERY HEADER THAT HAS A NEON PATH. A check
// that compiles only the tracker reports "aarch64 syntax OK" while a broken NEON
// kernel sits in another header -- which it did, until ops/fast.hpp gained one.
#include "bincv-cpp/io/pnm.hpp"
#include "bincv-cpp/ops/descriptor.hpp"
#include "bincv-cpp/ops/edge.hpp"
#include "bincv-cpp/ops/fast.hpp"
#include "bincv-cpp/ops/medianWide.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/ops/pack.hpp"

// The tracker at the shipped 1/2/2/2 ladder's depths and word type, so the NEON
// residual kernels are actually reached.
template void bincv::calcOpticalFlowPyrLK<uint32_t, 1, 2, 2, 2>(
    const bincv::LKLevels<uint32_t, 1, 2, 2, 2>&, const bincv::Point2f*, bincv::Point2f*,
    uint8_t*, float*, size_t, const bincv::LKParams&);

// ops/fast.hpp's vector path is inside `if constexpr (sizeof(SrcT) == 1)`, so only a
// uint8_t instantiation compiles it.
template size_t bincv::detectFast<uint8_t>(const uint8_t*, size_t, size_t, size_t, long long,
                                           bincv::FastCorner*, size_t, bool*, int);

// ops/pack.hpp's NEON fold, at both source widths and a word type on each side of the
// 32-bit boundary -- the split that the `uint64_t` bug lived in.
template void bincv::packBits<bincv::PackRule::GreaterThan, uint8_t, uint32_t>(
    const uint8_t*, size_t, size_t, size_t, bincv::BinMatView<uint32_t>, uint8_t);
template void bincv::packBits<bincv::PackRule::NonZero, uint16_t, uint64_t>(
    const uint16_t*, size_t, size_t, size_t, bincv::BinMatView<uint64_t>, uint16_t);
