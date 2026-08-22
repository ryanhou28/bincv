// Pyramid downsample (T3.4): the 2x2 box mean, subsampled, requantized to NOut
// bits -- ops/pyramid.hpp.
//
// CORE, AND MOSTLY NOT BY OMISSION. pyrDown is API TIER 2 (ARCHITECTURE 5.1):
// it has cv::pyrDown's name and role and deliberately different numerics, so
// there is no bit-exactness promise to check and no Tier 1 denominator. What
// stands behind it is three independent references, and only the third needs
// OpenCV:
//
//   1. A PER-PIXEL REFERENCE over at(), written from the operation's definition
//      and sharing no expression with the kernel. It runs in all four
//      configurations, including Debug -- the only one where pyrDown's
//      BINCV_ASSERT preconditions are live -- and -fno-exceptions, which is the
//      embedded claim.
//   2. THE REJECTED FORMULATION. impl::boxSum4Replicated reaches the same 2x2
//      sum through ops/bitslice.hpp's SINGLE-BIT adder network by replicating
//      plane p of each pixel 2^p times (k = 4 * (2^NIn - 1) inputs). It is the
//      exponential route T3.4 exists to replace, and keeping it under test is
//      what makes "the shipped route is linear in NIn" a comparison rather than
//      a claim -- the same reason ops/resample.hpp keeps E-8's losing arms.
//   3. THE REFERENCE PIPELINE'S BOX_2x2 PATH, behind BINCV_WITH_OPENCV:
//      cv::blur(2x2) then subsample with the Gaussian disabled, which is what
//      SEAL/src/keypoint_tracking/pyramids.cpp does. Two things are checked
//      there and neither is a tier promise: that binCV at NOut = 8 reproduces
//      the reference's value set exactly on the aligned block, and that
//      cv::blur's DEFAULT anchor really does shift its window half a pixel up
//      and to the left -- the deviation ops/pyramid.hpp documents.
//
// THE ARITHMETIC IS ENUMERATED, NOT SAMPLED. The requantizer's whole input space
// is the 4 * (2^NIn - 1) + 1 possible sums, so Pyramid.Requantize_* runs every
// one of them for every (NIn, NOut) pair -- packed into the LANES of the words
// under test, so the enumeration also proves the lanes stay independent. Same
// for the four bit-sliced primitives underneath it.
//
// CHECK GRANULARITY: ONE PER DESTINATION PIXEL for the sweeps. Per row would
// leave the CHECKS column blind to a shortened width sweep, which is this
// suite's most likely regression; the failure message is built only on failure.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/pyramid.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

// ---------------------------------------------------------------------------
// The allocation counter, in the idiom tests/test_storage.cpp established.
//
// ops/pyramid.hpp promises "no allocation and no scratch parameter". Half of that
// sentence used to carry a wrong number for the kernel's automatic storage (it
// quoted the widest single intermediate, NIn + NOut + 2 words, as the total); the
// stack half is now measured with -fstack-usage and recorded in the header and in
// EXPERIMENTS.md X-15, because it is a property of the emitted code rather than
// of the source. THE HEAP HALF IS CHECKABLE HERE, and is checked -- a kernel that
// grew a std::vector of scratch would still pass every value test in this file.
// ---------------------------------------------------------------------------
namespace {
std::size_t g_newCount = 0;

const void* volatile g_sink = nullptr;
inline void escape(const void* p) { g_sink = p; }

void* countedAllocate(std::size_t bytes) {
    ++g_newCount;
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    // Cannot throw std::bad_alloc: this file also builds with -fno-exceptions.
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) std::abort();
    return p;
}

void countedFree(void* p) noexcept { std::free(p); }

} // namespace

void* operator new(std::size_t bytes)   { return countedAllocate(bytes); }
void* operator new[](std::size_t bytes) { return countedAllocate(bytes); }

void operator delete(void* p) noexcept                 { countedFree(p); }
void operator delete[](void* p) noexcept               { countedFree(p); }
void operator delete(void* p, std::size_t) noexcept    { countedFree(p); }
void operator delete[](void* p, std::size_t) noexcept  { countedFree(p); }

namespace {

using bincv::pyrDown;
using bincv::pyrDownHeight;
using bincv::pyrDownWidth;
using bincv::QuantMat;

#define PYR_EXPECT(ok, what, detailExpr) \
    ::bincv::test::reportCheck((ok), (what), __FILE__, __LINE__, \
                               (ok) ? std::string() : (detailExpr))

// ---------------------------------------------------------------------------
// Inputs and the per-pixel reference
// ---------------------------------------------------------------------------

// splitmix64, so a failure reproduces exactly.
uint64_t nextRandom(uint64_t& state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

template <size_t N, typename WordType>
void fillRandom(QuantMat<N, WordType>& m, uint64_t seed) {
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            m.set(y, x, static_cast<unsigned>(nextRandom(state)) & QuantMat<N, WordType>::MaxValue);
        }
    }
}

/// The operation's definition, evaluated per pixel through at().
///
/// @note Deliberately the FIRST spelling from ops/pyramid.hpp's header --
///       "round(mean * (2^NOut - 1) / (2^NIn - 1))" written as an exact integer
///       rounding -- so that the kernel's second spelling (multiply, add, divide
///       in bit-sliced planes) has something to be equal to that is not itself.
/// @note The edge rule is written out as coordinates: 2y+1 and 2x+1 clamp to the
///       last row and column, which IS the replication ops/pyramid.hpp documents.
template <size_t NOut, size_t NIn, typename WordType>
unsigned referencePixel(const QuantMat<NIn, WordType>& src, size_t y, size_t x) {
    const size_t h = src.getHeight();
    const size_t w = src.getWidth();
    const size_t r0 = 2 * y;
    const size_t r1 = (2 * y + 1 < h) ? (2 * y + 1) : r0;
    const size_t c0 = 2 * x;
    const size_t c1 = (2 * x + 1 < w) ? (2 * x + 1) : c0;

    const unsigned sum =
        static_cast<unsigned>(src.at(static_cast<int>(r0), static_cast<int>(c0))) +
        static_cast<unsigned>(src.at(static_cast<int>(r0), static_cast<int>(c1))) +
        static_cast<unsigned>(src.at(static_cast<int>(r1), static_cast<int>(c0))) +
        static_cast<unsigned>(src.at(static_cast<int>(r1), static_cast<int>(c1)));

    const unsigned maxIn = (1u << NIn) - 1u;
    const unsigned maxOut = (1u << NOut) - 1u;
    // floor((S * A + 2M) / (4M)) == round(S / 4 * A / M), rounded half up.
    return (sum * maxOut + 2u * maxIn) / (4u * maxIn);
}

/// Every bit at or above `width` in every row's words of every plane, which D-13
/// requires to be zero. Returns the first offending (plane, row, bit), or "".
template <size_t N, typename WordType>
std::string paddingDirt(const QuantMat<N, WordType>& m) {
    constexpr size_t B = sizeof(WordType) * 8;
    const size_t rowWords = (m.getWidth() + B - 1) / B;
    for (size_t p = 0; p < N; ++p) {
        const bincv::BinMatConstView<WordType> plane = m.plane(p);
        for (size_t y = 0; y < plane.height; ++y) {
            for (size_t bit = plane.width; bit < rowWords * B; ++bit) {
                if (((plane.row(y)[bit / B] >> (bit % B)) & 1u) != 0) {
                    return "plane " + std::to_string(p) + " row " + std::to_string(y) +
                           " padding bit " + std::to_string(bit);
                }
            }
        }
    }
    return std::string();
}

template <size_t N, typename WordType>
void dirtyThePadding(QuantMat<N, WordType>& m) {
    constexpr size_t B = sizeof(WordType) * 8;
    const size_t width = m.getWidth();
    if (width == 0) return;
    const size_t rowWords = (width + B - 1) / B;
    for (size_t p = 0; p < N; ++p) {
        bincv::BinMatView<WordType> plane = m.plane(p);
        for (size_t y = 0; y < plane.height; ++y) {
            WordType* row = plane.row(y);
            for (size_t bit = width; bit < rowWords * B; ++bit) {
                row[bit / B] = static_cast<WordType>(
                    row[bit / B] |
                    static_cast<WordType>(static_cast<WordType>(1) << (bit % B)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 1. The reference sweep
// ---------------------------------------------------------------------------

std::vector<size_t> sweepWidths(size_t wordBits) {
    std::vector<size_t> widths;
    for (size_t w = 0; w <= 2 * wordBits + 3; ++w) widths.push_back(w);
    // The pyramid ladder a 640x480 frontend actually walks.
    for (size_t w : {size_t{94}, size_t{160}, size_t{320}}) widths.push_back(w);
    return widths;
}

template <size_t NOut, size_t NIn, typename WordType>
std::string caseLabel(const char* wordName, size_t width, size_t height, const char* suffix) {
    return std::string(wordName) + " " + std::to_string(NIn) + "->" + std::to_string(NOut) +
           " " + std::to_string(width) + "x" + std::to_string(height) + suffix;
}

/// One case: fill a source, run pyrDown, compare every destination pixel against
/// the per-pixel reference, and check the destination's padding bits.
template <size_t NOut, size_t NIn, typename WordType>
void checkCase(const char* wordName, size_t width, size_t height, uint64_t seed,
               bool dirtySource) {
    QuantMat<NIn, WordType> src(static_cast<int>(width), static_cast<int>(height));
    fillRandom(src, seed);
    if (dirtySource) dirtyThePadding(src);

    const size_t dstWidth = pyrDownWidth(width);
    const size_t dstHeight = pyrDownHeight(height);

    QuantMat<NOut, WordType> dst(static_cast<int>(dstWidth == 0 ? 1 : dstWidth),
                                 static_cast<int>(dstHeight == 0 ? 1 : dstHeight));
    // Pre-set every pixel, so a destination the kernel fails to write shows up as
    // a wrong value rather than as whatever a zero-filled allocation held.
    for (int y = 0; y < dst.rows(); ++y) {
        for (int x = 0; x < dst.cols(); ++x) dst.set(y, x, QuantMat<NOut, WordType>::MaxValue);
    }

    bincv::BinMatConstView<WordType> srcPlanes[NIn];
    bincv::BinMatView<WordType> dstPlanes[NOut];
    for (size_t p = 0; p < NIn; ++p) srcPlanes[p] = src.plane(p);
    for (size_t q = 0; q < NOut; ++q) {
        dstPlanes[q] = dst.plane(q);
        dstPlanes[q].width = dstWidth;
        dstPlanes[q].height = dstHeight;
    }
    pyrDown<NOut, NIn, WordType>(srcPlanes, dstPlanes);

    const std::string label =
        caseLabel<NOut, NIn, WordType>(wordName, width, height, dirtySource ? " dirty" : "");

    for (size_t y = 0; y < dstHeight; ++y) {
        for (size_t x = 0; x < dstWidth; ++x) {
            const unsigned got = dst.at(static_cast<int>(y), static_cast<int>(x));
            const unsigned want = referencePixel<NOut, NIn, WordType>(src, y, x);
            PYR_EXPECT(got == want, "pyrDown pixel matches the per-pixel reference",
                       label + " at (" + std::to_string(y) + "," + std::to_string(x) +
                           "): got " + std::to_string(got) + ", expected " +
                           std::to_string(want));
        }
    }

    if (dstWidth > 0 && dstHeight > 0 && dstWidth == dst.getWidth() &&
        dstHeight == dst.getHeight()) {
        const std::string dirt = paddingDirt(dst);
        PYR_EXPECT(dirt.empty(), "destination padding bits stay zero (D-13)", label + ": " + dirt);
    }
}

template <size_t NOut, size_t NIn, typename WordType>
void testAgainstReference(const char* wordName) {
    constexpr size_t B = sizeof(WordType) * 8;
    uint64_t seed = 0x9091u + static_cast<uint64_t>(B * 131 + NIn * 17 + NOut);
    for (size_t width : sweepWidths(B)) {
        for (size_t height : {size_t{1}, size_t{2}, size_t{3}, size_t{5}}) {
            checkCase<NOut, NIn, WordType>(wordName, width, height, seed++, false);
        }
    }
}

/// A source whose padding bits are all ones. Only the odd-width case can route a
/// padding bit anywhere near a live destination pixel -- and there the kernel
/// replicates the last live column over it before the arithmetic runs.
template <size_t NOut, size_t NIn, typename WordType>
void testDirtySource(const char* wordName) {
    constexpr size_t B = sizeof(WordType) * 8;
    uint64_t seed = 0xD147u + static_cast<uint64_t>(B * 7 + NIn);
    for (size_t width = 1; width <= 2 * B + 1; ++width) {
        if (width % B == 0) continue;
        checkCase<NOut, NIn, WordType>(wordName, width, 3, seed++, true);
    }
    checkCase<NOut, NIn, WordType>(wordName, 94, 5, seed++, true);
}

/// Strides that differ between source and destination -- which is exactly what a
/// pyramid ladder produces, since every level rounds its own row length up (D-5
/// says a kernel may not care).
template <size_t NOut, size_t NIn, typename WordType>
void testDifferingStrides(const char* wordName) {
    constexpr size_t B = sizeof(WordType) * 8;
    for (size_t width : {size_t{7}, B + 1, 2 * B, size_t{94}}) {
        const size_t height = 6;
        QuantMat<NIn, WordType> src(static_cast<int>(width), static_cast<int>(height),
                                    8 * sizeof(WordType));
        fillRandom(src, 0xA11u + width);
        QuantMat<NOut, WordType> dst(static_cast<int>(pyrDownWidth(width)),
                                     static_cast<int>(pyrDownHeight(height)));
        pyrDown<NOut, NIn, WordType>(src, dst);

        const std::string label = caseLabel<NOut, NIn, WordType>(wordName, width, height, " strides");
        PYR_EXPECT(src.getAlignedWidth() != dst.getAlignedWidth(),
                   "the case really does use differing strides", label);
        for (size_t y = 0; y < pyrDownHeight(height); ++y) {
            for (size_t x = 0; x < pyrDownWidth(width); ++x) {
                const unsigned got = dst.at(static_cast<int>(y), static_cast<int>(x));
                const unsigned want = referencePixel<NOut, NIn, WordType>(src, y, x);
                PYR_EXPECT(got == want, "differing strides do not change the result",
                           label + ": got " + std::to_string(got) + ", expected " +
                               std::to_string(want));
            }
        }
    }
}

/// Degenerate shapes must be a no-op rather than a crash.
template <size_t NOut, size_t NIn, typename WordType>
void testDegenerate(const char* wordName) {
    QuantMat<NIn, WordType> src(8, 4);
    fillRandom(src, 7);
    QuantMat<NOut, WordType> dst(4, 2);

    bincv::BinMatConstView<WordType> srcPlanes[NIn];
    bincv::BinMatView<WordType> dstPlanes[NOut];

    for (size_t p = 0; p < NIn; ++p) srcPlanes[p] = src.plane(p);
    for (size_t q = 0; q < NOut; ++q) dstPlanes[q] = dst.plane(q);
    for (size_t p = 0; p < NIn; ++p) srcPlanes[p].width = 0;
    for (size_t q = 0; q < NOut; ++q) dstPlanes[q].width = 0;
    pyrDown<NOut, NIn, WordType>(srcPlanes, dstPlanes);
    PYR_EXPECT(true, "zero-width pyrDown returns without touching anything", wordName);

    for (size_t p = 0; p < NIn; ++p) srcPlanes[p] = src.plane(p);
    for (size_t q = 0; q < NOut; ++q) dstPlanes[q] = dst.plane(q);
    for (size_t p = 0; p < NIn; ++p) srcPlanes[p].height = 0;
    for (size_t q = 0; q < NOut; ++q) dstPlanes[q].height = 0;
    pyrDown<NOut, NIn, WordType>(srcPlanes, dstPlanes);
    PYR_EXPECT(true, "zero-height pyrDown returns without touching anything", wordName);
}

// ---------------------------------------------------------------------------
// 2. The edge rule, checked a second way
// ---------------------------------------------------------------------------

/// An odd extent's missing partner is REPLICATED, and this checks that against a
/// construction rather than against the same clamp the reference uses: pyrDown of
/// a (2w+1) x (2h+1) image must equal pyrDown of the (2w+2) x (2h+2) image built
/// by duplicating its last column and row. If the kernel zero-filled, reflected,
/// or dropped the odd extent instead, the two would differ.
template <size_t NOut, size_t NIn, typename WordType>
void testEdgeReplication(const char* wordName) {
    for (size_t oddWidth : {size_t{1}, size_t{3}, size_t{2 * sizeof(WordType) * 8 - 1},
                            size_t{2 * sizeof(WordType) * 8 + 1}, size_t{95}}) {
        for (size_t oddHeight : {size_t{1}, size_t{5}}) {
            QuantMat<NIn, WordType> odd(static_cast<int>(oddWidth), static_cast<int>(oddHeight));
            fillRandom(odd, 0xEDDEu + oddWidth * 31 + oddHeight);

            QuantMat<NIn, WordType> padded(static_cast<int>(oddWidth + 1),
                                           static_cast<int>(oddHeight + 1));
            for (size_t y = 0; y <= oddHeight; ++y) {
                for (size_t x = 0; x <= oddWidth; ++x) {
                    const size_t sy = (y < oddHeight) ? y : oddHeight - 1;
                    const size_t sx = (x < oddWidth) ? x : oddWidth - 1;
                    padded.set(static_cast<int>(y), static_cast<int>(x),
                               odd.at(static_cast<int>(sy), static_cast<int>(sx)));
                }
            }

            QuantMat<NOut, WordType> fromOdd(static_cast<int>(pyrDownWidth(oddWidth)),
                                             static_cast<int>(pyrDownHeight(oddHeight)));
            QuantMat<NOut, WordType> fromPadded(static_cast<int>(pyrDownWidth(oddWidth + 1)),
                                                static_cast<int>(pyrDownHeight(oddHeight + 1)));
            pyrDown<NOut, NIn, WordType>(odd, fromOdd);
            pyrDown<NOut, NIn, WordType>(padded, fromPadded);

            const std::string label = caseLabel<NOut, NIn, WordType>(wordName, oddWidth,
                                                                     oddHeight, " edge");
            PYR_EXPECT(fromOdd.getWidth() == fromPadded.getWidth() &&
                           fromOdd.getHeight() == fromPadded.getHeight(),
                       "an odd extent and its edge-replicated pad give the same level size",
                       label);
            for (size_t y = 0; y < fromOdd.getHeight(); ++y) {
                for (size_t x = 0; x < fromOdd.getWidth(); ++x) {
                    const unsigned a = fromOdd.at(static_cast<int>(y), static_cast<int>(x));
                    const unsigned b = fromPadded.at(static_cast<int>(y), static_cast<int>(x));
                    PYR_EXPECT(a == b,
                               "an odd extent replicates its edge pixel into the missing half",
                               label + " at (" + std::to_string(y) + "," + std::to_string(x) +
                                   "): " + std::to_string(a) + " vs " + std::to_string(b));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. The bit-sliced arithmetic, enumerated
// ---------------------------------------------------------------------------

/// Packs one value per LANE, least significant plane first.
template <typename WordType>
void packLanes(const std::vector<unsigned>& values, WordType* planes, size_t n) {
    constexpr size_t B = sizeof(WordType) * 8;
    for (size_t p = 0; p < n; ++p) planes[p] = 0;
    for (size_t lane = 0; lane < values.size() && lane < B; ++lane) {
        for (size_t p = 0; p < n; ++p) {
            if (((values[lane] >> p) & 1u) != 0u) {
                planes[p] = static_cast<WordType>(
                    planes[p] | static_cast<WordType>(static_cast<WordType>(1) << lane));
            }
        }
    }
}

template <typename WordType>
unsigned unpackLane(const WordType* planes, size_t n, size_t lane) {
    unsigned value = 0;
    for (size_t p = 0; p < n; ++p) {
        if (((planes[p] >> lane) & 1u) != 0u) value |= (1u << p);
    }
    return value;
}

/// addPlanes, multiplyByAllOnes, addConstant and divideByConstant, over every
/// value each can see at the widths pyrDown instantiates them at. The values are
/// packed into the lanes of the words under test, so lane independence is proven
/// by the same enumeration.
template <typename WordType>
void testBitSlicedArithmetic(const char* wordName) {
    constexpr size_t B = sizeof(WordType) * 8;
    constexpr size_t n = 4;
    constexpr unsigned limit = 1u << n;

    for (unsigned base = 0; base < limit; ++base) {
        std::vector<unsigned> lhs;
        std::vector<unsigned> rhs;
        for (size_t lane = 0; lane < B; ++lane) {
            lhs.push_back((base + static_cast<unsigned>(lane)) % limit);
            rhs.push_back((base * 5u + static_cast<unsigned>(lane) * 3u + 1u) % limit);
        }

        WordType a[n];
        WordType b[n];
        packLanes<WordType>(lhs, a, n);
        packLanes<WordType>(rhs, b, n);

        WordType sum[n + 1];
        bincv::impl::addPlanes<WordType>(a, n, b, n, sum);

        WordType product[n + 3];
        bincv::impl::multiplyByAllOnes<WordType>(a, n, 3, product);

        WordType incremented[n + 1];
        for (size_t p = 0; p < n; ++p) incremented[p] = a[p];
        incremented[n] = 0;
        bincv::impl::addConstant<WordType>(incremented, n + 1, 11u);

        WordType dividend[n + 1];
        for (size_t p = 0; p < n; ++p) dividend[p] = a[p];
        dividend[n] = 0;
        WordType quotient[n];
        bincv::impl::divideByConstant<WordType>(dividend, n + 1, 3u, quotient, n);

        for (size_t lane = 0; lane < B; ++lane) {
            const unsigned x = lhs[lane];
            const unsigned y = rhs[lane];
            const std::string where =
                std::string(wordName) + " lane " + std::to_string(lane) + " x=" +
                std::to_string(x) + " y=" + std::to_string(y);

            PYR_EXPECT(unpackLane<WordType>(sum, n + 1, lane) == x + y,
                       "addPlanes adds each lane independently", where);
            PYR_EXPECT(unpackLane<WordType>(product, n + 3, lane) == x * 7u,
                       "multiplyByAllOnes multiplies by 2^shift - 1", where);
            PYR_EXPECT(unpackLane<WordType>(incremented, n + 1, lane) == x + 11u,
                       "addConstant adds a constant in every lane", where);
            PYR_EXPECT(unpackLane<WordType>(quotient, n, lane) == x / 3u,
                       "divideByConstant is exact floor division", where);
            PYR_EXPECT(unpackLane<WordType>(dividend, n + 1, lane) == x % 3u,
                       "divideByConstant leaves the remainder behind", where);
        }
    }
}

/// The requantizer over its ENTIRE input space: every sum from 0 to 4*(2^NIn - 1),
/// and the box sum over an exhaustive-in-value set of operand quadruples.
template <size_t NOut, size_t NIn, typename WordType>
void testRequantizeEnumerated(const char* wordName) {
    constexpr size_t B = sizeof(WordType) * 8;
    constexpr unsigned maxIn = (1u << NIn) - 1u;
    constexpr unsigned maxOut = (1u << NOut) - 1u;
    const unsigned sums = 4u * maxIn + 1u;

    for (unsigned base = 0; base < sums; base += static_cast<unsigned>(B)) {
        std::vector<unsigned> values;
        for (size_t lane = 0; lane < B; ++lane) {
            values.push_back((base + static_cast<unsigned>(lane)) % sums);
        }
        WordType sum[NIn + 2];
        packLanes<WordType>(values, sum, NIn + 2);

        WordType out[NOut];
        bincv::impl::requantizeBoxSum<NOut, NIn, WordType>(sum, out);

        for (size_t lane = 0; lane < B; ++lane) {
            const unsigned s = values[lane];
            const unsigned want = (s * maxOut + 2u * maxIn) / (4u * maxIn);
            const unsigned got = unpackLane<WordType>(out, NOut, lane);
            PYR_EXPECT(got == want, "requantizeBoxSum matches the exact rounded rescale",
                       std::string(wordName) + " " + std::to_string(NIn) + "->" +
                           std::to_string(NOut) + " S=" + std::to_string(s) + ": got " +
                           std::to_string(got) + ", expected " + std::to_string(want));
            PYR_EXPECT(got <= maxOut, "the requantized value fits in NOut bits",
                       std::string(wordName) + " S=" + std::to_string(s));
        }
    }
}

/// The shipped box sum against THE REJECTED ONE. Same four operands, one through
/// three ripple-carry additions (3*NIn + 1 stages), one through ops/bitslice.hpp's
/// single-bit network at k = 4 * (2^NIn - 1) inputs.
template <size_t NIn, typename WordType>
void testBoxSumAgainstReplicated(const char* wordName) {
    constexpr size_t B = sizeof(WordType) * 8;
    constexpr unsigned maxIn = (1u << NIn) - 1u;

    uint64_t state = 0xB0Cu + NIn * 977u + B;
    for (int trial = 0; trial < 8; ++trial) {
        std::vector<unsigned> values[4];
        WordType operand[4][NIn];
        for (size_t o = 0; o < 4; ++o) {
            for (size_t lane = 0; lane < B; ++lane) {
                values[o].push_back(static_cast<unsigned>(nextRandom(state)) & maxIn);
            }
            packLanes<WordType>(values[o], operand[o], NIn);
        }

        WordType direct[NIn + 2];
        WordType replicated[NIn + 2];
        bincv::impl::boxSum4<NIn, WordType>(operand[0], operand[1], operand[2], operand[3],
                                            direct);
        bincv::impl::boxSum4Replicated<NIn, WordType>(operand[0], operand[1], operand[2],
                                                      operand[3], replicated);

        for (size_t lane = 0; lane < B; ++lane) {
            const unsigned want = values[0][lane] + values[1][lane] + values[2][lane] +
                                  values[3][lane];
            const unsigned gotDirect = unpackLane<WordType>(direct, NIn + 2, lane);
            const unsigned gotRepl = unpackLane<WordType>(replicated, NIn + 2, lane);
            const std::string where = std::string(wordName) + " NIn=" + std::to_string(NIn) +
                                      " lane " + std::to_string(lane);
            PYR_EXPECT(gotDirect == want, "boxSum4 sums four NIn-bit operands exactly", where);
            PYR_EXPECT(gotRepl == want,
                       "the replicated single-bit route reaches the same sum", where);
        }
    }
}

/// THE WHOLE KERNEL, both ways. impl::pyrDownReplicated is pyrDown with the
/// exponential box sum substituted and nothing else changed, so a disagreement
/// anywhere in the sweep is a disagreement about the SUM -- the one thing T3.4
/// reformulated.
template <size_t NOut, size_t NIn, typename WordType>
void testRouteAgreement(const char* wordName) {
    constexpr size_t B = sizeof(WordType) * 8;
    uint64_t seed = 0x4047u + static_cast<uint64_t>(B * 31 + NIn * 5 + NOut);
    for (size_t width : {size_t{1}, size_t{7}, B - 1, B, B + 1, size_t{94}, size_t{160}}) {
        for (size_t height : {size_t{1}, size_t{3}, size_t{8}}) {
            QuantMat<NIn, WordType> src(static_cast<int>(width), static_cast<int>(height));
            fillRandom(src, seed++);
            QuantMat<NOut, WordType> direct(static_cast<int>(pyrDownWidth(width)),
                                            static_cast<int>(pyrDownHeight(height)));
            QuantMat<NOut, WordType> replicated(static_cast<int>(pyrDownWidth(width)),
                                                static_cast<int>(pyrDownHeight(height)));
            pyrDown<NOut, NIn, WordType>(src, direct);
            bincv::impl::pyrDownReplicated<NOut, NIn, WordType>(src, replicated);

            const std::string label =
                caseLabel<NOut, NIn, WordType>(wordName, width, height, " routes");
            for (size_t y = 0; y < direct.getHeight(); ++y) {
                for (size_t x = 0; x < direct.getWidth(); ++x) {
                    const unsigned a = direct.at(static_cast<int>(y), static_cast<int>(x));
                    const unsigned b = replicated.at(static_cast<int>(y), static_cast<int>(x));
                    PYR_EXPECT(a == b,
                               "the linear box sum and the exponential one give the same level",
                               label + " at (" + std::to_string(y) + "," + std::to_string(x) +
                                   "): " + std::to_string(a) + " vs " + std::to_string(b));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. The cost claim
// ---------------------------------------------------------------------------

/// T3.4's second blocking gap, stated as numbers. The shipped box is
/// 3*NIn + 1 full-adder stages; the replication route is 4*(2^NIn - 1) single-bit
/// inputs. A regression that reintroduced the exponential shape would break these.
void testCostIsNotExponential() {
    for (size_t nIn = 1; nIn <= 8; ++nIn) {
        const size_t linear = bincv::impl::boxSumFullAdders(nIn);
        const size_t exponential = bincv::impl::boxSum4ReplicatedInputs(nIn);
        const std::string where = "NIn=" + std::to_string(nIn);

        PYR_EXPECT(linear == 3 * nIn + 1, "the box sum costs 3*NIn + 1 full adders", where);
        PYR_EXPECT(exponential == 4 * ((size_t{1} << nIn) - 1),
                   "the replication route costs 4*(2^NIn - 1) inputs", where);
        // Linear means the FIRST DIFFERENCE is constant. Exponential means it
        // doubles. Checking the differences rather than the values is what makes
        // this a shape test and not a restatement of the formula above.
        if (nIn >= 2) {
            const size_t linearStep = linear - bincv::impl::boxSumFullAdders(nIn - 1);
            const size_t expStep = exponential - bincv::impl::boxSum4ReplicatedInputs(nIn - 1);
            PYR_EXPECT(linearStep == 3, "the shipped cost grows by a constant per bit", where);
            PYR_EXPECT(expStep == 4 * (size_t{1} << (nIn - 1)),
                       "the replicated cost doubles per bit", where);
        }
        // And the whole kernel, requantization included, stays polynomial: at a
        // fixed NOut its per-bit growth is constant too.
        for (size_t nOut = 1; nOut <= 8; ++nOut) {
            const size_t stages = bincv::impl::pyrDownAdderStages(nIn, nOut);
            const size_t expected = (3 * nIn + 1) + (nOut + 2) * (nIn + nOut + 2);
            PYR_EXPECT(stages == expected,
                       "pyrDown's total stage count is linear in NIn and quadratic in NOut",
                       where + " NOut=" + std::to_string(nOut));
            if (nIn >= 2) {
                const size_t step = stages - bincv::impl::pyrDownAdderStages(nIn - 1, nOut);
                PYR_EXPECT(step == 3 + (nOut + 2),
                           "adding a source bit adds a constant number of stages",
                           where + " NOut=" + std::to_string(nOut));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4b. The footprint claim
// ---------------------------------------------------------------------------

/// ops/pyramid.hpp's promise 2 has two halves, and only one of them was ever
/// right.
///
/// THE HEAP HALF -- "no allocation, no scratch parameter" -- is true and is
/// checked here by counting `operator new` across pyrDown at several (NIn, NOut,
/// WordType) and across Pyramid::build over the 1-3-4-5 ladder. `build()`
/// allocates nothing because every level already exists; the CONSTRUCTOR does,
/// which is why the arming happens after it.
///
/// THE STACK HALF was documented as "NIn + NOut + 2 words of automatic storage",
/// which is the widest SINGLE intermediate and not the total: the source declares
/// 8*NIn + 2*NOut + 6 words, and a -fstack-usage measurement of the emitted frame
/// is larger still (368 B at NIn=1/NOut=3/uint64_t where the old sentence implied
/// 48 B). impl::pyrDownAutomaticWords is that corrected inventory. A test cannot
/// read the emitted frame portably, so what is pinned here is the property whose
/// violation caused the bug: the budget must EXCEED the widest single
/// intermediate, so that quoting one for the other fails.
void testFootprintClaims() {
    // --- the automatic-storage inventory ---
    for (size_t nIn = 1; nIn <= 8; ++nIn) {
        for (size_t nOut = 1; nOut <= 8; ++nOut) {
            const std::string where =
                "NIn=" + std::to_string(nIn) + " NOut=" + std::to_string(nOut);
            const size_t words = bincv::impl::pyrDownAutomaticWords(nIn, nOut);

            // Named term by term, in the order ops/pyramid.hpp lists them, so a
            // reader can match this against the kernel's declarations.
            const size_t phases = 4 * nIn;              // topLeft..bottomRight
            const size_t partials = 2 * (nIn + 1);      // boxSum4's left, right
            const size_t sum = nIn + 2;                 // the 2x2 sum
            const size_t scaled = nIn + nOut + 2;       // requantize's value
            const size_t value = nOut;                  // the destination pixel
            PYR_EXPECT(words == phases + partials + sum + scaled + value,
                       "pyrDownAutomaticWords is the sum of the arrays the kernel declares",
                       where);

            // THE REGRESSION THIS FILE EXISTS TO CATCH. The header used to quote
            // the widest single intermediate as the whole budget. It is strictly
            // smaller for every supported (NIn, NOut), so the two can never again
            // be confused without this failing.
            PYR_EXPECT(words > nIn + nOut + 2,
                       "the automatic-storage budget is strictly larger than the widest "
                       "single intermediate -- quoting one for the other understated the "
                       "kernel's stack by 5x-10x",
                       where);
            // Larger by a FACTOR, not by a constant offset. The ratio runs from
            // 2.73x (NIn=1, NOut=8: 30 words against 11) to 6.9x (NIn=8, NOut=1:
            // 76 against 11), so 2x is the bound the whole supported range
            // actually supports -- and it is enough to make the substitution that
            // caused the bug a factor-of-two error at minimum.
            PYR_EXPECT(words >= 2 * (nIn + nOut + 2),
                       "and larger by a factor, not a constant: the widest single "
                       "intermediate is under half the budget everywhere in the "
                       "supported range",
                       where);

            // Linear in both, exponential in neither: the same shape claim
            // testCostIsNotExponential makes about time, made about space.
            if (nIn >= 2) {
                PYR_EXPECT(words - bincv::impl::pyrDownAutomaticWords(nIn - 1, nOut) == 8,
                           "a source bit costs a constant 8 words of stack", where);
            }
            if (nOut >= 2) {
                PYR_EXPECT(words - bincv::impl::pyrDownAutomaticWords(nIn, nOut - 1) == 2,
                           "an output bit costs a constant 2 words of stack", where);
            }
        }
    }

    // --- the heap half, measured ---
    {
        QuantMat<1, uint32_t> src(64, 32);
        QuantMat<3, uint32_t> dst(32, 16);
        for (int y = 0; y < 32; ++y)
            for (int x = 0; x < 64; ++x) src.set(y, x, static_cast<unsigned>((x ^ y) & 1));
        escape(&src);
        escape(&dst);
        const std::size_t before = g_newCount;
        pyrDown<3, 1, uint32_t>(src, dst);
        escape(&dst);
        PYR_EXPECT(g_newCount == before,
                   "pyrDown allocates nothing on the heap", "NIn=1 NOut=3 uint32_t");
    }
    {
        QuantMat<4, uint64_t> src(95, 61);
        QuantMat<5, uint64_t> dst(48, 31);
        for (int y = 0; y < 61; ++y)
            for (int x = 0; x < 95; ++x) src.set(y, x, static_cast<unsigned>((x * 7 + y) & 15));
        escape(&src);
        escape(&dst);
        const std::size_t before = g_newCount;
        pyrDown<5, 4, uint64_t>(src, dst);
        escape(&dst);
        PYR_EXPECT(g_newCount == before,
                   "pyrDown allocates nothing at an odd extent either",
                   "NIn=4 NOut=5 uint64_t");
    }
    {
        // The whole ladder. The constructor allocates; build() must not.
        bincv::Pyramid<uint32_t, 1, 3, 4, 5> pyramid(640, 480);
        auto& base = pyramid.level<0>();
        for (int y = 0; y < 480; y += 7)
            for (int x = 0; x < 640; x += 3) base.set(y, x, 1u);
        escape(&pyramid);
        const std::size_t before = g_newCount;
        pyramid.build();
        escape(&pyramid);
        PYR_EXPECT(g_newCount == before,
                   "Pyramid::build allocates nothing -- every level already exists and "
                   "pyrDown takes no scratch",
                   "1-3-4-5 at 640x480");
    }
}

// ---------------------------------------------------------------------------
// 5. The ladder
// ---------------------------------------------------------------------------

/// Pyramid<W, 1, 3, 4, 5> -- the ladder ARCHITECTURE 7.2 measured -- must size its
/// levels by ceil/2 and must produce exactly what four separate pyrDown calls do.
template <typename WordType>
void testPyramidLadder(const char* wordName) {
    for (size_t width : {size_t{640}, size_t{95}, size_t{17}}) {
        for (size_t height : {size_t{480}, size_t{61}, size_t{3}}) {
            bincv::Pyramid<WordType, 1, 3, 4, 5> pyramid(static_cast<int>(width),
                                                         static_cast<int>(height));
            const std::string label = std::string(wordName) + " " + std::to_string(width) +
                                      "x" + std::to_string(height);

            PYR_EXPECT(pyramid.Levels == 4, "the ladder has one level per bit depth", label);
            PYR_EXPECT(pyramid.template level<0>().getWidth() == width &&
                           pyramid.template level<0>().getHeight() == height,
                       "level 0 is the size it was constructed with", label);
            PYR_EXPECT(pyramid.template level<1>().getWidth() == pyrDownWidth(width),
                       "level 1 is ceil(w/2) wide", label);
            PYR_EXPECT(pyramid.template level<3>().getHeight() ==
                           pyrDownHeight(pyrDownHeight(pyrDownHeight(height))),
                       "level 3 halves the height three times", label);

            fillRandom(pyramid.template level<0>(), 0x1EAD0u + width * 13 + height);
            pyramid.build();

            // The same ladder, one call at a time, into separate containers.
            QuantMat<3, WordType> one(static_cast<int>(pyrDownWidth(width)),
                                      static_cast<int>(pyrDownHeight(height)));
            QuantMat<4, WordType> two(static_cast<int>(pyrDownWidth(one.getWidth())),
                                      static_cast<int>(pyrDownHeight(one.getHeight())));
            QuantMat<5, WordType> three(
                static_cast<int>(pyrDownWidth(two.getWidth())),
                static_cast<int>(pyrDownHeight(two.getHeight())));
            pyrDown<3, 1, WordType>(pyramid.template level<0>(), one);
            pyrDown<4, 3, WordType>(one, two);
            pyrDown<5, 4, WordType>(two, three);

            size_t disagreements = 0;
            for (size_t y = 0; y < three.getHeight(); ++y) {
                for (size_t x = 0; x < three.getWidth(); ++x) {
                    if (three.at(static_cast<int>(y), static_cast<int>(x)) !=
                        pyramid.template level<3>().at(static_cast<int>(y),
                                                       static_cast<int>(x))) {
                        ++disagreements;
                    }
                }
            }
            PYR_EXPECT(disagreements == 0,
                       "Pyramid::build is exactly the chain of pyrDown calls",
                       label + ": " + std::to_string(disagreements) + " pixels differ");

            const size_t total = pyramid.template level<0>().sizeInWords() +
                                 pyramid.template level<1>().sizeInWords() +
                                 pyramid.template level<2>().sizeInWords() +
                                 pyramid.template level<3>().sizeInWords();
            PYR_EXPECT(pyramid.sizeInWords() == total,
                       "the reported footprint is the sum of the levels", label);
            PYR_EXPECT(pyramid.sizeInBytes() == total * sizeof(WordType),
                       "the byte footprint follows the word footprint", label);

            for (const std::string& dirt : {paddingDirt(pyramid.template level<1>()),
                                           paddingDirt(pyramid.template level<2>()),
                                           paddingDirt(pyramid.template level<3>())}) {
                PYR_EXPECT(dirt.empty(), "every built level keeps its padding bits zero",
                           label + ": " + dirt);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. The OpenCV half: the reference pipeline's BOX_2x2 path
// ---------------------------------------------------------------------------

#ifdef BINCV_WITH_OPENCV

/// SEAL's BOX_2x2 pyrDown, transcribed.
///
/// @note This is a PORT of what SEAL/src/keypoint_tracking/pyramids.cpp does, not
///       a paraphrase: `cv::blur(_src, _src_new, cv::Size(2, 2))` with OpenCV's
///       DEFAULT anchor and border, then -- because `disableGaussian` is true for
///       every filter type except GAUSSIAN_5x5 -- PyrDownInvoker's early-out,
///       which is literally `dst[x * cn + c] = src[sx + c]` over rows `y * 2`.
cv::Mat referenceBox2x2(const cv::Mat& src) {
    cv::Mat blurred;
    cv::blur(src, blurred, cv::Size(2, 2));
    cv::Mat dst((src.rows + 1) / 2, (src.cols + 1) / 2, src.type());
    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            dst.at<uchar>(y, x) = blurred.at<uchar>(y * 2, x * 2);
        }
    }
    return dst;
}

/// The same, with the anchor moved to the block's top-left corner and the border
/// replicated -- binCV's GEOMETRY, expressed entirely in OpenCV calls, so that the
/// value comparison below is about arithmetic and not about phase.
cv::Mat alignedBox2x2(const cv::Mat& src) {
    cv::Mat blurred;
    cv::blur(src, blurred, cv::Size(2, 2), cv::Point(0, 0), cv::BORDER_REPLICATE);
    cv::Mat dst((src.rows + 1) / 2, (src.cols + 1) / 2, src.type());
    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            dst.at<uchar>(y, x) = blurred.at<uchar>(y * 2, x * 2);
        }
    }
    return dst;
}

cv::Mat randomBinaryMat(int width, int height, uint64_t seed) {
    cv::Mat m(height, width, CV_8U);
    uint64_t state = seed;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            m.at<uchar>(y, x) = ((nextRandom(state) & 1u) != 0) ? uchar{255} : uchar{0};
        }
    }
    return m;
}

/// **A MEASUREMENT, NOT AN ASSUMPTION.** `cv::blur` on CV_8U does not round the
/// exact mean to nearest: measured over every quadruple this sweep visits, its
/// 2x2 box is `ceil((a + b + c + d) / 4)` -- it rounds the mean UP. That is where
/// ARCHITECTURE 7.2's level-1 value set `{0, 64, 128, 192, 255}` comes from; the
/// exact means are `{0, 63.75, 127.5, 191.25, 255}`, and rounding those to
/// NEAREST would give 191, not 192.
///
/// binCV rounds once, half up, which is the deviation ops/pyramid.hpp records:
/// one LSB out of 255, on the value classes whose exact mean is not an integer.
/// This function is what turns that sentence into a check -- if OpenCV's rule
/// ever changes, the "ceil" claim fails here rather than being quietly absorbed.
int openCvBoxRule(int a, int b, int c, int d) { return (a + b + c + d + 3) / 4; }

/// binCV at NOut = 8 against the reference pipeline's own BOX_2x2 arithmetic, and
/// the two documented deviations, each pinned rather than asserted in prose.
template <typename WordType>
void testAgainstReferencePipeline(const char* wordName) {
    size_t exact = 0;
    size_t offByOne = 0;
    // The WHOLE-LEVEL distance to the reference pipeline, which is a different
    // quantity from the one-LSB arithmetic bound above and is measured here rather
    // than described. See check 6.
    size_t sealPixels = 0;
    size_t sealSame = 0;
    int sealMaxDelta = 0;
    for (int width : {16, 63, 64, 65, 128}) {
        for (int height : {8, 15, 16}) {
            const cv::Mat cvSrc = randomBinaryMat(
                width, height, UINT64_C(0xB0C0) + static_cast<uint64_t>(width * 131 + height));

            QuantMat<1, WordType> src(width, height);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    src.set(y, x, cvSrc.at<uchar>(y, x) != 0);
                }
            }
            QuantMat<8, WordType> dst(
                static_cast<int>(pyrDownWidth(static_cast<size_t>(width))),
                static_cast<int>(pyrDownHeight(static_cast<size_t>(height))));
            pyrDown<8, 1, WordType>(src, dst);

            const cv::Mat aligned = alignedBox2x2(cvSrc);
            const std::string label = std::string(wordName) + " " + std::to_string(width) + "x" +
                                      std::to_string(height);

            for (int y = 0; y < aligned.rows; ++y) {
                for (int x = 0; x < aligned.cols; ++x) {
                    const int r0 = 2 * y;
                    const int r1 = (2 * y + 1 < cvSrc.rows) ? (2 * y + 1) : r0;
                    const int c0 = 2 * x;
                    const int c1 = (2 * x + 1 < cvSrc.cols) ? (2 * x + 1) : c0;
                    const int a = cvSrc.at<uchar>(r0, c0);
                    const int b = cvSrc.at<uchar>(r0, c1);
                    const int c = cvSrc.at<uchar>(r1, c0);
                    const int d = cvSrc.at<uchar>(r1, c1);

                    const int got = static_cast<int>(dst.at(y, x));
                    const int openCv = static_cast<int>(aligned.at<uchar>(y, x));
                    const std::string where = label + " at (" + std::to_string(y) + "," +
                                              std::to_string(x) + "): binCV " +
                                              std::to_string(got) + ", cv::blur " +
                                              std::to_string(openCv);

                    // 1. OpenCV's rule, measured rather than assumed.
                    PYR_EXPECT(openCv == openCvBoxRule(a, b, c, d),
                               "cv::blur's CV_8U 2x2 box is ceil(sum / 4) -- it rounds the "
                               "mean UP, which is where ARCHITECTURE 7.2's 192 comes from",
                               where);
                    // 2. binCV's rule: the exact mean, rounded once, half up.
                    PYR_EXPECT(got == (a + b + c + d + 2) / 4,
                               "pyrDown<8, 1> is the exact 2x2 mean rounded once, half up",
                               where);
                    // 3. The two therefore agree to within one LSB out of 255, and
                    //    OpenCV is never the lower of the two. NOTE WHAT `openCv`
                    //    IS: alignedBox2x2, i.e. cv::blur at anchor (0, 0) with
                    //    BORDER_REPLICATE -- binCV's OWN geometry, so this bounds
                    //    the ARITHMETIC alone. The distance to the reference
                    //    pipeline's actual level (referenceBox2x2, default anchor)
                    //    is far larger and is entirely phase; checks 4 and 5 below
                    //    are what account for it.
                    PYR_EXPECT(openCv >= got && openCv - got <= 1,
                               "binCV and OpenCV's 2x2 box ARITHMETIC, on the aligned "
                               "block, agree to within one LSB -- OpenCV never lower "
                               "(this is not the whole-level distance to the reference "
                               "pipeline; see checks 4 and 5 for the phase)",
                               where);
                    if (got == openCv) {
                        ++exact;
                    } else {
                        ++offByOne;
                    }
                }
            }

            // 4. THE SAME DEVIATION AT THE LEVEL, not just inside the blur: the
            //    SEAL path's destination (y, x) is the 2x2 block at source rows
            //    2y-1..2y and columns 2x-1..2x, where binCV's is 2y..2y+1 by
            //    2x..2x+1. Checked on destination pixels whose shifted block is
            //    wholly inside the image, so the border rule is not the subject.
            const cv::Mat sealLevel = referenceBox2x2(cvSrc);
            for (int y = 1; y < sealLevel.rows; ++y) {
                for (int x = 1; x < sealLevel.cols; ++x) {
                    const int r0 = 2 * y - 1;
                    const int c0 = 2 * x - 1;
                    if (r0 + 1 >= cvSrc.rows || c0 + 1 >= cvSrc.cols) continue;
                    const int shifted = openCvBoxRule(
                        cvSrc.at<uchar>(r0, c0), cvSrc.at<uchar>(r0, c0 + 1),
                        cvSrc.at<uchar>(r0 + 1, c0), cvSrc.at<uchar>(r0 + 1, c0 + 1));
                    PYR_EXPECT(static_cast<int>(sealLevel.at<uchar>(y, x)) == shifted,
                               "the reference BOX_2x2 level is the 2x2 block one source "
                               "pixel up and to the left of binCV's",
                               label + " at (" + std::to_string(y) + "," + std::to_string(x) +
                                   ")");
                }
            }

            // 5. THE PHASE DEVIATION, OpenCV against OpenCV so that the two sides
            //    share a rounding rule and only the anchor differs: cv::blur's
            //    default anchor for an even kernel size is (1, 1), so its output at
            //    (r, c) is the ALIGNED block at (r-1, c-1). Interior only, where a
            //    border rule cannot be the explanation.
            cv::Mat blurDefault;
            cv::Mat blurAligned;
            cv::blur(cvSrc, blurDefault, cv::Size(2, 2));
            cv::blur(cvSrc, blurAligned, cv::Size(2, 2), cv::Point(0, 0), cv::BORDER_REPLICATE);
            size_t interior = 0;
            size_t agree = 0;
            for (int r = 1; r < cvSrc.rows; ++r) {
                for (int c = 1; c < cvSrc.cols; ++c) {
                    ++interior;
                    if (blurDefault.at<uchar>(r, c) == blurAligned.at<uchar>(r - 1, c - 1)) {
                        ++agree;
                    }
                }
            }
            PYR_EXPECT(interior > 0 && agree == interior,
                       "cv::blur's default anchor puts its 2x2 window half a pixel up and to "
                       "the left -- the phase deviation ops/pyramid.hpp records",
                       label + ": " + std::to_string(agree) + " of " + std::to_string(interior));

            // 6. REGRESSION GUARD ON THE DOCUMENTED DEVIATION ITSELF. Check 3
            //    bounds binCV against the ALIGNED block by one LSB; that bound was
            //    once described as agreement with "the reference pipeline's
            //    BOX_2x2", which it is not. Measure the actual whole-level
            //    distance to referenceBox2x2 (default anchor, the SEAL path) so
            //    the deviation is a number the header can quote, and assert that
            //    it is emphatically NOT within one LSB -- if it ever became so,
            //    the phase deviation would have silently disappeared and the three
            //    documented deviations would need re-deriving.
            for (int y = 0; y < sealLevel.rows; ++y) {
                for (int x = 0; x < sealLevel.cols; ++x) {
                    ++sealPixels;
                    const int delta = static_cast<int>(dst.at(y, x)) -
                                      static_cast<int>(sealLevel.at<uchar>(y, x));
                    const int mag = delta < 0 ? -delta : delta;
                    if (mag == 0) ++sealSame;
                    if (mag > sealMaxDelta) sealMaxDelta = mag;
                }
            }
        }
    }

    // Asserted once over the whole sweep, because it is a property of the
    // deviation and not of any one frame.
    PYR_EXPECT(sealPixels > 0 && sealMaxDelta > 1,
               "the distance to the reference pipeline's own BOX_2x2 level is NOT the "
               "one-LSB arithmetic bound -- it carries the phase deviation too, and a "
               "run where it did not would mean the phase deviation had vanished",
               std::string(wordName) + ": max |binCV - reference| = " +
                   std::to_string(sealMaxDelta));
    PYR_EXPECT(sealMaxDelta <= 255,
               "and it is bounded by full scale at NOut = 8",
               std::string(wordName) + ": " + std::to_string(sealMaxDelta));
    PYR_EXPECT(sealSame < sealPixels / 2,
               "most destination pixels differ from the reference level -- the phase "
               "deviation dominates, which is why check 3's bound is scoped to the "
               "aligned block",
               std::string(wordName) + ": " + std::to_string(sealSame) + " of " +
                   std::to_string(sealPixels) + " identical");

    std::cout << "\n--- pyrDown<8, 1> against OpenCV's 2x2 box ARITHMETIC on the ALIGNED "
                 "block (binCV's own geometry): "
              << wordName << " ---\n"
              << "    identical      " << exact << " destination pixels\n"
              << "    one LSB apart  " << offByOne
              << " (OpenCV rounds the mean up; binCV rounds to nearest)\n"
              << "--- and against the reference pipeline's WHOLE BOX_2x2 level "
                 "(default anchor), which also carries the phase deviation ---\n"
              << "    identical      " << sealSame << " of " << sealPixels
              << " destination pixels\n"
              << "    max |delta|    " << sealMaxDelta << " of 255 (all of it phase)\n";
}

#endif // BINCV_WITH_OPENCV

}  // namespace

#define PYRAMID_SWEEP(NIn, NOut, WordType, name)                                      \
    BINCV_TEST(Pyramid, Reference_##NIn##_##NOut##_##name) {                          \
        testAgainstReference<NOut, NIn, WordType>(#name);                             \
    }                                                                                 \
    BINCV_TEST(Pyramid, DirtySource_##NIn##_##NOut##_##name) {                        \
        testDirtySource<NOut, NIn, WordType>(#name);                                  \
    }                                                                                 \
    BINCV_TEST(Pyramid, Strides_##NIn##_##NOut##_##name) {                            \
        testDifferingStrides<NOut, NIn, WordType>(#name);                             \
    }                                                                                 \
    BINCV_TEST(Pyramid, Degenerate_##NIn##_##NOut##_##name) {                         \
        testDegenerate<NOut, NIn, WordType>(#name);                                   \
    }                                                                                 \
    BINCV_TEST(Pyramid, EdgeReplication_##NIn##_##NOut##_##name) {                    \
        testEdgeReplication<NOut, NIn, WordType>(#name);                              \
    }                                                                                 \
    BINCV_TEST(Pyramid, Requantize_##NIn##_##NOut##_##name) {                         \
        testRequantizeEnumerated<NOut, NIn, WordType>(#name);                         \
    }                                                                                 \
    BINCV_TEST(Pyramid, Routes_##NIn##_##NOut##_##name) {                             \
        testRouteAgreement<NOut, NIn, WordType>(#name);                               \
    }

// NIn = 8 gets every sweep except the route comparison: impl::boxSum4Replicated
// refuses to compile there, because at NIn = 8 its input array is 1020 words per
// destination word. That refusal IS the finding, so the suite records it as a
// missing arm rather than working around it.
#define PYRAMID_SWEEP_NO_ROUTES(NIn, NOut, WordType, name)                            \
    BINCV_TEST(Pyramid, Reference_##NIn##_##NOut##_##name) {                          \
        testAgainstReference<NOut, NIn, WordType>(#name);                             \
    }                                                                                 \
    BINCV_TEST(Pyramid, DirtySource_##NIn##_##NOut##_##name) {                        \
        testDirtySource<NOut, NIn, WordType>(#name);                                  \
    }                                                                                 \
    BINCV_TEST(Pyramid, Strides_##NIn##_##NOut##_##name) {                            \
        testDifferingStrides<NOut, NIn, WordType>(#name);                              \
    }                                                                                 \
    BINCV_TEST(Pyramid, Degenerate_##NIn##_##NOut##_##name) {                         \
        testDegenerate<NOut, NIn, WordType>(#name);                                   \
    }                                                                                 \
    BINCV_TEST(Pyramid, EdgeReplication_##NIn##_##NOut##_##name) {                    \
        testEdgeReplication<NOut, NIn, WordType>(#name);                              \
    }                                                                                 \
    BINCV_TEST(Pyramid, Requantize_##NIn##_##NOut##_##name) {                         \
        testRequantizeEnumerated<NOut, NIn, WordType>(#name);                         \
    }

// The ladder ARCHITECTURE 7.2 measured, at every word width: 1 -> 3 -> 4 -> 5.
PYRAMID_SWEEP(1, 3, uint8_t, uint8_t)
PYRAMID_SWEEP(1, 3, uint16_t, uint16_t)
PYRAMID_SWEEP(1, 3, uint32_t, uint32_t)
PYRAMID_SWEEP(1, 3, uint64_t, uint64_t)
PYRAMID_SWEEP(3, 4, uint8_t, uint8_t)
PYRAMID_SWEEP(3, 4, uint32_t, uint32_t)
PYRAMID_SWEEP(3, 4, uint64_t, uint64_t)
PYRAMID_SWEEP(4, 5, uint16_t, uint16_t)
PYRAMID_SWEEP(4, 5, uint32_t, uint32_t)

// The extremes of the (NIn, NOut) space, where the arithmetic's widths are most
// likely to be off by one: no cap at all, the widest cap, and a cap BELOW the
// source depth -- which is a real E-7 candidate, not a degenerate case.
PYRAMID_SWEEP(1, 1, uint32_t, uint32_t)
PYRAMID_SWEEP(1, 8, uint32_t, uint32_t)
PYRAMID_SWEEP_NO_ROUTES(8, 8, uint32_t, uint32_t)
PYRAMID_SWEEP(5, 2, uint32_t, uint32_t)

#define PYRAMID_ARITHMETIC(WordType, name)                                            \
    BINCV_TEST(Pyramid, Arithmetic_##name) { testBitSlicedArithmetic<WordType>(#name); } \
    BINCV_TEST(Pyramid, BoxSum_1_##name) { testBoxSumAgainstReplicated<1, WordType>(#name); } \
    BINCV_TEST(Pyramid, BoxSum_2_##name) { testBoxSumAgainstReplicated<2, WordType>(#name); } \
    BINCV_TEST(Pyramid, BoxSum_3_##name) { testBoxSumAgainstReplicated<3, WordType>(#name); } \
    BINCV_TEST(Pyramid, BoxSum_4_##name) { testBoxSumAgainstReplicated<4, WordType>(#name); } \
    BINCV_TEST(Pyramid, BoxSum_5_##name) { testBoxSumAgainstReplicated<5, WordType>(#name); } \
    BINCV_TEST(Pyramid, Ladder_##name) { testPyramidLadder<WordType>(#name); }

PYRAMID_ARITHMETIC(uint8_t, uint8_t)
PYRAMID_ARITHMETIC(uint16_t, uint16_t)
PYRAMID_ARITHMETIC(uint32_t, uint32_t)
PYRAMID_ARITHMETIC(uint64_t, uint64_t)

BINCV_TEST(Pyramid, CostIsNotExponential) { testCostIsNotExponential(); }
BINCV_TEST(Pyramid, FootprintClaims) { testFootprintClaims(); }

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Pyramid, ReferencePipeline_uint32_t) {
    testAgainstReferencePipeline<uint32_t>("uint32_t");
}
BINCV_TEST(Pyramid, ReferencePipeline_uint64_t) {
    testAgainstReferencePipeline<uint64_t>("uint64_t");
}
#endif


// ---------------------------------------------------------------------------
// X-39 / E-21: the FILTERED pyrDown routes against a per-pixel integer reference.
//
// X-39's rule asks each filter to reproduce ITS OWN definition exactly -- not
// OpenCV's, since these deliberately compute different functions and the border
// rule is binCV's (zero outside, the same rule the shipped route applies to source
// words past the row). That is what this checks, at several (NIn, NOut) pairs,
// because the requantization interacts with both.
// ---------------------------------------------------------------------------
namespace {

using FilterWord = uint32_t;

template <size_t NOut, size_t NIn>
size_t checkFilter(bincv::PyrDownFilter f, const char* name, int lo, int hi, const unsigned* w, unsigned ksum) {
    const int sw = 64, sh = 48;                      // even, so no replicate branch
    bincv::QuantMat<NIn, FilterWord> src(sw, sh);
    uint64_t st = 11;
    auto rnd=[&st]{ st=st*6364136223846793005ULL+1442695040888963407ULL; return (unsigned)(st>>33); };
    const unsigned maxIn = (1u << NIn) - 1u;
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < sw; ++x) src.set(y, x, rnd() % (maxIn + 1u));
    bincv::QuantMat<NOut, FilterWord> dst(32, 24);
    switch (f) {
      case bincv::PyrDownFilter::DirectSubsample: bincv::pyrDownFiltered<bincv::PyrDownFilter::DirectSubsample,NOut,NIn,FilterWord>(src,dst); break;
      case bincv::PyrDownFilter::Box2x2:          bincv::pyrDownFiltered<bincv::PyrDownFilter::Box2x2,NOut,NIn,FilterWord>(src,dst); break;
      case bincv::PyrDownFilter::Box3x3:          bincv::pyrDownFiltered<bincv::PyrDownFilter::Box3x3,NOut,NIn,FilterWord>(src,dst); break;
      case bincv::PyrDownFilter::Gaussian3x3:     bincv::pyrDownFiltered<bincv::PyrDownFilter::Gaussian3x3,NOut,NIn,FilterWord>(src,dst); break;
      default:                             bincv::pyrDownFiltered<bincv::PyrDownFilter::Gaussian5x5,NOut,NIn,FilterWord>(src,dst); break;
    }
    const unsigned maxOut = (1u << NOut) - 1u;
    const unsigned long long K2 = (unsigned long long)ksum * ksum;
    int bad = 0;
    for (int y = 0; y < dst.rows(); ++y) {
        for (int x = 0; x < dst.cols(); ++x) {
            unsigned long long sum = 0;
            for (int dy = lo; dy <= hi; ++dy)
                for (int dx = lo; dx <= hi; ++dx) {
                    const int sy = 2*y+dy, sx = 2*x+dx;
                    if (sy < 0 || sy >= sh || sx < 0 || sx >= sw) continue;   // zero outside
                    sum += (unsigned long long)w[dy-lo] * w[dx-lo] * src.at(sy, sx);
                }
            const unsigned long long num = sum * maxOut + (K2 / 2) * maxIn;
            const unsigned want = (unsigned)(num / (K2 * maxIn));
            const unsigned got = dst.at(y, x);
            if (got != want) {
                if (bad < 3) std::printf("    MISMATCH %s N=%zu->%zu at (%d,%d): got %u want %u\n",
                                    name, NIn, NOut, x, y, got, want);
                ++bad;
            }
        }
    }
    std::printf("  %-18s NIn=%zu NOut=%zu : %s\n", name, NIn, NOut, bad ? "FAIL" : "exact");
    return static_cast<size_t>(bad);
}


} // namespace

BINCV_TEST(Pyramid, FilteredRoutesMatchAPerPixelReference_uint32_t) {
    const unsigned w1[5]={1,0,0,0,0}, w2[5]={1,1,0,0,0}, w3[5]={1,1,1,0,0},
                   g3[5]={1,2,1,0,0}, g5[5]={1,4,6,4,1};
    size_t bad = 0;
    bad += checkFilter<1,1>(bincv::PyrDownFilter::DirectSubsample,"DIRECT_SUBSAMPLE",0,0,w1,1);
    bad += checkFilter<3,1>(bincv::PyrDownFilter::Box2x2,"BOX_2x2",0,1,w2,2);
    bad += checkFilter<3,1>(bincv::PyrDownFilter::Box3x3,"BOX_3x3",-1,1,w3,3);
    bad += checkFilter<5,1>(bincv::PyrDownFilter::Gaussian3x3,"GAUSSIAN_3x3",-1,1,g3,4);
    bad += checkFilter<7,1>(bincv::PyrDownFilter::Gaussian5x5,"GAUSSIAN_5x5",-2,2,g5,16);
    bad += checkFilter<5,3>(bincv::PyrDownFilter::Gaussian5x5,"GAUSSIAN_5x5",-2,2,g5,16);
    bad += checkFilter<2,2>(bincv::PyrDownFilter::Box3x3,"BOX_3x3",-1,1,w3,3);
    BINCV_CHECK_EQ(bad, size_t{0});
}

BINCV_TEST_MAIN("test_pyramid")
