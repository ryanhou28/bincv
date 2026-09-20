// Device-versus-host bit-exactness for cuda::erode / dilate / morphologyEx.
//
// The format is shared, so every case here is a raw comparison: run the HOST
// kernel and the DEVICE kernel on the same input, download, and the maps must
// match word for word. The host library is the truth -- it is itself Tier 1
// against cv::erode / cv::dilate / cv::morphologyEx, so the device arm inherits
// that claim transitively over the inputs this sweep covers, and over nothing
// else.
//
// A CUDA translation unit rather than a .cpp, for one case that cannot be
// written any other way: `impl::borderIndex` is a HOST header's function
// carrying BINCV_HOST_DEVICE, and the whole point of the annotation is that the
// kernel calls the Tier 1 promise itself instead of a transcription of it.
// Proving that means calling it ON THE DEVICE and comparing, which needs a
// kernel in this file.
//
// FIVE THINGS ARE CHECKED DIRECTLY RATHER THAN IMPLIED:
//
//   1. PADDING BITS. After every device call the destination's trailing partial
//      word has zero bits past `width`, asserted on its own. A word comparison
//      against the host would also catch it, but then the failure names the
//      wrong thing.
//   2. A DIRTY-PADDING SOURCE. A source whose trailing padding bits are 1,
//      written as raw words rather than through packBits, must still give the
//      host's answer. This is the extendedWord tail-mask path and it is the
//      single most likely place a forked kernel is silently wrong: it looks
//      correct on every frame binCV produced. ops/shift.hpp records the
//      concrete failure at 5 pixels wide.
//   3. SINGLE-CELL EQUIVALENCE. An element with exactly one set cell reduces
//      erode/dilate to a pure shift, so the device kernel must agree with the
//      HOST's ops/shift.hpp `shift` at every offset and all five border types
//      -- the host's own Morphology.SingleOffsetEqualsShift, replayed across
//      the bus. Two separately written recurrences meaning the same thing is
//      what that buys.
//   4. THE ARM MATRIX. All EIGHT combinations of the three runtime switches
//      produce the identical map, in ONE binary.
//   5. THE ACCEPTED DOMAIN. An element outside it must be REFUSED with
//      cudaErrorInvalidValue, not silently truncated. The gate's Debug
//      configuration is where the matching BINCV_ASSERT is compiled at all.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/morphology.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/morphology.hpp"
#include "bincv/ops/shift.hpp"
#include "test_util.hpp"

namespace {

using bincv::BinMat;
using bincv::BorderType;
using bincv::MorphOp;
using bincv::StructuringElement;

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// A random bit frame whose padding starts clean.
/// @note Packed by an explicit loop rather than through
/// `bincv::packBits<PackRule::GreaterThan>`, and NOT for convenience:
/// instantiating that template in a CUDA translation unit makes nvcc's
/// front end emit "missing return statement" for `impl::toPackCmp` and
/// `impl::packCmp` -- it does not see their switch over a two-value enum
/// as exhaustive -- and verify_cuda.sh fails the gate on any line
/// matching "warning:". The functor form (`packBitsIf`) has no such
/// problem, which is why the existing CUDA suites do not hit this.
BinMat<uint32_t> randomBits(size_t w, size_t h, uint64_t seed) {
    BinMat<uint32_t> m(static_cast<int>(w), static_cast<int>(h));
    for (size_t y = 0; y < h; ++y) {
        uint32_t* row = m.view().row(y);
        for (size_t x = 0; x < w; ++x) {
            const bool bit = static_cast<uint8_t>(splitmix(seed) >> 24) > 127u;
            if (bit) row[x >> 5] |= (uint32_t{1} << (x & 31u));
        }
    }
    return m;
}

size_t wordsOf(size_t width) { return (width + 31u) / 32u; }

/// Words that differ between a host matrix and a downloaded device result.
size_t mismatchWords(const BinMat<uint32_t>& expect, const BinMat<uint32_t>& got) {
    size_t bad = 0;
    const size_t words = wordsOf(expect.getWidth());
    for (size_t y = 0; y < expect.getHeight(); ++y) {
        const uint32_t* a = expect.constView().row(y);
        const uint32_t* b = got.constView().row(y);
        for (size_t i = 0; i < words; ++i)
            if (a[i] != b[i]) ++bad;
    }
    return bad;
}

/// CLAUDE.md's hard rule, checked on its own so a failure names it: no bit past
/// `width` may be set in any destination row.
size_t dirtyPaddingWords(const BinMat<uint32_t>& m) {
    const size_t words = wordsOf(m.getWidth());
    const uint32_t tail = (m.getWidth() % 32u) == 0u
                              ? 0xFFFFFFFFu
                              : ((uint32_t{1} << (m.getWidth() % 32u)) - 1u);
    size_t bad = 0;
    for (size_t y = 0; y < m.getHeight(); ++y)
        if ((m.constView().row(y)[words - 1] & ~tail) != 0u) ++bad;
    return bad;
}

/// The element list the sweep runs. The three parametric shapes are all
/// symmetric about their centre, and a suite built only from them cannot catch
/// an inverted offset sign -- negating a symmetric offset set gives the same set
/// back. The off-centre anchor and the two masks are what break that symmetry.
struct NamedElement {
    const char* name;
    StructuringElement se;
};

// Backing bytes for the masked elements. They must outlive every element that
// names them: StructuringElement::mask is a view, not a container.
const uint8_t kHoleMask[15] = {1, 0, 1, 0, 1,
                               0, 0, 0, 0, 0,   // a FULLY EMPTY row -- see below
                               1, 1, 0, 0, 1};
const uint8_t kCornerMask[9] = {1, 0, 0, 0, 0, 0, 0, 0, 1};

std::vector<NamedElement> elementList() {
    return {
        {"rect3x3", bincv::rect3x3()},
        {"cross3x3", bincv::cross3x3()},
        {"ellipse3x3 (a PLUS, OpenCV's surprise)", StructuringElement::ellipse(3, 3)},
        {"rect5x5", StructuringElement::rect(5, 5)},
        {"ellipse5x5 (17 cells, the CPU arm's loss)", StructuringElement::ellipse(5, 5)},
        {"rect1x1 (filled whatever the shape says)", StructuringElement::rect(1, 1)},
        {"rect3x3 anchored (0,0) -- asymmetric", StructuringElement::rect(3, 3, 0, 0)},
        {"custom 5x3 mask with holes AND an empty row",
         StructuringElement::custom(kHoleMask, 5, 3)},
        {"custom 3x3 diagonal corners", StructuringElement::custom(kCornerMask, 3, 3)},
        {"rect65x1 -- reachX 32, past the word-border gate",
         StructuringElement::rect(65, 1)},
        {"rect1x7 -- vertical only", StructuringElement::rect(1, 7)},
    };
}

const BorderType kBorders[5] = {bincv::BORDER_CONSTANT, bincv::BORDER_REPLICATE,
                                bincv::BORDER_REFLECT, bincv::BORDER_WRAP,
                                bincv::BORDER_REFLECT_101};

/// One comparison: host kernel against device kernel, plus the padding check.
/// @return true when the two agree and the padding is clean.
bool compareErodeDilate(const BinMat<uint32_t>& src, const StructuringElement& se,
                        BorderType border, bool isErode, bool borderValue,
                        size_t alignBytes = 4) {
    const int w = static_cast<int>(src.getWidth());
    const int h = static_cast<int>(src.getHeight());

    BinMat<uint32_t> expect(w, h);
    if (isErode)
        bincv::erode(src.constView(), expect.view(), se, border, borderValue);
    else
        bincv::dilate(src.constView(), expect.view(), se, border, borderValue);

    bincv::cuda::DeviceBinMat dSrc(w, h, alignBytes);
    bincv::cuda::DeviceBinMat dDst(w, h, alignBytes);
    if (bincv::cuda::upload(src.constView(), dSrc.view()) != cudaSuccess) return false;
    const auto el = bincv::cuda::toDeviceElement(se);
    const cudaError_t rc =
        isErode ? bincv::cuda::erode(dSrc.constView(), dDst.view(), el, border, borderValue)
                : bincv::cuda::dilate(dSrc.constView(), dDst.view(), el, border, borderValue);
    if (rc != cudaSuccess) return false;
    BinMat<uint32_t> got(w, h);
    if (bincv::cuda::download(dDst.constView(), got.view()) != cudaSuccess) return false;
    if (cudaDeviceSynchronize() != cudaSuccess) return false;

    return mismatchWords(expect, got) == 0 && dirtyPaddingWords(got) == 0;
}

bool compareMorphEx(const BinMat<uint32_t>& src, MorphOp op, const StructuringElement& se,
                    BorderType border) {
    const int w = static_cast<int>(src.getWidth());
    const int h = static_cast<int>(src.getHeight());

    BinMat<uint32_t> expect(w, h);
    BinMat<uint32_t> hostScratch(w, h);
    bincv::morphologyEx(src.constView(), expect.view(), op, se, hostScratch.view(), border);

    bincv::cuda::DeviceBinMat dSrc(w, h);
    bincv::cuda::DeviceBinMat dDst(w, h);
    bincv::cuda::DeviceBinMat dScratch(w, h);
    if (bincv::cuda::upload(src.constView(), dSrc.view()) != cudaSuccess) return false;
    const auto el = bincv::cuda::toDeviceElement(se);
    if (bincv::cuda::morphologyEx(dSrc.constView(), dDst.view(), op, el, dScratch.view(),
                                  border) != cudaSuccess)
        return false;
    BinMat<uint32_t> got(w, h);
    if (bincv::cuda::download(dDst.constView(), got.view()) != cudaSuccess) return false;
    if (cudaDeviceSynchronize() != cudaSuccess) return false;

    return mismatchWords(expect, got) == 0 && dirtyPaddingWords(got) == 0;
}

/// Calls the HOST header's BINCV_HOST_DEVICE borderIndex from a kernel.
__global__ void borderIndexKernel(const long long* p, size_t n, size_t len, int type,
                                  long long* out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = static_cast<long long>(bincv::impl::borderIndex(
            static_cast<ptrdiff_t>(p[i]), len, static_cast<BorderType>(type)));
    }
}

bool cudaDevicePresent() {
    int n = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return false;
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// The border mapping is SHARED, not transcribed -- and this is where that stops
// being an argument about source text.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaMorphology, BorderIndexOnDeviceEqualsHost) {
    const size_t lens[] = {1, 2, 3, 7, 31, 32, 33, 752};
    for (size_t len : lens) {
        std::vector<long long> probes;
        const long long L = static_cast<long long>(len);
        for (long long p = -3 * L; p <= 3 * L; ++p) probes.push_back(p);
        const size_t n = probes.size();

        long long* dIn = nullptr;
        long long* dOut = nullptr;
        BINCV_CHECK_EQ(cudaMalloc(&dIn, n * sizeof(long long)), cudaSuccess);
        BINCV_CHECK_EQ(cudaMalloc(&dOut, n * sizeof(long long)), cudaSuccess);
        BINCV_CHECK_EQ(cudaMemcpy(dIn, probes.data(), n * sizeof(long long),
                                  cudaMemcpyHostToDevice),
                       cudaSuccess);
        for (BorderType t : kBorders) {
            borderIndexKernel<<<16, 128>>>(dIn, n, len, static_cast<int>(t), dOut);
            std::vector<long long> got(n, -12345);
            BINCV_CHECK_EQ(cudaMemcpy(got.data(), dOut, n * sizeof(long long),
                                      cudaMemcpyDeviceToHost),
                           cudaSuccess);
            size_t bad = 0;
            for (size_t i = 0; i < n; ++i) {
                const long long want = static_cast<long long>(bincv::impl::borderIndex(
                    static_cast<ptrdiff_t>(probes[i]), len, t));
                if (got[i] != want) ++bad;
            }
            BINCV_CHECK_EQ(bad, 0u);
        }
        cudaFree(dIn);
        cudaFree(dOut);
    }
}

// ---------------------------------------------------------------------------
// THE SHAPE SWEEP. Widths straddle the word boundary from BOTH sides, because
// rowTailMask and the prev/next carries are where a forked word kernel
// diverges. 1023 and 1920 are in there specifically to cross several block
// seams with many words per row.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaMorphology, ErodeDilateMatchHostAcrossWidths) {
    const size_t widths[] = {1, 5, 31, 32, 33, 63, 64, 65, 97, 640, 752, 1023, 1920};
    const auto elements = elementList();
    uint64_t seed = 0x9E3779B9u;
    for (size_t w : widths) {
        const BinMat<uint32_t> src = randomBits(w, 13, seed++);
        for (const auto& ne : elements) {
            if (!bincv::cuda::deviceElementDomainOk(ne.se)) continue;
            for (BorderType border : kBorders) {
                BINCV_CHECK(compareErodeDilate(src, ne.se, border, true, true));
                BINCV_CHECK(compareErodeDilate(src, ne.se, border, false, false));
            }
        }
    }
}

// The NON-DEFAULT borderValue is swept on purpose: one fixed fill makes one of
// the two operations wrong at every edge, and the host records 28,862 of
// 298,541 checks failing when erode's fill is flipped. It is also the ONLY
// place a loose mask span is observable -- an element row with no set cell must
// contribute nothing, and a span of [0, cols) would fold the constant in.
BINCV_TEST(CudaMorphology, NonDefaultBorderValueAndEmptyMaskRows) {
    const size_t widths[] = {5, 33, 65, 200};
    const auto elements = elementList();
    uint64_t seed = 0x51ED2701u;
    for (size_t w : widths) {
        const BinMat<uint32_t> src = randomBits(w, 9, seed++);
        for (const auto& ne : elements) {
            if (!bincv::cuda::deviceElementDomainOk(ne.se)) continue;
            BINCV_CHECK(compareErodeDilate(src, ne.se, bincv::BORDER_CONSTANT, true, false));
            BINCV_CHECK(compareErodeDilate(src, ne.se, bincv::BORDER_CONSTANT, false, true));
        }
    }
}

// Nothing may silently assume stride == rowWords.
BINCV_TEST(CudaMorphology, OverAlignedStrideMatchesHost) {
    const size_t widths[] = {33, 65, 752};
    for (size_t w : widths) {
        const BinMat<uint32_t> src = randomBits(w, 11, w * 7 + 1);
        for (BorderType border : kBorders) {
            BINCV_CHECK(compareErodeDilate(src, bincv::rect3x3(), border, true, true, 128));
            BINCV_CHECK(compareErodeDilate(src, StructuringElement::ellipse(5, 5), border,
                                           false, false, 128));
        }
    }
}

// ---------------------------------------------------------------------------
// A DIRTY-PADDING SOURCE. Written as raw words, not through packBits, so the
// bits past `width` are 1. Every answer must be unchanged. A kernel that omits
// extendedWord's second half passes every other test in this file.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaMorphology, DirtyPaddingSourceChangesNothing) {
    const size_t widths[] = {1, 5, 31, 33, 65, 97, 751};
    const auto elements = elementList();
    for (size_t w : widths) {
        const size_t words = wordsOf(w);
        const uint32_t tail =
            (w % 32u) == 0u ? 0xFFFFFFFFu : ((uint32_t{1} << (w % 32u)) - 1u);

        BinMat<uint32_t> clean = randomBits(w, 7, w * 31 + 5);
        BinMat<uint32_t> dirty(static_cast<int>(w), 7);
        for (size_t y = 0; y < 7; ++y) {
            const uint32_t* a = clean.constView().row(y);
            uint32_t* b = dirty.view().row(y);
            for (size_t i = 0; i < words; ++i) b[i] = a[i];
            b[words - 1] |= ~tail;  // the padding bits, set
        }

        for (const auto& ne : elements) {
            if (!bincv::cuda::deviceElementDomainOk(ne.se)) continue;
            for (BorderType border : kBorders) {
                // The host answer is taken from the DIRTY source too: the claim
                // is that neither implementation reads padding as a pixel, and
                // comparing against the clean host answer as well would hide a
                // case where both read it.
                BINCV_CHECK(compareErodeDilate(dirty, ne.se, border, true, true));
                BINCV_CHECK(compareErodeDilate(dirty, ne.se, border, false, false));

                BinMat<uint32_t> fromClean(static_cast<int>(w), 7);
                BinMat<uint32_t> fromDirty(static_cast<int>(w), 7);
                bincv::erode(clean.constView(), fromClean.view(), ne.se, border, true);
                bincv::erode(dirty.constView(), fromDirty.view(), ne.se, border, true);
                BINCV_CHECK_EQ(mismatchWords(fromClean, fromDirty), 0u);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SINGLE-CELL EQUIVALENCE: an element with exactly one set cell is a pure
// shift, so the device kernel must agree with ops/shift.hpp's separately
// written recurrence -- at every offset in the element and all five borders.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaMorphology, SingleCellEqualsHostShift) {
    const size_t widths[] = {33, 65, 200};
    for (size_t w : widths) {
        const BinMat<uint32_t> src = randomBits(w, 9, w * 13 + 3);
        for (int ey = 0; ey < 5; ++ey) {
            for (int ex = 0; ex < 5; ++ex) {
                uint8_t cells[25] = {};
                cells[ey * 5 + ex] = 1;
                const StructuringElement se = StructuringElement::custom(cells, 5, 5);
                const ptrdiff_t dx = ex - 2;
                const ptrdiff_t dy = ey - 2;
                for (BorderType border : kBorders) {
                    // Erode with fill = true is shift with fill = true; dilate
                    // with fill = false is shift with fill = false.
                    BinMat<uint32_t> shifted(static_cast<int>(w), 9);
                    bincv::shift(src.constView(), shifted.view(), dx, dy, border, true);

                    bincv::cuda::DeviceBinMat dSrc(static_cast<int>(w), 9);
                    bincv::cuda::DeviceBinMat dDst(static_cast<int>(w), 9);
                    BINCV_CHECK_EQ(bincv::cuda::upload(src.constView(), dSrc.view()),
                                   cudaSuccess);
                    const auto el = bincv::cuda::toDeviceElement(se);
                    BINCV_CHECK_EQ(bincv::cuda::erode(dSrc.constView(), dDst.view(), el,
                                                      border, true),
                                   cudaSuccess);
                    BinMat<uint32_t> got(static_cast<int>(w), 9);
                    BINCV_CHECK_EQ(bincv::cuda::download(dDst.constView(), got.view()),
                                   cudaSuccess);
                    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
                    BINCV_CHECK_EQ(mismatchWords(shifted, got), 0u);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// All seven MorphOp values, every element, two borders.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaMorphology, MorphologyExMatchesHost) {
    const MorphOp ops[7] = {bincv::MORPH_ERODE,    bincv::MORPH_DILATE,
                            bincv::MORPH_OPEN,     bincv::MORPH_CLOSE,
                            bincv::MORPH_GRADIENT, bincv::MORPH_TOPHAT,
                            bincv::MORPH_BLACKHAT};
    const auto elements = elementList();
    const BinMat<uint32_t> src = randomBits(257, 67, 0xABCDEF01u);
    for (MorphOp op : ops) {
        for (const auto& ne : elements) {
            if (!bincv::cuda::deviceElementDomainOk(ne.se)) continue;
            BINCV_CHECK(compareMorphEx(src, op, ne.se, bincv::BORDER_CONSTANT));
            BINCV_CHECK(compareMorphEx(src, op, ne.se, bincv::BORDER_REPLICATE));
        }
    }
    // One reference-sized frame, so the multi-block path is not only exercised
    // at test-sized images.
    const BinMat<uint32_t> big = randomBits(752, 480, 0x1234u);
    for (MorphOp op : ops) {
        BINCV_CHECK(compareMorphEx(big, op, bincv::rect3x3(), bincv::BORDER_CONSTANT));
        BINCV_CHECK(compareMorphEx(big, op, StructuringElement::ellipse(5, 5),
                                   bincv::BORDER_REFLECT_101));
    }
}

// ---------------------------------------------------------------------------
// THE ARM MATRIX. Eight combinations of three switches, one binary, one answer.
// A fast arm that is faster and different is not an optimization.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaMorphology, AllEightArmCombinationsAgree) {
    const size_t widths[] = {65, 200, 1023};
    const StructuringElement elems[4] = {bincv::rect3x3(), bincv::cross3x3(),
                                         StructuringElement::ellipse(5, 5),
                                         StructuringElement::rect(65, 1)};
    const MorphOp ops[4] = {bincv::MORPH_ERODE, bincv::MORPH_OPEN, bincv::MORPH_GRADIENT,
                            bincv::MORPH_BLACKHAT};

    const bool saveFast = bincv::cuda::impl::morphFastArmEnabled();
    const bool saveBorder = bincv::cuda::impl::morphWordBorderEnabled();
    const bool saveFused = bincv::cuda::impl::morphAndNotFusedEnabled();

    for (size_t w : widths) {
        const BinMat<uint32_t> src = randomBits(w, 37, w + 99);
        for (const auto& se : elems) {
            for (BorderType border : kBorders) {
                for (MorphOp op : ops) {
                    for (int combo = 0; combo < 8; ++combo) {
                        bincv::cuda::impl::morphFastArmEnabled() = (combo & 1) != 0;
                        bincv::cuda::impl::morphWordBorderEnabled() = (combo & 2) != 0;
                        bincv::cuda::impl::morphAndNotFusedEnabled() = (combo & 4) != 0;
                        BINCV_CHECK(compareMorphEx(src, op, se, border));
                    }
                }
            }
        }
    }

    bincv::cuda::impl::morphFastArmEnabled() = saveFast;
    bincv::cuda::impl::morphWordBorderEnabled() = saveBorder;
    bincv::cuda::impl::morphAndNotFusedEnabled() = saveFused;
}

// The word-border arm's own gate excludes width < 64; the per-pixel band path
// is what runs there, and it must be right at every one of those widths.
BINCV_TEST(CudaMorphology, NarrowFramesTakeTheBandedPathAndMatch) {
    const bool saveBorder = bincv::cuda::impl::morphWordBorderEnabled();
    for (size_t w = 1; w <= 70; ++w) {
        const BinMat<uint32_t> src = randomBits(w, 5, w * 17 + 2);
        for (BorderType border : kBorders) {
            for (int on = 0; on < 2; ++on) {
                bincv::cuda::impl::morphWordBorderEnabled() = (on != 0);
                BINCV_CHECK(compareErodeDilate(src, bincv::rect3x3(), border, true, true));
                BINCV_CHECK(compareErodeDilate(src, StructuringElement::rect(3, 3, 0, 0),
                                               border, false, false));
            }
        }
    }
    bincv::cuda::impl::morphWordBorderEnabled() = saveBorder;
}

// ---------------------------------------------------------------------------
// THE ACCEPTED DOMAIN. Narrower than the host's, named in the header, and
// REFUSED rather than truncated. An op that quietly clipped a 33-row element to
// 32 would return a fully-formed wrong answer.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaMorphology, ElementsOutsideTheDeviceDomainAreRefused) {
    const StructuringElement tooTall = StructuringElement::rect(3, 33);
    const StructuringElement tooWide = StructuringElement::rect(513, 1);
    std::vector<uint8_t> bigMask(33 * 3, 1);
    const StructuringElement maskTooWide = StructuringElement::custom(bigMask.data(), 33, 3);

    BINCV_CHECK(!bincv::cuda::deviceElementDomainOk(tooTall));
    BINCV_CHECK(!bincv::cuda::deviceElementDomainOk(tooWide));
    BINCV_CHECK(!bincv::cuda::deviceElementDomainOk(maskTooWide));
    // The host accepts all three; that is the whole point of naming the
    // narrowing rather than pretending the two domains are the same.
    BINCV_CHECK(tooTall.valid() && tooWide.valid() && maskTooWide.valid());

    // Inside the domain right up to its edge.
    BINCV_CHECK(bincv::cuda::deviceElementDomainOk(StructuringElement::rect(3, 32)));
    BINCV_CHECK(bincv::cuda::deviceElementDomainOk(StructuringElement::rect(512, 1)));

    bincv::cuda::DeviceBinMat dSrc(200, 20);
    bincv::cuda::DeviceBinMat dDst(200, 20);
    for (const auto& se : {tooTall, tooWide, maskTooWide}) {
        const auto el = bincv::cuda::toDeviceElement(se);
        // toDeviceElement REPORTS the refusal rather than asserting it, which is
        // what lets a caller ask without risking the process. It is checked in
        // both configurations.
        BINCV_CHECK_EQ(el.valid, 0);

#if !BINCV_DEBUG_CHECKS
        // THE ERROR RETURN is checked only where the assertion is compiled OUT.
        // Owner ruling R4 requires a narrowed device domain to be BOTH asserted
        // and returned as an error, and BINCV_ASSERT aborts by design -- so
        // calling the launcher with a refused element under the gate's Debug
        // configuration would kill the suite rather than test it. The Debug
        // configuration's job here is that the assertion COMPILES for nvcc's
        // device pass; this configuration's job is that the contract a release
        // caller actually sees is the documented one.
        BINCV_CHECK_EQ(bincv::cuda::erode(dSrc.constView(), dDst.view(), el),
                       cudaErrorInvalidValue);
        BINCV_CHECK_EQ(bincv::cuda::dilate(dSrc.constView(), dDst.view(), el),
                       cudaErrorInvalidValue);
        // A refusal must leave no launch error behind for the next call to
        // inherit: it launches nothing, so nothing can have failed.
        BINCV_CHECK_EQ(cudaGetLastError(), cudaSuccess);
#endif
    }

    // Tall-but-legal, at the very edge of the domain, still has to be RIGHT.
    const BinMat<uint32_t> src = randomBits(200, 40, 0x777u);
    BINCV_CHECK(compareErodeDilate(src, StructuringElement::rect(3, 32),
                                   bincv::BORDER_REFLECT_101, true, true));
    BINCV_CHECK(compareErodeDilate(src, StructuringElement::rect(512, 1),
                                   bincv::BORDER_REPLICATE, false, false));
}

// ---------------------------------------------------------------------------
// The fused subtraction against the two-launch spelling the host composes, on
// its own rather than only through morphologyEx.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaMorphology, AndNotMatchesTheTwoLaunchSpelling) {
    const size_t widths[] = {1, 31, 33, 97, 1023};
    for (size_t w : widths) {
        const BinMat<uint32_t> a = randomBits(w, 17, w * 3 + 1);
        const BinMat<uint32_t> b = randomBits(w, 17, w * 5 + 2);

        BinMat<uint32_t> expect(static_cast<int>(w), 17);
        BinMat<uint32_t> tmp(static_cast<int>(w), 17);
        bincv::bitwiseNot(b.constView(), tmp.view());
        bincv::bitwiseAnd(a.constView(), tmp.constView(), expect.view());

        bincv::cuda::DeviceBinMat da(static_cast<int>(w), 17);
        bincv::cuda::DeviceBinMat db(static_cast<int>(w), 17);
        bincv::cuda::DeviceBinMat dd(static_cast<int>(w), 17);
        BINCV_CHECK_EQ(bincv::cuda::upload(a.constView(), da.view()), cudaSuccess);
        BINCV_CHECK_EQ(bincv::cuda::upload(b.constView(), db.view()), cudaSuccess);
        BINCV_CHECK_EQ(
            bincv::cuda::impl::andNot(da.constView(), db.constView(), dd.view()),
            cudaSuccess);
        BinMat<uint32_t> got(static_cast<int>(w), 17);
        BINCV_CHECK_EQ(bincv::cuda::download(dd.constView(), got.view()), cudaSuccess);
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        BINCV_CHECK_EQ(mismatchWords(expect, got), 0u);
        BINCV_CHECK_EQ(dirtyPaddingWords(got), 0u);
    }
}

#if BINCV_TEST_WITH_GTEST
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    const int summaryRc = ::bincv::test::summarize("CUDA morphology tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA morphology tests", argc, argv);
}
#endif
