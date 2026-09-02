// The reference pipeline's three-pixel median filter: denoiseMedian3.
//
// THE CORRECTNESS BAR IS NOT "MATCHES A FORMULA I WROTE DOWN". It is "matches
// the reference implementation pixel for pixel on binary input", and the
// reference is the reference frontend's denoiser. Two consequences shape
// this file:
//
// * The OpenCV half PORTS THE REFERENCE'S ACTUAL cv:: CALLS -- cv::Mat::zeros
// for both neighbour matrices, the two copyTo range assignments, then
// cv::min / cv::max in the reference's order. It does not reimplement the
// filter from the comment above them. That matters for exactly one reason:
// THE BORDER. `right_pixels` is written only on colRange(0, cols - 1) and
// `above_pixels` only on rowRange(1, rows), so the last column's right
// neighbour and the first row's above neighbour are the zeros the matrices
// were constructed with. Porting the construction rather than the reading of
// it means that border cannot be got wrong here in a way that agrees with a
// kernel that got it wrong the same way.
//
// * The CORE half has its own per-pixel reference with the neighbourhood
// written out as coordinates, so the three configurations without OpenCV --
// including Debug, the only one where denoiseMedian3's BINCV_ASSERTs are
// live, and -fno-exceptions, which is the embedded claim -- still check every
// pixel. The two references are independent: one is a sorting network over
// cv::Mat bytes, the other is `median of three bools` by counting.
//
// A THIRD CHECK NEITHER OF THOSE IS: Denoise.Composed_* runs
// shiftDown / shiftLeft / majority3 -- the spelling ops/bitslice.hpp and
// ops/shift.hpp were built to support -- and requires it to agree with the fused
// kernel on every pixel of every swept case. That is what keeps the fused
// version honest about being a fusion rather than a different operation, and it
// is the case that would fail first if the kernel's inline right-shift ever
// disagreed with ops/shift.hpp's about a word boundary.
//
// WHY THE CHECK COUNT IS NOT ONE PER PIXEL: a 640x480 case would contribute
// 307200 checks and drown the summary. Each swept case reports its DISAGREEMENT
// COUNT as a single check, so CHECKS tracks cases and a failure still says how
// badly it failed.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/bitslice.hpp"
#include "bincv-cpp/ops/denoise.hpp"
#include "bincv-cpp/ops/shift.hpp"
#include "test_util.hpp"

// OpenCV-only, and self-guarding: everything in it sits behind BINCV_WITH_OPENCV,
// so this include is a no-op in the three configurations that have no OpenCV.
#include "equivalence.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#endif

namespace {

using bincv::denoiseMedian3;

/// @def DENOISE_EXPECT
/// @brief One check, with a detail string built only when it fails.
#define DENOISE_EXPECT(ok, what, detailExpr) \
    ::bincv::test::reportCheck((ok), (what), __FILE__, __LINE__, \
                               (ok) ? std::string() : (detailExpr))

// ---------------------------------------------------------------------------
// Content: the same generator as tests/equivalence.hpp, minus OpenCV
// ---------------------------------------------------------------------------
//
// Duplicated rather than shared, for that work’s reason: a harness that shared a
// generator with the suite judging it could cancel a fault through both sides.

uint64_t nextRandom(uint64_t& state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

constexpr uint32_t FILL_SCALE = 1u << 24;

uint32_t fillThreshold(float fillRatio) {
    if (!(fillRatio > 0.0f)) return 0;
    if (fillRatio >= 1.0f) return FILL_SCALE;
    const double rounded = static_cast<double>(fillRatio) * static_cast<double>(FILL_SCALE) + 0.5;
    if (rounded >= static_cast<double>(FILL_SCALE)) return FILL_SCALE;
    return static_cast<uint32_t>(rounded);
}

/// @brief Fills through set, so the padding bits stay clear on entry.
template <typename WordType>
void fillRandom(bincv::BinMat<WordType>& m, float fillRatio, uint64_t seed) {
    uint64_t state = seed;
    const uint32_t threshold = fillThreshold(fillRatio);
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            if (static_cast<uint32_t>(nextRandom(state) >> 40) < threshold) m.set(y, x, true);
        }
    }
}

uint64_t caseSeed(int width, int height, size_t index) {
    return UINT64_C(0xDE0125E30000001) + static_cast<uint64_t>(width) * UINT64_C(1000003) +
           static_cast<uint64_t>(height) * UINT64_C(10007) + static_cast<uint64_t>(index);
}

std::string sizeLabel(const char* wordTypeName, int width, int height, const char* extra) {
    return std::string(wordTypeName) + " " + std::to_string(width) + "x" +
           std::to_string(height) + " " + extra;
}

/// @brief Set bits across the whole STRIDE, padding included.
/// @note Compared against countNonZero's per-pixel loop this is how a
/// padding-bit violation becomes visible; binCV exposes no per-word
/// popcount, so the test writes its own.
template <typename WordType>
int bitsAcrossStride(const bincv::BinMat<WordType>& m) {
    int bits = 0;
    for (int y = 0; y < m.rows(); ++y) {
        const WordType* row = m.ptr(y);
        for (size_t w = 0; w < m.getAlignedWidth(); ++w) {
            WordType v = row[w];
            while (v != 0) {
                bits += static_cast<int>(v & static_cast<WordType>(1));
                v = static_cast<WordType>(v >> 1);
            }
        }
    }
    return bits;
}

// ---------------------------------------------------------------------------
// The core reference: the neighbourhood written out as coordinates
// ---------------------------------------------------------------------------

/// @brief The median of three bools, by counting rather than by maj3.
/// @note Deliberately NOT (a&b)|(b&c)|(a&c) and not a sorting network either: a
/// reference that shares an expression with the code under test cannot
/// fail with it. "At least two of three" is the definition of the median
/// of three values drawn from a two-element set, and this counts them.
bool refMedian3(bool p1, bool p2, bool p3) {
    const int set = (p1 ? 1 : 0) + (p2 ? 1 : 0) + (p3 ? 1 : 0);
    return set >= 2;
}

/// @brief The reference filter's three-pixel L, per pixel, with the zero border.
/// @note THE NEIGHBOURHOOD AND THE BORDER, both from
/// the reference frontend's denoiser:
/// p1 = src[y - 1][x] -- `above_pixels`, whose rowRange(1, rows) is the
/// only part ever written, so row 0 reads the
/// cv::Mat::zeros it was built with.
/// p2 = src[y][x] -- the pixel itself.
/// p3 = src[y][x + 1] -- `right_pixels`, whose colRange(0, cols - 1) is
/// the only part ever written, so column
/// width - 1 reads zero.
/// Zero fill in both cases; not replicate, not reflect.
template <typename WordType>
bool refPixel(const bincv::BinMat<WordType>& src, int y, int x) {
    const bool p1 = (y > 0) && src.at(y - 1, x);
    const bool p2 = src.at(y, x);
    const bool p3 = (x + 1 < src.cols()) && src.at(y, x + 1);
    return refMedian3(p1, p2, p3);
}

/// @brief Pixels on which the kernel's result differs from that reference.
template <typename WordType>
int disagreements(const bincv::BinMat<WordType>& src, const bincv::BinMat<WordType>& dst) {
    int differing = 0;
    for (int y = 0; y < dst.rows(); ++y) {
        for (int x = 0; x < dst.cols(); ++x) {
            if (dst.at(y, x) != refPixel(src, y, x)) ++differing;
        }
    }
    return differing;
}

// The widths, plus 128 -- an exact multiple of every supported word width.
// 1 and 7 are the widths where the right-neighbour never crosses a word boundary
// at any word type, which is the geometry the kernel's trailing-word path owns.
const int WIDTHS[] = {1, 7, 31, 33, 40, 63, 65, 70, 128, 640};
const int HEIGHTS[] = {1, 2, 3, 17, 37};
const float FILLS[] = {0.0f, 0.01f, 0.5f, 0.99f, 1.0f};

// An over-aligned row stride (the design rule makes alignment a per-object choice).
constexpr size_t PADDED_ALIGNMENT = 32;

// ===========================================================================
// 1. The per-pixel reference, over the size and fill matrix
// ===========================================================================

template <typename WordType>
void testReference(const char* wordTypeName) {
    std::cout << "\n--- denoiseMedian3 vs a per-pixel reference: " << wordTypeName << " ---\n";

    for (int width : WIDTHS) {
        for (int height : HEIGHTS) {
            for (size_t f = 0; f < sizeof(FILLS) / sizeof(FILLS[0]); ++f) {
                const uint64_t seed = caseSeed(width, height, f);
                bincv::BinMat<WordType> src(width, height);
                fillRandom(src, FILLS[f], seed);

                bincv::BinMat<WordType> dst(width, height);
                denoiseMedian3(src.constView(), dst.view());

                const std::string label = sizeLabel(wordTypeName, width, height, "reference");
                DENOISE_EXPECT(disagreements(src, dst) == 0,
                               "denoiseMedian3 matches the per-pixel reference", label);

                // The invariant a pixel comparison is blind to.
                DENOISE_EXPECT(bitsAcrossStride(dst) == dst.countNonZero(),
                               "denoiseMedian3 leaves the padding bits zero", label);
            }
        }
    }
}

// ===========================================================================
// 2. THE BORDER, on its own, at the two edges the reference zero-fills
// ===========================================================================
//
// Section 1 would catch a wrong border -- it checks every pixel -- but it would
// report it as "1920 pixels differ" in a case labeled by size, and the two edges
// fail for different reasons. These cases isolate them, on content constructed so
// that a REPLICATE or REFLECT border gives a different answer from a zero one.
//
// The construction: an all-ones image. Then
// * every interior pixel is 1 (all three neighbours are 1),
// * row 0 becomes p2 & p3 -- 1 everywhere except the last column,
// * the last column becomes p1 & p2 -- 1 everywhere except row 0,
// * pixel (0, width - 1) has TWO zero neighbours and becomes 0.
// Under BORDER_REPLICATE every one of those would stay 1, so the expected image
// below is only correct for the zero fill the reference uses.

template <typename WordType>
void testBorder(const char* wordTypeName) {
    std::cout << "\n--- denoiseMedian3's zero border: " << wordTypeName << " ---\n";

    for (int width : WIDTHS) {
        for (int height : HEIGHTS) {
            bincv::BinMat<WordType> src(width, height);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) src.set(y, x, true);
            }

            bincv::BinMat<WordType> dst(width, height);
            denoiseMedian3(src.constView(), dst.view());

            int wrong = 0;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    // On an all-ones image a pixel survives iff at least two of
                    // {above exists, self, right exists} are 1 -- self always is,
                    // so it survives iff it has an above OR a right neighbour
                    // inside the image.
                    const bool expected = (y > 0) || (x + 1 < width);
                    if (dst.at(y, x) != expected) ++wrong;
                }
            }

            const std::string label = sizeLabel(wordTypeName, width, height, "all-ones border");
            DENOISE_EXPECT(wrong == 0,
                           "the top row and last column erode exactly as a zero border requires",
                           label + ", " + std::to_string(wrong) + " pixels wrong");

            // And the corner specifically, called out because it is the one pixel
            // with two out-of-image neighbours -- the case a border implementation
            // that handles the two edges separately gets wrong last.
            DENOISE_EXPECT(dst.at(0, width - 1) == false,
                           "the top-right pixel has two zero neighbours and clears", label);
        }
    }
}

// ===========================================================================
// 3. THE FUSION: shiftDown + shiftLeft + majority3 must give the same image
// ===========================================================================
//
// ops/denoise.hpp claims its one-pass kernel is exactly
//
// shiftDown(src, above, 1); shiftLeft(src, right, 1);
// majority3(above, src, right, dst);
//
// with the two frame-sized scratch buffers removed. This runs that composition
// -- through the shipped ops/shift.hpp and ops/bitslice.hpp, at their DEFAULT
// border, which is BORDER_CONSTANT with value false -- and requires the
// two images to agree on every pixel. A disagreement means either the kernel's
// inline right-shift or ops/shift.hpp's has a word-boundary bug, and the check
// is cheap because both sides already exist.

template <typename WordType>
void testComposed(const char* wordTypeName) {
    std::cout << "\n--- denoiseMedian3 == shiftDown + shiftLeft + majority3: " << wordTypeName
              << " ---\n";

    for (int width : WIDTHS) {
        for (int height : HEIGHTS) {
            for (size_t f = 0; f < sizeof(FILLS) / sizeof(FILLS[0]); ++f) {
                const uint64_t seed = caseSeed(width, height, 200 + f);
                bincv::BinMat<WordType> src(width, height);
                fillRandom(src, FILLS[f], seed);

                bincv::BinMat<WordType> fused(width, height);
                denoiseMedian3(src.constView(), fused.view());

                bincv::BinMat<WordType> above(width, height);
                bincv::BinMat<WordType> right(width, height);
                bincv::BinMat<WordType> composed(width, height);
                bincv::shiftDown(src.constView(), above.view(), 1);
                bincv::shiftLeft(src.constView(), right.view(), 1);
                bincv::majority3(above.constView(), src.constView(), right.constView(),
                                 composed.view());

                int differing = 0;
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        if (fused.at(y, x) != composed.at(y, x)) ++differing;
                    }
                }
                const std::string label = sizeLabel(wordTypeName, width, height, "composed");
                DENOISE_EXPECT(differing == 0,
                               "the fused kernel equals two shifts and a majority3",
                               label + ", " + std::to_string(differing) + " pixels differ");
            }
        }
    }
}

// ===========================================================================
// 4. Differing strides -- the kernel reads each view's own stride per row
// ===========================================================================

enum class Stride { Tight, Padded, Odd };

const char* strideName(Stride s) {
    switch (s) {
        case Stride::Tight:  return "tight";
        case Stride::Padded: return "padded";
        case Stride::Odd:    return "odd";
    }
    return "?";
}

template <typename WordType>
struct StridedMat {
    std::vector<WordType> buffer;   // used by Stride::Odd only
    bincv::BinMat<WordType> mat;
};

template <typename WordType>
void makeStrided(StridedMat<WordType>& out, Stride flavour, int width, int height) {
    constexpr size_t bits = bincv::BinMat<WordType>::WordBits;
    const size_t minWords = (static_cast<size_t>(width) + bits - 1) / bits;

    switch (flavour) {
        case Stride::Tight:
            out.mat = bincv::BinMat<WordType>(width, height);
            return;
        case Stride::Padded:
            out.mat = bincv::BinMat<WordType>(width, height, PADDED_ALIGNMENT);
            return;
        case Stride::Odd: {
            const size_t stride = minWords + 3;
            out.buffer.assign(stride * static_cast<size_t>(height), static_cast<WordType>(0));
            out.mat = bincv::BinMat<WordType>(out.buffer.data(), width, height, stride);
            return;
        }
    }
}

template <typename WordType>
void testDifferingStrides(const char* wordTypeName) {
    std::cout << "\n--- denoiseMedian3 across differing strides: " << wordTypeName << " ---\n";

    // All nine combinations: two views, three flavours each. Cheap, and it means
    // no "exactly one argument differs" shape is missing -- the omission that
    // left a stride bug green in tests/test_bitslice.cpp until it was found by
    // review.
    const Stride flavours[] = {Stride::Tight, Stride::Padded, Stride::Odd};

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            for (Stride srcFlavour : flavours) {
                for (Stride dstFlavour : flavours) {
                    StridedMat<WordType> src, dst;
                    makeStrided(src, srcFlavour, width, height);
                    makeStrided(dst, dstFlavour, width, height);
                    fillRandom(src.mat, 0.5f, caseSeed(width, height, 400));

                    denoiseMedian3(src.mat.constView(), dst.mat.view());

                    const std::string label =
                        sizeLabel(wordTypeName, width, height, strideName(srcFlavour)) + "/" +
                        strideName(dstFlavour);
                    DENOISE_EXPECT(disagreements(src.mat, dst.mat) == 0,
                                   "denoiseMedian3 matches the reference across strides", label);
                    DENOISE_EXPECT(bitsAcrossStride(dst.mat) == dst.mat.countNonZero(),
                                   "denoiseMedian3 leaves the padding bits zero", label);
                }
            }
        }
    }
}

// ===========================================================================
// 5. A source whose padding bits are already dirty
// ===========================================================================
//
// A wrapped buffer's padding belongs to its caller (BinMat's wrap constructor),
// so a source may legally arrive with every bit past `width` set. THIS IS THE
// CASE THAT PINS THE LAST COLUMN'S BORDER. The kernel forms pixel width - 1's
// right neighbour by shifting the row's trailing word down one bit, and bit
// `width % WordBits` of that word is PADDING -- so without the mask the kernel
// applies before the shift, the last column's neighbour would read the caller's
// junk instead of the zero the reference guarantees, on exactly one column of
// exactly the widths that do not end on a word boundary.

template <typename WordType>
void makeDirtyPadded(StridedMat<WordType>& out, int width, int height, float fillRatio,
                     uint64_t seed) {
    constexpr size_t bits = bincv::BinMat<WordType>::WordBits;
    const size_t minWords = (static_cast<size_t>(width) + bits - 1) / bits;
    const size_t stride = minWords + 1;
    out.buffer.assign(stride * static_cast<size_t>(height),
                      static_cast<WordType>(~static_cast<WordType>(0)));
    out.mat = bincv::BinMat<WordType>(out.buffer.data(), width, height, stride);

    uint64_t state = seed;
    const uint32_t threshold = fillThreshold(fillRatio);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            out.mat.set(y, x, static_cast<uint32_t>(nextRandom(state) >> 40) < threshold);
        }
    }
}

template <typename WordType>
void testDirtySource(const char* wordTypeName) {
    std::cout << "\n--- denoiseMedian3 with dirty source padding: " << wordTypeName << " ---\n";

    for (int width : WIDTHS) {
        for (int height : HEIGHTS) {
            for (size_t f = 0; f < sizeof(FILLS) / sizeof(FILLS[0]); ++f) {
                StridedMat<WordType> src;
                makeDirtyPadded(src, width, height, FILLS[f], caseSeed(width, height, 500 + f));

                bincv::BinMat<WordType> dst(width, height);
                denoiseMedian3(src.mat.constView(), dst.view());

                const std::string label = sizeLabel(wordTypeName, width, height, "dirty source");
                DENOISE_EXPECT(disagreements(src.mat, dst) == 0,
                               "denoiseMedian3 ignores the source's padding bits", label);
                DENOISE_EXPECT(bitsAcrossStride(dst) == dst.countNonZero(),
                               "denoiseMedian3 leaves the destination's padding zero", label);
            }
        }
    }
}

// ===========================================================================
// 6. Degenerate shapes, and two disjoint views over ONE buffer
// ===========================================================================

template <typename WordType>
void testDegenerateViews(const char* wordTypeName) {
    std::cout << "\n--- denoiseMedian3 on degenerate views: " << wordTypeName << " ---\n";

    const int shapes[][2] = {{0, 0}, {0, 5}, {5, 0}};
    for (const auto& shape : shapes) {
        bincv::BinMat<WordType> src(shape[0], shape[1]);
        bincv::BinMat<WordType> dst(shape[0], shape[1]);
        denoiseMedian3(src.constView(), dst.view());

        const std::string label = sizeLabel(wordTypeName, shape[0], shape[1], "degenerate");
        DENOISE_EXPECT(dst.countNonZero() == 0, "an empty denoiseMedian3 writes nothing", label);
    }
}

/// @brief Two views over ONE buffer that share no word must be ACCEPTED.
/// @note the aliasing predicate is exact and per row, not a bounding-box
/// test, so interleaved row bands over a single allocation are legal
/// arguments. Rejecting them would abort the Debug build on a view a caller
/// is entitled to build. Nothing here checks a pixel that the
/// reference sweep does not; the check is that the call RUNS.
template <typename WordType>
void testDisjointViewsAccepted(const char* wordTypeName) {
    std::cout << "\n--- denoiseMedian3 accepts disjoint views over one buffer: " << wordTypeName
              << " ---\n";

    constexpr size_t bits = bincv::BinMat<WordType>::WordBits;
    for (int width : {7, 65, 128}) {
        const int height = 5;
        const size_t rowWords = (static_cast<size_t>(width) + bits - 1) / bits;

        // Alternate row bands: src takes the even rows of a 2 * height-row
        // buffer, dst the odd ones. Same allocation, no shared word.
        std::vector<WordType> buffer(rowWords * 2 * static_cast<size_t>(height),
                                     static_cast<WordType>(0));
        bincv::BinMat<WordType> srcMat(buffer.data(), width, height, rowWords * 2);
        bincv::BinMat<WordType> dstMat(buffer.data() + rowWords, width, height, rowWords * 2);
        fillRandom(srcMat, 0.5f, caseSeed(width, height, 700));

        denoiseMedian3(srcMat.constView(), dstMat.view());

        const std::string label = sizeLabel(wordTypeName, width, height, "interleaved bands");
        DENOISE_EXPECT(disagreements(srcMat, dstMat) == 0,
                       "denoiseMedian3 accepts interleaved views over one buffer", label);

        // NO bitsAcrossStride CHECK HERE, deliberately. That helper walks the
        // whole STRIDE, and this view's stride spans the interleaved source rows
        // -- another view's live pixels, not padding. Counting them would report a
        // violation that is the test's construction rather than the kernel's
        // behavior. The padding invariant is asserted over ordinary views by
        // every other family in this file; what this case exists to check is that
        // the predicate ACCEPTS the view at all (a bounding-box test would
        // abort the Debug build here).
    }
}

// ===========================================================================
// 7. The OpenCV half: the reference implementation, ported call for call
// ===========================================================================

#ifdef BINCV_WITH_OPENCV

/// @brief The reference frontend's three-pixel median, transcribed.
///
/// @note This is a PORT, not a paraphrase. Every line below appears in
/// the reference frontend's denoiser in this order, including the
/// two `cv::Mat::zeros` constructions and the two range-limited copyTo
/// calls that are the entire border specification:
///
/// img.colRange(1, img.cols).copyTo(right_pixels.colRange(0, img.cols - 1));
/// img.rowRange(0, img.rows - 1).copyTo(above_pixels.rowRange(1, img.rows));
///
/// Column `cols - 1` of `right_pixels` and row 0 of `above_pixels` are
/// never assigned, so they keep the zeros they were built with. Writing
/// the border out by hand instead would put this file's reading of those
/// ranges on both sides of the comparison, which is the one thing the port
/// exists to avoid.
/// @note The only edit is guarding the two copies for a single-column or
/// single-row image, where cv::Range(1, 1) and cv::Range(0, 0) are empty.
/// The reference is only ever run on real frames; the the matrix includes
/// 1-pixel extents, and an empty range is a cv::Mat assertion rather than a
/// no-op.
cv::Mat referenceMedian3(const cv::Mat& img) {
    cv::Mat right_pixels = cv::Mat::zeros(img.size(), img.type());
    cv::Mat above_pixels = cv::Mat::zeros(img.size(), img.type());

    if (img.cols > 1) {
        img.colRange(1, img.cols).copyTo(right_pixels.colRange(0, img.cols - 1));
    }
    if (img.rows > 1) {
        img.rowRange(0, img.rows - 1).copyTo(above_pixels.rowRange(1, img.rows));
    }

    cv::Mat min_p1_p2 = cv::Mat::zeros(img.size(), img.type());
    cv::Mat max_p1_p2 = cv::Mat::zeros(img.size(), img.type());
    cv::Mat min_max = cv::Mat::zeros(img.size(), img.type());
    cv::Mat median_img = cv::Mat::zeros(img.size(), img.type());

    // | | p1 | |
    // | | p2 | p3 |
    // p1 is above_pixels, p3 is right_pixels
    // Median = max(min(p1, p2), min(max(p1, p2), p3))
    cv::min(above_pixels, img, min_p1_p2);
    cv::max(above_pixels, img, max_p1_p2);
    cv::min(max_p1_p2, right_pixels, min_max);
    cv::max(min_p1_p2, min_max, median_img);

    return median_img;
}

template <typename WordType>
void testAgainstReference(const char* wordTypeName) {
    std::cout << "\n--- denoiseMedian3 vs the reference implementation: " << wordTypeName
              << " ---\n";

    for (int width : bincv::test::equivalenceWidths()) {
        for (int height : bincv::test::equivalenceHeights()) {
            for (size_t f = 0; f < bincv::test::equivalenceFillRatios().size(); ++f) {
                const float fill = bincv::test::equivalenceFillRatios()[f];
                const uint64_t seed = caseSeed(width, height, 1100 + f);

                bincv::BinMat<WordType> src =
                    bincv::test::randomBinary<WordType>(width, height, fill, seed);

                // The harness's SECOND generator, which never touches the packing
                // or the unpacking path -- so the two sides of the comparison do
                // not share a conversion that could cancel (that work’s anchor).
                const cv::Mat cvSrc = bincv::test::randomCvMask(width, height, fill, seed);

                bincv::BinMat<WordType> dst(width, height);
                denoiseMedian3(src.constView(), dst.view());

                BINCV_EXPECT_BIT_EXACT(dst.constView(), referenceMedian3(cvSrc),
                                       bincv::test::caseLabel(wordTypeName, width, height, fill) +
                                           " [denoiseMedian3]");
            }
        }
    }
}

#endif // BINCV_WITH_OPENCV

} // namespace

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

BINCV_TEST(Denoise, Reference_uint8_t)  { testReference<uint8_t>("uint8_t"); }
BINCV_TEST(Denoise, Reference_uint16_t) { testReference<uint16_t>("uint16_t"); }
BINCV_TEST(Denoise, Reference_uint32_t) { testReference<uint32_t>("uint32_t"); }
BINCV_TEST(Denoise, Reference_uint64_t) { testReference<uint64_t>("uint64_t"); }

BINCV_TEST(Denoise, Border_uint8_t)  { testBorder<uint8_t>("uint8_t"); }
BINCV_TEST(Denoise, Border_uint16_t) { testBorder<uint16_t>("uint16_t"); }
BINCV_TEST(Denoise, Border_uint32_t) { testBorder<uint32_t>("uint32_t"); }
BINCV_TEST(Denoise, Border_uint64_t) { testBorder<uint64_t>("uint64_t"); }

BINCV_TEST(Denoise, Composed_uint8_t)  { testComposed<uint8_t>("uint8_t"); }
BINCV_TEST(Denoise, Composed_uint16_t) { testComposed<uint16_t>("uint16_t"); }
BINCV_TEST(Denoise, Composed_uint32_t) { testComposed<uint32_t>("uint32_t"); }
BINCV_TEST(Denoise, Composed_uint64_t) { testComposed<uint64_t>("uint64_t"); }

BINCV_TEST(Denoise, Strides_uint8_t)  { testDifferingStrides<uint8_t>("uint8_t"); }
BINCV_TEST(Denoise, Strides_uint16_t) { testDifferingStrides<uint16_t>("uint16_t"); }
BINCV_TEST(Denoise, Strides_uint32_t) { testDifferingStrides<uint32_t>("uint32_t"); }
BINCV_TEST(Denoise, Strides_uint64_t) { testDifferingStrides<uint64_t>("uint64_t"); }

BINCV_TEST(Denoise, DirtySource_uint8_t)  { testDirtySource<uint8_t>("uint8_t"); }
BINCV_TEST(Denoise, DirtySource_uint16_t) { testDirtySource<uint16_t>("uint16_t"); }
BINCV_TEST(Denoise, DirtySource_uint32_t) { testDirtySource<uint32_t>("uint32_t"); }
BINCV_TEST(Denoise, DirtySource_uint64_t) { testDirtySource<uint64_t>("uint64_t"); }

BINCV_TEST(Denoise, DegenerateViews_uint8_t)  { testDegenerateViews<uint8_t>("uint8_t"); }
BINCV_TEST(Denoise, DegenerateViews_uint16_t) { testDegenerateViews<uint16_t>("uint16_t"); }
BINCV_TEST(Denoise, DegenerateViews_uint32_t) { testDegenerateViews<uint32_t>("uint32_t"); }
BINCV_TEST(Denoise, DegenerateViews_uint64_t) { testDegenerateViews<uint64_t>("uint64_t"); }

BINCV_TEST(Denoise, DisjointViews_uint8_t)  { testDisjointViewsAccepted<uint8_t>("uint8_t"); }
BINCV_TEST(Denoise, DisjointViews_uint16_t) { testDisjointViewsAccepted<uint16_t>("uint16_t"); }
BINCV_TEST(Denoise, DisjointViews_uint32_t) { testDisjointViewsAccepted<uint32_t>("uint32_t"); }
BINCV_TEST(Denoise, DisjointViews_uint64_t) { testDisjointViewsAccepted<uint64_t>("uint64_t"); }

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Denoise, OpenCv_uint8_t)  { testAgainstReference<uint8_t>("uint8_t"); }
BINCV_TEST(Denoise, OpenCv_uint16_t) { testAgainstReference<uint16_t>("uint16_t"); }
BINCV_TEST(Denoise, OpenCv_uint32_t) { testAgainstReference<uint32_t>("uint32_t"); }
BINCV_TEST(Denoise, OpenCv_uint64_t) { testAgainstReference<uint64_t>("uint64_t"); }
#endif

BINCV_TEST_MAIN("test_denoise")
