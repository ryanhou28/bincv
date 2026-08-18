// Threshold / binarize (T3.2): the 1-bit frame from a higher-precision source.
//
// TWO HALVES, and they are two different tiers rather than two different
// configurations.
//
//   1. The CORE half (everything up to the OpenCV guard) covers `binarize`, the
//      QuantMat<N> -> 1 bit kernel. **API TIER 3**: OpenCV has no N-bit image
//      type, so there is no cv:: expression to be bit-exact against, and the
//      reference is per pixel -- `src.at(y, x) > thresh`, the comparison
//      ops/threshold.hpp documents. It runs in all four verification
//      configurations, including Debug, the only one where binarize's
//      BINCV_ASSERTs are live.
//
//   2. The OPENCV half covers `threshold`, the CV_8U -> 1 bit kernel, and it is
//      the tier promise: bit-exact against
//      cv::threshold(src, dst, thresh, 255, THRESH_BINARY) through T2.1's
//      harness, across its full size matrix.
//
// THE COMPARISON IS STRICTLY GREATER THAN, AND THAT IS WHAT THIS FILE IS FOR.
// cv::threshold's THRESH_BINARY sets dst where `src > thresh`; an implementation
// using `>=` differs on exactly the pixels EQUAL to thresh. At a mid-range
// threshold on random content that is roughly one pixel in 256 -- a sweep that
// samples a few thresholds on a few images can pass with the comparison
// backwards. So the boundary is not sampled here, it is ENUMERATED:
//
//   Threshold.Ramp_*  a 256-pixel ramp holding every uint8 value exactly once,
//                     thresholded at every value 0..255 and at 255 fractional
//                     thresholds between them, each compared against
//                     cv::threshold on the same cv::Mat. That is the ENTIRE
//                     (pixel value, integer threshold) space of this operation,
//                     not a sample of it.
//   Threshold.Ends_*  thresh = 0, 254 and 255 on real content, where an
//                     off-by-one is a whole-image difference rather than a
//                     scattering: `>= 0` selects every pixel, `> 255` selects
//                     none, and `>= 255` versus `> 254` differ on the value 255
//                     alone.
//   BinarizeSweep_*   every threshold from 0 to MaxValue + 1 on the N-bit side,
//                     for the same reason and by the same argument.
//
// WHY THE INPUT IS NOT PACKED CONTENT ON THE TIER 1 SIDE: both sides of that
// comparison read ONE cv::Mat. cv::threshold reads it and bincv::threshold reads
// it, so there is no packing on the input path at all and no shared conversion
// that could cancel a fault (T2.1's argument). Only the binCV OUTPUT is unpacked,
// by the harness, and that path is anchored by tests/test_equivalence.cpp.
//
// WHY THE CHECK COUNT IS NOT ONE PER PIXEL: each swept case reports its
// disagreement count as a single check, so CHECKS tracks cases.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/threshold.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "test_util.hpp"

// OpenCV-only, and self-guarding.
#include "equivalence.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {

using bincv::binarize;

/// @def THRESHOLD_EXPECT
/// @brief One check, with a detail string built only when it fails.
#define THRESHOLD_EXPECT(ok, what, detailExpr) \
    ::bincv::test::reportCheck((ok), (what), __FILE__, __LINE__, \
                               (ok) ? std::string() : (detailExpr))

// ---------------------------------------------------------------------------
// Content
// ---------------------------------------------------------------------------

uint64_t nextRandom(uint64_t& state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

uint64_t caseSeed(int width, int height, size_t index) {
    return UINT64_C(0x7A3E5A01D0000001) + static_cast<uint64_t>(width) * UINT64_C(1000003) +
           static_cast<uint64_t>(height) * UINT64_C(10007) + static_cast<uint64_t>(index);
}

std::string sizeLabel(const char* wordTypeName, int width, int height, const std::string& extra) {
    return std::string(wordTypeName) + " " + std::to_string(width) + "x" +
           std::to_string(height) + " " + extra;
}

/// @brief Set bits across the whole STRIDE, padding included.
/// @note Against countNonZero()'s per-pixel loop this is how a padding-bit
///       violation becomes visible. It is the check that matters most for
///       `binarize`: thresholdGE answers EVERY lane in a word, including the
///       ones past `width`, and at threshold 0 it answers "yes" to all of them
///       whatever the planes hold (ops/bitslice.hpp). Without the tail mask the
///       kernel applies, `binarize(src, dst, 0)` would leave up to WordBits - 1
///       phantom pixels per row for the next reduction to count (D-13).
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

/// @brief Random N-bit pixel values, written through set() so the padding stays
///        clear on entry.
template <size_t N, typename WordType>
void fillRandom(bincv::QuantMat<N, WordType>& m, uint64_t seed) {
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            const unsigned v =
                static_cast<unsigned>(nextRandom(state) >> 40) & bincv::QuantMat<N, WordType>::MaxValue;
            m.set(y, x, v);
        }
    }
}

/// @brief Pixels on which binarize's result differs from `src.at(y, x) > thresh`.
/// @note The reference is the comparison ops/threshold.hpp documents, written as
///       an ordinary unsigned `>` on the value at() reassembles from the planes --
///       not as a bit-sliced expression. A reference sharing thresholdGE's
///       formulation could not fail with it.
template <size_t N, typename WordType>
int disagreements(const bincv::QuantMat<N, WordType>& src, const bincv::BinMat<WordType>& dst,
                  unsigned thresh) {
    int differing = 0;
    for (int y = 0; y < dst.rows(); ++y) {
        for (int x = 0; x < dst.cols(); ++x) {
            // The cast is not decoration: QuantMat<1>::at() returns bool
            // (core/types.hpp), and `bool > unsigned` is an int/unsigned
            // comparison -- which -Wextra reports and which would make the N == 1
            // instantiation the one that does not compile.
            const bool expected = static_cast<unsigned>(src.at(y, x)) > thresh;
            if (dst.at(y, x) != expected) ++differing;
        }
    }
    return differing;
}

const int WIDTHS[] = {1, 7, 31, 33, 40, 63, 65, 70, 128, 640};

// ===========================================================================
// 1. EVERY threshold, at the sizes where a trailing partial word lives
// ===========================================================================
//
// 0 through MaxValue + 1 inclusive. Both ends are reached by a caller sweeping
// rather than by choice, and both are where an off-by-one shows as a whole-image
// answer: threshold 0 selects every non-zero pixel (and would select ALL of them
// if the kernel used `>=`), threshold MaxValue selects nothing (and would select
// the saturated pixels if it used `>=`), threshold MaxValue + 1 is the shortcut
// path in ops/threshold.hpp that exists so `thresh + 1` cannot wrap.

template <size_t N, typename WordType>
void testBinarizeSweep(const char* wordTypeName) {
    std::cout << "\n--- binarize<" << N << "> at every threshold: " << wordTypeName << " ---\n";

    constexpr unsigned maxValue = bincv::QuantMat<N, WordType>::MaxValue;

    for (int width : {1, 7, 31, 33, 63, 65, 70, 128}) {
        for (int height : {1, 3}) {
            bincv::QuantMat<N, WordType> src(width, height);
            fillRandom(src, caseSeed(width, height, N));

            for (unsigned thresh = 0; thresh <= maxValue + 1u; ++thresh) {
                bincv::BinMat<WordType> dst(width, height);
                binarize(src, dst.view(), thresh);

                const std::string label =
                    sizeLabel(wordTypeName, width, height,
                              "N=" + std::to_string(N) + " thresh=" + std::to_string(thresh));
                THRESHOLD_EXPECT(disagreements(src, dst, thresh) == 0,
                                 "binarize matches src.at(y, x) > thresh", label);
                THRESHOLD_EXPECT(bitsAcrossStride(dst) == dst.countNonZero(),
                                 "binarize leaves the padding bits zero", label);
            }
        }
    }
}

// ===========================================================================
// 2. The full width list, at the boundary thresholds
// ===========================================================================

template <size_t N, typename WordType>
void testBinarizeSizes(const char* wordTypeName) {
    std::cout << "\n--- binarize<" << N << "> across the size matrix: " << wordTypeName << " ---\n";

    constexpr unsigned maxValue = bincv::QuantMat<N, WordType>::MaxValue;
    const unsigned thresholds[] = {0u, maxValue / 2u, maxValue - 1u, maxValue, maxValue + 1u};

    for (int width : WIDTHS) {
        for (int height : {1, 2, 3, 17, 37}) {
            bincv::QuantMat<N, WordType> src(width, height);
            fillRandom(src, caseSeed(width, height, 100 + N));

            for (unsigned thresh : thresholds) {
                bincv::BinMat<WordType> dst(width, height);
                binarize(src, dst.view(), thresh);

                const std::string label =
                    sizeLabel(wordTypeName, width, height,
                              "N=" + std::to_string(N) + " thresh=" + std::to_string(thresh));
                THRESHOLD_EXPECT(disagreements(src, dst, thresh) == 0,
                                 "binarize matches src.at(y, x) > thresh", label);
                THRESHOLD_EXPECT(bitsAcrossStride(dst) == dst.countNonZero(),
                                 "binarize leaves the padding bits zero", label);
            }
        }
    }
}

// ===========================================================================
// 3. The saturated image, which is where `>` and `>=` differ by a whole image
// ===========================================================================
//
// An image whose every pixel is MaxValue. At `thresh == MaxValue` the answer is
// EMPTY under `>` and FULL under `>=`, so this single case separates the two
// spellings on every pixel rather than on one in 2^N of them.
// And its mirror: an all-zero image at thresh 0, empty under `>` and full under
// `>=`.

template <size_t N, typename WordType>
void testBinarizeSaturated(const char* wordTypeName) {
    std::cout << "\n--- binarize<" << N << "> on saturated and empty images: " << wordTypeName
              << " ---\n";

    constexpr unsigned maxValue = bincv::QuantMat<N, WordType>::MaxValue;

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            bincv::QuantMat<N, WordType> full(width, height);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) full.set(y, x, maxValue);
            }
            const bincv::QuantMat<N, WordType> zero(width, height);

            bincv::BinMat<WordType> dst(width, height);
            const std::string label =
                sizeLabel(wordTypeName, width, height, "N=" + std::to_string(N));

            // Saturated at MaxValue: `> MaxValue` is false everywhere.
            binarize(full, dst.view(), maxValue);
            THRESHOLD_EXPECT(dst.countNonZero() == 0,
                             "an all-MaxValue image at thresh = MaxValue selects nothing",
                             label + " [saturated]");
            THRESHOLD_EXPECT(bitsAcrossStride(dst) == 0,
                             "and leaves no padding bit set", label + " [saturated]");

            // Saturated one below: `> MaxValue - 1` is true everywhere. (At N == 1
            // this is thresh 0 on an all-ones image, which is the same statement.)
            binarize(full, dst.view(), maxValue - 1u);
            THRESHOLD_EXPECT(dst.countNonZero() == width * height,
                             "an all-MaxValue image at thresh = MaxValue - 1 selects everything",
                             label + " [saturated-1]");
            THRESHOLD_EXPECT(bitsAcrossStride(dst) == width * height,
                             "and still leaves no padding bit set", label + " [saturated-1]");

            // All zero at thresh 0: `> 0` is false everywhere.
            binarize(zero, dst.view(), 0u);
            THRESHOLD_EXPECT(dst.countNonZero() == 0,
                             "an all-zero image at thresh = 0 selects nothing",
                             label + " [zero]");
            THRESHOLD_EXPECT(bitsAcrossStride(dst) == 0,
                             "and leaves no padding bit set -- thresholdGE would answer every "
                             "lane at threshold 0",
                             label + " [zero]");

            // THE LARGEST `unsigned` THERE IS, on the saturated image. This is
            // the case ops/threshold.hpp's `thresh >= maxValue` shortcut exists
            // for: without it, `thresh + 1` wraps to 0 and thresholdGE answers
            // "every lane" -- an ALL-ONES result for the threshold that should
            // select least of all. Measured: deleting the shortcut turns this
            // family red and leaves every other family in the file green,
            // because no ordinary sweep reaches a threshold that can wrap.
            binarize(full, dst.view(), ~0u);
            THRESHOLD_EXPECT(dst.countNonZero() == 0,
                             "thresh = UINT_MAX selects nothing rather than wrapping to 0",
                             label + " [uint-max]");
            THRESHOLD_EXPECT(bitsAcrossStride(dst) == 0,
                             "and leaves no padding bit set", label + " [uint-max]");
        }
    }
}

// ===========================================================================
// 4. Differing strides, dirty source padding, degenerate shapes
// ===========================================================================

constexpr size_t PADDED_ALIGNMENT = 32;

template <size_t N, typename WordType>
void testBinarizeStrides(const char* wordTypeName) {
    std::cout << "\n--- binarize<" << N << "> across differing strides: " << wordTypeName
              << " ---\n";

    constexpr unsigned maxValue = bincv::QuantMat<N, WordType>::MaxValue;

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            for (int srcPadded = 0; srcPadded < 2; ++srcPadded) {
                for (int dstPadded = 0; dstPadded < 2; ++dstPadded) {
                    bincv::QuantMat<N, WordType> src(
                        width, height,
                        srcPadded ? PADDED_ALIGNMENT
                                  : bincv::QuantMat<N, WordType>::DefaultRowAlignment);
                    bincv::BinMat<WordType> dst(
                        width, height,
                        dstPadded ? PADDED_ALIGNMENT
                                  : bincv::BinMat<WordType>::DefaultRowAlignment);
                    fillRandom(src, caseSeed(width, height, 300 + N));

                    const unsigned thresh = maxValue / 2u;
                    binarize(src, dst.view(), thresh);

                    const std::string label = sizeLabel(
                        wordTypeName, width, height,
                        "N=" + std::to_string(N) + (srcPadded ? " padded-src" : " tight-src") +
                            (dstPadded ? " padded-dst" : " tight-dst"));
                    THRESHOLD_EXPECT(disagreements(src, dst, thresh) == 0,
                                     "binarize matches the reference across strides", label);
                    THRESHOLD_EXPECT(bitsAcrossStride(dst) == dst.countNonZero(),
                                     "binarize leaves the padding bits zero", label);
                }
            }
        }
    }
}

/// @brief Source planes whose padding bits are ALREADY SET -- a legal
///        construction (a wrapped buffer's padding belongs to its caller).
/// @note This is what makes the tail mask in ops/threshold.hpp observable. With
///       every padding bit set in every plane, an unmasked store would leave the
///       destination's padding set for every threshold the dirty lanes pass,
///       which is most of them.
template <size_t N, typename WordType>
void testBinarizeDirtyPlanes(const char* wordTypeName) {
    std::cout << "\n--- binarize<" << N << "> with dirty source padding: " << wordTypeName
              << " ---\n";

    constexpr size_t bits = bincv::BinMat<WordType>::WordBits;
    constexpr unsigned maxValue = bincv::QuantMat<N, WordType>::MaxValue;

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            const size_t minWords = (static_cast<size_t>(width) + bits - 1) / bits;
            const size_t stride = minWords + 1;
            std::vector<WordType> buffer(stride * static_cast<size_t>(height) * N,
                                         static_cast<WordType>(~static_cast<WordType>(0)));
            bincv::QuantMat<N, WordType> src(buffer.data(), width, height, stride);

            // Every bit is set, padding included; now write the pixels, leaving
            // the bits past `width` exactly as dirty as they started.
            uint64_t state = caseSeed(width, height, 400 + N);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    src.set(y, x, static_cast<unsigned>(nextRandom(state) >> 40) & maxValue);
                }
            }

            for (unsigned thresh : {0u, maxValue / 2u, maxValue - 1u, maxValue}) {
                bincv::BinMat<WordType> dst(width, height);
                binarize(src, dst.view(), thresh);

                const std::string label =
                    sizeLabel(wordTypeName, width, height,
                              "N=" + std::to_string(N) + " dirty thresh=" + std::to_string(thresh));
                THRESHOLD_EXPECT(disagreements(src, dst, thresh) == 0,
                                 "binarize ignores the source planes' padding bits", label);
                THRESHOLD_EXPECT(bitsAcrossStride(dst) == dst.countNonZero(),
                                 "binarize leaves the destination's padding zero", label);
            }
        }
    }
}

template <size_t N, typename WordType>
void testBinarizeDegenerate(const char* wordTypeName) {
    std::cout << "\n--- binarize<" << N << "> on degenerate views: " << wordTypeName << " ---\n";

    const int shapes[][2] = {{0, 0}, {0, 5}, {5, 0}};
    for (const auto& shape : shapes) {
        const bincv::QuantMat<N, WordType> src(shape[0], shape[1]);
        bincv::BinMat<WordType> dst(shape[0], shape[1]);
        binarize(src, dst.view(), 0u);

        const std::string label =
            sizeLabel(wordTypeName, shape[0], shape[1], "N=" + std::to_string(N) + " degenerate");
        THRESHOLD_EXPECT(dst.countNonZero() == 0, "an empty binarize writes nothing", label);
    }
}

/// @brief The plane-view entry point, called with the array spelling directly.
/// @note The QuantMat overload is documented as a thin wrapper over it (D-5), and
///       this is what stops that from being only a claim: the two are called on
///       the same content and required to produce the same image. A caller
///       holding views rather than a container takes this path.
template <size_t N, typename WordType>
void testBinarizePlaneViews(const char* wordTypeName) {
    std::cout << "\n--- binarize<" << N << "> through plane views: " << wordTypeName << " ---\n";

    constexpr unsigned maxValue = bincv::QuantMat<N, WordType>::MaxValue;

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            bincv::QuantMat<N, WordType> src(width, height);
            fillRandom(src, caseSeed(width, height, 500 + N));

            bincv::BinMatConstView<WordType> planes[N];
            for (size_t p = 0; p < N; ++p) planes[p] = src.constPlane(p);

            for (unsigned thresh : {0u, maxValue / 2u, maxValue}) {
                bincv::BinMat<WordType> viaContainer(width, height);
                bincv::BinMat<WordType> viaViews(width, height);
                binarize(src, viaContainer.view(), thresh);
                binarize(planes, viaViews.view(), thresh);

                int differing = 0;
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        if (viaContainer.at(y, x) != viaViews.at(y, x)) ++differing;
                    }
                }
                const std::string label =
                    sizeLabel(wordTypeName, width, height,
                              "N=" + std::to_string(N) + " views thresh=" + std::to_string(thresh));
                THRESHOLD_EXPECT(differing == 0,
                                 "the QuantMat overload is the plane-view kernel", label);
            }
        }
    }
}

// ===========================================================================
// 4b. THIRTY-TWO PLANES: the widest cutoff an `unsigned` threshold can express
// ===========================================================================
//
// The plane-view entry point takes N from its argument and QuantMat's N <= 8 cap
// does not reach it, so N here is whatever a caller writes. THE TOP OF THE
// `unsigned` RANGE IS WHERE THAT MATTERS, and nothing else in this file goes
// near it with more than 8 planes:
//
//   * `thresh == MaxValue` must select NOTHING and must not wrap `thresh + 1`.
//   * `thresh == MaxValue - 1` must select exactly the saturated pixels.
//
// Those two cases one apart are the ones a shortcut written in terms of a
// SATURATED MaxValue answers identically and wrongly. Measured, on the code as it
// stood before this family existed: with 33 planes and a pixel holding 2^32,
// thresh = UINT_MAX - 1 selected it and thresh = UINT_MAX did not -- an answer
// that is not monotone in the threshold. 33 planes is now a compile error
// (ops/threshold.hpp static_asserts N <= 32, because the cutoff such a caller
// needs is 2^32 and the parameter cannot hold it), and 32 -- the widest N that IS
// expressible -- is what this family pins.

template <typename WordType>
void testBinarizeWideCutoff(const char* wordTypeName) {
    std::cout << "\n--- binarize<32> at the top of the unsigned range: " << wordTypeName
              << " ---\n";

    constexpr size_t N = 32;
    constexpr size_t bits = bincv::BinMat<WordType>::WordBits;

    for (int width : {1, 7, 33, 65, 70}) {
        for (int height : {1, 3}) {
            const size_t rowWords = (static_cast<size_t>(width) + bits - 1) / bits;
            // One buffer, N plane-sized slabs inside it, all with the same stride.
            std::vector<WordType> buffer(rowWords * static_cast<size_t>(height) * N,
                                         static_cast<WordType>(0));

            // Pixel values chosen so that both ends of the range are present:
            // 0, 1, UINT_MAX - 1 and UINT_MAX, cycling across the row.
            const unsigned values[] = {0u, 1u, ~0u - 1u, ~0u, 0x80000000u, 0x7FFFFFFFu};
            std::vector<unsigned> value(static_cast<size_t>(width) *
                                        static_cast<size_t>(height));
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const unsigned v =
                        values[static_cast<size_t>(x + y * 3) % (sizeof(values) / sizeof(values[0]))];
                    value[static_cast<size_t>(y) * static_cast<size_t>(width) +
                          static_cast<size_t>(x)] = v;
                    for (size_t pl = 0; pl < N; ++pl) {
                        if (((v >> pl) & 1u) == 0u) continue;
                        WordType* row = buffer.data() + pl * rowWords * static_cast<size_t>(height) +
                                        static_cast<size_t>(y) * rowWords;
                        const size_t w = static_cast<size_t>(x) / bits;
                        row[w] = static_cast<WordType>(
                            row[w] | static_cast<WordType>(static_cast<WordType>(1)
                                                          << (static_cast<size_t>(x) % bits)));
                    }
                }
            }

            bincv::BinMatConstView<WordType> planes[N];
            for (size_t pl = 0; pl < N; ++pl) {
                planes[pl] = bincv::BinMatConstView<WordType>{
                    buffer.data() + pl * rowWords * static_cast<size_t>(height),
                    static_cast<size_t>(width), static_cast<size_t>(height), rowWords};
            }

            for (unsigned thresh : {0u, 1u, 0x7FFFFFFFu, ~0u - 2u, ~0u - 1u, ~0u}) {
                bincv::BinMat<WordType> dst(width, height);
                binarize(planes, dst.view(), thresh);

                int differing = 0;
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const bool expected =
                            value[static_cast<size_t>(y) * static_cast<size_t>(width) +
                                  static_cast<size_t>(x)] > thresh;
                        if (dst.at(y, x) != expected) ++differing;
                    }
                }

                const std::string label =
                    sizeLabel(wordTypeName, width, height,
                              "N=32 thresh=" + std::to_string(thresh));
                THRESHOLD_EXPECT(differing == 0,
                                 "binarize<32> matches value > thresh at the top of the range",
                                 label + ", " + std::to_string(differing) + " pixels differ");
                THRESHOLD_EXPECT(bitsAcrossStride(dst) == dst.countNonZero(),
                                 "binarize<32> leaves the padding bits zero", label);
            }
        }
    }
}

// ===========================================================================
// 5. The OpenCV half: TIER 1 against cv::threshold with THRESH_BINARY
// ===========================================================================

#ifdef BINCV_WITH_OPENCV

/// @brief What OpenCV produces: THRESH_BINARY with maxval 255, so the bytes are
///        {0, 255} -- exactly what the harness unpacks a BinMat into.
cv::Mat openCvThreshold(const cv::Mat& src, double thresh) {
    cv::Mat out;
    cv::threshold(src, out, thresh, 255.0, cv::THRESH_BINARY);
    return out;
}

/// @brief Random CV_8U content over the FULL value range, not a binary mask.
/// @note tests/equivalence.hpp's generators produce {0, 255} only, which is the
///       right content for the Tier 1 logic and shift kernels and the wrong
///       content here: a threshold whose sources are only ever 0 or 255 agrees
///       with itself for every threshold in 1..254. This generator is local for
///       that reason, and it feeds BOTH sides of the comparison.
cv::Mat randomGray(int width, int height, uint64_t seed) {
    cv::Mat out = cv::Mat::zeros(height, width, CV_8U);
    uint64_t state = seed;
    for (int y = 0; y < height; ++y) {
        uint8_t* row = out.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            row[x] = static_cast<uint8_t>((nextRandom(state) >> 33) & 0xFFu);
        }
    }
    return out;
}

/// @brief The T2.1 size matrix, at the thresholds where an off-by-one is a
///        whole-image difference rather than a scattering.
/// @note BOTH DESTINATION ALIGNMENTS, and that is not padding for its own sake.
///       At DefaultRowAlignment == sizeof(WordType) an image's aligned width IS
///       its minimum row width, so `dst.row(y)` and `dst.ptr + y * words` are the
///       same address and the kernel's row addressing is invisible. Measured:
///       replacing `dst.row(y)` in ops/threshold.hpp with the stride-free
///       expression left all 27040 checks green before this loop existed. D-4 is
///       provisional, so an over-aligned destination is a supported shape, and
///       the Tier 3 half already swept it (testBinarizeStrides) -- the Tier 1
///       half not doing so was an asymmetry rather than a decision.
template <typename WordType>
void testThresholdSizes(const char* wordTypeName) {
    std::cout << "\n--- threshold vs cv::threshold across the size matrix: " << wordTypeName
              << " ---\n";

    // 0 and 255 are the two ends cv::threshold itself special-cases before
    // dispatching; 254 is the one below the top, where `>= 255` and `> 254`
    // differ on the value 255 alone; 127 and 128 straddle the middle.
    const double thresholds[] = {0.0, 1.0, 127.0, 128.0, 254.0, 255.0};

    for (int width : bincv::test::equivalenceWidths()) {
        for (int height : bincv::test::equivalenceHeights()) {
            const cv::Mat src = randomGray(width, height, caseSeed(width, height, 900));

            for (double thresh : thresholds) {
                for (int dstPadded = 0; dstPadded < 2; ++dstPadded) {
                    bincv::BinMat<WordType> dst(
                        width, height,
                        dstPadded ? PADDED_ALIGNMENT
                                  : bincv::BinMat<WordType>::DefaultRowAlignment);
                    bincv::threshold(src, dst.view(), thresh);

                    BINCV_EXPECT_BIT_EXACT(
                        dst.constView(), openCvThreshold(src, thresh),
                        sizeLabel(wordTypeName, width, height,
                                  "thresh=" + std::to_string(static_cast<int>(thresh)) +
                                      (dstPadded ? " padded-dst" : " tight-dst")));
                }
            }
        }
    }
}

/// @brief THE BOUNDARY, ENUMERATED. A ramp holding every uint8 value exactly
///        once, thresholded at every integer 0..255 -- so every (pixel value,
///        threshold) pair this operation has is compared against cv::threshold.
/// @note An implementation using `>=` instead of `>` differs from OpenCV on the
///        single pixel whose value equals the threshold, for 256 of these 256
///        cases. A sampled test can miss that; this cannot.
/// @note The fractional thresholds are the second half. cv::threshold FLOORS
///        `thresh` for a CV_8U source before dispatching, so `t - 0.5` must
///        behave as `t - 1` and `t + 0.5` as `t`. ops/threshold.hpp reduces the
///        double to the same integer cutoff, and this is what says so.
template <typename WordType>
void testThresholdRamp(const char* wordTypeName) {
    std::cout << "\n--- threshold over a full 0..255 ramp, every threshold: " << wordTypeName
              << " ---\n";

    // Three rows: the ramp, its reverse, and the ramp again -- so a kernel that
    // used row 0's data for every row, or that mixed up its stride, fails here as
    // well as in the size sweep.
    const int width = 256;
    const int height = 3;
    cv::Mat src = cv::Mat::zeros(height, width, CV_8U);
    for (int x = 0; x < width; ++x) {
        src.at<uint8_t>(0, x) = static_cast<uint8_t>(x);
        src.at<uint8_t>(1, x) = static_cast<uint8_t>(255 - x);
        src.at<uint8_t>(2, x) = static_cast<uint8_t>(x);
    }

    for (int t = 0; t <= 255; ++t) {
        bincv::BinMat<WordType> dst(width, height);
        bincv::threshold(src, dst.view(), static_cast<double>(t));
        BINCV_EXPECT_BIT_EXACT(dst.constView(), openCvThreshold(src, static_cast<double>(t)),
                               std::string(wordTypeName) + " ramp thresh=" + std::to_string(t));
    }

    // Fractional and out-of-range thresholds, against the same reference.
    const double odd[] = {-1.0, -0.5, 0.5, 126.5, 127.5, 254.5, 255.5, 300.0};
    for (double thresh : odd) {
        bincv::BinMat<WordType> dst(width, height);
        bincv::threshold(src, dst.view(), thresh);
        BINCV_EXPECT_BIT_EXACT(dst.constView(), openCvThreshold(src, thresh),
                               std::string(wordTypeName) + " ramp fractional thresh=" +
                                   std::to_string(thresh));
    }
}

/// @brief The ramp again at widths that are NOT a multiple of any word size, so
///        the trailing partial word carries live boundary values rather than
///        padding.
template <typename WordType>
void testThresholdRampWidths(const char* wordTypeName) {
    std::cout << "\n--- threshold over ramps at awkward widths: " << wordTypeName << " ---\n";

    for (int width : bincv::test::equivalenceWidths()) {
        for (int height : {1, 3, 17}) {
            cv::Mat src = cv::Mat::zeros(height, width, CV_8U);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    src.at<uint8_t>(y, x) = static_cast<uint8_t>((x + y * 7) & 0xFF);
                }
            }
            // Thresholds chosen so that the LAST column's value is exactly the
            // threshold in some case, which is the pixel a trailing-word bug and
            // an off-by-one bug both land on.
            for (int t : {0, 1, 63, 127, 128, 254, 255}) {
                for (int dstPadded = 0; dstPadded < 2; ++dstPadded) {
                    bincv::BinMat<WordType> dst(
                        width, height,
                        dstPadded ? PADDED_ALIGNMENT
                                  : bincv::BinMat<WordType>::DefaultRowAlignment);
                    bincv::threshold(src, dst.view(), static_cast<double>(t));
                    BINCV_EXPECT_BIT_EXACT(dst.constView(),
                                           openCvThreshold(src, static_cast<double>(t)),
                                           sizeLabel(wordTypeName, width, height,
                                                     "ramp thresh=" + std::to_string(t) +
                                                         (dstPadded ? " padded-dst"
                                                                    : " tight-dst")));
                }
            }
        }
    }
}

/// @brief A cv::Mat ROI -- `step != cols` -- against cv::threshold on the same
///        ROI.
/// @note EVERY OTHER cv::Mat IN THIS FILE IS FRESHLY ALLOCATED, hence continuous,
///       so `src.step` is the same as `src.cols` and the kernel's row addressing
///       is not being tested at all. Measured: replacing
///       `src.ptr<uint8_t>(y)` in ops/threshold.hpp with
///       `src.ptr<uint8_t>(0) + y * src.cols` left all 27040 checks green before
///       this family existed. A cropped frame is the natural way a VIO frontend
///       hands a region to a kernel, so this is a supported shape and not an
///       exotic one.
/// @note The ROI is taken at a non-zero x offset as well as a non-zero y offset,
///       so a kernel that got the row pitch right and the row ORIGIN wrong still
///       fails here.
template <typename WordType>
void testThresholdRoi(const char* wordTypeName) {
    std::cout << "\n--- threshold on a cv::Mat ROI (step != cols): " << wordTypeName << " ---\n";

    const cv::Mat big = randomGray(200, 64, caseSeed(200, 64, 950));

    // {x, y, width, height} -- widths that leave a partial trailing word at every
    // supported word size, at offsets that are not multiples of any of them.
    const int rois[][4] = {
        {0, 0, 70, 9},   {37, 4, 70, 9},   {1, 1, 1, 1},     {13, 7, 63, 33},
        {5, 0, 65, 1},   {129, 31, 71, 33}, {3, 2, 128, 17}, {0, 63, 200, 1},
    };

    for (const auto& r : rois) {
        const cv::Mat src = big(cv::Rect(r[0], r[1], r[2], r[3]));
        for (double thresh : {0.0, 1.0, 127.0, 128.0, 254.0, 255.0}) {
            for (int dstPadded = 0; dstPadded < 2; ++dstPadded) {
                bincv::BinMat<WordType> dst(
                    r[2], r[3],
                    dstPadded ? PADDED_ALIGNMENT : bincv::BinMat<WordType>::DefaultRowAlignment);
                bincv::threshold(src, dst.view(), thresh);

                BINCV_EXPECT_BIT_EXACT(
                    dst.constView(), openCvThreshold(src, thresh),
                    sizeLabel(wordTypeName, r[2], r[3],
                              "roi@" + std::to_string(r[0]) + "," + std::to_string(r[1]) +
                                  " thresh=" + std::to_string(static_cast<int>(thresh)) +
                                  (dstPadded ? " padded-dst" : " tight-dst")));
            }
        }
    }
}

/// @brief Thresholds OUTSIDE the domain of the Tier 1 promise, pinned against the
///        ARITHMETIC rather than against cv::threshold.
/// @note THIS IS THE ONE FAMILY IN THE OPENCV HALF THAT DOES NOT USE
///       cv::threshold AS ITS REFERENCE, and the reason is written into
///       ops/threshold.hpp: for a CV_8U source cv::threshold reduces its double
///       with cvFloor, whose `(int)value` conversion is undefined once the value
///       leaves `int`'s range. Measured on OpenCV 4.5.4 / x86-64 over a 0..255
///       ramp: cv::threshold sets EVERY pixel at +1e300 and CLEARS every pixel at
///       -1e300, i.e. the exact opposite of the comparison, in both directions.
///       Comparing against that would be pinning a compiler's conversion
///       behaviour.
/// @note So the reference here is `double(pixel) > thresh` per pixel, which is
///       what binCV computes, and the family exists so that binCV's CHOICE at the
///       ends -- everything below, nothing above, nothing for NaN -- cannot drift
///       silently. The suite's previous "out-of-range" values (-1.0 and 300.0) are
///       both inside int range and never reached this.
template <typename WordType>
void testThresholdOutOfDomain(const char* wordTypeName) {
    std::cout << "\n--- threshold beyond cv::threshold's domain: " << wordTypeName << " ---\n";

    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double thresholds[] = {
        1e300, -1e300, inf, -inf, nan,
        2147483648.0,    // +2^31: the first value outside int
        -2147483649.0,   // one below -2^31
        4294967296.0,    // +2^32
        -4294967296.0,
    };

    for (int width : {1, 7, 65, 70, 256}) {
        for (int height : {1, 3}) {
            cv::Mat src = cv::Mat::zeros(height, width, CV_8U);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    src.at<uint8_t>(y, x) = static_cast<uint8_t>((x + y * 7) & 0xFF);
                }
            }

            for (double thresh : thresholds) {
                // The arithmetic, per pixel, in the {0, 255} bytes the harness
                // unpacks a BinMat into. `>` against NaN is false for every pixel,
                // which is why NaN selects nothing.
                cv::Mat expected = cv::Mat::zeros(height, width, CV_8U);
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        if (static_cast<double>(src.at<uint8_t>(y, x)) > thresh) {
                            expected.at<uint8_t>(y, x) = 255;
                        }
                    }
                }

                bincv::BinMat<WordType> dst(width, height);
                bincv::threshold(src, dst.view(), thresh);
                BINCV_EXPECT_BIT_EXACT(dst.constView(), expected,
                                       sizeLabel(wordTypeName, width, height,
                                                 "out-of-domain thresh=" +
                                                     std::to_string(thresh)));
            }
        }
    }
}

#endif // BINCV_WITH_OPENCV

} // namespace

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------
//
// N = 1, 2, 3 and 5 rather than every N up to 8: 1 is BinMat (core/types.hpp),
// and 3 and 5 are the pyramid levels ARCHITECTURE 7.2 actually reaches. 2 is the
// smallest N where a plane loop can be wrong about plane order.

BINCV_TEST(Threshold, BinarizeSweep_uint8_t) {
    testBinarizeSweep<1, uint8_t>("uint8_t");
    testBinarizeSweep<2, uint8_t>("uint8_t");
    testBinarizeSweep<3, uint8_t>("uint8_t");
    testBinarizeSweep<5, uint8_t>("uint8_t");
}
BINCV_TEST(Threshold, BinarizeSweep_uint16_t) {
    testBinarizeSweep<1, uint16_t>("uint16_t");
    testBinarizeSweep<2, uint16_t>("uint16_t");
    testBinarizeSweep<3, uint16_t>("uint16_t");
    testBinarizeSweep<5, uint16_t>("uint16_t");
}
BINCV_TEST(Threshold, BinarizeSweep_uint32_t) {
    testBinarizeSweep<1, uint32_t>("uint32_t");
    testBinarizeSweep<2, uint32_t>("uint32_t");
    testBinarizeSweep<3, uint32_t>("uint32_t");
    testBinarizeSweep<5, uint32_t>("uint32_t");
}
BINCV_TEST(Threshold, BinarizeSweep_uint64_t) {
    testBinarizeSweep<1, uint64_t>("uint64_t");
    testBinarizeSweep<2, uint64_t>("uint64_t");
    testBinarizeSweep<3, uint64_t>("uint64_t");
    testBinarizeSweep<5, uint64_t>("uint64_t");
}

BINCV_TEST(Threshold, BinarizeSizes_uint8_t) {
    testBinarizeSizes<1, uint8_t>("uint8_t");
    testBinarizeSizes<3, uint8_t>("uint8_t");
    testBinarizeSizes<5, uint8_t>("uint8_t");
}
BINCV_TEST(Threshold, BinarizeSizes_uint16_t) {
    testBinarizeSizes<1, uint16_t>("uint16_t");
    testBinarizeSizes<3, uint16_t>("uint16_t");
    testBinarizeSizes<5, uint16_t>("uint16_t");
}
BINCV_TEST(Threshold, BinarizeSizes_uint32_t) {
    testBinarizeSizes<1, uint32_t>("uint32_t");
    testBinarizeSizes<3, uint32_t>("uint32_t");
    testBinarizeSizes<5, uint32_t>("uint32_t");
}
BINCV_TEST(Threshold, BinarizeSizes_uint64_t) {
    testBinarizeSizes<1, uint64_t>("uint64_t");
    testBinarizeSizes<3, uint64_t>("uint64_t");
    testBinarizeSizes<5, uint64_t>("uint64_t");
}

BINCV_TEST(Threshold, BinarizeSaturated_uint8_t) {
    testBinarizeSaturated<1, uint8_t>("uint8_t");
    testBinarizeSaturated<3, uint8_t>("uint8_t");
    testBinarizeSaturated<5, uint8_t>("uint8_t");
}
BINCV_TEST(Threshold, BinarizeSaturated_uint16_t) {
    testBinarizeSaturated<1, uint16_t>("uint16_t");
    testBinarizeSaturated<3, uint16_t>("uint16_t");
    testBinarizeSaturated<5, uint16_t>("uint16_t");
}
BINCV_TEST(Threshold, BinarizeSaturated_uint32_t) {
    testBinarizeSaturated<1, uint32_t>("uint32_t");
    testBinarizeSaturated<3, uint32_t>("uint32_t");
    testBinarizeSaturated<5, uint32_t>("uint32_t");
}
BINCV_TEST(Threshold, BinarizeSaturated_uint64_t) {
    testBinarizeSaturated<1, uint64_t>("uint64_t");
    testBinarizeSaturated<3, uint64_t>("uint64_t");
    testBinarizeSaturated<5, uint64_t>("uint64_t");
}

BINCV_TEST(Threshold, BinarizeStrides_uint8_t) {
    testBinarizeStrides<1, uint8_t>("uint8_t");
    testBinarizeStrides<5, uint8_t>("uint8_t");
}
BINCV_TEST(Threshold, BinarizeStrides_uint16_t) {
    testBinarizeStrides<1, uint16_t>("uint16_t");
    testBinarizeStrides<5, uint16_t>("uint16_t");
}
BINCV_TEST(Threshold, BinarizeStrides_uint32_t) {
    testBinarizeStrides<1, uint32_t>("uint32_t");
    testBinarizeStrides<5, uint32_t>("uint32_t");
}
BINCV_TEST(Threshold, BinarizeStrides_uint64_t) {
    testBinarizeStrides<1, uint64_t>("uint64_t");
    testBinarizeStrides<5, uint64_t>("uint64_t");
}

BINCV_TEST(Threshold, BinarizeDirty_uint8_t) {
    testBinarizeDirtyPlanes<1, uint8_t>("uint8_t");
    testBinarizeDirtyPlanes<3, uint8_t>("uint8_t");
    testBinarizeDirtyPlanes<5, uint8_t>("uint8_t");
}
BINCV_TEST(Threshold, BinarizeDirty_uint16_t) {
    testBinarizeDirtyPlanes<1, uint16_t>("uint16_t");
    testBinarizeDirtyPlanes<3, uint16_t>("uint16_t");
    testBinarizeDirtyPlanes<5, uint16_t>("uint16_t");
}
BINCV_TEST(Threshold, BinarizeDirty_uint32_t) {
    testBinarizeDirtyPlanes<1, uint32_t>("uint32_t");
    testBinarizeDirtyPlanes<3, uint32_t>("uint32_t");
    testBinarizeDirtyPlanes<5, uint32_t>("uint32_t");
}
BINCV_TEST(Threshold, BinarizeDirty_uint64_t) {
    testBinarizeDirtyPlanes<1, uint64_t>("uint64_t");
    testBinarizeDirtyPlanes<3, uint64_t>("uint64_t");
    testBinarizeDirtyPlanes<5, uint64_t>("uint64_t");
}

BINCV_TEST(Threshold, BinarizeViews_uint8_t) {
    testBinarizePlaneViews<1, uint8_t>("uint8_t");
    testBinarizePlaneViews<5, uint8_t>("uint8_t");
}
BINCV_TEST(Threshold, BinarizeViews_uint16_t) {
    testBinarizePlaneViews<1, uint16_t>("uint16_t");
    testBinarizePlaneViews<5, uint16_t>("uint16_t");
}
BINCV_TEST(Threshold, BinarizeViews_uint32_t) {
    testBinarizePlaneViews<1, uint32_t>("uint32_t");
    testBinarizePlaneViews<5, uint32_t>("uint32_t");
}
BINCV_TEST(Threshold, BinarizeViews_uint64_t) {
    testBinarizePlaneViews<1, uint64_t>("uint64_t");
    testBinarizePlaneViews<5, uint64_t>("uint64_t");
}

BINCV_TEST(Threshold, BinarizeWideCutoff_uint8_t)  { testBinarizeWideCutoff<uint8_t>("uint8_t"); }
BINCV_TEST(Threshold, BinarizeWideCutoff_uint16_t) { testBinarizeWideCutoff<uint16_t>("uint16_t"); }
BINCV_TEST(Threshold, BinarizeWideCutoff_uint32_t) { testBinarizeWideCutoff<uint32_t>("uint32_t"); }
BINCV_TEST(Threshold, BinarizeWideCutoff_uint64_t) { testBinarizeWideCutoff<uint64_t>("uint64_t"); }

BINCV_TEST(Threshold, BinarizeDegenerate_uint8_t) {
    testBinarizeDegenerate<1, uint8_t>("uint8_t");
    testBinarizeDegenerate<5, uint8_t>("uint8_t");
}
BINCV_TEST(Threshold, BinarizeDegenerate_uint16_t) {
    testBinarizeDegenerate<1, uint16_t>("uint16_t");
    testBinarizeDegenerate<5, uint16_t>("uint16_t");
}
BINCV_TEST(Threshold, BinarizeDegenerate_uint32_t) {
    testBinarizeDegenerate<1, uint32_t>("uint32_t");
    testBinarizeDegenerate<5, uint32_t>("uint32_t");
}
BINCV_TEST(Threshold, BinarizeDegenerate_uint64_t) {
    testBinarizeDegenerate<1, uint64_t>("uint64_t");
    testBinarizeDegenerate<5, uint64_t>("uint64_t");
}

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Threshold, Sizes_uint8_t)  { testThresholdSizes<uint8_t>("uint8_t"); }
BINCV_TEST(Threshold, Sizes_uint16_t) { testThresholdSizes<uint16_t>("uint16_t"); }
BINCV_TEST(Threshold, Sizes_uint32_t) { testThresholdSizes<uint32_t>("uint32_t"); }
BINCV_TEST(Threshold, Sizes_uint64_t) { testThresholdSizes<uint64_t>("uint64_t"); }

BINCV_TEST(Threshold, Ramp_uint8_t)  { testThresholdRamp<uint8_t>("uint8_t"); }
BINCV_TEST(Threshold, Ramp_uint16_t) { testThresholdRamp<uint16_t>("uint16_t"); }
BINCV_TEST(Threshold, Ramp_uint32_t) { testThresholdRamp<uint32_t>("uint32_t"); }
BINCV_TEST(Threshold, Ramp_uint64_t) { testThresholdRamp<uint64_t>("uint64_t"); }

BINCV_TEST(Threshold, RampWidths_uint8_t)  { testThresholdRampWidths<uint8_t>("uint8_t"); }
BINCV_TEST(Threshold, RampWidths_uint16_t) { testThresholdRampWidths<uint16_t>("uint16_t"); }
BINCV_TEST(Threshold, RampWidths_uint32_t) { testThresholdRampWidths<uint32_t>("uint32_t"); }
BINCV_TEST(Threshold, RampWidths_uint64_t) { testThresholdRampWidths<uint64_t>("uint64_t"); }

BINCV_TEST(Threshold, Roi_uint8_t)  { testThresholdRoi<uint8_t>("uint8_t"); }
BINCV_TEST(Threshold, Roi_uint16_t) { testThresholdRoi<uint16_t>("uint16_t"); }
BINCV_TEST(Threshold, Roi_uint32_t) { testThresholdRoi<uint32_t>("uint32_t"); }
BINCV_TEST(Threshold, Roi_uint64_t) { testThresholdRoi<uint64_t>("uint64_t"); }

BINCV_TEST(Threshold, OutOfDomain_uint8_t)  { testThresholdOutOfDomain<uint8_t>("uint8_t"); }
BINCV_TEST(Threshold, OutOfDomain_uint16_t) { testThresholdOutOfDomain<uint16_t>("uint16_t"); }
BINCV_TEST(Threshold, OutOfDomain_uint32_t) { testThresholdOutOfDomain<uint32_t>("uint32_t"); }
BINCV_TEST(Threshold, OutOfDomain_uint64_t) { testThresholdOutOfDomain<uint64_t>("uint64_t"); }
#endif

BINCV_TEST_MAIN("test_threshold")
