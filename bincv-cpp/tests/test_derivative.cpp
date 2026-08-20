// The binarized spatial derivative (T3.5): derivativeX / derivativeY.
//
// THE CORRECTNESS BAR IS THE REFERENCE IMPLEMENTATION, NOT A FORMULA. The
// operation is SEAL/src/keypoint_tracking/gradients.cpp's calcBinarizedDeriv, and
// that function is two cv::filter2D calls with [-1, 0, 1] as a 1x3 and a 3x1.
// Two properties of cv::filter2D decide whether binCV agrees with it, and both
// are the kind that produce a plausible-looking image when got backwards:
//
//   * filter2D CORRELATES. dst(x) = src(x+1) - src(x-1). A convolution would
//     negate every gradient -- and NOTHING DOWNSTREAM WOULD NOTICE. It is
//     tempting to say T3.6's cross term would catch it, since sumXX and sumYY are
//     magnitude popcounts and sumXY is the only entry that reads the sign planes;
//     but the inversion negates BOTH derivatives and (-Ix)(-Iy) = IxIy, so the
//     whole 2x2 covariance is invariant under it (pinned in
//     tests/test_covariance.cpp). Derivative.OpenCvFilter2D_Direction is
//     therefore the ONLY guard on the direction. It pins it against the real
//     cv::filter2D on a single step edge, where the two readings differ by sign
//     at a known column.
//   * filter2D's default border is BORDER_REFLECT_101, NOT zero.
//     Derivative.OpenCvFilter2D_BorderDefault pins that too, by running the same
//     input through the default and through all three of BORDER_CONSTANT,
//     BORDER_REFLECT_101 and BORDER_REPLICATE explicitly and requiring the
//     default to equal exactly one of them.
//
// Neither probe is a comment about OpenCV; each is a check that fails if OpenCV
// ever disagrees with what ops/derivative.hpp was written against.
//
// FOUR HALVES, and only the last needs OpenCV:
//
//   CORE, per-pixel reference. Every swept case is compared against a reference
//     written in COORDINATES rather than in words -- `a = tap(+1)`,
//     `b = tap(-1)`, `value = a - b` -- over an INDEPENDENTLY WRITTEN border
//     mapping (a do-while, the shape OpenCV uses, not the closed form
//     ops/shift.hpp ships). Both axes, all five BorderTypes, both fill values,
//     N = 1, 2 and 3, all four word widths.
//   CORE, structural. The canonical-zero rule swept over whole frames; the
//     padding-bit invariant in every destination plane INCLUDING the sign plane;
//     sources whose padding is already dirty; differing strides; degenerate
//     shapes; and the cost constants.
//   CORE, agreement between formulations. The ternary spelling against the
//     generic ripple at N = 1 (impl::derivativeXGeneric), and the fused kernel
//     against the COMPOSED spelling -- shiftLeft/shiftRight plus ops/logic.hpp --
//     which is what keeps the inline shift honest about word boundaries.
//   OPENCV. calcBinarizedDeriv PORTED -- its own cv::filter2D calls, its own
//     scale factor of 16 -- compared at every pixel, borders included, after
//     dividing by 4080. The division is required to be EXACT, which is what makes
//     "the scale factor is representational" a checked claim rather than an
//     assertion: if any other value ever appeared, the division would not be.
//     Plus the same comparison for N-bit sources, where cv::filter2D on CV_8U
//     holding the pixel VALUES needs no scale factor at all.
//
// WHY THE CHECK COUNT IS NOT ONE PER PIXEL: a 129x17 case would contribute 2193
// checks and drown the summary. Each swept case reports its DISAGREEMENT COUNT as
// a single check, so CHECKS tracks cases and a failure still says how badly it
// failed.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/logic.hpp"
#include "bincv-cpp/ops/shift.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {

using bincv::BorderType;
using bincv::BORDER_CONSTANT;
using bincv::BORDER_REFLECT;
using bincv::BORDER_REFLECT_101;
using bincv::BORDER_REPLICATE;
using bincv::BORDER_WRAP;

/// @def DERIV_EXPECT
/// @brief One check, with a detail string built only when it fails.
#define DERIV_EXPECT(ok, what, detailExpr) \
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

/// @brief The pixel VALUES of a frame, row-major -- the reference's whole state.
/// @note Deliberately a plain vector rather than a view onto the container under
///       test: a reference that read its input back out of the object it is
///       judging could cancel a packing fault through both sides (T2.1's
///       argument for the equivalence harness's second generator).
struct Frame {
    int width = 0;
    int height = 0;
    unsigned maxValue = 1;
    std::vector<unsigned> pixels;

    unsigned at(int y, int x) const {
        return pixels[static_cast<size_t>(y) * static_cast<size_t>(width) +
                      static_cast<size_t>(x)];
    }
};

Frame makeFrame(int width, int height, unsigned maxValue, uint64_t seed) {
    Frame f;
    f.width = width;
    f.height = height;
    f.maxValue = maxValue;
    f.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height), 0u);
    uint64_t state = seed;
    for (size_t i = 0; i < f.pixels.size(); ++i) {
        f.pixels[i] = static_cast<unsigned>(nextRandom(state) % (maxValue + 1u));
    }
    return f;
}

/// @brief A frame with a single step edge, so a sign error is unmissable.
Frame makeStepFrame(int width, int height, unsigned maxValue, bool vertical) {
    Frame f;
    f.width = width;
    f.height = height;
    f.maxValue = maxValue;
    f.pixels.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0u);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const bool high = vertical ? (y >= height / 2) : (x >= width / 2);
            if (high) f.pixels[static_cast<size_t>(y) * static_cast<size_t>(width) +
                               static_cast<size_t>(x)] = maxValue;
        }
    }
    return f;
}

template <size_t N, typename WordType>
void loadFrame(const Frame& f, bincv::QuantMat<N, WordType>& m) {
    for (int y = 0; y < f.height; ++y) {
        for (int x = 0; x < f.width; ++x) m.set(y, x, f.at(y, x));
    }
}

// BinMat's set() takes bool rather than an unsigned, so N == 1 needs its own
// spelling (quantMat.hpp says why the binary case deserves bool).
template <typename WordType>
void loadFrame(const Frame& f, bincv::BinMat<WordType>& m) {
    for (int y = 0; y < f.height; ++y) {
        for (int x = 0; x < f.width; ++x) m.set(y, x, f.at(y, x) != 0u);
    }
}

// ---------------------------------------------------------------------------
// The per-pixel reference
// ---------------------------------------------------------------------------

/// @brief cv::borderInterpolate, written as OpenCV writes it: a do-while.
/// @note INDEPENDENT of impl::borderIndex on purpose. That function is a closed
///       form and is pinned against cv::borderInterpolate by tests/test_shift.cpp;
///       reusing it here would make this suite blind to a border regression in
///       the one operation whose edges T3.6 reads.
int referenceBorderIndex(int p, int len, BorderType type) {
    if (len == 1) return (type == BORDER_CONSTANT) ? -1 : 0;
    if (p >= 0 && p < len) return p;
    switch (type) {
        case BORDER_REPLICATE:
            return p < 0 ? 0 : len - 1;
        case BORDER_REFLECT:
            do {
                if (p < 0) p = -p - 1;
                else p = len - 1 - (p - len);
            } while (p < 0 || p >= len);
            return p;
        case BORDER_REFLECT_101:
            do {
                if (p < 0) p = -p;
                else p = len - 2 - (p - len);
            } while (p < 0 || p >= len);
            return p;
        case BORDER_WRAP:
            while (p < 0) p += len;
            while (p >= len) p -= len;
            return p;
        case BORDER_CONSTANT:
        default:
            return -1;
    }
}

/// @brief One tap of the [-1, 0, 1] kernel, in coordinates.
int referenceTap(const Frame& f, int y, int x, bool horizontal, int offset, BorderType type,
                 bool borderValue) {
    if (horizontal) {
        const int sx = referenceBorderIndex(x + offset, f.width, type);
        if (sx < 0) return borderValue ? static_cast<int>(f.maxValue) : 0;
        return static_cast<int>(f.at(y, sx));
    }
    const int sy = referenceBorderIndex(y + offset, f.height, type);
    if (sy < 0) return borderValue ? static_cast<int>(f.maxValue) : 0;
    return static_cast<int>(f.at(sy, x));
}

/// @brief The reference derivative: RIGHT/BELOW tap minus LEFT/ABOVE tap.
/// @note This one line is the correlation direction. cv::filter2D with the anchor
///       at the centre computes exactly it (Derivative.OpenCvFilter2D_Direction
///       pins that), and a convolution would compute its negation.
int referenceDeriv(const Frame& f, int y, int x, bool horizontal, BorderType type,
                   bool borderValue) {
    return referenceTap(f, y, x, horizontal, +1, type, borderValue) -
           referenceTap(f, y, x, horizontal, -1, type, borderValue);
}

// ---------------------------------------------------------------------------
// Reading the result back
// ---------------------------------------------------------------------------

/// @brief The RAW planes of a signed pixel: magnitude bits, and the sign bit
///        separately, WITHOUT the canonical-zero reading at() applies.
/// @note at() hides a set sign over a zero magnitude by design. This suite has to
///       be able to SEE one, because "the kernel never writes one" is the claim.
template <size_t N, typename WordType>
void rawSigned(const bincv::SignedQuantMat<N, WordType>& m, int y, int x, unsigned& magnitude,
               bool& signBit) {
    const unsigned raw = m.planes().at(y, x);
    magnitude = raw & bincv::SignedQuantMat<N, WordType>::MaxMagnitude;
    signBit = ((raw >> bincv::SignedQuantMat<N, WordType>::SignPlaneIndex) & 1u) != 0u;
}

/// @brief Bits set past `width` in a plane's trailing word, over the whole image.
/// @note The padding-bit invariant (D-13) as a number. Read plane by plane so the
///       SIGN plane is covered too -- it is the one a per-value check cannot see,
///       because at() reports 0 for a zero magnitude whatever the sign holds.
template <size_t N, typename WordType>
int dirtyPaddingBits(const bincv::SignedQuantMat<N, WordType>& m) {
    const size_t words = bincv::impl::minRowWords<WordType>(m.getWidth());
    const WordType tail = bincv::impl::rowTailMask<WordType>(m.getWidth());
    const WordType pad = static_cast<WordType>(~tail);
    int bits = 0;
    for (size_t p = 0; p < N + 1; ++p) {
        const bincv::BinMatConstView<WordType> plane = m.planes().plane(p);
        for (size_t y = 0; y < plane.height; ++y) {
            WordType v = static_cast<WordType>(plane.row(y)[words - 1] & pad);
            while (v != 0) {
                bits += static_cast<int>(v & static_cast<WordType>(1));
                v = static_cast<WordType>(v >> 1);
            }
        }
    }
    return bits;
}

// ---------------------------------------------------------------------------
// The sweep matrix
// ---------------------------------------------------------------------------

// Chosen so that every word width sees a row that ends mid-word, a row that ends
// exactly on a word boundary, and a row spanning one, two and many words.
const std::vector<int>& sweepWidths() {
    static const std::vector<int> w = {1,  2,  3,  7,  8,   9,   15,  16, 17,
                                       31, 32, 33, 63, 64,  65,  66,  94, 127,
                                       128, 129};
    return w;
}

const std::vector<int>& sweepHeights() {
    static const std::vector<int> h = {1, 2, 3, 4, 8, 9, 17};
    return h;
}

struct BorderCase {
    BorderType type;
    bool value;
    const char* name;
};

const std::vector<BorderCase>& sweepBorders() {
    static const std::vector<BorderCase> b = {
        {BORDER_REFLECT_101, false, "reflect101"}, {BORDER_CONSTANT, false, "constant0"},
        {BORDER_CONSTANT, true, "constant1"},      {BORDER_REPLICATE, false, "replicate"},
        {BORDER_REFLECT, false, "reflect"},        {BORDER_WRAP, false, "wrap"}};
    return b;
}

std::string caseLabel(const char* wordName, size_t n, const char* axis, const char* border,
                      int width, int height) {
    return std::string(wordName) + " N=" + std::to_string(n) + " " + axis + " " + border + " " +
           std::to_string(width) + "x" + std::to_string(height);
}

uint64_t caseSeed(int width, int height, size_t n, size_t index) {
    return UINT64_C(0xDE21A71E0000001) + static_cast<uint64_t>(width) * UINT64_C(1000003) +
           static_cast<uint64_t>(height) * UINT64_C(10007) + n * UINT64_C(131) + index;
}

// ---------------------------------------------------------------------------
// One swept case: run both axes and compare every pixel against the reference
// ---------------------------------------------------------------------------

/// @brief Runs one (size, border, axis) case and returns how many pixels differ.
/// @note Checks the VALUE, the MAGNITUDE and the SIGN BIT separately rather than
///       only at(): at() applies the canonical-zero reading, so a kernel that
///       wrote a set sign over a zero magnitude would agree on every value and be
///       wrong in the plane T3.6's cross term reads.
template <size_t N, typename WordType>
int runCase(const Frame& f, bool horizontal, BorderType type, bool borderValue, int& signViolations,
            int& padding) {
    bincv::QuantMat<N, WordType> src(f.width, f.height);
    loadFrame(f, src);
    bincv::SignedQuantMat<N, WordType> dst(f.width, f.height);

    if (horizontal) {
        bincv::derivativeX(src, dst, type, borderValue);
    } else {
        bincv::derivativeY(src, dst, type, borderValue);
    }

    int mismatches = 0;
    signViolations = 0;
    for (int y = 0; y < f.height; ++y) {
        for (int x = 0; x < f.width; ++x) {
            const int expected = referenceDeriv(f, y, x, horizontal, type, borderValue);
            unsigned magnitude = 0;
            bool signBit = false;
            rawSigned(dst, y, x, magnitude, signBit);

            const int expectedMagnitude = expected < 0 ? -expected : expected;
            if (dst.at(y, x) != expected) ++mismatches;
            if (static_cast<int>(magnitude) != expectedMagnitude) ++mismatches;
            if (signBit != (expected < 0)) ++mismatches;
            // The canonical-zero rule, per pixel: magnitude 0 must never carry a
            // set sign bit. Counted separately so a violation is not lost in the
            // mismatch total.
            if (magnitude == 0u && signBit) ++signViolations;
        }
    }
    padding = dirtyPaddingBits(dst);
    return mismatches;
}

// ---------------------------------------------------------------------------
// The per-pixel sweep
// ---------------------------------------------------------------------------

template <size_t N, typename WordType>
void sweepReference(const char* wordName) {
    size_t index = 0;
    for (const BorderCase& border : sweepBorders()) {
        for (int height : sweepHeights()) {
            for (int width : sweepWidths()) {
                const Frame f = makeFrame(width, height, (1u << N) - 1u,
                                          caseSeed(width, height, N, index++));
                for (int axis = 0; axis < 2; ++axis) {
                    const bool horizontal = (axis == 0);
                    int signViolations = 0;
                    int padding = 0;
                    const int bad = runCase<N, WordType>(f, horizontal, border.type, border.value,
                                                         signViolations, padding);
                    const std::string label = caseLabel(wordName, N, horizontal ? "dx" : "dy",
                                                        border.name, width, height);
                    DERIV_EXPECT(bad == 0 && signViolations == 0 && padding == 0,
                                 "derivative matches the per-pixel reference",
                                 label + ": " + std::to_string(bad) + " mismatches, " +
                                     std::to_string(signViolations) + " canonical-zero violations, " +
                                     std::to_string(padding) + " dirty padding bits");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Formulation agreement
// ---------------------------------------------------------------------------

/// @brief The ternary route against the generic ripple, at N == 1.
/// @note ARCHITECTURE 7.4 says ternary is the N = 1 instance of the general
///       signed form "and not a separate code path". This is that sentence as a
///       test: impl::derivativeXGeneric takes the ripple even at N = 1, and the
///       two must produce identical images -- every plane, every word.
template <typename WordType>
void sweepRoutesAgree(const char* wordName) {
    size_t index = 0;
    for (const BorderCase& border : sweepBorders()) {
        for (int height : sweepHeights()) {
            for (int width : sweepWidths()) {
                const Frame f = makeFrame(width, height, 1u, caseSeed(width, height, 1, index++));
                bincv::BinMat<WordType> src(width, height);
                loadFrame(f, src);

                for (int axis = 0; axis < 2; ++axis) {
                    bincv::TernaryMat<WordType> fast(width, height);
                    bincv::TernaryMat<WordType> generic(width, height);

                    bincv::BinMatConstView<WordType> srcPlanes[1] = {src.constPlane(0)};
                    bincv::BinMatView<WordType> fastMag[1] = {fast.magnitude(0)};
                    bincv::BinMatView<WordType> genericMag[1] = {generic.magnitude(0)};

                    if (axis == 0) {
                        bincv::derivativeX<1, WordType>(srcPlanes, fastMag, fast.sign(),
                                                        border.type, border.value);
                        bincv::impl::derivativeXGeneric<1, WordType>(
                            srcPlanes, genericMag, generic.sign(), border.type, border.value);
                    } else {
                        bincv::derivativeY<1, WordType>(srcPlanes, fastMag, fast.sign(),
                                                        border.type, border.value);
                        bincv::impl::derivativeYGeneric<1, WordType>(
                            srcPlanes, genericMag, generic.sign(), border.type, border.value);
                    }

                    int differ = 0;
                    for (size_t w = 0; w < fast.sizeInWords(); ++w) {
                        if (fast.data()[w] != generic.data()[w]) ++differ;
                    }
                    DERIV_EXPECT(differ == 0,
                                 "ternary route equals the generic ripple at N = 1",
                                 caseLabel(wordName, 1, axis == 0 ? "dx" : "dy", border.name,
                                           width, height) +
                                     ": " + std::to_string(differ) + " words differ");
                }
            }
        }
    }
}

/// @brief The fused kernel against the COMPOSED spelling at level 0.
/// @note shiftLeft(src, 1) is the src(x+1) tap and shiftRight(src, 1) the
///       src(x-1) tap -- ops/shift.hpp's convention, stated at the top of that
///       file. This case is what keeps the inline one-bit shift in
///       ops/derivative.hpp honest about word boundaries and about the border:
///       the two implementations share no code below the border mapping.
template <typename WordType>
void sweepComposed(const char* wordName) {
    size_t index = 0;
    for (const BorderCase& border : sweepBorders()) {
        for (int height : sweepHeights()) {
            for (int width : sweepWidths()) {
                const Frame f = makeFrame(width, height, 1u, caseSeed(width, height, 7, index++));
                bincv::BinMat<WordType> src(width, height);
                loadFrame(f, src);

                for (int axis = 0; axis < 2; ++axis) {
                    bincv::TernaryMat<WordType> fused(width, height);
                    if (axis == 0) {
                        bincv::derivativeX(src, fused, border.type, border.value);
                    } else {
                        bincv::derivativeY(src, fused, border.type, border.value);
                    }

                    // pos = a & ~b;  neg = b & ~a;  mag = pos | neg;  sign = neg
                    bincv::BinMat<WordType> a(width, height);
                    bincv::BinMat<WordType> b(width, height);
                    bincv::BinMat<WordType> notA(width, height);
                    bincv::BinMat<WordType> notB(width, height);
                    bincv::BinMat<WordType> pos(width, height);
                    bincv::BinMat<WordType> neg(width, height);
                    if (axis == 0) {
                        bincv::shiftLeft<WordType>(src.constPlane(0), a.plane(0), 1, border.type,
                                                   border.value);
                        bincv::shiftRight<WordType>(src.constPlane(0), b.plane(0), 1, border.type,
                                                    border.value);
                    } else {
                        bincv::shiftUp<WordType>(src.constPlane(0), a.plane(0), 1, border.type,
                                                 border.value);
                        bincv::shiftDown<WordType>(src.constPlane(0), b.plane(0), 1, border.type,
                                                   border.value);
                    }
                    bincv::bitwiseNot<WordType>(a.constPlane(0), notA.plane(0));
                    bincv::bitwiseNot<WordType>(b.constPlane(0), notB.plane(0));
                    bincv::bitwiseAnd<WordType>(a.constPlane(0), notB.constPlane(0), pos.plane(0));
                    bincv::bitwiseAnd<WordType>(b.constPlane(0), notA.constPlane(0), neg.plane(0));

                    int differ = 0;
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            unsigned magnitude = 0;
                            bool signBit = false;
                            rawSigned(fused, y, x, magnitude, signBit);
                            const bool expectedMag = pos.at(y, x) || neg.at(y, x);
                            const bool expectedSign = neg.at(y, x);
                            if ((magnitude != 0u) != expectedMag) ++differ;
                            if (signBit != expectedSign) ++differ;
                        }
                    }
                    DERIV_EXPECT(differ == 0, "fused derivative equals the composed spelling",
                                 caseLabel(wordName, 1, axis == 0 ? "dx" : "dy", border.name,
                                           width, height) +
                                     ": " + std::to_string(differ) + " pixels differ");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Structural properties
// ---------------------------------------------------------------------------

/// @brief A source whose PADDING bits are already set, wrapped rather than owned.
/// @note BinMat's wrap constructor documents that a caller's padding is the
///       caller's (sensor DMA, a sub-region of a wider frame). The horizontal
///       kernel is the one that can leak such a bit into a live pixel: `cur >> 1`
///       moves bit `width % WordBits` into pixel `width - 1`.
/// @note WHAT THESE CASES STAND BEHIND IS THE RIGHT-BORDER FIXUP, NOT A MASK.
///       ops/derivative.hpp does NOT mask the source's trailing word -- the mask
///       was measured dead and REMOVED, because the bit it would clear lands on
///       exactly the bit the fixup overwrites a moment later (48444 of 48444
///       checks pass with it and without it). Do not "restore" it; measurement
///       rejected it.
///
///       What is load-bearing is the fixup, and these cases are how the coupling
///       between the two is guarded from this end. Deleting the fixup from the
///       shipped kernel fails 16772 checks, of which 28 are DirtyPadding cases
///       that pass when the mask is put back -- i.e. the mask is dead BECAUSE the
///       fixup is there. Anything that narrows the fixup must re-establish the
///       last live column's right tap AND the padding invariant explicitly, which
///       is the coupling ops/derivative.hpp's item 3 states from the other end.
template <size_t N, typename WordType>
void sweepDirtyPadding(const char* wordName) {
    size_t index = 0;
    for (const BorderCase& border : sweepBorders()) {
        for (int height : sweepHeights()) {
            for (int width : sweepWidths()) {
                const size_t stride = bincv::impl::minRowWords<WordType>(
                    static_cast<size_t>(width));
                const size_t total = stride * static_cast<size_t>(height) * (N);
                std::vector<WordType> buffer(total,
                                             static_cast<WordType>(~static_cast<WordType>(0)));
                bincv::QuantMat<N, WordType> src(buffer.data(), width, height, stride);

                const Frame f = makeFrame(width, height, (1u << N) - 1u,
                                          caseSeed(width, height, N, index++ + 4096));
                loadFrame(f, src);

                for (int axis = 0; axis < 2; ++axis) {
                    bincv::SignedQuantMat<N, WordType> dst(width, height);
                    if (axis == 0) {
                        bincv::derivativeX(src, dst, border.type, border.value);
                    } else {
                        bincv::derivativeY(src, dst, border.type, border.value);
                    }
                    int bad = 0;
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            if (dst.at(y, x) !=
                                referenceDeriv(f, y, x, axis == 0, border.type, border.value)) {
                                ++bad;
                            }
                        }
                    }
                    const int padding = dirtyPaddingBits(dst);
                    DERIV_EXPECT(bad == 0 && padding == 0,
                                 "a dirty source padding bit never reaches a live pixel",
                                 caseLabel(wordName, N, axis == 0 ? "dx" : "dy", border.name,
                                           width, height) +
                                     ": " + std::to_string(bad) + " mismatches, " +
                                     std::to_string(padding) + " dirty padding bits");
                }
            }
        }
    }
}

/// @brief Over-aligned rows (D-4's opt-in), so the strides differ from the width.
template <size_t N, typename WordType>
void sweepStrides(const char* wordName) {
    const size_t alignments[] = {sizeof(WordType), 8 * sizeof(WordType)};
    size_t index = 0;
    for (size_t srcAlign : alignments) {
        for (size_t dstAlign : alignments) {
            for (int width : {7, 33, 65, 94}) {
                for (int height : {1, 5}) {
                    const Frame f = makeFrame(width, height, (1u << N) - 1u,
                                              caseSeed(width, height, N, index++ + 8192));
                    bincv::QuantMat<N, WordType> src(width, height, srcAlign);
                    loadFrame(f, src);
                    for (int axis = 0; axis < 2; ++axis) {
                        bincv::SignedQuantMat<N, WordType> dst(width, height, dstAlign);
                        if (axis == 0) {
                            bincv::derivativeX(src, dst);
                        } else {
                            bincv::derivativeY(src, dst);
                        }
                        int bad = 0;
                        for (int y = 0; y < height; ++y) {
                            for (int x = 0; x < width; ++x) {
                                if (dst.at(y, x) != referenceDeriv(f, y, x, axis == 0,
                                                                   BORDER_REFLECT_101, false)) {
                                    ++bad;
                                }
                            }
                        }
                        DERIV_EXPECT(bad == 0 && dirtyPaddingBits(dst) == 0,
                                     "differing strides do not change the result",
                                     caseLabel(wordName, N, axis == 0 ? "dx" : "dy", "align",
                                               width, height) +
                                         ": " + std::to_string(bad) + " mismatches");
                    }
                }
            }
        }
    }
}

/// @brief The border columns and rows, named explicitly rather than swept.
/// @note The sweep above already covers these, but a failure there reports "17
///       mismatches" on a 94x9 case. These say WHICH property broke.
template <typename WordType>
void checkBorderIdentities(const char* wordName) {
    // Reflect-101 makes both taps read the same source pixel at the first and last
    // column, so the derivative is EXACTLY ZERO there whatever the image holds.
    // That is the property T3.7's corner response depends on -- a zero fill would
    // put a full-strength edge all the way around the frame.
    {
        const int width = 65;
        const int height = 9;
        const Frame f = makeFrame(width, height, 1u, 0x5eed01u);
        bincv::BinMat<WordType> src(width, height);
        loadFrame(f, src);
        bincv::TernaryMat<WordType> dx(width, height);
        bincv::TernaryMat<WordType> dy(width, height);
        bincv::derivativeX(src, dx);
        bincv::derivativeY(src, dy);

        int nonZeroEdge = 0;
        for (int y = 0; y < height; ++y) {
            if (dx.at(y, 0) != 0) ++nonZeroEdge;
            if (dx.at(y, width - 1) != 0) ++nonZeroEdge;
        }
        for (int x = 0; x < width; ++x) {
            if (dy.at(0, x) != 0) ++nonZeroEdge;
            if (dy.at(height - 1, x) != 0) ++nonZeroEdge;
        }
        DERIV_EXPECT(nonZeroEdge == 0,
                     "BORDER_REFLECT_101 makes the derivative zero on the outer edge",
                     std::string(wordName) + ": " + std::to_string(nonZeroEdge) +
                         " non-zero edge pixels");
    }

    // A zero fill does NOT have that property, which is why the default matters.
    // One deterministic frame with column 1 and row 1 set proves the difference is
    // real rather than a statement about this file's default.
    {
        const int width = 33;
        const int height = 5;
        bincv::BinMat<WordType> src(width, height);
        for (int y = 0; y < height; ++y) src.set(y, 1, true);
        for (int x = 0; x < width; ++x) src.set(1, x, true);
        bincv::TernaryMat<WordType> dxZero(width, height);
        bincv::TernaryMat<WordType> dxReflect(width, height);
        bincv::derivativeX(src, dxZero, BORDER_CONSTANT, false);
        bincv::derivativeX(src, dxReflect, BORDER_REFLECT_101, false);
        DERIV_EXPECT(dxZero.at(2, 0) == +1 && dxReflect.at(2, 0) == 0,
                     "a zero fill manufactures an edge at column 0 where reflect-101 does not",
                     std::string(wordName) + ": zero=" + std::to_string(dxZero.at(2, 0)) +
                         " reflect=" + std::to_string(dxReflect.at(2, 0)));

        bincv::TernaryMat<WordType> dyZero(width, height);
        bincv::TernaryMat<WordType> dyReflect(width, height);
        bincv::derivativeY(src, dyZero, BORDER_CONSTANT, false);
        bincv::derivativeY(src, dyReflect, BORDER_REFLECT_101, false);
        DERIV_EXPECT(dyZero.at(0, 3) == +1 && dyReflect.at(0, 3) == 0,
                     "the same at row 0 on the vertical axis",
                     std::string(wordName) + ": zero=" + std::to_string(dyZero.at(0, 3)) +
                         " reflect=" + std::to_string(dyReflect.at(0, 3)));
    }

    // The correlation direction, on a step edge, in binCV's own terms: a 0 -> 1
    // step between columns c-1 and c gives +1 at column c-1 (the right tap sees
    // the high side first). A convolution would give -1 there.
    {
        const int width = 64;
        const int height = 3;
        const Frame f = makeStepFrame(width, height, 1u, false);
        bincv::BinMat<WordType> src(width, height);
        loadFrame(f, src);
        bincv::TernaryMat<WordType> dx(width, height);
        bincv::derivativeX(src, dx);
        const int c = width / 2;
        DERIV_EXPECT(dx.at(1, c - 1) == +1 && dx.at(1, c) == +1,
                     "a rising step gives a POSITIVE derivative (correlation, not convolution)",
                     std::string(wordName) + ": " + std::to_string(dx.at(1, c - 1)) + ", " +
                         std::to_string(dx.at(1, c)));

        // Vertically the frame has to be tall enough that the step is not itself
        // at the border, where reflect-101 forces a zero whatever the content is.
        const int tall = 8;
        const Frame fv = makeStepFrame(width, tall, 1u, true);
        bincv::BinMat<WordType> srcv(width, tall);
        loadFrame(fv, srcv);
        bincv::TernaryMat<WordType> dy(width, tall);
        bincv::derivativeY(srcv, dy);
        const int r = tall / 2;
        DERIV_EXPECT(dy.at(r - 1, 3) == +1 && dy.at(r, 3) == +1,
                     "the same on the vertical axis: the +1 tap is the row BELOW",
                     std::string(wordName) + ": " + std::to_string(dy.at(r - 1, 3)) + ", " +
                         std::to_string(dy.at(r, 3)));
    }

    // width == 1 and height == 1: impl::borderIndex answers 0 for both reflect
    // flavours, so both taps read the only pixel and the derivative is 0. That is
    // cv::filter2D's answer on a 1x1 image too (checked in the OpenCV half).
    {
        bincv::BinMat<WordType> one(1, 1);
        one.set(0, 0, true);
        bincv::TernaryMat<WordType> dx(1, 1);
        bincv::TernaryMat<WordType> dy(1, 1);
        bincv::derivativeX(one, dx);
        bincv::derivativeY(one, dy);
        DERIV_EXPECT(dx.at(0, 0) == 0 && dy.at(0, 0) == 0,
                     "a 1x1 image has a zero derivative under reflect-101",
                     std::string(wordName));
    }
}

/// @brief Empty views are a no-op, not an error.
template <typename WordType>
void checkDegenerate(const char* wordName) {
    bincv::BinMatConstView<WordType> src[1] = {bincv::BinMatConstView<WordType>{}};
    bincv::BinMatView<WordType> mag[1] = {bincv::BinMatView<WordType>{}};
    bincv::BinMatView<WordType> sign{};
    bincv::derivativeX<1, WordType>(src, mag, sign);
    bincv::derivativeY<1, WordType>(src, mag, sign);
    DERIV_EXPECT(true, "empty views are a no-op", std::string(wordName));

    // A zero-height image with a real width, and a zero-width one with a real
    // height: both reach the kernel's early return by a different field.
    bincv::BinMat<WordType> zeroRows(17, 0);
    bincv::TernaryMat<WordType> outRows(17, 0);
    bincv::derivativeX(zeroRows, outRows);
    bincv::derivativeY(zeroRows, outRows);
    DERIV_EXPECT(outRows.empty(), "a zero-height image is a no-op", std::string(wordName));
}

// ---------------------------------------------------------------------------
// The OpenCV half: the reference's own calls
// ---------------------------------------------------------------------------

#ifdef BINCV_WITH_OPENCV

/// @brief calcBinarizedDeriv, PORTED -- its kernels, its ddepth, its scale.
/// @note Not a reimplementation. The point of porting rather than paraphrasing is
///       that the border and the correlation direction come from cv::filter2D
///       itself, so binCV cannot agree with a misreading of them.
void portedCalcBinarizedDeriv(const cv::Mat& src, cv::Mat& binarizedX, cv::Mat& binarizedY) {
    const int ddepth = CV_16S;
    cv::Mat kernelX = (cv::Mat_<int>(1, 3) << -1, 0, 1);
    cv::Mat kernelY = (cv::Mat_<int>(3, 1) << -1, 0, 1);
    cv::filter2D(src, binarizedX, ddepth, kernelX);
    cv::filter2D(src, binarizedY, ddepth, kernelY);
    const int scaleFactor = 16;
    binarizedX *= scaleFactor;
    binarizedY *= scaleFactor;
}

cv::Mat toCv8U(const Frame& f, int scale) {
    cv::Mat m(f.height, f.width, CV_8U);
    for (int y = 0; y < f.height; ++y) {
        for (int x = 0; x < f.width; ++x) {
            m.at<uchar>(y, x) = static_cast<uchar>(f.at(y, x) * static_cast<unsigned>(scale));
        }
    }
    return m;
}

/// @brief binCV against the ported reference at every pixel, borders included.
/// @note THE SCALE FACTOR IS DIVIDED OUT AND THE DIVISION IS REQUIRED TO BE
///       EXACT. 255 (the reference's "white") times 16 is 4080, so the reference's
///       only possible values are {-4080, 0, +4080}; if any other ever appeared,
///       the exactness check would fail and the "representational, not semantic"
///       claim in ops/derivative.hpp would be false rather than merely
///       unsupported.
template <typename WordType>
void sweepAgainstReference(const char* wordName) {
    size_t index = 0;
    for (int height : sweepHeights()) {
        for (int width : sweepWidths()) {
            const Frame f = makeFrame(width, height, 1u, caseSeed(width, height, 1, index++ + 555));
            const cv::Mat src8 = toCv8U(f, 255);
            cv::Mat refX, refY;
            portedCalcBinarizedDeriv(src8, refX, refY);

            bincv::BinMat<WordType> src(width, height);
            loadFrame(f, src);
            bincv::TernaryMat<WordType> dx(width, height);
            bincv::TernaryMat<WordType> dy(width, height);
            bincv::derivativeX(src, dx);
            bincv::derivativeY(src, dy);

            int bad = 0;
            int inexact = 0;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const int rx = refX.at<short>(y, x);
                    const int ry = refY.at<short>(y, x);
                    if (rx % 4080 != 0 || ry % 4080 != 0) ++inexact;
                    if (rx / 4080 != dx.at(y, x)) ++bad;
                    if (ry / 4080 != dy.at(y, x)) ++bad;
                }
            }
            DERIV_EXPECT(bad == 0 && inexact == 0,
                         "binCV equals the ported calcBinarizedDeriv, scale divided out",
                         caseLabel(wordName, 1, "dx+dy", "reference", width, height) + ": " +
                             std::to_string(bad) + " mismatches, " + std::to_string(inexact) +
                             " values not a multiple of 4080");
        }
    }
}

/// @brief The same comparison for an N-BIT source, where no scale factor applies.
/// @note cv::filter2D on a CV_8U image holding the pixel VALUES 0..2^N-1 produces
///       exactly the N-bit derivative into CV_16S. There is no reference
///       implementation for the N-bit case -- the reference pipeline never
///       binarizes above level 0 -- so cv::filter2D itself is the denominator,
///       which is also what makes the border and the correlation direction the
///       same ones the level-0 comparison is judged against.
template <size_t N, typename WordType>
void sweepAgainstFilter2D(const char* wordName) {
    size_t index = 0;
    for (int height : sweepHeights()) {
        for (int width : sweepWidths()) {
            const Frame f = makeFrame(width, height, (1u << N) - 1u,
                                      caseSeed(width, height, N, index++ + 777));
            const cv::Mat src8 = toCv8U(f, 1);
            cv::Mat refX, refY;
            cv::Mat kernelX = (cv::Mat_<int>(1, 3) << -1, 0, 1);
            cv::Mat kernelY = (cv::Mat_<int>(3, 1) << -1, 0, 1);
            cv::filter2D(src8, refX, CV_16S, kernelX);
            cv::filter2D(src8, refY, CV_16S, kernelY);

            bincv::QuantMat<N, WordType> src(width, height);
            loadFrame(f, src);
            bincv::SignedQuantMat<N, WordType> dx(width, height);
            bincv::SignedQuantMat<N, WordType> dy(width, height);
            bincv::derivativeX(src, dx);
            bincv::derivativeY(src, dy);

            int bad = 0;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    if (refX.at<short>(y, x) != dx.at(y, x)) ++bad;
                    if (refY.at<short>(y, x) != dy.at(y, x)) ++bad;
                }
            }
            DERIV_EXPECT(bad == 0, "binCV equals cv::filter2D on an N-bit source",
                         caseLabel(wordName, N, "dx+dy", "filter2D", width, height) + ": " +
                             std::to_string(bad) + " mismatches");
        }
    }
}

#endif // BINCV_WITH_OPENCV

} // namespace

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

BINCV_TEST(Derivative, Stages) {
    // "Linear in N" as a number, next to the exponential route it replaces --
    // ops/pyramid.hpp keeps the same pair for the box sum, for the same reason.
    static_assert(bincv::derivativeAdderStages(1) == 2, "");
    static_assert(bincv::derivativeAdderStages(3) == 6, "");
    static_assert(bincv::derivativeAdderStages(8) == 16, "");
    static_assert(bincv::derivativeReplicatedInputs(1) == 2, "");
    static_assert(bincv::derivativeReplicatedInputs(4) == 30, "");
    static_assert(bincv::derivativeReplicatedInputs(8) == 510, "");
    for (size_t n = 1; n <= 8; ++n) {
        BINCV_CHECK(bincv::derivativeAdderStages(n) <= bincv::derivativeReplicatedInputs(n));
    }
    // THE DOMAIN, pinned. derivativeReplicatedInputs is a public constexpr taking
    // a size_t, and its body is `size_t{1} << n`, which is UNDEFINED at
    // n >= 64 rather than ill-formed -- so before the guard it returned 0 for
    // n = 64 and 137438953470 for n = 100 under -fsanitize=undefined, and
    // -Wconversion cannot see it. The saturation is the only value that keeps
    // "the replication route is worse" true out of domain, so it is what is
    // checked. Both branches of the guard are exercised, because a guard that is
    // never taken and a guard that is always taken fail the same way.
    static_assert(bincv::derivativeReplicatedInputs(63) ==
                      2 * ((size_t{1} << 63) - 1),
                  "the last in-domain n must still compute the real value");
    static_assert(bincv::derivativeReplicatedInputs(sizeof(size_t) * 8) ==
                      static_cast<size_t>(-1),
                  "out of domain must saturate rather than shift by the word width");
    for (size_t n = sizeof(size_t) * 8; n <= sizeof(size_t) * 8 + 40; ++n) {
        BINCV_CHECK(bincv::derivativeReplicatedInputs(n) == static_cast<size_t>(-1));
        BINCV_CHECK(bincv::derivativeAdderStages(8) < bincv::derivativeReplicatedInputs(n));
    }
    // The destination width is exactly right rather than rounded up: the
    // difference of two N-bit values needs N magnitude planes and one sign.
    static_assert(bincv::SignedQuantMat<1, uint32_t>::Planes == 2, "");
    static_assert(bincv::SignedQuantMat<3, uint32_t>::MaxMagnitude == 7, "");
    static_assert(bincv::QuantMat<3, uint32_t>::MaxValue ==
                      bincv::SignedQuantMat<3, uint32_t>::MaxMagnitude,
                  "");
    BINCV_CHECK(true);
}

#define DERIVATIVE_WORD_CASES(WordType)                                                   \
    BINCV_TEST(Derivative, Reference1_##WordType) { sweepReference<1, WordType>(#WordType); } \
    BINCV_TEST(Derivative, Reference2_##WordType) { sweepReference<2, WordType>(#WordType); } \
    BINCV_TEST(Derivative, Reference3_##WordType) { sweepReference<3, WordType>(#WordType); } \
    BINCV_TEST(Derivative, RoutesAgree_##WordType) { sweepRoutesAgree<WordType>(#WordType); }  \
    BINCV_TEST(Derivative, Composed_##WordType) { sweepComposed<WordType>(#WordType); }        \
    BINCV_TEST(Derivative, DirtyPadding1_##WordType) { sweepDirtyPadding<1, WordType>(#WordType); } \
    BINCV_TEST(Derivative, DirtyPadding3_##WordType) { sweepDirtyPadding<3, WordType>(#WordType); } \
    BINCV_TEST(Derivative, Strides_##WordType) {                                          \
        sweepStrides<1, WordType>(#WordType);                                             \
        sweepStrides<2, WordType>(#WordType);                                             \
    }                                                                                     \
    BINCV_TEST(Derivative, Borders_##WordType) { checkBorderIdentities<WordType>(#WordType); } \
    BINCV_TEST(Derivative, Degenerate_##WordType) { checkDegenerate<WordType>(#WordType); }

DERIVATIVE_WORD_CASES(uint8_t)
DERIVATIVE_WORD_CASES(uint16_t)
DERIVATIVE_WORD_CASES(uint32_t)
DERIVATIVE_WORD_CASES(uint64_t)

#ifdef BINCV_WITH_OPENCV

BINCV_TEST(Derivative, OpenCvFilter2D_Direction) {
    // filter2D CORRELATES. A 1x8 row stepping 0 -> 255 between columns 3 and 4
    // gives dx(3) = src(4) - src(2) = +255; a convolution predicts -255. This
    // check is the ONLY guard on the direction: the inversion negates both
    // derivatives, and T3.6's covariance -- cross term included -- is invariant
    // under that, so no downstream test can see it (test_covariance.cpp pins the
    // invariance).
    cv::Mat src = cv::Mat::zeros(1, 8, CV_8U);
    for (int x = 4; x < 8; ++x) src.at<uchar>(0, x) = 255;
    cv::Mat kernelX = (cv::Mat_<int>(1, 3) << -1, 0, 1);
    cv::Mat dx;
    cv::filter2D(src, dx, CV_16S, kernelX);
    BINCV_CHECK_EQ(static_cast<int>(dx.at<short>(0, 3)), 255);
    BINCV_CHECK_EQ(static_cast<int>(dx.at<short>(0, 4)), 255);
    BINCV_CHECK_EQ(static_cast<int>(dx.at<short>(0, 2)), 0);

    // The same for the 3x1 kernel: the +1 tap is the row BELOW.
    cv::Mat srcV = cv::Mat::zeros(8, 1, CV_8U);
    for (int y = 4; y < 8; ++y) srcV.at<uchar>(y, 0) = 255;
    cv::Mat kernelY = (cv::Mat_<int>(3, 1) << -1, 0, 1);
    cv::Mat dy;
    cv::filter2D(srcV, dy, CV_16S, kernelY);
    BINCV_CHECK_EQ(static_cast<int>(dy.at<short>(3, 0)), 255);
}

BINCV_TEST(Derivative, OpenCvFilter2D_BorderDefault) {
    // The default border is BORDER_REFLECT_101, not zero. A single set pixel at
    // column 1 separates the two: reflect-101 gives dx(0) = src(1) - src(1) = 0,
    // a zero fill gives dx(0) = src(1) - 0 = +255.
    cv::Mat src = cv::Mat::zeros(1, 8, CV_8U);
    src.at<uchar>(0, 1) = 255;
    cv::Mat kernelX = (cv::Mat_<int>(1, 3) << -1, 0, 1);
    cv::Mat defaulted, constant, reflect101, replicate;
    cv::filter2D(src, defaulted, CV_16S, kernelX);
    cv::filter2D(src, constant, CV_16S, kernelX, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(src, reflect101, CV_16S, kernelX, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
    cv::filter2D(src, replicate, CV_16S, kernelX, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    BINCV_CHECK_EQ(static_cast<int>(defaulted.at<short>(0, 0)), 0);
    BINCV_CHECK_EQ(static_cast<int>(reflect101.at<short>(0, 0)), 0);
    BINCV_CHECK_EQ(static_cast<int>(constant.at<short>(0, 0)), 255);
    BINCV_CHECK_EQ(static_cast<int>(replicate.at<short>(0, 0)), 255);
    // ...and binCV's BORDER_REFLECT_101 is the same enumerator value OpenCV's is,
    // which is what lets a caller pass one through to the other unchanged.
    BINCV_CHECK_EQ(static_cast<int>(bincv::BORDER_REFLECT_101),
                   static_cast<int>(cv::BORDER_REFLECT_101));
    BINCV_CHECK_EQ(static_cast<int>(cv::BORDER_DEFAULT), static_cast<int>(cv::BORDER_REFLECT_101));

    // A 1x1 image: both taps resolve to the only pixel, so the answer is 0.
    cv::Mat one = cv::Mat::ones(1, 1, CV_8U) * 255;
    cv::Mat oneDx;
    cv::filter2D(one, oneDx, CV_16S, kernelX);
    BINCV_CHECK_EQ(static_cast<int>(oneDx.at<short>(0, 0)), 0);
}

BINCV_TEST(Derivative, Reference_uint32_t_OpenCv) { sweepAgainstReference<uint32_t>("uint32_t"); }
BINCV_TEST(Derivative, Reference_uint64_t_OpenCv) { sweepAgainstReference<uint64_t>("uint64_t"); }
BINCV_TEST(Derivative, Filter2D_uint32_t_OpenCv) {
    sweepAgainstFilter2D<2, uint32_t>("uint32_t");
    sweepAgainstFilter2D<3, uint32_t>("uint32_t");
}
BINCV_TEST(Derivative, Filter2D_uint8_t_OpenCv) {
    sweepAgainstFilter2D<2, uint8_t>("uint8_t");
    sweepAgainstFilter2D<3, uint8_t>("uint8_t");
}

#endif // BINCV_WITH_OPENCV

BINCV_TEST_MAIN("test_derivative")
