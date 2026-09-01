// Shift kernels ( horizontal, vertical and borders).
//
// TWO HALVES, exactly as tests/test_logic.cpp has:
//
// 1. The CORE half (everything above the OpenCV guard) needs no OpenCV, so it
// runs in all four verification configurations -- including Debug, the only
// one where the kernels' BINCV_ASSERT preconditions are live, and
// -fno-exceptions, which is the embedded claim. It checks the kernels against
// a NAIVE PER-PIXEL REFERENCE, over differing strides, at the degenerate
// sizes, against sources whose padding bits are dirty, and against the
// padding-bit invariant that no pixel comparison can see.
//
// 2. The OPENCV half asserts what the border semantics actually promise:
// the same mapping cv::borderInterpolate computes, and the same images
// cv::copyMakeBorder produces, over that work’s size matrix at all four word
// widths. That is what makes morphology bit-exact later.
//
// THE ORACLE IS THE PER-PIXEL REFERENCE, AND IT SHARES NO CODE WITH THE KERNEL.
//
// refPixel below indexes the source one pixel at a time and knows nothing about
// words, carries or masks -- so a mistake in the word recurrence cannot cancel
// through it. Its border mapping is written as OpenCV's documented do-while loop,
// where impl::borderIndex is a closed-form modulo: two different algorithms for
// one function, compared against each other over a wide range of coordinates by
// Shift.BorderIndex_Reference, and both compared against cv::borderInterpolate in
// the OpenCV half. A shared border helper would have made the border half of this
// suite a tautology, which is exactly what tests/equivalence.hpp property 2 warns
// about for the conversion path.
//
// WHY THE CHECK COUNT IS NOT ONE PER PIXEL: a swept case reports its DISAGREEMENT
// COUNT as a single check, so CHECKS tracks cases and a failure still says how
// badly it failed.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/shift.hpp"
#include "test_util.hpp"

// OpenCV-only, and self-guarding: everything in it sits behind BINCV_WITH_OPENCV,
// so this include is a no-op in the three configurations that have no OpenCV.
#include "equivalence.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#if __has_include(<opencv2/imgproc.hpp>)
#define BINCV_TEST_HAVE_IMGPROC 1
#include <opencv2/imgproc.hpp>
#else
#define BINCV_TEST_HAVE_IMGPROC 0
#endif
#endif

namespace {

using bincv::BORDER_CONSTANT;
using bincv::BORDER_REFLECT;
using bincv::BORDER_REFLECT_101;
using bincv::BORDER_REPLICATE;
using bincv::BORDER_WRAP;
using bincv::BorderType;

// ---------------------------------------------------------------------------
// The border types, as data
// ---------------------------------------------------------------------------

const BorderType ALL_BORDERS[] = {BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,
                                  BORDER_REFLECT_101, BORDER_WRAP};

const char* borderName(BorderType t) {
    switch (t) {
        case BORDER_CONSTANT:    return "constant";
        case BORDER_REPLICATE:   return "replicate";
        case BORDER_REFLECT:     return "reflect";
        case BORDER_REFLECT_101: return "reflect101";
        case BORDER_WRAP:        return "wrap";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// The reference border mapping: OpenCV's do-while, not the library's closed form
// ---------------------------------------------------------------------------
//
// Transcribed from cv::borderInterpolate's documented algorithm, deliberately in
// its ITERATIVE shape. impl::borderIndex reaches the same fixed point with one
// signed modulo into the pattern's period, so the two agree only if the closed
// form is right -- which is the whole reason this is not a call into the library.
//
// -1 means "outside, and the border is a constant", exactly as OpenCV returns.

long refBorderIndex(long p, long len, BorderType type) {
    if (p >= 0 && p < len) return p;
    switch (type) {
        case BORDER_REPLICATE:
            return p < 0 ? 0 : len - 1;

        case BORDER_REFLECT:
        case BORDER_REFLECT_101: {
            if (len == 1) return 0;
            const long delta = (type == BORDER_REFLECT_101) ? 1 : 0;
            do {
                if (p < 0) {
                    p = -p - 1 + delta;
                } else {
                    p = len - 1 - (p - len) - delta;
                }
            } while (p < 0 || p >= len);
            return p;
        }

        case BORDER_WRAP: {
            while (p < 0) p += len;
            while (p >= len) p -= len;
            return p;
        }

        case BORDER_CONSTANT:
            break;
    }
    return -1;
}

/// @brief What dst(y, x) must hold after shift(src, dst, dx, dy, type, value).
/// @note One pixel at a time, through at. No word, no mask, no carry -- which is
/// what makes it an independent verdict on the packed implementation.
template <typename WordType>
bool refPixel(const bincv::BinMat<WordType>& src, long y, long x, long dx, long dy,
              BorderType type, bool value) {
    const long sy = refBorderIndex(y + dy, static_cast<long>(src.rows()), type);
    const long sx = refBorderIndex(x + dx, static_cast<long>(src.cols()), type);
    if (sy < 0 || sx < 0) return value;
    return src.at(static_cast<int>(sy), static_cast<int>(sx));
}

/// @brief Pixels on which the kernel's result differs from the reference.
template <typename WordType>
int disagreements(const bincv::BinMat<WordType>& src, const bincv::BinMat<WordType>& dst,
                  long dx, long dy, BorderType type, bool value) {
    int differing = 0;
    for (int y = 0; y < dst.rows(); ++y) {
        for (int x = 0; x < dst.cols(); ++x) {
            if (dst.at(y, x) != refPixel(src, static_cast<long>(y), static_cast<long>(x), dx, dy,
                                         type, value)) {
                ++differing;
            }
        }
    }
    return differing;
}

// ---------------------------------------------------------------------------
// Content: the same generator as tests/equivalence.hpp, minus OpenCV
// ---------------------------------------------------------------------------
//
// Duplicated rather than shared for the reason tests/test_logic.cpp gives:
// randomBinary lives behind BINCV_WITH_OPENCV, and three of the four
// configurations this suite runs in have no OpenCV. Same SplitMix64, same
// threshold mapping, so a case that fails here reproduces there.

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
    const double rounded =
        static_cast<double>(fillRatio) * static_cast<double>(FILL_SCALE) + 0.5;
    if (rounded >= static_cast<double>(FILL_SCALE)) return FILL_SCALE;
    return static_cast<uint32_t>(rounded);
}

/// @brief Fills a matrix through set, so its padding bits stay clear.
template <typename WordType>
void fillRandom(bincv::BinMat<WordType>& m, float fillRatio, uint64_t seed) {
    uint64_t state = seed;
    const uint32_t threshold = fillThreshold(fillRatio);
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            m.set(y, x, static_cast<uint32_t>(nextRandom(state) >> 40) < threshold);
        }
    }
}

uint64_t caseSeed(int width, int height, size_t index) {
    return UINT64_C(0x5417E0000000003) + static_cast<uint64_t>(width) * UINT64_C(1000003) +
           static_cast<uint64_t>(height) * UINT64_C(10007) + static_cast<uint64_t>(index);
}

// ---------------------------------------------------------------------------
// Observers
// ---------------------------------------------------------------------------

/// @brief Set bits across the whole STRIDE, padding included.
/// @note Not a library operation -- binCV exposes no per-word popcount.
/// Comparing it against countNonZero's per-pixel loop is how a padding-bit
/// violation becomes visible: they agree only when every bit past `width` is
/// zero. A shift with a border value of `true` sets every padding bit of the
/// trailing word before the mask removes them, so this is the check that
/// fails first if the mask is dropped.
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

std::string label(const char* wordTypeName, int width, int height, long dx, long dy,
                  BorderType type, bool value, const std::string& extra) {
    return std::string(wordTypeName) + " " + std::to_string(width) + "x" +
           std::to_string(height) + " d=(" + std::to_string(dx) + "," + std::to_string(dy) +
           ") " + borderName(type) + (value ? "/1" : "/0") + " " + extra;
}

/// @brief Runs a pure horizontal or vertical shift through its DIRECTIONAL entry
/// point, and anything else through shift.
/// @note The directional wrappers are where a sign error hides: they are four
/// one-line calls into shift and each one can have its direction backwards
/// without any word arithmetic being wrong. Routing the axis-aligned sweeps
/// through them is what makes that a test failure.
template <typename WordType>
void runShift(const bincv::BinMat<WordType>& src, bincv::BinMat<WordType>& dst, long dx,
              long dy, BorderType type, bool value) {
    if (dy == 0 && dx >= 0) {
        bincv::shiftLeft(src.constView(), dst.view(), static_cast<size_t>(dx), type, value);
    } else if (dy == 0) {
        bincv::shiftRight(src.constView(), dst.view(), static_cast<size_t>(-dx), type, value);
    } else if (dx == 0 && dy > 0) {
        bincv::shiftUp(src.constView(), dst.view(), static_cast<size_t>(dy), type, value);
    } else if (dx == 0) {
        bincv::shiftDown(src.constView(), dst.view(), static_cast<size_t>(-dy), type, value);
    } else {
        bincv::shift(src.constView(), dst.view(), static_cast<ptrdiff_t>(dx),
                     static_cast<ptrdiff_t>(dy), type, value);
    }
}

/// @brief One swept case: run it, compare against the reference, check the padding.
template <typename WordType>
void checkOneCase(const char* wordTypeName, const bincv::BinMat<WordType>& src,
                  bincv::BinMat<WordType>& dst, long dx, long dy, BorderType type, bool value,
                  const char* extra) {
    runShift(src, dst, dx, dy, type, value);

    ::bincv::test::reportCheck(
        disagreements(src, dst, dx, dy, type, value) == 0, "matches the per-pixel reference",
        __FILE__, __LINE__,
        label(wordTypeName, dst.cols(), dst.rows(), dx, dy, type, value, extra));

    // The invariant a pixel comparison is blind to. With borderValue == true the
    // fill writes ones into the trailing word's padding bits, so this is not a
    // formality here the way it is for AND.
    ::bincv::test::reportCheck(
        bitsAcrossStride(dst) == dst.countNonZero(), "padding bits are zero past width",
        __FILE__, __LINE__,
        label(wordTypeName, dst.cols(), dst.rows(), dx, dy, type, value, extra));
}

// ---------------------------------------------------------------------------
// The sweep matrix
// ---------------------------------------------------------------------------
//
// The widths. All but 640 are non-multiples of at least one supported word
// width, which is where the cross-word carry and the trailing-word mask meet.

const int WIDTHS[] = {1, 7, 31, 33, 40, 63, 65, 70};
const float FILLS[] = {0.0f, 0.01f, 0.5f, 0.99f, 1.0f};

// An over-aligned row stride (the design rule makes alignment a per-object choice): 32 bytes
// is a whole number of 1-, 2-, 4- and 8-byte words, so every word type gets a
// stride strictly larger than the ceil(width / WordBits) words its rows need.
constexpr size_t PADDED_ALIGNMENT = 32;

// ===========================================================================
// 1. The k sweep: 0 through 2 * WordBits + 1, both directions
// ===========================================================================
//
// The range names, and it is chosen to straddle every boundary the
// recurrence has: k == 0 and k == WordBits and k == 2 * WordBits are the
// bitShift == 0 cases, where `x << (WordBits - bitShift)` would be a shift by
// WordBits -- undefined behavior, and on x86 the natural encoding masks the count
// and returns `x` instead of 0, so the bug is invisible at every other k.

template <typename WordType>
void testHorizontalSweep(const char* wordTypeName) {
    std::cout << "\n--- shift: k from 0 to 2*WordBits+1, both directions: " << wordTypeName
              << " ---\n";

    constexpr size_t wordBits = bincv::BinMat<WordType>::WordBits;

    for (int width : WIDTHS) {
        for (int height : {1, 3}) {
            bincv::BinMat<WordType> src(width, height);
            fillRandom(src, 0.5f, caseSeed(width, height, 1));

            for (size_t k = 0; k <= 2 * wordBits + 1; ++k) {
                const long kk = static_cast<long>(k);
                bincv::BinMat<WordType> dst(width, height);
                checkOneCase(wordTypeName, src, dst, kk, 0, BORDER_CONSTANT, false, "left");
                bincv::BinMat<WordType> dst2(width, height);
                checkOneCase(wordTypeName, src, dst2, -kk, 0, BORDER_CONSTANT, false, "right");
            }
        }
    }
}

// ===========================================================================
// 2. Wide rows: the word-multiple widths, and k beyond the image
// ===========================================================================

template <typename WordType>
void testWideRows(const char* wordTypeName) {
    std::cout << "\n--- shift over wide rows: " << wordTypeName << " ---\n";

    constexpr long wordBits = static_cast<long>(bincv::BinMat<WordType>::WordBits);

    for (int width : {128, 640}) {
        for (int height : {1, 17}) {
            bincv::BinMat<WordType> src(width, height);
            fillRandom(src, 0.5f, caseSeed(width, height, 2));

            const long ks[] = {0,
                               1,
                               7,
                               wordBits - 1,
                               wordBits,
                               wordBits + 1,
                               2 * wordBits,
                               2 * wordBits + 1,
                               63,
                               64,
                               65,
                               static_cast<long>(width) - 1,
                               static_cast<long>(width),
                               static_cast<long>(width) + 1,
                               2 * static_cast<long>(width)};

            for (long k : ks) {
                bincv::BinMat<WordType> dst(width, height);
                checkOneCase(wordTypeName, src, dst, k, 0, BORDER_CONSTANT, false, "left");
                bincv::BinMat<WordType> dst2(width, height);
                checkOneCase(wordTypeName, src, dst2, -k, 0, BORDER_CONSTANT, false, "right");
            }
        }
    }
}

// ===========================================================================
// 3. Vertical shifts, including offsets that exceed the height
// ===========================================================================

template <typename WordType>
void testVertical(const char* wordTypeName) {
    std::cout << "\n--- vertical shift, offsets past the height: " << wordTypeName << " ---\n";

    for (int width : {7, 33, 70}) {
        for (int height : {1, 2, 3, 17}) {
            bincv::BinMat<WordType> src(width, height);
            fillRandom(src, 0.5f, caseSeed(width, height, 3));

            // 0.. 2*height+1 covers every in-range offset and the first few past
            // it; 3*height+2 is well past, which is where a clamp that should have
            // been a modulo (BORDER_WRAP) stops agreeing with a reflection.
            std::vector<long> ks;
            for (long k = 0; k <= 2 * static_cast<long>(height) + 1; ++k) ks.push_back(k);
            ks.push_back(3 * static_cast<long>(height) + 2);

            for (long k : ks) {
                bincv::BinMat<WordType> dst(width, height);
                checkOneCase(wordTypeName, src, dst, 0, k, BORDER_CONSTANT, false, "up");
                bincv::BinMat<WordType> dst2(width, height);
                checkOneCase(wordTypeName, src, dst2, 0, -k, BORDER_CONSTANT, false, "down");
            }
        }
    }
}

// ===========================================================================
// 4. Every BorderType, both fill values, in both axes at once
// ===========================================================================
//
// Against the per-pixel reference here; against cv::copyMakeBorder in the OpenCV
// half. Both are needed: this one runs in the three configurations that have no
// OpenCV, and OpenCV is the only thing that can say the mapping is OpenCV's.
//
// borderValue is swept for EVERY type, not only for BORDER_CONSTANT, because
// "the other four ignore it" is itself a property -- and it is OpenCV's, whose
// borderValue argument is documented as used with BORDER_CONSTANT alone.

template <typename WordType>
void testBorders(const char* wordTypeName) {
    std::cout << "\n--- shift with every BorderType: " << wordTypeName << " ---\n";

    constexpr long wordBits = static_cast<long>(bincv::BinMat<WordType>::WordBits);

    for (int width : {1, 7, 33, 65}) {
        for (int height : {1, 3, 5}) {
            bincv::BinMat<WordType> src(width, height);
            fillRandom(src, 0.5f, caseSeed(width, height, 4));

            const long offsets[][2] = {
                {0, 0},
                {1, 0},   {-1, 0},   {0, 1},   {0, -1},
                {1, 1},   {-1, -1},  {1, -1},  {-1, 1},
                {wordBits - 1, 0},   {wordBits, 0},   {wordBits + 1, 0},   {-wordBits, 0},
                {0, 3},   {0, -3},
                {static_cast<long>(width) + 3, 0},
                {0, static_cast<long>(height) + 3},
                {3, -2},
            };

            for (const auto& off : offsets) {
                for (BorderType type : ALL_BORDERS) {
                    for (int v = 0; v < 2; ++v) {
                        bincv::BinMat<WordType> dst(width, height);
                        checkOneCase(wordTypeName, src, dst, off[0], off[1], type, v != 0,
                                     "borders");
                    }
                }
            }
        }
    }
}

// ===========================================================================
// 5. Degenerate shapes: empty, and 1x1
// ===========================================================================

template <typename WordType>
void testDegenerate(const char* wordTypeName) {
    std::cout << "\n--- shift at degenerate sizes: " << wordTypeName << " ---\n";

    // Empty is a no-op, not an error. All three shapes of empty, since a kernel
    // could plausibly guard one and not the others -- and an empty image has no
    // border to interpolate, so a shift that reached borderIndex would divide by
    // zero or spin.
    const int emptyShapes[][2] = {{0, 0}, {0, 5}, {5, 0}};
    for (const auto& shape : emptyShapes) {
        bincv::BinMat<WordType> src(shape[0], shape[1]);
        bincv::BinMat<WordType> dst(shape[0], shape[1]);
        for (BorderType type : ALL_BORDERS) {
            bincv::shift(src.constView(), dst.view(), 3, -2, type, true);
            bincv::shiftLeft(src.constView(), dst.view(), 1, type, true);
            bincv::shiftUp(src.constView(), dst.view(), 1, type, true);
            BINCV_CHECK(dst.empty());
            BINCV_CHECK_EQ(dst.countNonZero(), 0);
        }
    }

    // An empty DESTINATION must not be written through even when its buffer
    // exists: a 5x0 matrix wrapping a live buffer is the shape a kernel that
    // trusted `width` alone would scribble on.
    {
        std::vector<WordType> buffer(4, static_cast<WordType>(0x2Du));
        bincv::BinMat<WordType> dst(buffer.data(), 5, 0, 1);
        bincv::BinMat<WordType> src(buffer.data(), 5, 0, 1);
        bincv::shift(src.constView(), dst.view(), 1, 1, BORDER_REPLICATE, true);
        bool untouched = true;
        for (WordType w : buffer) untouched = untouched && (w == static_cast<WordType>(0x2Du));
        BINCV_CHECK(untouched);
    }

    // 1x1: one live bit and WordBits - 1 padding bits, which is where the tail
    // mask is at its most aggressive. It is also the only shape where
    // BORDER_REFLECT_101 has no room to reflect -- OpenCV answers 0, and the
    // closed form would divide by a period of zero if it did not say so first.
    for (int pixel = 0; pixel < 2; ++pixel) {
        bincv::BinMat<WordType> src(1, 1);
        src.set(0, 0, pixel != 0);
        for (BorderType type : ALL_BORDERS) {
            for (int v = 0; v < 2; ++v) {
                for (long d = 0; d <= 2; ++d) {
                    bincv::BinMat<WordType> dst(1, 1);
                    checkOneCase(wordTypeName, src, dst, d, 0, type, v != 0, "1x1 h");
                    bincv::BinMat<WordType> dst2(1, 1);
                    checkOneCase(wordTypeName, src, dst2, 0, -d, type, v != 0, "1x1 v");
                }
            }
        }
    }

    std::cout << " " << wordTypeName << ": empty and 1x1 shapes handled\n";
}

// ===========================================================================
// 6. Differing strides between src and dst
// ===========================================================================
//
// The failure this exists to catch: a kernel that walks its arguments as one dense
// run is correct whenever both were built the same way, and wrong the moment one
// is over-aligned or wraps a caller's buffer with its own stride.

enum class Stride { Tight, Padded, Odd };

const char* strideName(Stride s) {
    switch (s) {
        case Stride::Tight:  return "tight";
        case Stride::Padded: return "padded";
        case Stride::Odd:    return "odd";
    }
    return "?";
}

/// @brief A matrix at the requested stride flavour, plus the buffer behind it.
template <typename WordType>
struct StridedMat {
    std::vector<WordType> buffer;   // used by Stride::Odd only
    bincv::BinMat<WordType> mat;
};

template <typename WordType>
void makeStrided(StridedMat<WordType>& out, Stride flavour, int width, int height) {
    constexpr size_t wordBits = bincv::BinMat<WordType>::WordBits;
    const size_t minWords = (static_cast<size_t>(width) + wordBits - 1) / wordBits;

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
    std::cout << "\n--- shift across differing strides: " << wordTypeName << " ---\n";

    constexpr long wordBits = static_cast<long>(bincv::BinMat<WordType>::WordBits);
    const Stride combos[][2] = {
        {Stride::Tight,  Stride::Padded},
        {Stride::Padded, Stride::Tight},
        {Stride::Odd,    Stride::Tight},
        {Stride::Tight,  Stride::Odd},
        {Stride::Padded, Stride::Odd},
        {Stride::Odd,    Stride::Padded},
    };

    const long offsets[][2] = {{1, 0}, {-1, 0}, {wordBits, 0},     {wordBits + 1, 0},
                               {0, 1}, {0, -2}, {wordBits - 1, 1}, {-3, -1}};

    for (int width : {7, 33, 65, 70}) {
        for (int height : {1, 3, 17}) {
            for (size_t f = 0; f < sizeof(FILLS) / sizeof(FILLS[0]); ++f) {
                for (const auto& combo : combos) {
                    StridedMat<WordType> src, dst;
                    makeStrided(src, combo[0], width, height);
                    fillRandom(src.mat, FILLS[f], caseSeed(width, height, f + 500));

                    for (const auto& off : offsets) {
                        makeStrided(dst, combo[1], width, height);
                        const std::string extra =
                            std::string(strideName(combo[0])) + "/" + strideName(combo[1]);
                        checkOneCase(wordTypeName, src.mat, dst.mat, off[0], off[1],
                                     BORDER_REPLICATE, false, extra.c_str());
                    }
                }
            }
        }
    }
}

// ===========================================================================
// 7. SOURCES WHOSE PADDING BITS ARE ALREADY DIRTY
// ===========================================================================
//
// The case the spec's word recurrence does not cover on its own, and the one
// place a shift differs from a pointwise kernel.
//
// "Out-of-range source WORDS read as zero" is not the whole rule. A row's trailing
// partial word holds pixels in its low bits and PADDING in its high ones, and
// those padding bits sit at column indices past `width` -- outside the image. A
// shiftLeft by k moves them to columns width-k.. width-1, which are live
// destination pixels. BinMat's wrap constructor documents that a caller's padding
// belongs to the caller, so a source with dirty padding is a supported
// construction (tests/test_logic.cpp sweeps the same one).
//
// MEASURED, with the `fill` extension removed from impl::extendedRowWord so that
// the trailing word is read as-is: at width 5, height 3, uint8_t, over an all-ones
// buffer, shiftLeft by 1 reports 3 wrong pixels per row where the border says
// zero. The recurrence alone cannot see it -- both words it reads are in range.

/// @brief A matrix wrapping a buffer whose PADDING bits are all ones.
template <typename WordType>
void makeDirtyPadded(StridedMat<WordType>& out, int width, int height, float fillRatio,
                     uint64_t seed) {
    constexpr size_t wordBits = bincv::BinMat<WordType>::WordBits;
    const size_t minWords = (static_cast<size_t>(width) + wordBits - 1) / wordBits;
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));

    out.buffer.assign(minWords * static_cast<size_t>(height), allOnes);
    out.mat = bincv::BinMat<WordType>(out.buffer.data(), width, height, minWords);

    // Through set, so the pixel bits are the drawn content and every bit past
    // `width` keeps the 1 it was born with.
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
    std::cout << "\n--- shift with dirty source padding: " << wordTypeName << " ---\n";

    constexpr long wordBits = static_cast<long>(bincv::BinMat<WordType>::WordBits);

    for (int width : {1, 5, 7, 31, 33, 63, 65, 70}) {
        for (int height : {1, 3}) {
            for (size_t f = 0; f < sizeof(FILLS) / sizeof(FILLS[0]); ++f) {
                StridedMat<WordType> src;
                makeDirtyPadded(src, width, height, FILLS[f], caseSeed(width, height, f + 600));

                const long ks[] = {1, 2, 3, wordBits - 1, wordBits, wordBits + 1};
                for (long k : ks) {
                    for (BorderType type : ALL_BORDERS) {
                        bincv::BinMat<WordType> dst(width, height);
                        checkOneCase(wordTypeName, src.mat, dst, k, 0, type, false, "dirty left");
                        bincv::BinMat<WordType> dst2(width, height);
                        checkOneCase(wordTypeName, src.mat, dst2, -k, 0, type, true,
                                     "dirty right");
                    }
                }
            }
        }
    }
}

// ===========================================================================
// 8. The kernel writes nothing outside the destination it was given
// ===========================================================================

template <typename WordType>
void testGuardWords(const char* wordTypeName) {
    std::cout << "\n--- shift writes only its destination: " << wordTypeName << " ---\n";

    constexpr size_t wordBits = bincv::BinMat<WordType>::WordBits;
    const WordType sentinel = static_cast<WordType>(0xA5A5A5A5A5A5A5A5ull);

    for (int width : WIDTHS) {
        const int height = 3;
        const size_t minWords = (static_cast<size_t>(width) + wordBits - 1) / wordBits;
        const size_t stride = minWords + 2;          // two padding words per row
        const size_t lead = 2;                       // words before the matrix
        const size_t trail = 2;                      // words after it

        for (int borderValue = 0; borderValue < 2; ++borderValue) {
            std::vector<WordType> buffer(lead + stride * static_cast<size_t>(height) + trail,
                                         sentinel);
            bincv::BinMat<WordType> dst(buffer.data() + lead, width, height, stride);
            // The wrapped destination starts dirty, sentinel bits and all, which is
            // also what makes this a test of the trailing-word mask: the kernel has
            // to CLEAR bits it never sets.
            bincv::BinMat<WordType> src(width, height);
            fillRandom(src, 0.5f, caseSeed(width, height, 700));

            bincv::shift(src.constView(), dst.view(), 3, 1, BORDER_CONSTANT, borderValue != 0);

            bool guardsIntact = true;
            for (size_t i = 0; i < lead; ++i) guardsIntact = guardsIntact && buffer[i] == sentinel;
            for (size_t i = buffer.size() - trail; i < buffer.size(); ++i) {
                guardsIntact = guardsIntact && buffer[i] == sentinel;
            }
            // The words INSIDE the destination's own stride, past the words its
            // pixels need. They belong to the caller; for a destination that
            // windows a taller image they are a neighbour's pixels.
            for (int y = 0; y < height; ++y) {
                for (size_t w = minWords; w < stride; ++w) {
                    guardsIntact =
                        guardsIntact &&
                        buffer[lead + static_cast<size_t>(y) * stride + w] == sentinel;
                }
            }
            ::bincv::test::reportCheck(
                guardsIntact, "guard words outside dst are untouched", __FILE__, __LINE__,
                label(wordTypeName, width, height, 3, 1, BORDER_CONSTANT, borderValue != 0,
                      "guards"));
            ::bincv::test::reportCheck(
                disagreements(src, dst, 3, 1, BORDER_CONSTANT, borderValue != 0) == 0,
                "matches the per-pixel reference", __FILE__, __LINE__,
                label(wordTypeName, width, height, 3, 1, BORDER_CONSTANT, borderValue != 0,
                      "wrapped dst"));

            // The trailing partial word must be clean even though the buffer was
            // handed over dirty. Counted over the PIXEL words only; the stride
            // padding past them is the caller's and is checked above.
            int pixelWordBits = 0;
            for (int y = 0; y < height; ++y) {
                const WordType* row = dst.ptr(y);
                for (size_t w = 0; w < minWords; ++w) {
                    WordType v = row[w];
                    while (v != 0) {
                        pixelWordBits += static_cast<int>(v & static_cast<WordType>(1));
                        v = static_cast<WordType>(v >> 1);
                    }
                }
            }
            ::bincv::test::reportCheck(
                pixelWordBits == dst.countNonZero(), "padding bits are zero past width",
                __FILE__, __LINE__,
                label(wordTypeName, width, height, 3, 1, BORDER_CONSTANT, borderValue != 0,
                      "wrapped dst"));
        }
    }
}

// ===========================================================================
// 9. Views over one buffer that share no word are ACCEPTED
// ===========================================================================
//
// shift.hpp refuses an overlapping destination, and "overlapping" has to mean per
// row rather than per bounding box. Two views over one buffer can interleave
// without sharing a byte -- alternate row bands are what a pyramid downsample
// takes (the design notes), left/right column tiles are how a frame is split
// across a loop -- and the design rule says a kernel takes any {ptr, width, height, stride}.
// The equivalent predicate in ops/logic.hpp got this wrong once and aborted every
// Debug build on a call that was correct in release; this case is what would say
// so if the shared predicate regressed.
//
// Meaningful only in the Debug configuration, where BINCV_ASSERT is live. It runs
// everywhere so that a release build still proves the arithmetic.

template <typename WordType>
void testDisjointViewsOverOneBuffer(const char* wordTypeName) {
    std::cout << "\n--- shift accepts disjoint views over one buffer: " << wordTypeName
              << " ---\n";

    constexpr size_t wordBits = bincv::BinMat<WordType>::WordBits;
    const size_t rowWords = 2;
    const int width = static_cast<int>(2 * wordBits);   // exactly two words, no tail

    // (a) Interleaved ROW BANDS: dst is physical rows 0, 2, 4 and src is rows
    // 1, 3, 5 of one 6-row image. Same stride, half a stride apart.
    {
        const int height = 3;
        std::vector<WordType> buffer(rowWords * 6, static_cast<WordType>(0));
        for (size_t i = 0; i < buffer.size(); ++i) buffer[i] = static_cast<WordType>(0x5Au + i);
        const std::vector<WordType> before = buffer;

        const bincv::BinMatConstView<WordType> src{buffer.data() + rowWords,
                                                   static_cast<size_t>(width),
                                                   static_cast<size_t>(height), 2 * rowWords};
        const bincv::BinMatView<WordType> dst{buffer.data(), static_cast<size_t>(width),
                                              static_cast<size_t>(height), 2 * rowWords};
        bincv::shiftLeft(src, dst, wordBits);           // exactly one word

        bool correct = true;
        for (size_t y = 0; y < static_cast<size_t>(height); ++y) {
            const size_t srcIndex = (2 * y + 1) * rowWords;
            const size_t dstIndex = (2 * y) * rowWords;
            correct = correct && buffer[dstIndex] == before[srcIndex + 1];
            correct = correct && buffer[dstIndex + 1] == static_cast<WordType>(0);
            correct = correct && buffer[srcIndex] == before[srcIndex];          // src untouched
            correct = correct && buffer[srcIndex + 1] == before[srcIndex + 1];
        }
        ::bincv::test::reportCheck(correct, "interleaved row bands are accepted and correct",
                                   __FILE__, __LINE__,
                                   label(wordTypeName, width, height, static_cast<long>(wordBits),
                                         0, BORDER_CONSTANT, false, "row bands"));
    }

    // (b) Left and right COLUMN TILES of one image: same stride, one row apart.
    {
        const int height = 4;
        const size_t stride = 2 * rowWords;
        std::vector<WordType> buffer(stride * static_cast<size_t>(height),
                                     static_cast<WordType>(0));
        for (size_t i = 0; i < buffer.size(); ++i) buffer[i] = static_cast<WordType>(0x33u + i);
        const std::vector<WordType> before = buffer;

        const bincv::BinMatConstView<WordType> src{buffer.data(), static_cast<size_t>(width),
                                                   static_cast<size_t>(height), stride};
        const bincv::BinMatView<WordType> dst{buffer.data() + rowWords,
                                              static_cast<size_t>(width),
                                              static_cast<size_t>(height), stride};
        bincv::shiftRight(src, dst, wordBits);

        bool correct = true;
        for (size_t y = 0; y < static_cast<size_t>(height); ++y) {
            const size_t srcIndex = y * stride;
            const size_t dstIndex = y * stride + rowWords;
            correct = correct && buffer[dstIndex] == static_cast<WordType>(0);
            correct = correct && buffer[dstIndex + 1] == before[srcIndex];
            correct = correct && buffer[srcIndex] == before[srcIndex];
            correct = correct && buffer[srcIndex + 1] == before[srcIndex + 1];
        }
        ::bincv::test::reportCheck(correct, "column tiles are accepted and correct", __FILE__,
                                   __LINE__,
                                   label(wordTypeName, width, height,
                                         -static_cast<long>(wordBits), 0, BORDER_CONSTANT, false,
                                         "column tiles"));
    }
}

// ===========================================================================
// 10. The closed-form border mapping against the iterative one
// ===========================================================================
//
// impl::borderIndex is a modulo into the pattern's period; refBorderIndex above is
// OpenCV's do-while. Two algorithms, one function -- and this is the only check in
// the core half that can catch a closed form that is off by one at exactly one
// coordinate, because everything else in the file sees the mapping only through
// images whose interiors dominate.
//
// Reaches into impl:: deliberately: the mapping is the whole Tier 1 promise, and
// testing it only through the shifts built on it would leave the two reflect
// flavours distinguishable by one column at one edge.

void testBorderIndexAgainstReference() {
    std::cout << "\n--- borderIndex: closed form vs the iterative reference ---\n";

    const long lengths[] = {1, 2, 3, 4, 5, 7, 8, 31, 32, 33, 63, 64, 65, 70, 640};

    for (long len : lengths) {
        for (BorderType type : ALL_BORDERS) {
            int differing = 0;
            long compared = 0;
            for (long p = -3 * len - 3; p <= 3 * len + 3; ++p) {
                const long expected = refBorderIndex(p, len, type);
                const long actual = static_cast<long>(
                    bincv::impl::borderIndex(static_cast<ptrdiff_t>(p),
                                             static_cast<size_t>(len), type));
                if (actual != expected) ++differing;
                ++compared;
            }
            ::bincv::test::reportCheck(
                differing == 0, "borderIndex matches the iterative reference", __FILE__, __LINE__,
                std::string(borderName(type)) + " len=" + std::to_string(len) + ": " +
                    std::to_string(differing) + " of " + std::to_string(compared) +
                    " coordinates differ");
        }
    }
}

// ===========================================================================
// 11. Tier 1: the border mapping and the shifted images, against OpenCV
// ===========================================================================

#ifdef BINCV_WITH_OPENCV

/// @brief impl::borderIndex against cv::borderInterpolate, coordinate by coordinate.
/// @note This is the check that makes "border semantics match OpenCV exactly" a
/// measurement. Everything downstream -- morphology's bit-exactness against
/// cv::erode -- rests on this one mapping, and an image comparison can only
/// see it where the border actually reaches.
void testBorderInterpolateAgainstOpenCv() {
    std::cout << "\n--- borderIndex vs cv::borderInterpolate ---\n";

    const int lengths[] = {1, 2, 3, 4, 5, 7, 8, 31, 32, 33, 63, 64, 65, 70, 640};
    const int cvTypes[] = {cv::BORDER_CONSTANT, cv::BORDER_REPLICATE, cv::BORDER_REFLECT,
                           cv::BORDER_REFLECT_101, cv::BORDER_WRAP};
    const BorderType ourTypes[] = {BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,
                                   BORDER_REFLECT_101, BORDER_WRAP};

    for (int len : lengths) {
        for (size_t t = 0; t < sizeof(cvTypes) / sizeof(cvTypes[0]); ++t) {
            int differing = 0;
            int compared = 0;
            for (int p = -3 * len - 3; p <= 3 * len + 3; ++p) {
                const int expected = cv::borderInterpolate(p, len, cvTypes[t]);
                const int actual = static_cast<int>(bincv::impl::borderIndex(
                    static_cast<ptrdiff_t>(p), static_cast<size_t>(len), ourTypes[t]));
                if (actual != expected) ++differing;
                ++compared;
            }
            ::bincv::test::reportCheck(
                differing == 0, "borderIndex is bit-exact against cv::borderInterpolate",
                __FILE__, __LINE__,
                std::string(borderName(ourTypes[t])) + " len=" + std::to_string(len) + ": " +
                    std::to_string(differing) + " of " + std::to_string(compared) +
                    " coordinates differ");
        }
    }
}

int cvBorderType(BorderType type) {
    switch (type) {
        case BORDER_CONSTANT:    return cv::BORDER_CONSTANT;
        case BORDER_REPLICATE:   return cv::BORDER_REPLICATE;
        case BORDER_REFLECT:     return cv::BORDER_REFLECT;
        case BORDER_REFLECT_101: return cv::BORDER_REFLECT_101;
        case BORDER_WRAP:        return cv::BORDER_WRAP;
    }
    return cv::BORDER_CONSTANT;
}

/// @brief The shifted image OpenCV produces, via cv::copyMakeBorder and a crop.
///
/// @note OpenCV has no shift, so the denominator has to be built. It is exact:
/// dst(y, x) = extended-src(y + dy, x + dx), and copyMakeBorder is the
/// function that materializes "extended src". Pad by just enough that the
/// window lands inside, then take the window --
/// top = max(0, -dy), bottom = max(0, dy), and the ROI origin is
/// (max(dx, 0), max(dy, 0)).
/// @note This is the requirement stated literally: "every border type matches
/// cv::copyMakeBorder on equivalent content".
cv::Mat openCvShift(const cv::Mat& src, long dx, long dy, BorderType type, bool value) {
    const int left = static_cast<int>(dx < 0 ? -dx : 0);
    const int right = static_cast<int>(dx > 0 ? dx : 0);
    const int top = static_cast<int>(dy < 0 ? -dy : 0);
    const int bottom = static_cast<int>(dy > 0 ? dy : 0);

    cv::Mat padded;
    cv::copyMakeBorder(src, padded, top, bottom, left, right, cvBorderType(type),
                       cv::Scalar::all(value ? 255 : 0));

    const int x0 = static_cast<int>(dx > 0 ? dx : 0);
    const int y0 = static_cast<int>(dy > 0 ? dy : 0);
    return padded(cv::Rect(x0, y0, src.cols, src.rows)).clone();
}

/// @brief "0.50" -- the fill ratio as a failing case's label spells it.
/// @note Two decimals, matching tests/equivalence.hpp's caseLabel, so a failure
/// here and a failure there name the same case the same way. std::to_string
/// would print 0.010000.
std::string fillText(float fill) {
    char text[16];
    std::snprintf(text, sizeof(text), "%.2f", static_cast<double>(fill));
    return std::string(text);
}

/// @brief Tier 1 border semantics over the size matrix.
/// @note OpenCV's input comes from randomCvMask, the harness's SECOND generator,
/// which writes CV_8U bytes directly and never touches the packing or the
/// unpacking path. tests/test_logic.cpp measured what the obvious spelling
/// costs: with both sides built through toCvMask, a one-column fault in the
/// conversion cancelled exactly and the suite passed 56044 of 56044.
/// @note **The WHOLE the matrix -- widths, heights AND fill ratios.** It used to
/// be equivalenceWidths x {1, 3, 17} at a hard-coded fill of 0.5, which is
/// a subset while the surrounding prose claimed the matrix. Both omissions
/// cost coverage: height 2 is the smallest case where a wrong stride is not
/// invisible (tests/equivalence.hpp says so in as many words), and 0.0 and
/// 1.0 are the exact all-clear and all-set frames, which are where a masked
/// or short-circuited kernel stops being exercised and where the border
/// stands at maximum contrast against the image.
/// @note **The last six offsets are relative to the EXTENTS, and that is the point
/// of declaring the table inside the size loops.** that work’s second done-when
/// clause is "vertical shifts correct for offsets exceeding the image
/// height", and the fixed table this replaces reached |dy| = 3 against
/// heights of 1, 3 and 17 -- so `dy > height` at a height greater than 1 was
/// never asked of cv::copyMakeBorder at all. Measured, with a clamp injected
/// into ops/shift.hpp's row loop that is wrong ONLY past the height
/// (`dy > src.height ? src.height : dy` feeding borderIndex): the core
/// reference half went red, 112589 of 112733, and all four Shift.OpenCv_*
/// cases stayed GREEN. With the offsets below the same clamp fails 2768
/// checks and takes every Shift.OpenCv_* with it. The horizontal analogue
/// (clamping dx to the width) always did fail here -- {wordBits + 1, -1}
/// exceeds a width of 1 -- so the hole was one axis wide, which is exactly
/// the shape of gap an "it is symmetric" reading of the file would miss.
template <typename WordType>
void testOpenCvBorders(const char* wordTypeName) {
    std::cout << "\n--- shift vs cv::copyMakeBorder: " << wordTypeName << " ---\n";

    constexpr long wordBits = static_cast<long>(bincv::BinMat<WordType>::WordBits);
    const std::vector<float>& fills = bincv::test::equivalenceFillRatios();

    for (int width : bincv::test::equivalenceWidths()) {
        for (int height : bincv::test::equivalenceHeights()) {
            const long w = static_cast<long>(width);
            const long h = static_cast<long>(height);

            const long offsets[][2] = {{0, 0},
                                       {1, 0},
                                       {-1, 0},
                                       {0, 1},
                                       {0, -1},
                                       {1, 1},
                                       {-2, 3},
                                       {wordBits, 0},
                                       {wordBits + 1, -1},
                                       {-wordBits - 1, 2},
                                       // Past the extent on each axis, and both at
                                       // once -- the regime names, where a
                                       // reflection's period and a clamp part ways.
                                       {0, h + 2},
                                       {0, -(h + 2)},
                                       {w + 2, 0},
                                       {-(w + 2), 0},
                                       {w + 2, h + 2},
                                       {-(w + 2), -(h + 2)}};

            for (size_t f = 0; f < fills.size(); ++f) {
                const float fill = fills[f];
                const uint64_t seed = caseSeed(width, height, 800 + f);

                bincv::BinMat<WordType> src =
                    bincv::test::randomBinary<WordType>(width, height, fill, seed);
                const cv::Mat cvSrc = bincv::test::randomCvMask(width, height, fill, seed);

                for (const auto& off : offsets) {
                    for (BorderType type : ALL_BORDERS) {
                        for (int v = 0; v < 2; ++v) {
                            if (type != BORDER_CONSTANT && v == 1) continue;  // value ignored
                            bincv::BinMat<WordType> dst(width, height);
                            bincv::shift(src.constView(), dst.view(),
                                         static_cast<ptrdiff_t>(off[0]),
                                         static_cast<ptrdiff_t>(off[1]), type, v != 0);
                            BINCV_EXPECT_BIT_EXACT(
                                dst.constView(), openCvShift(cvSrc, off[0], off[1], type, v != 0),
                                label(wordTypeName, width, height, off[0], off[1], type, v != 0,
                                      "vs copyMakeBorder fill=" + fillText(fill)));
                        }
                    }
                }
            }
        }
    }
}

/// @brief The premise the fill decision rests on, measured rather than asserted.
///
/// @note ops/shift.hpp says erode needs ones outside the image and dilate needs
/// zeros, and claims OpenCV encodes the same asymmetry through
/// morphologyDefaultBorderValue. That is a statement about OpenCV, so it
/// is checked against OpenCV: eroding an all-white frame with the default
/// border must NOT produce a black frame around the edge, and dilating an
/// all-black one must NOT produce a white one.
/// @note If this case ever fails, the paragraph in ops/shift.hpp is wrong and
/// that work’s border defaults follow it -- which is exactly why it is here rather
/// than in the morphology task that will consume it.
#if BINCV_TEST_HAVE_IMGPROC
void testMorphologyFillPremise() {
    std::cout << "\n--- the erode/dilate fill asymmetry, as OpenCV implements it ---\n";

    const cv::Mat white(8, 8, CV_8U, cv::Scalar(255));
    const cv::Mat black = cv::Mat::zeros(8, 8, CV_8U);

    cv::Mat eroded;
    cv::erode(white, eroded, cv::Mat());          // 3x3 rect, default border
    ::bincv::test::reportCheck(cv::countNonZero(eroded) == 64,
                               "cv::erode's default border behaves as ONES outside", __FILE__,
                               __LINE__,
                               "eroding an all-white frame left " +
                                   std::to_string(cv::countNonZero(eroded)) + " of 64 pixels set");

    cv::Mat dilated;
    cv::dilate(black, dilated, cv::Mat());
    ::bincv::test::reportCheck(cv::countNonZero(dilated) == 0,
                               "cv::dilate's default border behaves as ZEROS outside", __FILE__,
                               __LINE__,
                               "dilating an all-black frame set " +
                                   std::to_string(cv::countNonZero(dilated)) + " of 64 pixels");

    // And the other direction, so the two cases above cannot both pass because
    // the border was never consulted: with the fills swapped, each result changes.
    cv::Mat erodedZero;
    cv::erode(white, erodedZero, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT,
              cv::Scalar::all(0));
    ::bincv::test::reportCheck(cv::countNonZero(erodedZero) == 36,
                               "a ZERO border does erode the frame's edge away", __FILE__,
                               __LINE__,
                               "eroding an all-white frame against a zero border left " +
                                   std::to_string(cv::countNonZero(erodedZero)) + " of 64");
}
#endif // BINCV_TEST_HAVE_IMGPROC

#endif // BINCV_WITH_OPENCV

} // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

BINCV_TEST(Shift, Horizontal_uint8_t)  { testHorizontalSweep<uint8_t>("uint8_t"); }
BINCV_TEST(Shift, Horizontal_uint16_t) { testHorizontalSweep<uint16_t>("uint16_t"); }
BINCV_TEST(Shift, Horizontal_uint32_t) { testHorizontalSweep<uint32_t>("uint32_t"); }
BINCV_TEST(Shift, Horizontal_uint64_t) { testHorizontalSweep<uint64_t>("uint64_t"); }

BINCV_TEST(Shift, WideRows_uint8_t)  { testWideRows<uint8_t>("uint8_t"); }
BINCV_TEST(Shift, WideRows_uint16_t) { testWideRows<uint16_t>("uint16_t"); }
BINCV_TEST(Shift, WideRows_uint32_t) { testWideRows<uint32_t>("uint32_t"); }
BINCV_TEST(Shift, WideRows_uint64_t) { testWideRows<uint64_t>("uint64_t"); }

BINCV_TEST(Shift, Vertical_uint8_t)  { testVertical<uint8_t>("uint8_t"); }
BINCV_TEST(Shift, Vertical_uint16_t) { testVertical<uint16_t>("uint16_t"); }
BINCV_TEST(Shift, Vertical_uint32_t) { testVertical<uint32_t>("uint32_t"); }
BINCV_TEST(Shift, Vertical_uint64_t) { testVertical<uint64_t>("uint64_t"); }

BINCV_TEST(Shift, Borders_uint8_t)  { testBorders<uint8_t>("uint8_t"); }
BINCV_TEST(Shift, Borders_uint16_t) { testBorders<uint16_t>("uint16_t"); }
BINCV_TEST(Shift, Borders_uint32_t) { testBorders<uint32_t>("uint32_t"); }
BINCV_TEST(Shift, Borders_uint64_t) { testBorders<uint64_t>("uint64_t"); }

BINCV_TEST(Shift, Degenerate_uint8_t)  { testDegenerate<uint8_t>("uint8_t"); }
BINCV_TEST(Shift, Degenerate_uint16_t) { testDegenerate<uint16_t>("uint16_t"); }
BINCV_TEST(Shift, Degenerate_uint32_t) { testDegenerate<uint32_t>("uint32_t"); }
BINCV_TEST(Shift, Degenerate_uint64_t) { testDegenerate<uint64_t>("uint64_t"); }

BINCV_TEST(Shift, Strides_uint8_t)  { testDifferingStrides<uint8_t>("uint8_t"); }
BINCV_TEST(Shift, Strides_uint16_t) { testDifferingStrides<uint16_t>("uint16_t"); }
BINCV_TEST(Shift, Strides_uint32_t) { testDifferingStrides<uint32_t>("uint32_t"); }
BINCV_TEST(Shift, Strides_uint64_t) { testDifferingStrides<uint64_t>("uint64_t"); }

BINCV_TEST(Shift, DirtySource_uint8_t)  { testDirtySource<uint8_t>("uint8_t"); }
BINCV_TEST(Shift, DirtySource_uint16_t) { testDirtySource<uint16_t>("uint16_t"); }
BINCV_TEST(Shift, DirtySource_uint32_t) { testDirtySource<uint32_t>("uint32_t"); }
BINCV_TEST(Shift, DirtySource_uint64_t) { testDirtySource<uint64_t>("uint64_t"); }

BINCV_TEST(Shift, GuardWords_uint8_t)  { testGuardWords<uint8_t>("uint8_t"); }
BINCV_TEST(Shift, GuardWords_uint16_t) { testGuardWords<uint16_t>("uint16_t"); }
BINCV_TEST(Shift, GuardWords_uint32_t) { testGuardWords<uint32_t>("uint32_t"); }
BINCV_TEST(Shift, GuardWords_uint64_t) { testGuardWords<uint64_t>("uint64_t"); }

BINCV_TEST(Shift, DisjointViews_uint8_t)  { testDisjointViewsOverOneBuffer<uint8_t>("uint8_t"); }
BINCV_TEST(Shift, DisjointViews_uint16_t) { testDisjointViewsOverOneBuffer<uint16_t>("uint16_t"); }
BINCV_TEST(Shift, DisjointViews_uint32_t) { testDisjointViewsOverOneBuffer<uint32_t>("uint32_t"); }
BINCV_TEST(Shift, DisjointViews_uint64_t) { testDisjointViewsOverOneBuffer<uint64_t>("uint64_t"); }

BINCV_TEST(Shift, BorderIndex_Reference) { testBorderIndexAgainstReference(); }

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Shift, BorderInterpolate_OpenCv) { testBorderInterpolateAgainstOpenCv(); }

BINCV_TEST(Shift, OpenCv_uint8_t)  { testOpenCvBorders<uint8_t>("uint8_t"); }
BINCV_TEST(Shift, OpenCv_uint16_t) { testOpenCvBorders<uint16_t>("uint16_t"); }
BINCV_TEST(Shift, OpenCv_uint32_t) { testOpenCvBorders<uint32_t>("uint32_t"); }
BINCV_TEST(Shift, OpenCv_uint64_t) { testOpenCvBorders<uint64_t>("uint64_t"); }

#if BINCV_TEST_HAVE_IMGPROC
BINCV_TEST(Shift, MorphologyFillPremise) { testMorphologyFillPremise(); }
#endif
#endif

BINCV_TEST_MAIN("BinMat shift kernel tests")
