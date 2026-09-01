// Horizontal decimation by two: destination bit j is source bit 2j.
//
// CORE ONLY, AND NOT BY OMISSION. OpenCV has no operation that subsamples columns
// without also filtering -- cv::resize(INTER_NEAREST) resamples both axes on a
// byte image, and its column mapping for an odd width is a rounding rule rather
// than "keep the even columns" -- so there is no Tier 1 denominator to compare
// against and nothing here sits behind BINCV_WITH_OPENCV. ops/resample.hpp is
// Tier 3 (the design notes), and this file is the whole of what stands behind it.
//
// THREE IMPLEMENTATIONS, ONE REFERENCE
//
// compares three routes to the same destination, and the protocol
// (EXPERIMENTS.md) requires every one of them to be correct BEFORE any of them is
// timed -- a benchmark between a right answer and a wrong one is not a
// measurement. So each variant is checked against refDecimate, which is a
// per-pixel loop over at: `expected(y, j) = src.at(y, 2j)`, sharing no
// expression with any of the three kernels. Whether the three also agree with
// EACH OTHER is then implied rather than asserted separately.
//
// A impl::decimateColumnsBy2Gather per-pixel gather loop
// B decimateColumnsBy2 word-local Morton deinterleave --
// the PUBLIC entry point, because
// chose it, so the suite
// tests the shipped function rather
// than a copy of it
// C impl::decimateColumnsBy2FrameMasked big-integer masked unshuffle
//
// WHAT THE WIDTH SWEEP IS FOR
//
// Every interesting failure of a decimation kernel is at a width that is not a
// multiple of the word size, and there are two independent ways to be off by one:
// the SOURCE tail (an odd source width keeps its last column, so dst.width is
// ceil(src.width / 2)) and the DESTINATION tail (whose padding bits must stay
// zero,). The sweep therefore runs every width from 0 to 3*WordBits + 3
// PLUS the pyramid ladder's own widths, at all four word types, so the boundary
// cases are covered by construction rather than by choosing them.
//
// Variant C carries a third: its recurrence pads the row to a power-of-two bit
// count, so 20 words become 32 and four of the padding words are pure invention.
// A width like 640 at uint32_t (20 words) is the case that catches a padded-word
// zeroing bug, and it is in the ladder below.
//
// CHECK GRANULARITY: ONE PER DESTINATION PIXEL, PER VARIANT
//
// Per row would leave the CHECKS column blind to a shortened width sweep, which
// is this suite's most likely regression. The failure message is built only when
// a check fails.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/resample.hpp"
#include "test_util.hpp"

namespace {

using bincv::BinMat;
using bincv::BinMatConstView;
using bincv::BinMatView;
using bincv::decimatedWidth;
using bincv::rowsDecimatedBy2;

#define RESAMPLE_EXPECT(ok, what, detailExpr) \
    ::bincv::test::reportCheck((ok), (what), __FILE__, __LINE__, \
                               (ok) ? std::string() : (detailExpr))

// ---------------------------------------------------------------------------
// Inputs and the reference
// ---------------------------------------------------------------------------

// splitmix64, so a failure reproduces exactly. Deliberately not std::mt19937: the
// point is a repeatable bit pattern, not a distribution.
uint64_t nextRandom(uint64_t& state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

template <typename WordType>
void fillRandom(BinMat<WordType>& m, uint64_t seed) {
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            m.set(y, x, (nextRandom(state) & 1u) != 0);
        }
    }
}

/// The reference: keep every even column. One at per destination pixel, and no
/// word arithmetic anywhere, so it cannot fail the same way a kernel does.
template <typename WordType>
std::vector<bool> refDecimate(const BinMat<WordType>& src) {
    const size_t outWidth = decimatedWidth(src.getWidth());
    std::vector<bool> out(outWidth * src.getHeight(), false);
    for (size_t y = 0; y < src.getHeight(); ++y) {
        for (size_t j = 0; j < outWidth; ++j) {
            out[y * outWidth + j] = src.at(static_cast<int>(y), static_cast<int>(2 * j));
        }
    }
    return out;
}

/// Reads a destination pixel straight out of the view's words, rather than through
/// BinMat::at, so that a kernel and its checker do not share an accessor.
template <typename WordType>
bool viewBit(const BinMatConstView<WordType>& v, size_t y, size_t x) {
    constexpr size_t B = sizeof(WordType) * 8;
    return ((v.row(y)[x / B] >> (x % B)) & 1u) != 0;
}

/// Every bit at or above `width` in every row's words, which the design rule requires to be
/// zero. Returns the first offending (row, bit) as a string, or "" if clean.
template <typename WordType>
std::string paddingDirt(const BinMatConstView<WordType>& v) {
    constexpr size_t B = sizeof(WordType) * 8;
    const size_t rowWords = (v.width + B - 1) / B;
    for (size_t y = 0; y < v.height; ++y) {
        for (size_t bit = v.width; bit < rowWords * B; ++bit) {
            if (((v.row(y)[bit / B] >> (bit % B)) & 1u) != 0) {
                return "row " + std::to_string(y) + " padding bit " + std::to_string(bit);
            }
        }
    }
    return std::string();
}

// ---------------------------------------------------------------------------
// One (width, height) case, all three variants
// ---------------------------------------------------------------------------

enum class Variant { Gather, Unshuffle, FrameMasked };

const char* variantName(Variant v) {
    switch (v) {
        case Variant::Gather: return "gather";
        case Variant::Unshuffle: return "decimateColumnsBy2";
        case Variant::FrameMasked: return "frame-masked";
    }
    return "?";
}

template <typename WordType>
void runVariant(Variant variant, const BinMatConstView<WordType>& src,
                const BinMatView<WordType>& dst) {
    switch (variant) {
        case Variant::Gather:
            bincv::impl::decimateColumnsBy2Gather(src, dst);
            return;
        case Variant::Unshuffle:
            bincv::decimateColumnsBy2(src, dst);
            return;
        case Variant::FrameMasked: {
            // The mask table and scratch are the caller's (no heap in a kernel),
            // which is exactly the footprint weighs against zero for the
            // other two. Built per call here on purpose: a plan reused across
            // calls is the benchmark's job, and a test that built it once would
            // not notice buildFrameMaskedPlan writing the wrong number of words.
            std::vector<WordType> masks(
                bincv::impl::frameMaskedPlanWords<WordType>(src.width) + 1, WordType{0});
            std::vector<WordType> scratch(
                bincv::impl::frameMaskedRowWords<WordType>(src.width) + 1, WordType{0});
            bincv::impl::buildFrameMaskedPlan<WordType>(src.width, masks.data());
            bincv::impl::decimateColumnsBy2FrameMasked(src, dst, masks.data(), scratch.data());
            return;
        }
    }
}

/// One case: fill a source, decimate it with `variant`, compare every destination
/// pixel against the reference, and check the destination's padding.
/// @param dirtySource sets every source bit at or above the width to 1 first, so
/// that "a dirty source cannot reach a live destination pixel" is checked
/// rather than argued (see the file header of ops/resample.hpp).
template <typename WordType>
void checkCase(Variant variant, const char* wordName, size_t width, size_t height,
               uint64_t seed, bool dirtySource) {
    constexpr size_t B = sizeof(WordType) * 8;

    BinMat<WordType> src(static_cast<int>(width), static_cast<int>(height));
    fillRandom(src, seed);

    if (dirtySource && width > 0) {
        const size_t rowWords = (width + B - 1) / B;
        for (size_t y = 0; y < height; ++y) {
            WordType* row = src.ptr(static_cast<int>(y));
            for (size_t bit = width; bit < rowWords * B; ++bit) {
                row[bit / B] = static_cast<WordType>(
                    row[bit / B] | static_cast<WordType>(static_cast<WordType>(1) << (bit % B)));
            }
        }
    }

    const std::vector<bool> expected = refDecimate(src);
    const size_t outWidth = decimatedWidth(width);

    // A destination one row taller and one word wider than needed, pre-filled with
    // ones: anything the kernel fails to write shows up as a wrong value rather
    // than as whatever zero-initialised memory happened to hold, and the guard row
    // catches a kernel that walks past its height.
    BinMat<WordType> dstStore(static_cast<int>(outWidth == 0 ? 1 : outWidth) + static_cast<int>(B),
                              static_cast<int>(height == 0 ? 1 : height) + 1);
    dstStore.fill(true);

    BinMatView<WordType> dst{dstStore.data(), outWidth, height, dstStore.getAlignedWidth()};
    runVariant<WordType>(variant, src.constView(), dst);

    const std::string label = std::string(variantName(variant)) + " " + wordName + " " +
                              std::to_string(width) + "x" + std::to_string(height) +
                              (dirtySource ? " dirty" : "");

    const BinMatConstView<WordType> dstConst = dst;
    for (size_t y = 0; y < height; ++y) {
        for (size_t j = 0; j < outWidth; ++j) {
            const bool got = viewBit(dstConst, y, j);
            const bool want = expected[y * outWidth + j];
            RESAMPLE_EXPECT(got == want, "decimated pixel matches the per-pixel reference",
                            label + " at (" + std::to_string(y) + "," + std::to_string(j) +
                                "): got " + (got ? "1" : "0") + ", expected " +
                                (want ? "1" : "0"));
        }
    }

    if (outWidth > 0 && height > 0) {
        const std::string dirt = paddingDirt(dstConst);
        RESAMPLE_EXPECT(dirt.empty(), "destination padding bits stay zero",
                        label + ": " + dirt);
    }
}

// ---------------------------------------------------------------------------
// The sweeps
// ---------------------------------------------------------------------------

std::vector<size_t> sweepWidths(size_t wordBits) {
    std::vector<size_t> widths;
    for (size_t w = 0; w <= 3 * wordBits + 3; ++w) widths.push_back(w);
    // The pyramid ladder this will actually call this with, plus one odd width
    // whose word count is not a power of two (which is variant C's padding case).
    for (size_t w : {size_t{94}, size_t{160}, size_t{320}, size_t{640}}) widths.push_back(w);
    return widths;
}

template <typename WordType>
void testAgainstReference(const char* wordName, Variant variant) {
    constexpr size_t B = sizeof(WordType) * 8;
    uint64_t seed = 0x5EED0000u + static_cast<uint64_t>(B);
    for (size_t width : sweepWidths(B)) {
        for (size_t height : {size_t{1}, size_t{3}}) {
            checkCase<WordType>(variant, wordName, width, height, seed++, false);
        }
    }
}

template <typename WordType>
void testDirtySource(const char* wordName, Variant variant) {
    constexpr size_t B = sizeof(WordType) * 8;
    uint64_t seed = 0xD147Fu + static_cast<uint64_t>(B);
    // Only widths whose last word is partial can have dirty padding at all.
    for (size_t width = 1; width <= 2 * B + 1; ++width) {
        if (width % B == 0) continue;
        checkCase<WordType>(variant, wordName, width, 3, seed++, true);
    }
    checkCase<WordType>(variant, wordName, 94, 3, seed++, true);
}

/// Strides that differ between source and destination, which is the layout a
/// pyramid produces (the design rule says a kernel may not care) -- here by over-allocating
/// the source's row alignment so its stride is longer than its rows need.
template <typename WordType>
void testDifferingStrides(const char* wordName, Variant variant) {
    constexpr size_t B = sizeof(WordType) * 8;
    for (size_t width : {size_t{7}, B + 1, 2 * B, size_t{94}}) {
        const size_t height = 4;
        BinMat<WordType> src(static_cast<int>(width), static_cast<int>(height),
                             8 * sizeof(WordType));
        fillRandom(src, 0xA11 + width);
        const std::vector<bool> expected = refDecimate(src);
        const size_t outWidth = decimatedWidth(width);

        BinMat<WordType> dst(static_cast<int>(outWidth), static_cast<int>(height));
        runVariant<WordType>(variant, src.constView(), dst.view());

        const std::string label = std::string(variantName(variant)) + " " + wordName +
                                  " strides " + std::to_string(width);
        RESAMPLE_EXPECT(src.getAlignedWidth() != dst.getAlignedWidth(),
                        "the case really does use differing strides", label);
        for (size_t y = 0; y < height; ++y) {
            for (size_t j = 0; j < outWidth; ++j) {
                const bool got = dst.at(static_cast<int>(y), static_cast<int>(j));
                RESAMPLE_EXPECT(got == expected[y * outWidth + j],
                                "differing strides do not change the result", label);
            }
        }
    }
}

/// Degenerate shapes must be no-ops rather than crashes, exactly as in
/// tests/test_logic.cpp: a zero width or height addresses no pixels.
template <typename WordType>
void testDegenerate(const char* wordName, Variant variant) {
    BinMat<WordType> src(8, 4);
    fillRandom(src, 7);
    BinMat<WordType> dst(4, 4);

    const std::string label = std::string(variantName(variant)) + " " + wordName;

    BinMatConstView<WordType> emptyWidth = src.constView();
    emptyWidth.width = 0;
    BinMatView<WordType> dstEmptyWidth = dst.view();
    dstEmptyWidth.width = 0;
    runVariant<WordType>(variant, emptyWidth, dstEmptyWidth);
    RESAMPLE_EXPECT(true, "zero-width decimation returns without touching anything", label);

    BinMatConstView<WordType> emptyHeight = src.constView();
    emptyHeight.height = 0;
    BinMatView<WordType> dstEmptyHeight = dst.view();
    dstEmptyHeight.height = 0;
    runVariant<WordType>(variant, emptyHeight, dstEmptyHeight);
    RESAMPLE_EXPECT(true, "zero-height decimation returns without touching anything", label);
}

/// The free half: rowsDecimatedBy2 is a view, so it must alias the source's own
/// memory and read rows 0, 2, 4,... -- no copy, no allocation.
template <typename WordType>
void testRowView(const char* wordName) {
    for (size_t height : {size_t{1}, size_t{2}, size_t{5}, size_t{8}}) {
        BinMat<WordType> src(37, static_cast<int>(height));
        fillRandom(src, 0x0FFE0u + height);
        const BinMatConstView<WordType> halved = rowsDecimatedBy2(src.constView());

        const std::string label = std::string(wordName) + " h=" + std::to_string(height);
        RESAMPLE_EXPECT(halved.ptr == src.constView().ptr, "the row view aliases the source",
                        label);
        RESAMPLE_EXPECT(halved.height == (height + 1) / 2, "the row view halves the height",
                        label);
        RESAMPLE_EXPECT(halved.width == src.getWidth(), "the row view keeps the width", label);

        for (size_t y = 0; y < halved.height; ++y) {
            for (size_t x = 0; x < halved.width; ++x) {
                const bool got = viewBit(halved, y, x);
                const bool want = src.at(static_cast<int>(2 * y), static_cast<int>(x));
                RESAMPLE_EXPECT(got == want, "the row view reads every other row", label);
            }
        }
    }
}

/// The plan sizes variant C reports must be the ones it uses. Checked directly,
/// because the kernel indexes `masks + k * rowWords` and an under-sized table
/// would read past a caller's buffer without any test noticing.
template <typename WordType>
void testPlanSizes(const char* wordName) {
    constexpr size_t B = sizeof(WordType) * 8;
    for (size_t width : {size_t{1}, B, B + 1, size_t{94}, size_t{640}}) {
        const size_t rowWords = bincv::impl::frameMaskedRowWords<WordType>(width);
        const size_t passes = bincv::impl::frameMaskedPasses<WordType>(width);
        const size_t plan = bincv::impl::frameMaskedPlanWords<WordType>(width);
        const std::string label = std::string(wordName) + " w=" + std::to_string(width);

        RESAMPLE_EXPECT(rowWords >= (width + B - 1) / B, "the padded row holds the real row",
                        label);
        RESAMPLE_EXPECT((rowWords & (rowWords - 1)) == 0, "the padded row is a power of two",
                        label);
        RESAMPLE_EXPECT(plan == passes * rowWords, "the plan is passes x padded row words",
                        label);

        // One pass per doubling of the period, from 2 up to the padded bit count.
        size_t expectedPasses = 0;
        for (size_t p = 2; p <= rowWords * B; p <<= 1) ++expectedPasses;
        RESAMPLE_EXPECT(passes == expectedPasses, "one pass per period doubling", label);
    }
}

}  // namespace

#define RESAMPLE_TESTS(WordType, name)                                                  \
    BINCV_TEST(Resample, Reference_Gather_##name) {                                      \
        testAgainstReference<WordType>(#name, Variant::Gather);                          \
    }                                                                                    \
    BINCV_TEST(Resample, Reference_Unshuffle_##name) {                                   \
        testAgainstReference<WordType>(#name, Variant::Unshuffle);                       \
    }                                                                                    \
    BINCV_TEST(Resample, Reference_FrameMasked_##name) {                                 \
        testAgainstReference<WordType>(#name, Variant::FrameMasked);                     \
    }                                                                                    \
    BINCV_TEST(Resample, DirtySource_Gather_##name) {                                    \
        testDirtySource<WordType>(#name, Variant::Gather);                               \
    }                                                                                    \
    BINCV_TEST(Resample, DirtySource_Unshuffle_##name) {                                 \
        testDirtySource<WordType>(#name, Variant::Unshuffle);                            \
    }                                                                                    \
    BINCV_TEST(Resample, DirtySource_FrameMasked_##name) {                               \
        testDirtySource<WordType>(#name, Variant::FrameMasked);                          \
    }                                                                                    \
    BINCV_TEST(Resample, Strides_Gather_##name) {                                        \
        testDifferingStrides<WordType>(#name, Variant::Gather);                          \
    }                                                                                    \
    BINCV_TEST(Resample, Strides_Unshuffle_##name) {                                     \
        testDifferingStrides<WordType>(#name, Variant::Unshuffle);                       \
    }                                                                                    \
    BINCV_TEST(Resample, Strides_FrameMasked_##name) {                                   \
        testDifferingStrides<WordType>(#name, Variant::FrameMasked);                     \
    }                                                                                    \
    BINCV_TEST(Resample, Degenerate_Gather_##name) {                                     \
        testDegenerate<WordType>(#name, Variant::Gather);                                \
    }                                                                                    \
    BINCV_TEST(Resample, Degenerate_Unshuffle_##name) {                                  \
        testDegenerate<WordType>(#name, Variant::Unshuffle);                             \
    }                                                                                    \
    BINCV_TEST(Resample, Degenerate_FrameMasked_##name) {                                \
        testDegenerate<WordType>(#name, Variant::FrameMasked);                           \
    }                                                                                    \
    BINCV_TEST(Resample, RowView_##name) { testRowView<WordType>(#name); }               \
    BINCV_TEST(Resample, PlanSizes_##name) { testPlanSizes<WordType>(#name); }

RESAMPLE_TESTS(uint8_t, uint8_t)
RESAMPLE_TESTS(uint16_t, uint16_t)
RESAMPLE_TESTS(uint32_t, uint32_t)
RESAMPLE_TESTS(uint64_t, uint64_t)

BINCV_TEST_MAIN("test_resample")
