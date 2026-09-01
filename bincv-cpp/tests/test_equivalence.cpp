// The equivalence harness's own test suite.
//
// Two jobs, and the second is the one that matters:
//
// 1. DEMONSTRATE the harness end to end on an already-implemented operation --
// countNonZero against cv::countNonZero -- across the full size and
// fill matrix, so that.. inherit something known to work rather
// than something newly written alongside the kernel it is meant to judge.
//
// 2. PROVE THE HARNESS CAN FAIL. A harness that passes everything is worse
// than no harness, because it licenses every kernel built on it. The
// Comparator* cases below feed it deliberate faults -- a one-column
// off-by-one, a dropped trailing partial word, a transposed row/column, a
// single flipped pixel, a size change -- and require it to report each one,
// at the right place, with the right count. tests/CMakeLists.txt goes
// further and rebuilds this whole file with each fault compiled INTO the
// conversion, registered WILL_FAIL, so "the harness detects a real
// discrepancy" is three ctest results rather than a paragraph.
//
// Requires OpenCV; tests/CMakeLists.txt only builds it when OpenCV was found.
// The core-only, no-exceptions and Debug configurations never see this file.

// The guard comes FIRST, before any OpenCV header. It exists to explain a
// core-only build of this file, and it cannot do that from underneath an
// #include <opencv2/...> that the same build has no include path for: the
// preprocessor stops at the missing header and the explanation never prints.
// Measured, with only the standard opencv4 layout on the include path (which is
// what the build system passes via -isystem): "fatal error: opencv2/core.hpp: No
// such file or directory" instead of the message below.
#ifndef BINCV_WITH_OPENCV
#error "test_equivalence.cpp needs OpenCV; it must only be built when BINCV_OPENCV_FOUND"
#endif

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "bincv-cpp/binMat.hpp"
// QuantMat, for the bit-plane row of the packing anchor: a plane view is the one
// conversion that work’s per-plane Tier 1 checks depend on, and it is not a BinMat view.
#include "bincv-cpp/quantMat.hpp"
#include "equivalence.hpp"
#include "test_util.hpp"

namespace {

using bincv::test::caseLabel;
using bincv::test::compareMasks;
using bincv::test::ConversionFault;
using bincv::test::equivalenceFillRatios;
using bincv::test::equivalenceHeights;
using bincv::test::equivalenceWidths;
using bincv::test::Mismatch;
using bincv::test::randomBinary;
using bincv::test::randomCvMask;
using bincv::test::toCvMask;
using bincv::test::unpackTo8U;

// ---------------------------------------------------------------------------
// Case seeds
//
// Derived from the case rather than fixed, so neighbouring shapes do not all
// carry the same picture, and reproducible because the derivation is pure.
// ---------------------------------------------------------------------------
uint64_t caseSeed(int width, int height, size_t fillIndex) {
    return UINT64_C(0x51ED5EEDC0FFEE01) +
           static_cast<uint64_t>(width) * UINT64_C(1000003) +
           static_cast<uint64_t>(height) * UINT64_C(10007) +
           static_cast<uint64_t>(fillIndex);
}

// An opt-in row alignment large enough that every supported word type gets
// a stride strictly greater than the ceil(width / WordBits) words its rows need:
// 32 bytes is a whole number of 1-, 2-, 4- and 8-byte words, and 8 words even at
// uint64_t. This is what puts a stride > minimum into the sweep.
constexpr size_t PADDED_ROW_ALIGNMENT = 32;

// ---------------------------------------------------------------------------
// Deliberate faults, applied to a cv::Mat and nothing else
//
// These perturb an ALREADY CORRECT mask rather than corrupting the conversion,
// which is what keeps these cases meaningful in the faulted builds too: they do
// not touch unpackTo8U, so they mean the same thing whether or not
// BINCV_EQUIVALENCE_INJECT_FAULT is set.
// ---------------------------------------------------------------------------
enum class MatFault {
    ColumnRotate,          // every pixel moves one column left, cyclically
    DropTrailingColumns,   // the final partial word's columns read as zero
    TransposeContent,      // rows and columns swapped (square input)
    TransposeShape,        // rows and columns swapped (non-square input)
    FlipOnePixel,          // exactly one pixel inverted, away from the edges
    ShrinkOneRow           // one row shorter than it should be
};

const char* faultName(MatFault fault) {
    switch (fault) {
        case MatFault::ColumnRotate:        return "off-by-one column (cyclic rotate)";
        case MatFault::DropTrailingColumns: return "dropped trailing partial word";
        case MatFault::TransposeContent:    return "transposed row/column (square)";
        case MatFault::TransposeShape:      return "transposed row/column (non-square)";
        case MatFault::FlipOnePixel:        return "one flipped pixel";
        case MatFault::ShrinkOneRow:        return "one row short";
    }
    return "?";
}

cv::Mat perturb(const cv::Mat& src, MatFault fault, size_t wordBits) {
    switch (fault) {
        case MatFault::ColumnRotate: {
            cv::Mat out = src.clone();
            for (int y = 0; y < src.rows; ++y) {
                const uint8_t* in = src.ptr<uint8_t>(y);
                uint8_t* row = out.ptr<uint8_t>(y);
                for (int x = 0; x < src.cols; ++x) row[x] = in[(x + 1) % src.cols];
            }
            return out;
        }
        case MatFault::DropTrailingColumns: {
            cv::Mat out = src.clone();
            const size_t cols = static_cast<size_t>(src.cols);
            const size_t tailStart = (cols % wordBits == 0) ? cols : (cols - cols % wordBits);
            for (int y = 0; y < src.rows; ++y) {
                uint8_t* row = out.ptr<uint8_t>(y);
                for (size_t x = tailStart; x < cols; ++x) row[x] = 0;
            }
            return out;
        }
        case MatFault::TransposeContent:
        case MatFault::TransposeShape: {
            cv::Mat out;
            cv::transpose(src, out);
            return out;
        }
        case MatFault::FlipOnePixel: {
            cv::Mat out = src.clone();
            const int y = src.rows * 2 / 3;
            const int x = src.cols * 3 / 4;
            uint8_t& pixel = out.ptr<uint8_t>(y)[x];
            pixel = (pixel != 0) ? static_cast<uint8_t>(0) : static_cast<uint8_t>(255);
            return out;
        }
        case MatFault::ShrinkOneRow:
            return src.rowRange(0, src.rows - 1).clone();
    }
    return src.clone();
}

/// @brief Independently confirms that (mm.row, mm.col) really is the FIRST pixel
/// at which the two masks differ, scanning row-major.
/// @note The locator is the part of this harness a reader will trust without
/// checking -- "it said row 12, col 65, so that is where to look". This is
/// what makes that trustworthy: a separate scan, not a re-run of the same
/// loop, which also has to agree about the byte values it reported.
bool locationIsGenuinelyFirst(const cv::Mat& a, const cv::Mat& b, const Mismatch& mm) {
    if (!mm.found || mm.shapeMismatch) return false;
    for (int y = 0; y < a.rows; ++y) {
        const uint8_t* rowA = a.ptr<uint8_t>(y);
        const uint8_t* rowB = b.ptr<uint8_t>(y);
        for (int x = 0; x < a.cols; ++x) {
            if (rowA[x] == rowB[x]) continue;
            return y == mm.row && x == mm.col &&
                   static_cast<int>(rowA[x]) == mm.actual &&
                   static_cast<int>(rowB[x]) == mm.expected;
        }
    }
    return false;   // nothing differs, so no location can be right
}

/// @brief FNV-1a over the row-major pixel stream, as a portability witness.
/// @note uint64_t throughout, and it reads one bit per pixel rather than the
/// byte, so it says the same thing about a {0,1} mask and a {0,255} one.
uint64_t pixelDigest(const cv::Mat& mask) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (int y = 0; y < mask.rows; ++y) {
        const uint8_t* row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < mask.cols; ++x) {
            hash ^= static_cast<uint64_t>(row[x] != 0 ? 1u : 0u);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

/// @brief Counts pixels on which two BinMats of the same shape disagree.
/// @note Through at, deliberately: it is a third reader, independent of both
/// the view arithmetic in unpackTo8U and the packing in randomBinary.
template <typename A, typename B>
int pixelDisagreements(const A& lhs, const B& rhs) {
    if (lhs.getWidth() != rhs.getWidth() || lhs.getHeight() != rhs.getHeight()) return -1;
    int differing = 0;
    for (int y = 0; y < lhs.rows(); ++y)
        for (int x = 0; x < lhs.cols(); ++x)
            if (lhs.at(y, x) != rhs.at(y, x)) ++differing;
    return differing;
}

// ===========================================================================
// 1. The generator
// ===========================================================================

void testGeneratorReproducible() {
    std::cout << "\n--- randomBinary is reproducible ---\n";

    // Same seed, same matrix -- the property every recorded failure depends on.
    for (float fill : equivalenceFillRatios()) {
        bincv::BinMat32 a = randomBinary<uint32_t>(70, 37, fill, UINT64_C(0xA5A5A5A5));
        bincv::BinMat32 b = randomBinary<uint32_t>(70, 37, fill, UINT64_C(0xA5A5A5A5));
        BINCV_CHECK_EQ(pixelDisagreements(a, b), 0);
    }

    // A different seed must give different content, or "seeded" would be a
    // synonym for "constant" and the sweep below would test one picture 240 times.
    bincv::BinMat32 s1 = randomBinary<uint32_t>(640, 480, 0.5f, 1);
    bincv::BinMat32 s2 = randomBinary<uint32_t>(640, 480, 0.5f, 2);
    BINCV_CHECK(pixelDisagreements(s1, s2) > 100000);

    // Seed 0 is not special. SplitMix64 advances by a fixed gamma before mixing,
    // so a zero seed is an ordinary starting point rather than a degenerate one.
    bincv::BinMat32 z = randomBinary<uint32_t>(640, 480, 0.5f, 0);
    BINCV_CHECK(z.countNonZero() > 100000);
    BINCV_CHECK(z.countNonZero() < 200000);
}

void testGeneratorDegenerateDimensions() {
    std::cout << "\n--- randomBinary and randomCvMask agree on degenerate sizes ---\n";

    // The two generators document this contract in identical words, so they have
    // to mean the same thing by it. They did not: randomBinary constructed the
    // BinMat BEFORE its own `width <= 0 || height <= 0` guard, so a negative
    // dimension was reported by BinMat's precondition -- std::invalid_argument
    // ("BinMat dimensions must be non-negative"), or an abort under
    // BINCV_NO_EXCEPTIONS -- while randomCvMask returned an empty cv::Mat. The
    // guard was dead for every negative value it named.
    //
    // These calls throwing rather than returning empty is what a regression here
    // looks like; the case fails either way.
    const int degenerate[][2] = {{0, 5}, {5, 0}, {0, 0}, {-1, 5}, {5, -1}, {-3, -3}};
    for (const auto& d : degenerate) {
        bincv::BinMat32 m = randomBinary<uint32_t>(d[0], d[1], 0.5f, 1);
        const cv::Mat mask = randomCvMask(d[0], d[1], 0.5f, 1);
        BINCV_CHECK(m.empty());
        BINCV_CHECK(mask.empty());
        BINCV_CHECK_EQ(m.countNonZero(), 0);
    }

    // A 1x1 matrix is not degenerate, and must still carry content.
    bincv::BinMat32 one = randomBinary<uint32_t>(1, 1, 1.0f, 1);
    BINCV_CHECK(!one.empty());
    BINCV_CHECK_EQ(one.countNonZero(), 1);
}

void testGeneratorWordTypeIndependent() {
    std::cout << "\n--- randomBinary is word-type independent ---\n";

    // The same seed must give the same PICTURE at 8, 16, 32 and 64 bits per word.
    // Drawing whole words instead of pixels would have made this false, and the
    // the matrix exists partly to compare word widths against each other -- which
    // is only meaningful if they are looking at the same image.
    const int sizes[][2] = {{1, 1}, {7, 3}, {31, 17}, {33, 2}, {65, 37}, {70, 17}, {640, 3}};
    for (const auto& size : sizes) {
        for (size_t f = 0; f < equivalenceFillRatios().size(); ++f) {
            const float fill = equivalenceFillRatios()[f];
            const uint64_t seed = caseSeed(size[0], size[1], f);
            bincv::BinMat8  m8  = randomBinary<uint8_t>(size[0], size[1], fill, seed);
            bincv::BinMat16 m16 = randomBinary<uint16_t>(size[0], size[1], fill, seed);
            bincv::BinMat32 m32 = randomBinary<uint32_t>(size[0], size[1], fill, seed);
            bincv::BinMat64 m64 = randomBinary<uint64_t>(size[0], size[1], fill, seed);
            BINCV_CHECK_EQ(pixelDisagreements(m8, m32), 0);
            BINCV_CHECK_EQ(pixelDisagreements(m16, m32), 0);
            BINCV_CHECK_EQ(pixelDisagreements(m64, m32), 0);
        }
    }
}

void testGeneratorFillRatios() {
    std::cout << "\n--- randomBinary honours its fill ratio ---\n";

    // The endpoints are exact, not approximate: an all-zero and an all-ones frame
    // are where a short-circuiting kernel stops being exercised.
    for (int width : equivalenceWidths()) {
        for (int height : equivalenceHeights()) {
            bincv::BinMat32 empty = randomBinary<uint32_t>(width, height, 0.0f, 7);
            bincv::BinMat32 full = randomBinary<uint32_t>(width, height, 1.0f, 7);
            BINCV_CHECK_EQ(empty.countNonZero(), 0);
            BINCV_CHECK_EQ(full.countNonZero(), width * height);
        }
    }

    // The interior ratios only have to be close, and at 307200 pixels "close"
    // is tight: three standard deviations of a 0.5 fill is under 0.003.
    const double pixels = 640.0 * 480.0;
    const float interior[] = {0.01f, 0.5f, 0.99f};
    for (float fill : interior) {
        bincv::BinMat32 m = randomBinary<uint32_t>(640, 480, fill, 99);
        const double observed = static_cast<double>(m.countNonZero()) / pixels;
        const double error = observed - static_cast<double>(fill);
        std::cout << " fill " << fill << " -> " << observed << "\n";
        BINCV_CHECK(error > -0.005 && error < 0.005);
    }
}

void testGeneratorGoldenValues() {
    std::cout << "\n--- randomBinary golden values ---\n";

    // Recorded values, not self-consistency. Everything above would still pass if
    // the generator changed; these are what say it did not, and they are what
    // "the same seed on any platform" means in practice -- a platform whose
    // arithmetic differs fails HERE, with an obvious cause, rather than somewhere
    // downstream that looks like a packing bug.
    //
    // Generated by tests/equivalence.hpp itself. Regenerate deliberately, never to
    // make a red run green.
    struct Golden { int width, height; float fill; uint64_t seed; int nonZero; uint64_t digest; };
    const Golden golden[] = {
        {70,  37,  0.5f,  UINT64_C(0xB1C0DE00000001), 1289,   UINT64_C(0x51872ED8E4B54C78)},
        {65,  17,  0.01f, UINT64_C(0xB1C0DE00000002), 16,     UINT64_C(0x2842A0187D163F9B)},
        {31,  3,   0.99f, UINT64_C(0xB1C0DE00000003), 93,     UINT64_C(0x83C42ABD6076558A)},
        {640, 480, 0.5f,  UINT64_C(0xB1C0DE00000004), 153664, UINT64_C(0xAC2450E7532E1A4D)},
    };
    for (const Golden& g : golden) {
        // Through randomCvMask, so the golden values describe the RNG and its
        // fill mapping alone -- no BinMat, no packing, no unpacking.
        const cv::Mat mask = randomCvMask(g.width, g.height, g.fill, g.seed);
        BINCV_CHECK_EQ(cv::countNonZero(mask), g.nonZero);
        BINCV_CHECK(pixelDigest(mask) == g.digest);

        // And the BinMat generator reproduces exactly the same picture.
        bincv::BinMat32 packed = randomBinary<uint32_t>(g.width, g.height, g.fill, g.seed);
        BINCV_CHECK_EQ(packed.countNonZero(), g.nonZero);
    }

    // The fill mapping is hand-written precisely so these are fixed numbers
    // rather than whatever the local standard library happens to do
    // (std::uniform_int_distribution is not portable across implementations).
    BINCV_CHECK_EQ(bincv::test::fillThreshold(0.0f), 0u);
    BINCV_CHECK_EQ(bincv::test::fillThreshold(0.01f), 167772u);
    BINCV_CHECK_EQ(bincv::test::fillThreshold(0.5f), 8388608u);
    BINCV_CHECK_EQ(bincv::test::fillThreshold(0.99f), 16609444u);
    BINCV_CHECK_EQ(bincv::test::fillThreshold(1.0f), bincv::test::FILL_SCALE);
    BINCV_CHECK_EQ(bincv::test::fillThreshold(-1.0f), 0u);
    BINCV_CHECK_EQ(bincv::test::fillThreshold(2.0f), bincv::test::FILL_SCALE);
}

// ===========================================================================
// 2. The comparator must be able to fail
// ===========================================================================

/// @brief Feeds the comparator one deliberate fault and requires it to report it.
/// @tparam WordType The word width the case is built at -- ACTUALLY built at.
/// This used to be a `size_t wordBits` parameter while the body
/// hardcoded BinMat32, so the three "at each supported word width"
/// DropTrailingColumns cases were the same case three times: no
/// BinMat8/16/64 was ever constructed, and at width 70 the tail begins
/// at column 64 for all four word widths (64 being a multiple of every
/// one of them), so perturb zeroed identical columns in each.
template <typename WordType>
void expectComparatorCatches(MatFault fault, const char* wordTypeName, int width, int height,
                             float fill, bool countPreserving) {
    constexpr size_t wordBits = bincv::BinMat<WordType>::WordBits;
    std::cout << "\n--- comparator catches: " << faultName(fault)
              << " (" << wordTypeName << ") ---\n";

    bincv::BinMat<WordType> m =
        randomBinary<WordType>(width, height, fill, caseSeed(width, height, 3));
    // ConversionFault::None explicitly: this case must mean the same thing in the
    // faulted builds, where the default would already be wrong on purpose.
    const cv::Mat good = unpackTo8U(m.constView(), ConversionFault::None);
    const cv::Mat bad = perturb(good, fault, wordBits);

    const Mismatch mm = compareMasks(good, bad);
    std::cout << " " << caseLabel(wordTypeName, width, height, fill)
              << " -> " << mm.describe() << "\n";

    BINCV_CHECK(mm.found);

    if (good.rows == bad.rows && good.cols == bad.cols) {
        BINCV_CHECK(!mm.shapeMismatch);

        // OpenCV is the independent oracle for how many pixels differ, so the
        // comparator's count is checked against something that is not itself.
        const int oracle = cv::countNonZero(good != bad);
        BINCV_CHECK(oracle > 0);
        BINCV_CHECK_EQ(static_cast<int>(mm.differing), oracle);
        BINCV_CHECK_EQ(static_cast<int>(mm.total),
                       static_cast<int>(good.rows * good.cols));

        //... and the location it reports is really the first difference.
        BINCV_CHECK(locationIsGenuinelyFirst(good, bad, mm));
    } else {
        BINCV_CHECK(mm.shapeMismatch);
        BINCV_CHECK_EQ(mm.row, -1);
    }

    if (countPreserving) {
        // The reason bit-exactness is the Tier 1 rule and "the counts agree" is
        // not. A cyclic column rotation preserves countNonZero EXACTLY, so a
        // popcount-only check passes on an image that is visibly wrong. The
        // comparator still reports it.
        BINCV_CHECK_EQ(cv::countNonZero(good), cv::countNonZero(bad));
        BINCV_CHECK(mm.found);
    }
}

void testComparatorCatchesFaults() {
    // 70 wide is a 6-bit tail at 32 bits per word, so the dropped-trailing-word
    // fault has something to drop. At a width that divides the word size it would
    // be a no-op -- which is itself worth knowing, and is why the sweep in
    // testPackingAnchor spans both kinds of width.
    expectComparatorCatches<uint32_t>(MatFault::ColumnRotate,        "uint32_t", 70, 37, 0.5f, true);
    expectComparatorCatches<uint32_t>(MatFault::DropTrailingColumns, "uint32_t", 70, 37, 0.5f, false);
    expectComparatorCatches<uint32_t>(MatFault::TransposeContent,    "uint32_t", 37, 37, 0.5f, true);
    expectComparatorCatches<uint32_t>(MatFault::TransposeShape,      "uint32_t", 70, 37, 0.5f, false);
    expectComparatorCatches<uint32_t>(MatFault::FlipOnePixel,        "uint32_t", 70, 37, 0.5f, false);
    expectComparatorCatches<uint32_t>(MatFault::FlipOnePixel,        "uint32_t", 640, 480, 0.5f, false);
    expectComparatorCatches<uint32_t>(MatFault::ShrinkOneRow,        "uint32_t", 70, 37, 0.5f, false);

    // A dropped trailing word at each supported word width, and at a width where
    // the four word widths genuinely disagree about where the tail starts. 122 is
    // chosen for that: 8 -> 120, 16 -> 112, 32 -> 96, 64 -> 64, four distinct
    // columns. At 70 all four start at 64 and these were one case repeated.
    expectComparatorCatches<uint8_t>(MatFault::DropTrailingColumns,  "uint8_t",  122, 17, 0.5f, false);
    expectComparatorCatches<uint16_t>(MatFault::DropTrailingColumns, "uint16_t", 122, 17, 0.5f, false);
    expectComparatorCatches<uint32_t>(MatFault::DropTrailingColumns, "uint32_t", 122, 17, 0.5f, false);
    expectComparatorCatches<uint64_t>(MatFault::DropTrailingColumns, "uint64_t", 122, 17, 0.5f, false);
}

void testComparatorAcceptsIdentical() {
    std::cout << "\n--- comparator accepts identical content ---\n";

    // The other half of "it can fail": it must not report a difference that is
    // not there, or every kernel would be red and the harness would be turned off.
    for (int width : equivalenceWidths()) {
        for (size_t f = 0; f < equivalenceFillRatios().size(); ++f) {
            const float fill = equivalenceFillRatios()[f];
            bincv::BinMat32 m = randomBinary<uint32_t>(width, 17, fill, caseSeed(width, 17, f));
            const cv::Mat mask = unpackTo8U(m.constView(), ConversionFault::None);
            const Mismatch mm = compareMasks(mask, mask.clone());
            BINCV_CHECK(!mm.found);
            BINCV_CHECK_EQ(static_cast<int>(mm.differing), 0);
        }
    }

    // An empty result and an empty expectation agree; they do not "differ in
    // shape" just because both are 0x0.
    bincv::BinMat32 nothing;
    const Mismatch emptyCase = compareMasks(unpackTo8U(nothing.constView(),
                                                       ConversionFault::None), cv::Mat());
    BINCV_CHECK(!emptyCase.found);

    // A non-CV_8U expectation is a mistake in the test, not a pass.
    const cv::Mat floats = cv::Mat::zeros(4, 4, CV_32FC1);
    bincv::BinMat32 small = randomBinary<uint32_t>(4, 4, 0.5f, 5);
    const Mismatch wrongType =
        compareMasks(unpackTo8U(small.constView(), ConversionFault::None), floats);
    BINCV_CHECK(wrongType.found);
    BINCV_CHECK(wrongType.shapeMismatch);
}

// ===========================================================================
// 2b. expectBitExact itself must be able to fail
//
// The comparator cases above assert on compareMasks directly and never call
// expectBitExact, which left the harness's PRIMARY entry point -- and the large
// majority of this suite's checks -- with its failure path unexecuted. Measured:
// rewriting expectBitExact's `if (!m.found)` to `if (true)`, so that it reported
// success unconditionally, still produced "3392/3392 checks passed" and exit 0,
// and compiled warning-free under -Werror. Two of the three WILL_FAIL fault
// builds then went green (i.e. their ctest cases went red), so the property
// survived only by the coincidence of which injected faults happened to route
// through expectBitExact -- and those targets can be deleted with a green gate.
//
// VerdictCapture closes that: it swaps the sink expectBitExact reports through,
// so these cases drive its real failure path in-process and assert on the verdict
// it produced. They are counted in CHECKS and floored in expected-checks.txt,
// which the WILL_FAIL targets are not.
// ===========================================================================

/// @brief Runs one expectBitExact call with its verdicts captured, not reported.
std::vector<bincv::test::CapturedVerdict> captureBitExact(
        const bincv::BinMatConstView<uint32_t>& actual, const cv::Mat& expected,
        const std::string& context) {
    bincv::test::VerdictCapture capture;
    bincv::test::expectBitExact(actual, expected, context);
    return capture.verdicts();
}

bool noteMentions(const std::vector<bincv::test::CapturedVerdict>& verdicts,
                  const std::string& needle) {
    for (const bincv::test::CapturedVerdict& v : verdicts) {
        if (!v.ok && v.note.find(needle) != std::string::npos) return true;
    }
    return false;
}

size_t failureCountOf(const std::vector<bincv::test::CapturedVerdict>& verdicts) {
    size_t n = 0;
    for (const bincv::test::CapturedVerdict& v : verdicts) {
        if (!v.ok) ++n;
    }
    return n;
}

void testExpectBitExactReportsFailures() {
    std::cout << "\n--- expectBitExact reports the failures it is given ---\n";

    // The seam must still be wired to the suite in an ordinary run. Without this,
    // a default reporter quietly replaced by a no-op would make every case below
    // pass while silencing every real kernel failure.
    BINCV_CHECK(bincv::test::checkReporter() == &bincv::test::reportCheck);

    bincv::BinMat32 m = randomBinary<uint32_t>(70, 37, 0.5f, UINT64_C(0xBE11A5));
    const cv::Mat good = unpackTo8U(m.constView(), ConversionFault::None);

    // (a) identical content -> two verdicts, both passing (pixels, then padding).
    {
        const auto verdicts = captureBitExact(m.constView(), good, "self-test [identical]");
        BINCV_CHECK_EQ(static_cast<int>(verdicts.size()), 2);
        BINCV_CHECK_EQ(static_cast<int>(failureCountOf(verdicts)), 0);
    }

    // (b) one flipped pixel -> reported, and the note carries both the caller's
    // context and the LOCATION. A failure that does not say where is not
    // actionable over a 640x480 frame, which is why firstMismatch exists.
    {
        const cv::Mat bad = perturb(good, MatFault::FlipOnePixel, 32);
        const auto verdicts = captureBitExact(m.constView(), bad, "self-test [flipped]");
        BINCV_CHECK_EQ(static_cast<int>(failureCountOf(verdicts)), 1);
        BINCV_CHECK(noteMentions(verdicts, "first mismatch at row"));
        BINCV_CHECK(noteMentions(verdicts, "self-test [flipped]"));
        BINCV_CHECK(noteMentions(verdicts, "1 of 2590 pixels differ"));
    }

    // (c) a cyclic column rotation -> reported, though countNonZero cannot see it.
    {
        const cv::Mat bad = perturb(good, MatFault::ColumnRotate, 32);
        const auto verdicts = captureBitExact(m.constView(), bad, "self-test [rotated]");
        BINCV_CHECK_EQ(static_cast<int>(failureCountOf(verdicts)), 1);
        BINCV_CHECK_EQ(cv::countNonZero(good), cv::countNonZero(bad));
    }

    // (d) a size change -> reported as a shape mismatch, not as an overlap compare.
    {
        const cv::Mat bad = perturb(good, MatFault::ShrinkOneRow, 32);
        const auto verdicts = captureBitExact(m.constView(), bad, "self-test [short]");
        BINCV_CHECK_EQ(static_cast<int>(failureCountOf(verdicts)), 1);
        BINCV_CHECK(noteMentions(verdicts, "shape mismatch"));
    }

    // (e) THE ONE THE PIXEL COMPARISON CANNOT SEE. Same pixels, dirty padding:
    // 70 columns at 32 bits per word leaves bits 6..31 of word 2 as padding.
    // This is the defect CLAUDE.md names as a hard rule and the shape a
    // word-wise kernel takes when it forgets clearTrailingBits.
    {
        bincv::BinMat32 dirty = randomBinary<uint32_t>(70, 37, 0.5f, UINT64_C(0xBE11A5));
        dirty.ptr(0)[2] = static_cast<uint32_t>(dirty.ptr(0)[2] | 0xFFFFFFC0u);

        // The pixel comparison alone still says these are identical...
        const Mismatch pixelsOnly = bincv::test::firstMismatch(dirty.constView(), good);
        BINCV_CHECK(!pixelsOnly.found);
        BINCV_CHECK_EQ(dirty.countNonZero(), cv::countNonZero(good));

        //... and expectBitExact still reports it, through the padding verdict.
        const auto verdicts = captureBitExact(dirty.constView(), good, "self-test [dirty padding]");
        BINCV_CHECK_EQ(static_cast<int>(verdicts.size()), 2);
        BINCV_CHECK_EQ(static_cast<int>(failureCountOf(verdicts)), 1);
        BINCV_CHECK(noteMentions(verdicts, "padding bit set at row 0, bit 70"));
        BINCV_CHECK(noteMentions(verdicts, "26 padding bits set in total"));
    }
}

/// @brief The padding check, over every word width and both stride regimes.
/// @note Its own case rather than a line inside the anchor, because it is the
/// check whose absence was invisible: with it removed, a stand-in
/// word-wise bitwiseNot passed 240 of 240 swept cases at uint64_t while
/// leaving 826,200 phantom set bits behind.
template <typename WordType>
void testPaddingInvariant(const char* wordTypeName) {
    std::cout << "\n--- padding invariant: " << wordTypeName << " ---\n";
    constexpr size_t wordBits = bincv::BinMat<WordType>::WordBits;

    for (int width : equivalenceWidths()) {
        for (int height : {1, 3, 17}) {
            for (size_t f = 0; f < equivalenceFillRatios().size(); ++f) {
                const float fill = equivalenceFillRatios()[f];
                const uint64_t seed = caseSeed(width, height, f);
                const std::string label = caseLabel(wordTypeName, width, height, fill);

                // Minimum stride, and an over-aligned one: a dirty bit can hide in
                // either the trailing partial word or the whole padding words past it.
                bincv::BinMat<WordType> tight = randomBinary<WordType>(width, height, fill, seed);
                BINCV_EXPECT_PADDING_CLEAN(tight.constView(), label + " [tight]");

                bincv::BinMat<WordType> padded =
                    randomBinary<WordType>(width, height, fill, seed, PADDED_ROW_ALIGNMENT);
                BINCV_EXPECT_PADDING_CLEAN(padded.constView(), label + " [padded]");

                // A whole-word popcount over the stride must agree with the
                // per-pixel count. This is the reduction the hard rule protects,
                // and the arithmetic that a phantom bit actually corrupts.
                size_t overStride = 0;
                for (int y = 0; y < padded.rows(); ++y) {
                    const WordType* row = padded.ptr(y);
                    for (size_t w = 0; w < padded.getAlignedWidth(); ++w) {
                        WordType v = row[w];
                        while (v != 0) {
                            overStride += static_cast<size_t>(v & static_cast<WordType>(1));
                            v = static_cast<WordType>(v >> 1);
                        }
                    }
                }
                BINCV_CHECK_EQ(static_cast<int>(overStride), padded.countNonZero());
            }
        }
    }

    // The stride the sweep is built on really is larger than the minimum, or the
    // [padded] half above would be a second copy of the [tight] half.
    bincv::BinMat<WordType> probe(70, 3, PADDED_ROW_ALIGNMENT);
    const size_t minWords = (70 + wordBits - 1) / wordBits;
    BINCV_CHECK(probe.getAlignedWidth() > minWords);

    // And a hand-dirtied padding bit is detected at every word width, in both the
    // trailing partial word and the words past it.
    {
        bincv::BinMat<WordType> dirty = randomBinary<WordType>(70, 3, 0.5f, 1, PADDED_ROW_ALIGNMENT);
        BINCV_CHECK(!bincv::test::firstDirtyPaddingBit(dirty.constView()).found);
        dirty.ptr(1)[dirty.getAlignedWidth() - 1] = static_cast<WordType>(1);
        const bincv::test::PaddingViolation p =
            bincv::test::firstDirtyPaddingBit(dirty.constView());
        BINCV_CHECK(p.found);
        BINCV_CHECK_EQ(static_cast<int>(p.row), 1);
        BINCV_CHECK_EQ(static_cast<int>(p.dirtyBits), 1);
        BINCV_CHECK_EQ(static_cast<int>(p.bit),
                       static_cast<int>((dirty.getAlignedWidth() - 1) * wordBits));
    }
}

/// @brief A faulted build must actually behave differently from an unfaulted one.
/// @note The only use of faultInjected, and the reason it is not dead code: a
/// CMake typo that stopped passing BINCV_EQUIVALENCE_INJECT_FAULT would
/// leave a WILL_FAIL target that fails for no reason at all, and an
/// inverted result looks identical either way.
void testInjectedFaultIsLive() {
    std::cout << "\n--- the injected fault is live (or absent) as declared ---\n";

    bool conversionDiffers = false;
    bool paddingDirty = false;
    const int shapes[][2] = {{70, 37}, {65, 17}, {31, 3}};
    for (const auto& shape : shapes) {
        bincv::BinMat32 m =
            randomBinary<uint32_t>(shape[0], shape[1], 0.5f, caseSeed(shape[0], shape[1], 1));
        const cv::Mat asBuilt = unpackTo8U(m.constView());                        // INJECTED_FAULT
        const cv::Mat pristine = unpackTo8U(m.constView(), ConversionFault::None);
        if (compareMasks(asBuilt, pristine).found) conversionDiffers = true;
        if (bincv::test::firstDirtyPaddingBit(m.constView()).found) paddingDirty = true;
    }

    std::cout << " faultInjected()=" << (bincv::test::faultInjected() ? "true" : "false")
              << " conversionDiffers=" << conversionDiffers
              << " paddingDirty=" << paddingDirty << "\n";

    if (bincv::test::faultInjected()) {
        // Something must be visibly wrong, or the WILL_FAIL target is passing for
        // a reason unrelated to the fault it names.
        BINCV_CHECK(conversionDiffers || paddingDirty);
    } else {
        // And in an ordinary build the default conversion IS the None conversion,
        // and the generator leaves the padding alone.
        BINCV_CHECK(!conversionDiffers);
        BINCV_CHECK(!paddingDirty);
    }
}

// ===========================================================================
// 3. The anchor: the conversion, pinned against content it never touched
// ===========================================================================

/// @brief The full the sweep, per word type.
/// @note This is what stops the harness being circular. randomCvMask builds the
/// same content directly as CV_8U without constructing a BinMat or calling
/// unpackTo8U, so an off-by-one or a dropped tail in the conversion shows up
/// here even though it would CANCEL inside any pointwise-kernel test built
/// on the same conversion (see equivalence.hpp).
template <typename WordType>
void testPackingAnchor(const char* wordTypeName) {
    std::cout << "\n--- packing anchor: " << wordTypeName << " ---\n";

    for (int width : equivalenceWidths()) {
        for (int height : equivalenceHeights()) {
            for (size_t f = 0; f < equivalenceFillRatios().size(); ++f) {
                const float fill = equivalenceFillRatios()[f];
                const uint64_t seed = caseSeed(width, height, f);
                const std::string label = caseLabel(wordTypeName, width, height, fill);

                const cv::Mat reference = randomCvMask(width, height, fill, seed);

                // (a) the packed generator agrees with the CV_8U one
                bincv::BinMat<WordType> generated =
                    randomBinary<WordType>(width, height, fill, seed);
                BINCV_EXPECT_BIT_EXACT(generated.constView(), reference, label + " [generated]");

                // (b) so does content that entered through fromCVMat -- an
                // already-implemented operation, and the one every future
                // Tier 1 test will use to build its inputs
                bincv::BinMat<WordType> loaded;
                loaded.fromCVMat(reference);
                BINCV_EXPECT_BIT_EXACT(loaded.constView(), reference, label + " [fromCVMat]");

                // (c) and so does the same content in a matrix whose rows are
                // OVER-ALIGNED, so the view's stride exceeds the
                // ceil(width / WordBits) words the row needs.
                //
                // Measured on the default-alignment sweep: stride was the
                // minimum in 48 of 48 cases, so nothing here exercised a
                // stride a kernel could get wrong -- and "strides may differ
                // between arguments" is precisely what warn
                // about. A stand-in kernel that walked src and dst as one
                // dense run and ignored view.stride was reported identical
                // on every matrix this sweep used to build.
                bincv::BinMat<WordType> padded =
                    randomBinary<WordType>(width, height, fill, seed, PADDED_ROW_ALIGNMENT);
                BINCV_EXPECT_BIT_EXACT(padded.constView(), reference, label + " [padded stride]");

                // (d) and so does a QuantMat<3> BIT PLANE, which is a different
                // view entirely: same word type and stride, but offset by the
                // plane pitch (height * strideWords) rather than starting at
                // the allocation.
                //
                // Added because the anchor pinned only BinMat views while
                // that work’s plane overloads are checked through constPlane ->
                // unpackTo8U on the binCV side -- so the one conversion those
                // 2592 checks depend on was the one conversion nothing here
                // pinned. A plane pitch off by a row would be invisible to any
                // test that also built its expectation from constPlane.
                //
                // Each plane carries INDEPENDENT content, so a plane loop that
                // read the wrong plane cannot pass by symmetry.
                if (width > 0 && height > 0) {
                    cv::Mat planeMasks[3];
                    for (size_t p = 0; p < 3; ++p) {
                        planeMasks[p] =
                            randomCvMask(width, height, fill,
                                         seed + UINT64_C(0x51ED270F) * (p + 1));
                    }
                    bincv::QuantMat<3, WordType> quant(width, height);
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            unsigned value = 0;
                            for (size_t p = 0; p < 3; ++p) {
                                if (planeMasks[p].ptr<uint8_t>(y)[x] != 0) {
                                    value |= (1u << p);
                                }
                            }
                            quant.set(y, x, value);
                        }
                    }
                    for (size_t p = 0; p < 3; ++p) {
                        BINCV_EXPECT_BIT_EXACT(quant.constPlane(p), planeMasks[p],
                                               label + " [quantmat plane " +
                                                   std::to_string(p) + "]");
                    }
                }
            }
        }
    }
}

// ===========================================================================
// 4. The demonstration: countNonZero against cv::countNonZero
// ===========================================================================

/// @brief that work’s "done when": the harness, on an operation that already exists.
/// @note The denominator is the design notes's -- OpenCV performing the same
/// semantic operation on the SAME binary content stored as CV_8U, which is
/// what toCvMask produces.
/// @note Note what this case can and cannot see. countNonZero reduces an image to
/// one number, so it agrees under any permutation of the pixels: the
/// comparator cases above show a cyclic column rotation passing a
/// count-only check. Bit-exactness of the CONTENT is testPackingAnchor's
/// job, and the two together are what make this demonstration mean
/// something.
template <typename WordType>
void testCountNonZeroEquivalence(const char* wordTypeName) {
    std::cout << "\n--- countNonZero vs cv::countNonZero: " << wordTypeName << " ---\n";

    for (int width : equivalenceWidths()) {
        for (int height : equivalenceHeights()) {
            for (size_t f = 0; f < equivalenceFillRatios().size(); ++f) {
                const float fill = equivalenceFillRatios()[f];
                bincv::BinMat<WordType> m =
                    randomBinary<WordType>(width, height, fill, caseSeed(width, height, f));

                const cv::Mat equivalent = toCvMask(m.constView());
                BINCV_CHECK_EQ(m.countNonZero(), cv::countNonZero(equivalent));
            }
        }
    }
}

// ===========================================================================
// 5. A second demonstration, on an operation that PRODUCES an image
// ===========================================================================

/// @brief transposed against cv::transpose, through expectBitExact.
/// @note countNonZero returns a scalar, so it cannot exercise the pixel-by-pixel
/// comparison that every Tier 1 kernel in earlier work..this will assert through.
/// This does: an already-implemented binCV operation, its exact OpenCV
/// equivalent, and the two compared through the harness.
/// @note Deliberately smaller than the full sweep -- transposed is a naive
/// per-pixel copy (its @todo says so), and this file is on the critical
/// path of a gate that should stay cheap to run.
/// @note MEASURED, and it is the clearest evidence for why testPackingAnchor
/// exists: with ConversionFault::TransposeRowCol compiled in, this case
/// still PASSES. The fault transposes toCvMask's output AND unpackTo8U's,
/// cv::transpose sits between them, and the two cancel exactly. The
/// countNonZero case likewise survives ColumnOffByOne, a cyclic
/// permutation being count-preserving. A conversion bug is invisible to a
/// test whose two sides share the conversion; only the anchor, whose
/// expectation comes from randomCvMask, can see it.
template <typename WordType>
void testTransposeEquivalence(const char* wordTypeName) {
    std::cout << "\n--- transposed() vs cv::transpose: " << wordTypeName << " ---\n";

    const int widths[] = {1, 7, 31, 33, 63, 65, 70};
    const int heights[] = {1, 2, 3, 17, 37};
    for (int width : widths) {
        for (int height : heights) {
            const float fill = 0.5f;
            const std::string label = caseLabel(wordTypeName, width, height, fill);
            bincv::BinMat<WordType> src =
                randomBinary<WordType>(width, height, fill, caseSeed(width, height, 2));

            cv::Mat expected;
            cv::transpose(toCvMask(src.constView()), expected);

            bincv::BinMat<WordType> actual = src.transposed();
            BINCV_EXPECT_BIT_EXACT(actual.constView(), expected, label + " [transposed]");
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Registration
//
// One case per (behaviour, word type), matching the naming the rest of the
// suites use so a --gtest_filter written for one narrows the others.
// ---------------------------------------------------------------------------

BINCV_TEST(Equivalence, GeneratorReproducible)        { testGeneratorReproducible(); }
BINCV_TEST(Equivalence, GeneratorDegenerateDimensions){ testGeneratorDegenerateDimensions(); }
BINCV_TEST(Equivalence, GeneratorWordTypeIndependent) { testGeneratorWordTypeIndependent(); }
BINCV_TEST(Equivalence, GeneratorFillRatios)          { testGeneratorFillRatios(); }
BINCV_TEST(Equivalence, GeneratorGoldenValues)        { testGeneratorGoldenValues(); }

BINCV_TEST(Equivalence, ComparatorCatchesFaults)      { testComparatorCatchesFaults(); }
BINCV_TEST(Equivalence, ComparatorAcceptsIdentical)   { testComparatorAcceptsIdentical(); }

BINCV_TEST(Equivalence, ExpectBitExactReportsFailures) { testExpectBitExactReportsFailures(); }
BINCV_TEST(Equivalence, InjectedFaultIsLive)           { testInjectedFaultIsLive(); }

BINCV_TEST(Equivalence, PaddingInvariant_uint8_t)  { testPaddingInvariant<uint8_t>("uint8_t"); }
BINCV_TEST(Equivalence, PaddingInvariant_uint16_t) { testPaddingInvariant<uint16_t>("uint16_t"); }
BINCV_TEST(Equivalence, PaddingInvariant_uint32_t) { testPaddingInvariant<uint32_t>("uint32_t"); }
BINCV_TEST(Equivalence, PaddingInvariant_uint64_t) { testPaddingInvariant<uint64_t>("uint64_t"); }

BINCV_TEST(Equivalence, PackingAnchor_uint8_t)  { testPackingAnchor<uint8_t>("uint8_t"); }
BINCV_TEST(Equivalence, PackingAnchor_uint16_t) { testPackingAnchor<uint16_t>("uint16_t"); }
BINCV_TEST(Equivalence, PackingAnchor_uint32_t) { testPackingAnchor<uint32_t>("uint32_t"); }
BINCV_TEST(Equivalence, PackingAnchor_uint64_t) { testPackingAnchor<uint64_t>("uint64_t"); }

BINCV_TEST(Equivalence, CountNonZero_uint8_t)  { testCountNonZeroEquivalence<uint8_t>("uint8_t"); }
BINCV_TEST(Equivalence, CountNonZero_uint16_t) { testCountNonZeroEquivalence<uint16_t>("uint16_t"); }
BINCV_TEST(Equivalence, CountNonZero_uint32_t) { testCountNonZeroEquivalence<uint32_t>("uint32_t"); }
BINCV_TEST(Equivalence, CountNonZero_uint64_t) { testCountNonZeroEquivalence<uint64_t>("uint64_t"); }

BINCV_TEST(Equivalence, Transpose_uint8_t)  { testTransposeEquivalence<uint8_t>("uint8_t"); }
BINCV_TEST(Equivalence, Transpose_uint16_t) { testTransposeEquivalence<uint16_t>("uint16_t"); }
BINCV_TEST(Equivalence, Transpose_uint32_t) { testTransposeEquivalence<uint32_t>("uint32_t"); }
BINCV_TEST(Equivalence, Transpose_uint64_t) { testTransposeEquivalence<uint64_t>("uint64_t"); }

BINCV_TEST_MAIN("BinMat equivalence harness tests")
