#pragma once

/// @file equivalence.hpp
/// @brief The Tier 1 equivalence harness (T2.1): bit-exactness against OpenCV.
///
/// ARCHITECTURE 10.2 says every Tier 1 operation ships with a test asserting
/// bit-exactness against the equivalent OpenCV expression on the same content.
/// This header is the machinery that makes that a *claim* rather than an
/// assertion, and it is deliberately built BEFORE the kernels of T2.2-T2.7, so
/// that no kernel can be written to fit a weak test.
///
/// Two entry points, per the T2.1 spec:
///
///   expectBitExact(actual, expected)   -- convert a BinMat view to CV_8U {0,255}
///                                         and compare EVERY pixel against what
///                                         OpenCV produced, AND require the
///                                         result's padding bits to be zero.
///   randomBinary<W>(w, h, fill, seed)  -- reproducible random content.
///
/// and two things the spec did not ask for. A 640x480 failure is unactionable
/// without firstMismatch(), which reports WHERE the two disagree -- row, column,
/// expected byte, actual byte, and how many pixels differ in total. And a pixel
/// comparison alone cannot see a violation of the project's padding rule, so
/// firstDirtyPaddingBit() / expectPaddingClean() report that separately --
/// see property 4 below.
///
/// ---------------------------------------------------------------------------
/// WHY THIS HARNESS IS NOT VACUOUS
///
/// A harness that passes everything is worse than no harness, because it
/// licenses every kernel built on it. Five structural properties keep this one
/// able to fail, and each is checked by tests/test_equivalence.cpp rather than
/// argued for here:
///
///  1. **The anchor.** The obvious way to write this harness is circular:
///     generate content in a BinMat, convert it to CV_8U with unpackTo8U(), and
///     then compare the BinMat against that same conversion. That tautology
///     passes even if unpackTo8U reads the wrong column. So the harness carries
///     a SECOND generator, randomCvMask(), which produces the same content
///     directly as CV_8U without touching a BinMat or the unpacking path at all.
///     Equivalence.PackingAnchor pins unpackTo8U against it across the whole size
///     and fill matrix. That is the test that catches a conversion bug.
///
///  2. **Shared conversions can CANCEL, and the anchor is what stops them.**
///     A future logic kernel is tested as: cvA = toCvMask(a), cvB = toCvMask(b),
///     cv::bitwise_and(cvA, cvB, cvExpected), bincv::bitwiseAnd(a, b, dst),
///     expectBitExact(dst, cvExpected). If toCvMask carried a one-column shift,
///     every one of those five values would be shifted identically and the
///     comparison would still pass -- the fault cancels through a pointwise
///     operation. Nothing inside such a test can notice. Only property 1 can,
///     which is why the anchor runs over the same size matrix and is not a
///     one-off smoke test.
///
///  3. **Injectable faults.** BINCV_EQUIVALENCE_INJECT_FAULT compiles a
///     deliberate bug into the harness itself -- into the unpacking path, or into
///     the generator's padding -- and tests/CMakeLists.txt builds
///     tests/test_equivalence.cpp once per fault as a WILL_FAIL ctest case. A
///     harness whose ability to fail is asserted in a report decays; one whose
///     ability to fail is four ctest cases does not.
///
///  4. **A pixel comparison is BLIND to padding bits, so it is not the only
///     check.** unpackTo8U() never reads a bit at or past `width`, which means a
///     kernel that writes whole words and forgets to clear the trailing partial
///     one produces a result that is bit-exact pixel for pixel and still breaks
///     CLAUDE.md's hard rule ("padding bits stay zero ... or word-wise reductions
///     over-count"). MEASURED, before this check existed: a word-wise bitwiseNot
///     over the full T2.1 sweep passed 240 of 240 cases at uint64_t while leaving
///     826,200 phantom set bits behind, and countNonZero -- which loops x < width
///     -- agreed with OpenCV throughout. expectBitExact() therefore reports TWO
///     verdicts per call, the pixel comparison and expectPaddingClean(), so the
///     two failures stay distinguishable in the log.
///
///  5. **The verdicts go through a seam the suite can capture.** expectBitExact()
///     does not call reportCheck() directly; it calls checkReporter(), which is
///     reportCheck() by default and which VerdictCapture swaps out. That is what
///     lets tests/test_equivalence.cpp drive expectBitExact's own FAILURE path
///     in-process -- feeding it a flipped pixel, a shape change and a dirty
///     padding word and requiring it to report each -- without failing itself.
///     MEASURED, before the seam existed: neutering expectBitExact so that it
///     reported success unconditionally still produced "3392/3392 checks passed"
///     and exit 0, and compiled warning-free under -Werror.
///
/// ---------------------------------------------------------------------------
/// SCOPE
///
/// OpenCV-only, guarded by BINCV_WITH_OPENCV: the core-only, no-exceptions and
/// Debug configurations never compile it and are unaffected by it. It lives in
/// tests/ and not in include/ on purpose -- it depends on cv::Mat, and nothing
/// the embedded configurations build may.
///
/// @note This header stays in plain `bincv::test` rather than opening
///       `inline namespace BINCV_ABI_NAMESPACE` (D-10), matching
///       tests/test_util.hpp. Opening it here would create a second `test`
///       namespace inside the inline one, and `bincv::test` would then be
///       ambiguous between the two in every suite that includes both. D-10's
///       hazard -- one configuration's inline function bodies silently merging
///       with another's at link time -- does not arise for a test-side header
///       that only ever appears in a single translation unit per binary.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV

#include <opencv2/core.hpp>

namespace bincv {
namespace test {

// ---------------------------------------------------------------------------
// Reproducible pseudo-randomness
// ---------------------------------------------------------------------------

/// @brief SplitMix64 -- the harness's only source of randomness.
///
/// @note NOT std::random_device and NOT an unseeded generator: content that is
///       not reproducible turns a failing case into an anecdote. Every matrix
///       this harness produces is a pure function of (width, height, fillRatio,
///       seed).
///
/// @note NOT std::uniform_int_distribution either, and this is the part that is
///       easy to get wrong. The standard specifies what a distribution must
///       *mean*, not how it must consume its engine, so the same engine, seed and
///       distribution give DIFFERENT sequences on libstdc++, libc++ and MSVC. A
///       golden value recorded on one platform would then fail on another and the
///       failure would look like a packing bug. The mapping from random bits to a
///       pixel is therefore written out by hand below, in exact-width unsigned
///       integer arithmetic, which is the one thing the standard does pin down.
///
/// @note The engine itself is chosen for the same reason: SplitMix64 is four
///       operations on a uint64_t, all of them modular by definition, with no
///       table, no state array and no implementation freedom. std::mt19937_64
///       would also be portable as an *engine*; SplitMix64 is simply smaller and
///       leaves nothing to look up.
class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}

    /// @brief The next 64 random bits.
    /// @note Every operation here is on uint64_t, so it is modular arithmetic
    ///       with a fixed width -- identical on every conforming implementation,
    ///       32-bit and 64-bit hosts alike, big-endian and little-endian alike
    ///       (no bytes are ever reinterpreted).
    uint64_t next() {
        state_ += UINT64_C(0x9E3779B97F4A7C15);
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
        return z ^ (z >> 31);
    }

private:
    uint64_t state_;
};

/// @brief Denominator of the fill-ratio comparison: 2^24.
/// @note A power of two, and small enough that a float fill ratio scales into it
///       without any rounding at all -- see fillThreshold().
constexpr uint32_t FILL_SCALE = 1u << 24;

/// @brief Maps a fill ratio to the integer threshold a pixel draw is compared to.
///
/// @return A value in [0, FILL_SCALE]. A pixel is set iff its 24-bit draw is
///         strictly less than this, so 0 means "never" and FILL_SCALE means
///         "always" (the largest possible draw is FILL_SCALE - 1).
///
/// @note Portable by construction, which is the whole reason it is written out:
///       float -> double is exact, multiplying by 2^24 only changes the exponent
///       so it is exact too, the single `+ 0.5` is the only rounding in the
///       function, and truncation of a value below 2^25 is exact. No
///       implementation-defined step remains, so 0.01f gives the same threshold
///       on every IEEE-754 host.
/// @note The endpoints are exact rather than probabilistic: fillRatio <= 0
///       produces an all-zero matrix and fillRatio >= 1 an all-ones one. The
///       T2.1 fill matrix includes both, and "approximately empty" would be a
///       different test.
inline uint32_t fillThreshold(float fillRatio) {
    if (!(fillRatio > 0.0f)) return 0;               // <= 0, and NaN
    if (fillRatio >= 1.0f) return FILL_SCALE;
    const double rounded =
        static_cast<double>(fillRatio) * static_cast<double>(FILL_SCALE) + 0.5;
    if (rounded >= static_cast<double>(FILL_SCALE)) return FILL_SCALE;
    return static_cast<uint32_t>(rounded);
}

/// @brief One pixel: draws 24 bits and compares them against `threshold`.
/// @note The top 24 bits are used because SplitMix64's low bits are the ones its
///       final xorshift mixes least.
/// @note A draw happens for EVERY pixel, including at fill ratios 0.0 and 1.0
///       where the answer is already known. That keeps the position in the
///       sequence a function of (row, column) alone, so the same seed at
///       different fill ratios produces matrices that are comparable pixel for
///       pixel rather than shifted relative to one another.
inline bool nextPixel(SplitMix64& rng, uint32_t threshold) {
    return static_cast<uint32_t>(rng.next() >> 40) < threshold;
}

// ---------------------------------------------------------------------------
// Injectable faults
// ---------------------------------------------------------------------------

/// @brief A deliberate bug compiled into the harness's own machinery.
/// @note Not a debugging aid. tests/CMakeLists.txt compiles
///       tests/test_equivalence.cpp once per non-None value and registers each
///       build WILL_FAIL, so "this harness can detect a real discrepancy" is a
///       ctest result rather than a claim in a commit message.
/// @note Three of the four corrupt the BinMat -> CV_8U conversion, which is what
///       every pixel comparison flows through. The fourth, DirtyPadding, corrupts
///       the GENERATOR instead, because the invariant it breaks is one no pixel
///       comparison can see -- the conversion never reads a bit at or past
///       `width`. It is the fault that proves expectPaddingClean() can fail.
enum class ConversionFault {
    None = 0,
    ColumnOffByOne = 1,    ///< reads column (x + 1) mod width instead of column x
    DropTrailingWord = 2,  ///< pixels in the final PARTIAL word read as zero
    TransposeRowCol = 3,   ///< emits the transpose: (x, y) where (y, x) belongs
    DirtyPadding = 4       ///< randomBinary() leaves every bit past `width` SET
};

/// @def BINCV_EQUIVALENCE_INJECT_FAULT
/// @brief 0 in every ordinary build. Set to a ConversionFault value to compile a
///        deliberate bug into the harness, which every faulted build must fail on.
#ifndef BINCV_EQUIVALENCE_INJECT_FAULT
#  define BINCV_EQUIVALENCE_INJECT_FAULT 0
#endif

inline constexpr ConversionFault INJECTED_FAULT =
    static_cast<ConversionFault>(BINCV_EQUIVALENCE_INJECT_FAULT);

/// @brief True when this translation unit was built with a fault compiled in.
/// @note Used by Equivalence.InjectedFaultIsLive, which requires a faulted build
///       to actually BEHAVE differently from an unfaulted one. Without that, a
///       CMake typo that stopped passing BINCV_EQUIVALENCE_INJECT_FAULT would
///       turn a WILL_FAIL target into a target that fails for no reason at all,
///       and the inverted result would look exactly the same.
inline constexpr bool faultInjected() { return INJECTED_FAULT != ConversionFault::None; }

/// @brief Sets every padding bit of every row: the DirtyPadding fault's payload.
/// @note Deliberately writes through raw row pointers rather than set(), which
///       refuses to address a pixel past `width` -- breaking the invariant is the
///       entire point. Compiled in every build so it stays type-checked and
///       warning-clean; only the WILL_FAIL target ever calls it.
template <typename WordType>
void dirtyThePadding(BinMat<WordType>& m) {
    constexpr size_t wordBits = BinMat<WordType>::WordBits;
    if (m.empty()) return;

    const size_t minWords = (m.getWidth() + wordBits - 1) / wordBits;
    const size_t tailBits = m.getWidth() % wordBits;
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));

    for (int y = 0; y < m.rows(); ++y) {
        WordType* row = m.ptr(y);
        if (tailBits != 0) {
            const WordType paddingMask = static_cast<WordType>(allOnes << tailBits);
            row[minWords - 1] = static_cast<WordType>(row[minWords - 1] | paddingMask);
        }
        for (size_t w = minWords; w < m.getAlignedWidth(); ++w) row[w] = allOnes;
    }
}

/// @brief A random binary matrix at a given fill ratio, seeded and reproducible.
/// @param width Width in pixels; <= 0 yields an empty matrix.
/// @param height Height in pixels; <= 0 yields an empty matrix.
/// @param fillRatio Expected fraction of set pixels. 0.0 and 1.0 are exact.
/// @param seed Any 64-bit value. The same seed always gives the same matrix.
/// @param rowAlignment Row-stride alignment in bytes, forwarded to BinMat. The
///        default is word granularity (D-4), i.e. the MINIMUM stride, which is
///        the one case a kernel must not be allowed to assume: pass a larger
///        value to build a view whose stride exceeds ceil(width / WordBits).
///
/// @note The `<= 0 yields an empty matrix` contract is enforced BEFORE the
///       BinMat is constructed, deliberately. Constructing first meant a negative
///       dimension was reported by BinMat's own precondition (a throw, or an
///       abort under BINCV_NO_EXCEPTIONS) and the guard below was dead for every
///       negative value it named -- so this generator and randomCvMask(), which
///       document the contract in identical words, disagreed on it.
///
/// @note **Word-type independent.** Pixels are drawn in row-major order and
///       written through set(), one draw per pixel, so BinMat<uint8_t> and
///       BinMat<uint64_t> built from the same seed hold the same picture. Drawing
///       whole words instead would have tied the content to the word width and
///       made "the same seed on any word type" false -- and the T2.1 size matrix
///       exists precisely to compare word widths against each other.
/// @note Written through set(), so bits past `width` are never touched and the
///       padding-bit invariant holds on return -- asserted over the whole size,
///       height, fill and word-type matrix by Equivalence.PaddingInvariant_*,
///       and not merely intended. (Unless ConversionFault::DirtyPadding is
///       compiled in, whose whole purpose is to break it.)
/// @note Row-major and one draw per pixel also means a 640-wide matrix's first
///       row is the first 640 draws of the sequence, which makes a failing case
///       reproducible in isolation.
template <typename WordType>
BinMat<WordType> randomBinary(int width, int height, float fillRatio, uint64_t seed,
                              size_t rowAlignment = BinMat<WordType>::DefaultRowAlignment) {
    if (width <= 0 || height <= 0) return BinMat<WordType>();

    BinMat<WordType> m(width, height, rowAlignment);
    SplitMix64 rng(seed);
    const uint32_t threshold = fillThreshold(fillRatio);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (nextPixel(rng, threshold)) m.set(y, x, true);
        }
    }

    // The DirtyPadding fault, and the only fault that lives outside the
    // conversion: it makes the generator break the invariant that
    // expectPaddingClean() exists to enforce, which is what turns "the padding
    // check can fail" into a ctest result. Compiled (and type-checked) in every
    // build; INJECTED_FAULT is None in all but the WILL_FAIL targets.
    if (INJECTED_FAULT == ConversionFault::DirtyPadding) {
        dirtyThePadding(m);
    }
    return m;
}

/// @brief The same content as randomBinary(), produced directly as CV_8U {0,255}.
///
/// @note This is the harness's independent side, and it is what keeps the whole
///       thing from being circular. It never constructs a BinMat and never calls
///       the unpacking path, so comparing randomBinary() against it actually
///       tests the packing -- see the "anchor" note at the top of this file.
/// @note It must stay a separate loop over the same primitives. Refactoring the
///       two generators to share a fill routine would reintroduce exactly the
///       circularity it exists to break.
inline cv::Mat randomCvMask(int width, int height, float fillRatio, uint64_t seed) {
    if (width <= 0 || height <= 0) return cv::Mat();

    cv::Mat out = cv::Mat::zeros(height, width, CV_8U);
    SplitMix64 rng(seed);
    const uint32_t threshold = fillThreshold(fillRatio);
    for (int y = 0; y < height; ++y) {
        uint8_t* row = out.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            row[x] = nextPixel(rng, threshold) ? static_cast<uint8_t>(255)
                                               : static_cast<uint8_t>(0);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// BinMat -> CV_8U
// ---------------------------------------------------------------------------

/// @brief Unpacks a view to a CV_8U matrix whose pixels are 0 or 255.
/// @param v The view to convert.
/// @param fault Normally INJECTED_FAULT, i.e. None. See ConversionFault.
///
/// @note 255 rather than 1 because that is the value OpenCV's own binary
///       operations produce and compare against -- cv::threshold, cv::bitwise_*
///       on a mask, cv::erode. Comparing against a {0,1} matrix would make the
///       harness agree with OpenCV only after a normalisation step, and a
///       normalisation step is somewhere for a discrepancy to hide.
/// @note Reads the view's words directly rather than going through BinMat::at(),
///       so what it validates is the LAYOUT the view describes -- which is what
///       kernels will bind to (D-5) -- and not the container's accessor.
/// @note Pixels at or past `width` are never read, so a dirty padding bit cannot
///       reach the output. That keeps the two failures separable -- a wrong PIXEL
///       and a dirty PADDING BIT are different bugs and get different messages --
///       but it also means this function alone CANNOT see a padding violation.
///       expectBitExact() therefore pairs it with expectPaddingClean(); see
///       property 4 at the top of this file for the measurement that shows why a
///       pixel comparison on its own licenses a broken kernel.
template <typename WordType>
cv::Mat unpackTo8U(const BinMatConstView<WordType>& v,
                   ConversionFault fault = INJECTED_FAULT) {
    constexpr size_t wordBits = BinMatConstView<WordType>::WordBits;
    if (v.width == 0 || v.height == 0 || v.ptr == nullptr) return cv::Mat();

    const int w = static_cast<int>(v.width);
    const int h = static_cast<int>(v.height);
    const bool transposed = (fault == ConversionFault::TransposeRowCol);

    cv::Mat out = cv::Mat::zeros(transposed ? w : h, transposed ? h : w, CV_8U);

    // First column of the trailing PARTIAL word, or `width` when the row ends on
    // a word boundary and there is no partial word to drop.
    const size_t tailStart =
        (v.width % wordBits == 0) ? v.width : (v.width - (v.width % wordBits));

    for (size_t y = 0; y < v.height; ++y) {
        const WordType* rowWords = v.row(y);
        for (size_t x = 0; x < v.width; ++x) {
            size_t src = x;
            if (fault == ConversionFault::ColumnOffByOne) src = (x + 1) % v.width;

            const WordType mask =
                static_cast<WordType>(static_cast<WordType>(1) << (src % wordBits));
            bool bit = (rowWords[src / wordBits] & mask) != 0;

            if (fault == ConversionFault::DropTrailingWord && x >= tailStart) bit = false;

            const uint8_t value = bit ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
            if (transposed) {
                out.ptr<uint8_t>(static_cast<int>(x))[y] = value;
            } else {
                out.ptr<uint8_t>(static_cast<int>(y))[x] = value;
            }
        }
    }
    return out;
}

/// @brief The benchmark and equivalence denominator: this binary content, as CV_8U.
/// @note ARCHITECTURE 10.3 -- "OpenCV performing the same semantic operation on
///       the same binary content stored as CV_8U" -- so this is the function that
///       builds OpenCV's *inputs* in a Tier 1 equivalence test, while
///       expectBitExact() checks its *output*.
template <typename WordType>
cv::Mat toCvMask(const BinMatConstView<WordType>& v) {
    return unpackTo8U(v);
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// @brief Where two masks first disagree, and how badly.
/// @note "Not equal" is not a usable failure message for a 640x480 image. This is
///       what makes one: the first differing pixel in row-major order, both byte
///       values, and the total number of differing pixels -- which distinguishes
///       a single-bit bug from a whole-image one at a glance.
struct Mismatch {
    bool found = false;         ///< true if the two disagree in any way
    bool shapeMismatch = false; ///< dimensions or element type differ
    int row = -1;               ///< row of the first differing pixel
    int col = -1;               ///< column of the first differing pixel
    int expected = -1;          ///< OpenCV's byte at (row, col)
    int actual = -1;            ///< binCV's byte at (row, col)
    size_t differing = 0;       ///< how many pixels differ in total
    size_t total = 0;           ///< how many pixels were compared
    int actualWidth = 0, actualHeight = 0;
    int expectedWidth = 0, expectedHeight = 0;
    std::string note;           ///< set for shape mismatches

    /// @brief A one-line, actionable description.
    std::string describe() const {
        if (!found) return "identical";
        if (shapeMismatch) {
            std::string s = "shape mismatch: binCV " + std::to_string(actualWidth) + "x" +
                            std::to_string(actualHeight) + ", OpenCV " +
                            std::to_string(expectedWidth) + "x" + std::to_string(expectedHeight);
            if (!note.empty()) s += " (" + note + ")";
            return s;
        }
        return "first mismatch at row " + std::to_string(row) + ", col " + std::to_string(col) +
               ": expected " + std::to_string(expected) + ", actual " + std::to_string(actual) +
               "; " + std::to_string(differing) + " of " + std::to_string(total) +
               " pixels differ";
    }
};

/// @brief Compares two CV_8U masks pixel by pixel. Every pixel, no sampling.
/// @note Scans to the end even after the first difference, because the COUNT is
///       what separates "one bit is wrong" from "the whole image is shifted", and
///       those two failures are debugged differently.
inline Mismatch compareMasks(const cv::Mat& actual, const cv::Mat& expected) {
    Mismatch m;
    m.actualWidth = actual.cols;
    m.actualHeight = actual.rows;
    m.expectedWidth = expected.cols;
    m.expectedHeight = expected.rows;

    if (!expected.empty() && expected.type() != CV_8UC1) {
        m.found = true;
        m.shapeMismatch = true;
        m.note = "OpenCV matrix is not CV_8UC1";
        return m;
    }
    if (!actual.empty() && actual.type() != CV_8UC1) {
        m.found = true;
        m.shapeMismatch = true;
        m.note = "binCV matrix is not CV_8UC1";
        return m;
    }
    if (actual.rows != expected.rows || actual.cols != expected.cols) {
        m.found = true;
        m.shapeMismatch = true;
        return m;
    }

    m.total = static_cast<size_t>(actual.rows) * static_cast<size_t>(actual.cols);
    for (int y = 0; y < actual.rows; ++y) {
        const uint8_t* a = actual.ptr<uint8_t>(y);
        const uint8_t* e = expected.ptr<uint8_t>(y);
        for (int x = 0; x < actual.cols; ++x) {
            if (a[x] == e[x]) continue;
            ++m.differing;
            if (!m.found) {
                m.found = true;
                m.row = y;
                m.col = x;
                m.actual = a[x];
                m.expected = e[x];
            }
        }
    }
    return m;
}

/// @brief Where a binCV result first disagrees with an OpenCV one.
/// @note The overload a kernel test calls when it wants to inspect the failure
///       rather than assert on it.
/// @note PIXELS ONLY. This cannot see a dirty padding bit -- unpackTo8U never
///       reads one. Pair it with firstDirtyPaddingBit(), or call expectBitExact(),
///       which does both.
template <typename WordType>
Mismatch firstMismatch(const BinMatConstView<WordType>& actual, const cv::Mat& expected) {
    return compareMasks(unpackTo8U(actual), expected);
}

// ---------------------------------------------------------------------------
// The padding-bit invariant
// ---------------------------------------------------------------------------

/// @brief Where a view's padding bits are dirty, and how badly.
/// @note CLAUDE.md, hard rules: "Padding bits stay zero. Any operation that
///       writes whole words past `width` must clear them, or word-wise reductions
///       over-count." A pixel comparison is structurally blind to this, so it gets
///       its own report with its own message rather than being folded into
///       Mismatch -- a wrong pixel and an uncleared tail are fixed differently.
struct PaddingViolation {
    bool found = false;      ///< true if any bit at or past `width` is set
    size_t row = 0;          ///< row of the first dirty bit
    size_t bit = 0;          ///< its bit index within the row, counting from pixel 0
    size_t dirtyBits = 0;    ///< how many padding bits are set in total
    size_t width = 0;        ///< the view's width in pixels
    size_t rowWords = 0;     ///< words scanned per row

    /// @brief A one-line, actionable description.
    std::string describe() const {
        if (!found) return "padding clean";
        return "padding bit set at row " + std::to_string(row) + ", bit " +
               std::to_string(bit) + " (width " + std::to_string(width) + ", " +
               std::to_string(rowWords) + " words/row); " + std::to_string(dirtyBits) +
               " padding bits set in total";
    }
};

/// @brief The first bit at or past `width` that is set, scanning row-major.
///
/// @note Scans through the view's STRIDE, not just the ceil(width / WordBits)
///       words a row inherently needs, so an over-aligned row (rowAlignment > word
///       size, D-4) has its whole tail checked and not merely its trailing partial
///       word. Matches what tests/test_binMat.cpp's countBitsAcrossStride does for
///       the container, but takes a VIEW -- which is what kernels write through
///       (D-5), and the reason the container-side check could not cover them.
/// @note A single-row view may legitimately carry stride == 0 (see
///       BinMatConstView::row), so the words-per-row figure is the larger of the
///       stride and the minimum. Otherwise a 1-row view's trailing partial word
///       would go unscanned -- which is exactly the word most likely to be dirty.
template <typename WordType>
PaddingViolation firstDirtyPaddingBit(const BinMatConstView<WordType>& v) {
    constexpr size_t wordBits = BinMatConstView<WordType>::WordBits;
    PaddingViolation p;
    if (v.ptr == nullptr || v.width == 0 || v.height == 0) return p;

    const size_t minWords = (v.width + wordBits - 1) / wordBits;
    const size_t rowWords = (v.stride > minWords) ? v.stride : minWords;
    const size_t tailBits = v.width % wordBits;
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));

    p.width = v.width;
    p.rowWords = rowWords;

    for (size_t y = 0; y < v.height; ++y) {
        const WordType* row = v.row(y);
        for (size_t w = 0; w < rowWords; ++w) {
            WordType padding;
            if (w + 1 < minWords) {
                continue;                            // wholly pixels
            } else if (w + 1 == minWords) {
                if (tailBits == 0) continue;         // a full word of pixels
                padding = static_cast<WordType>(row[w] &
                          static_cast<WordType>(allOnes << tailBits));
            } else {
                padding = row[w];                    // past the pixel words entirely
            }
            if (padding == 0) continue;

            size_t firstBit = 0;
            while (((padding >> firstBit) & 1u) == 0u) ++firstBit;
            if (!p.found) {
                p.found = true;
                p.row = y;
                p.bit = w * wordBits + firstBit;
            }
            while (padding != 0) {
                p.dirtyBits += static_cast<size_t>(padding & static_cast<WordType>(1));
                padding = static_cast<WordType>(padding >> 1);
            }
        }
    }
    return p;
}

// ---------------------------------------------------------------------------
// The reporting seam
// ---------------------------------------------------------------------------

/// @brief The signature of tests/test_util.hpp's reportCheck().
using CheckReporter = void (*)(bool ok, const char* expr, const char* file, int line,
                               const std::string& note);

/// @brief Where expectBitExact() and expectPaddingClean() send their verdicts.
///
/// @note reportCheck() by default, and it must STAY reportCheck() in any ordinary
///       run -- Equivalence.HarnessSeamIsWiredToTheSuite asserts exactly that, so
///       a default swapped for a no-op is a named failure rather than a silent
///       one. The indirection exists for one reason: without it, the suite could
///       not drive expectBitExact's FAILURE path without failing itself, and a
///       failure path no test executes is a failure path that can be deleted.
///       Measured -- neutering expectBitExact to report success unconditionally
///       left this suite at "3392/3392 checks passed" and exit 0.
inline CheckReporter& checkReporter() {
    static CheckReporter reporter = &reportCheck;
    return reporter;
}

/// @brief One verdict, as expectBitExact() would have handed it to reportCheck().
struct CapturedVerdict {
    bool ok = false;
    std::string expr;
    std::string note;
};

inline std::vector<CapturedVerdict>& capturedVerdicts() {
    static std::vector<CapturedVerdict> verdicts;
    return verdicts;
}

inline void captureReporter(bool ok, const char* expr, const char* /*file*/,
                            int /*line*/, const std::string& note) {
    capturedVerdicts().push_back(
        CapturedVerdict{ok, expr != nullptr ? std::string(expr) : std::string(), note});
}

/// @brief RAII: routes expectBitExact()'s verdicts into a buffer instead of the suite.
/// @note Only the two expect* functions below go through the seam. BINCV_CHECK and
///       friends call reportCheck() directly and are unaffected, so a check made
///       while a capture is active still counts and still fails the suite.
class VerdictCapture {
public:
    VerdictCapture() : previous_(checkReporter()) {
        capturedVerdicts().clear();
        checkReporter() = &captureReporter;
    }
    ~VerdictCapture() { checkReporter() = previous_; }

    VerdictCapture(const VerdictCapture&) = delete;
    VerdictCapture& operator=(const VerdictCapture&) = delete;

    /// @brief Every verdict recorded since this capture began.
    const std::vector<CapturedVerdict>& verdicts() const { return capturedVerdicts(); }

    /// @brief How many of them were failures.
    size_t failures() const {
        size_t n = 0;
        for (const CapturedVerdict& v : capturedVerdicts()) {
            if (!v.ok) ++n;
        }
        return n;
    }

    /// @brief The note of the first failing verdict, or "" if none failed.
    std::string firstFailureNote() const {
        for (const CapturedVerdict& v : capturedVerdicts()) {
            if (!v.ok) return v.note;
        }
        return std::string();
    }

private:
    CheckReporter previous_;
};

/// @brief "context -- detail", or just "detail" when there is no context.
inline std::string annotate(const std::string& context, const std::string& detail) {
    return context.empty() ? detail : (context + " -- " + detail);
}

/// @brief Asserts that every bit at or past `width` in a view's rows is zero.
/// @param v The view to check -- normally a kernel's DESTINATION.
/// @param context Free text naming the case. Only printed on failure.
/// @param file, line Defaulted; BINCV_EXPECT_PADDING_CLEAN fills in the call site.
///
/// @note This is the check that a Tier 1 kernel test cannot skip. A kernel that
///       writes whole words -- which is the only reason to store bits packed --
///       is bit-exact pixel for pixel whether or not it clears its trailing
///       partial word, so the pixel comparison licenses the broken version. The
///       first word-wise reduction built on top then over-counts.
/// @note Reported as its OWN check, so the log distinguishes "the kernel computed
///       the wrong pixel" from "the kernel computed the right pixels and left the
///       tail dirty".
template <typename WordType>
void expectPaddingClean(const BinMatConstView<WordType>& v,
                        const std::string& context = std::string(),
                        const char* file = __FILE__, int line = __LINE__) {
    const PaddingViolation p = firstDirtyPaddingBit(v);
    checkReporter()(!p.found, "padding bits are zero past width", file, line,
                    p.found ? annotate(context, p.describe()) : std::string());
}

/// @def BINCV_EXPECT_PADDING_CLEAN(view, context)
/// @brief expectPaddingClean() attributed to the call site rather than this header.
#define BINCV_EXPECT_PADDING_CLEAN(view, context) \
    ::bincv::test::expectPaddingClean((view), (context), __FILE__, __LINE__)

/// @brief Asserts that a binCV Tier 1 result is bit-exact against OpenCV's.
/// @param actual The binCV result, as a view (D-5).
/// @param expected What OpenCV produced on the same content stored as CV_8U.
/// @param context Free text naming the case -- word type, size, fill ratio. Only
///        printed on failure, and worth setting: 2000 identical failure lines
///        that do not say which of 240 shapes produced them are not a report.
/// @param file, line Defaulted so the two-argument spelling in the T2.1 spec
///        compiles; BINCV_EXPECT_BIT_EXACT fills them in with the CALL SITE,
///        which is what a failure should point at rather than this header.
///
/// @note Converts `actual` to CV_8U {0,255} and compares EVERY pixel. A size
///       difference is a failure, not a reason to compare the overlap.
/// @note Then checks the padding, because the pixel comparison cannot: a kernel
///       that writes whole words and never clears the trailing partial one is
///       bit-exact on every pixel and still breaks CLAUDE.md's hard rule. Both
///       properties are what "bit-exact against OpenCV" has to mean for a
///       bit-packed container, so both are asserted here rather than left to a
///       companion call a kernel author may forget.
/// @note The `actual` view is the one checked for padding, never `expected` --
///       `expected` is a cv::Mat and has no padding bits to speak of.
/// @note Counts as exactly TWO checks in the suite's CHECKS total, whatever the
///       image size, so the per-suite floor in tests/expected-checks.txt tracks
///       cases rather than pixels.
/// @note A view that deliberately WINDOWS a larger image has its neighbours'
///       pixels sitting in the region this calls padding, and would fail here.
///       That is the correct answer for a Tier 1 destination -- a kernel writing
///       whole words into a window is clobbering its neighbours -- and no such
///       destination exists in the test recipe today. If one ever does, it is a
///       decision to record, not a check to loosen quietly.
template <typename WordType>
void expectBitExact(const BinMatConstView<WordType>& actual, const cv::Mat& expected,
                    const std::string& context = std::string(),
                    const char* file = __FILE__, int line = __LINE__) {
    const Mismatch m = firstMismatch(actual, expected);
    checkReporter()(!m.found, "bit-exact against OpenCV", file, line,
                    m.found ? annotate(context, m.describe()) : std::string());
    expectPaddingClean(actual, context, file, line);
}

/// @def BINCV_EXPECT_BIT_EXACT(actual, expected, context)
/// @brief expectBitExact() attributed to the call site rather than to this header.
#define BINCV_EXPECT_BIT_EXACT(actual, expected, context) \
    ::bincv::test::expectBitExact((actual), (expected), (context), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// The size and fill matrix
// ---------------------------------------------------------------------------

/// @brief Widths every Tier 1 equivalence test sweeps. T2.1 names these.
/// @note All but 640 are NON-MULTIPLES of at least one supported word width, and
///       that is the entire point: packing bugs live in the trailing partial
///       word. 1 is the degenerate row; 7 is shorter than the narrowest word; 31,
///       33, 63 and 65 straddle the 32- and 64-bit boundaries from both sides; 70
///       leaves a 6-bit tail at 32 bits and a 6-bit tail at 8 bits; 640 is the
///       real frame width and the one case where every word type divides evenly.
/// @note 40 covers the configuration none of the others reach: a width where ONE
///       word type ends a row exactly on a word boundary while another does not.
///       Measured over the original list -- {w : w % 8 == 0 and w % 64 != 0} was
///       EMPTY, so at uint8_t and uint16_t the no-trailing-partial-word path was
///       only reached at 640, where every word type is simultaneously exact. A
///       tail branch that is wrong only when one word width has a tail and
///       another does not had no case that separated them. 40 is 5 whole uint8_t
///       words with no tail, an 8-bit tail at uint16_t and uint32_t, and a 40-bit
///       tail at uint64_t.
inline const std::vector<int>& equivalenceWidths() {
    static const std::vector<int> widths{1, 7, 31, 33, 40, 63, 65, 70, 640};
    return widths;
}

/// @brief Heights every Tier 1 equivalence test sweeps.
/// @note Rows are whole strides, so height cannot produce a packing bug the way
///       width can -- but it can produce a stride bug, and 3, 17 and 37 are
///       multiples of nothing in sight. 1 is the single-row view, where a wrong
///       stride is invisible; 2 is the smallest case where it is not.
inline const std::vector<int>& equivalenceHeights() {
    static const std::vector<int> heights{1, 2, 3, 17, 37, 480};
    return heights;
}

/// @brief Fill ratios every Tier 1 equivalence test sweeps. T2.1 names these.
/// @note 0.0 and 1.0 are exact, not approximate: an all-zero and an all-ones
///       image are where a masked or short-circuited kernel stops being
///       exercised. 0.01 and 0.99 are the sparse and dense cases either side.
inline const std::vector<float>& equivalenceFillRatios() {
    static const std::vector<float> fills{0.0f, 0.01f, 0.5f, 0.99f, 1.0f};
    return fills;
}

/// @brief "uint32_t 70x37 fill=0.50" -- the context string for a swept case.
inline std::string caseLabel(const char* wordTypeName, int width, int height, float fill) {
    char fillText[16];
    std::snprintf(fillText, sizeof(fillText), "%.2f", static_cast<double>(fill));
    return std::string(wordTypeName) + " " + std::to_string(width) + "x" +
           std::to_string(height) + " fill=" + fillText;
}

} // namespace test
} // namespace bincv

#endif // BINCV_WITH_OPENCV
