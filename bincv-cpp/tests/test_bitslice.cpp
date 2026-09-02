// Bit-sliced small-count arithmetic: maj3 / bitSlicedSum / thresholdGE,
// and the view-level majority3 kernel built on the first of them.
//
// TWO HALVES, the same split as tests/test_logic.cpp and tests/test_reduce.cpp.
//
// 1. The CORE half (everything up to the OpenCV guard) needs no OpenCV, so it
// runs in all four verification configurations -- including the Debug one,
// which is the only place majority3's BINCV_ASSERT preconditions are live,
// and the -fno-exceptions one, which is the embedded claim.
//
// 2. The OPENCV half checks majority3 against the reference pipeline's own
// three-pixel median, written as OpenCV calls: max(min(a,b), min(max(a,b), c))
// over CV_8U (the reference frontend's denoiser). That is a SECOND
// REFERENCE, not a tier promise -- bit-sliced arithmetic is Tier 3
// (the design notes) and OpenCV has no pointwise median of three images.
// It is here because must match that formula, and the cheapest way to
// be sure the majority IS that median is to run the formula.
//
// EXHAUSTIVE, NOT SAMPLED -- WHICH IS AFFORDABLE HERE AND NOWHERE ELSE
//
// Every word-level primitive in this file is pointwise in the bit LANE: lane i of
// the output depends on lane i of the inputs and on nothing else. So for k inputs
// there are exactly 2^k distinct per-lane input patterns, and the whole input
// space can be enumerated rather than sampled. It is, for
//
// maj3 k = 3 all 8 patterns
// bitSlicedSum k = 1, 2, 3, 4, 9 all 2, 4, 8, 16 and 512 patterns
// bitSlicedSum k = 16 all 65536 patterns
// thresholdGE nPlanes = 0..5 every value 0..2^n-1 against every
// threshold 0..2^n+1, which is that
// function's ENTIRE input space
//
// at all four word widths. k = 3, 4 and 9 are the MVP's shapes (median of 3, box
// 2x2, 3x3 median); k = 1 and 2 are the degenerate ones; 16 is the larger case
// asks for, and 2^16 patterns is still under a second.
//
// The patterns are packed into the LANES of the words under test -- pattern
// base+L in lane L -- so one call covers WordBits patterns and the enumeration
// also proves the lanes stay independent. A kernel that leaked a carry from lane
// L into lane L+1 would pass a per-pattern-at-a-time test that used lane 0 only.
//
// CHECK GRANULARITY: ONE CHECK PER (PATTERN) OR PER (PATTERN, THRESHOLD)
//
// Per lane-bit would be the same number here (one lane holds one pattern), and
// per case would leave the CHECKS column blind to a pattern list or a threshold
// list that got shorter -- which is this suite's most likely regression, since
// exhaustiveness is its whole claim. The failure message is built only when a
// check fails (see BITSLICE_EXPECT).
//
// WHAT THE REFERENCES ARE, AND WHY THEY ARE NOT THE IMPLEMENTATION RESTATED
//
// refMedian3 is the sorting network -- max(min(a,b), min(max(a,b),c)) -- not
// (a&b)|(b&c)|(a&c). refCount counts set inputs with an ordinary loop over the
// input WORDS (not over the pattern index it was built from), and refThresholdGE
// is `value >= threshold` on unsigned ints. None of the three shares an
// expression with the code under test, which is the point: a reference derived
// from the implementation cannot fail with it.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/bitslice.hpp"
#include "test_util.hpp"

// OpenCV-only, and self-guarding: everything in it sits behind BINCV_WITH_OPENCV,
// so this include is a no-op in the three configurations that have no OpenCV.
#include "equivalence.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#endif

namespace {

using bincv::bitSlicedSum;
using bincv::bitSlicedSumPlanes;
using bincv::maj3;
using bincv::majority3;
using bincv::thresholdGE;

// The largest k this file enumerates, and the planes it needs. Stack arrays, so
// that the tests allocate exactly as little as the kernels do.
constexpr size_t MAX_INPUTS = 16;
constexpr size_t MAX_PLANES = 5;   // bitSlicedSumPlanes(16)

static_assert(bitSlicedSumPlanes(0) == 0, "no inputs need no planes");
static_assert(bitSlicedSumPlanes(1) == 1, "");
static_assert(bitSlicedSumPlanes(2) == 2, "");
static_assert(bitSlicedSumPlanes(3) == 2, "");
static_assert(bitSlicedSumPlanes(4) == 3, "the 2x2 box sum");
static_assert(bitSlicedSumPlanes(9) == 4, "the 3x3 median");
static_assert(bitSlicedSumPlanes(16) == 5, "");
static_assert(bitSlicedSumPlanes(MAX_INPUTS) == MAX_PLANES, "MAX_PLANES sizes the arrays below");

// ---------------------------------------------------------------------------
// One check, whose message costs nothing unless it fails
// ---------------------------------------------------------------------------
//
// `ok` is evaluated twice, so it is always passed as an already-computed bool.
#define BITSLICE_EXPECT(ok, what, detailExpr) \
    ::bincv::test::reportCheck((ok), (what), __FILE__, __LINE__, \
                               (ok) ? std::string() : (detailExpr))

// ---------------------------------------------------------------------------
// Lane access, written once
// ---------------------------------------------------------------------------

template <typename WordType>
bool bitAt(WordType w, size_t lane) {
    return ((w >> lane) & static_cast<WordType>(1)) != 0;
}

template <typename WordType>
void setBit(WordType& w, size_t lane) {
    w = static_cast<WordType>(w | static_cast<WordType>(static_cast<WordType>(1) << lane));
}

template <typename WordType>
constexpr size_t wordBits() {
    return sizeof(WordType) * 8;
}

// ---------------------------------------------------------------------------
// The per-pixel references -- written before the kernels, and independent of them
// ---------------------------------------------------------------------------

/// @brief Median of three binary pixels, as the reference pipeline computes it.
/// @note max(min(a, b), min(max(a, b), c)), i.e. the reference frontend's three-pixel median
/// with cv::min / cv::max read as && / || on {0, 255}. Deliberately NOT
/// (a&b)|(b&c)|(a&c): a reference that restates the implementation cannot
/// disagree with it.
bool refMedian3(bool a, bool b, bool c) {
    const bool minAB = a && b;
    const bool maxAB = a || b;
    const bool minMaxC = maxAB && c;
    return minAB || minMaxC;
}

/// @brief How many of the k inputs have lane `lane` set -- one loop, no popcount.
/// @note Reads the input WORDS rather than the pattern index they were built
/// from, so a bug in the pattern builder cannot cancel against a bug in the
/// adder.
template <typename WordType>
unsigned refCount(const WordType* inputs, size_t k, size_t lane) {
    unsigned n = 0;
    for (size_t j = 0; j < k; ++j) {
        if (bitAt(inputs[j], lane)) ++n;
    }
    return n;
}

/// @brief The unsigned value a lane's planes encode, LSB plane first.
template <typename WordType>
unsigned laneValue(const WordType* planes, size_t nPlanes, size_t lane) {
    unsigned v = 0;
    for (size_t p = 0; p < nPlanes; ++p) {
        if (bitAt(planes[p], lane)) v |= (1u << p);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Pattern enumeration
// ---------------------------------------------------------------------------

/// @brief Packs the integers [base, base + lanes) into `n` bit-sliced words.
/// @param words `n` words, rebuilt from zero: bit `lane` of `words[i]` becomes
/// bit i of the integer base + lane.
/// @param n How many bits of each integer to keep -- the input count k when the
/// integer is a PATTERN index (input j's bit is bit j), and the plane
/// count when it is a VALUE (plane p's bit is bit p). The two
/// enumerations this file runs are the same packing operation, so they are
/// the same function.
/// @return How many lanes were filled -- WordBits, or fewer for the last chunk.
/// @note Lane L holds integer base + L, which is what makes one call cover
/// WordBits patterns and what makes the sweep prove the lanes stay
/// independent: a kernel that leaked a carry from lane L into lane L+1
/// would pass a test that used lane 0 only.
template <typename WordType>
size_t packLanes(WordType* words, size_t n, uint64_t base, uint64_t total) {
    const uint64_t remaining = total - base;
    const uint64_t lanes64 = remaining < wordBits<WordType>() ? remaining : wordBits<WordType>();
    const size_t lanes = static_cast<size_t>(lanes64);

    for (size_t i = 0; i < n; ++i) words[i] = static_cast<WordType>(0);
    for (size_t lane = 0; lane < lanes; ++lane) {
        const uint64_t value = base + lane;
        for (size_t i = 0; i < n; ++i) {
            if (((value >> i) & UINT64_C(1)) != 0) setBit(words[i], lane);
        }
    }
    return lanes;
}

std::string patternText(const char* wordTypeName, size_t k, uint64_t pattern) {
    return std::string(wordTypeName) + " k=" + std::to_string(k) + " pattern=" +
           std::to_string(pattern);
}

// ===========================================================================
// 1. maj3, exhaustively over all 8 patterns
// ===========================================================================

template <typename WordType>
void testMaj3(const char* wordTypeName) {
    std::cout << "\n--- maj3 vs a per-pixel median: " << wordTypeName << " ---\n";

    const uint64_t total = 8;   // 2^3 patterns, and every word type holds all 8
    WordType inputs[3];

    for (uint64_t base = 0; base < total; base += wordBits<WordType>()) {
        const size_t lanes = packLanes(inputs, 3, base, total);
        const WordType got = maj3<WordType>(inputs[0], inputs[1], inputs[2]);

        // The same three inputs through the OTHER route the MVP has for a
        // majority: count them, then threshold at 2. maj3 and the adder network
        // are independent code paths, and this is the only case that binds them.
        WordType planes[MAX_PLANES];
        bitSlicedSum(inputs, 3, planes);
        const WordType viaSum = thresholdGE(planes, bitSlicedSumPlanes(3), 2u);

        for (size_t lane = 0; lane < lanes; ++lane) {
            const bool expected = refMedian3(bitAt(inputs[0], lane), bitAt(inputs[1], lane),
                                             bitAt(inputs[2], lane));
            const bool okMaj = (bitAt(got, lane) == expected);
            BITSLICE_EXPECT(okMaj, "maj3 matches the per-pixel median",
                            patternText(wordTypeName, 3, base + lane) + ": got " +
                                std::to_string(bitAt(got, lane) ? 1 : 0) + ", expected " +
                                std::to_string(expected ? 1 : 0));

            const bool okSame = (bitAt(viaSum, lane) == bitAt(got, lane));
            BITSLICE_EXPECT(okSame, "maj3 agrees with thresholdGE(bitSlicedSum(3), 2)",
                            patternText(wordTypeName, 3, base + lane));
        }
    }
}

// ===========================================================================
// 2. bitSlicedSum against a per-pixel count, exhaustively
// ===========================================================================

template <typename WordType>
void sweepSum(const char* wordTypeName, size_t k) {
    const size_t nPlanes = bitSlicedSumPlanes(k);
    const uint64_t total = UINT64_C(1) << k;

    WordType inputs[MAX_INPUTS];
    WordType planes[MAX_PLANES];

    for (uint64_t base = 0; base < total; base += wordBits<WordType>()) {
        const size_t lanes = packLanes(inputs, k, base, total);
        bitSlicedSum(inputs, k, planes);

        for (size_t lane = 0; lane < lanes; ++lane) {
            const unsigned expected = refCount(inputs, k, lane);
            const unsigned actual = laneValue(planes, nPlanes, lane);
            const bool ok = (actual == expected);
            BITSLICE_EXPECT(ok, "bit-sliced sum matches the per-pixel count",
                            patternText(wordTypeName, k, base + lane) + ": got " +
                                std::to_string(actual) + ", expected " + std::to_string(expected));
        }
    }
}

template <typename WordType>
void testSumSmall(const char* wordTypeName) {
    std::cout << "\n--- bitSlicedSum vs a per-pixel count: " << wordTypeName << " ---\n";
    // 1 and 2 are the degenerate counts, 3 is the median, 4 the 2x2 box, 9 the
    // 3x3 median. Every pattern of every one of them.
    for (size_t k : {size_t(1), size_t(2), size_t(3), size_t(4), size_t(9)}) {
        sweepSum<WordType>(wordTypeName, k);
    }
}

template <typename WordType>
void testSumWide(const char* wordTypeName) {
    std::cout << "\n--- bitSlicedSum at k=16, all 65536 patterns: " << wordTypeName << " ---\n";
    sweepSum<WordType>(wordTypeName, 16);
}

// ===========================================================================
// 3. thresholdGE over its ENTIRE input space
// ===========================================================================
//
// The function's input is (value, threshold, nPlanes), and for nPlanes <= 5 all
// three are enumerated: every value the planes can hold, against every threshold
// from 0 to one past the maximum. 5 planes is what k = 16 produces, so no k this
// file tests reaches a plane count this sweep has not covered completely.
//
// The two ends are the ones a caller reaches by arithmetic rather than by choice
// and are therefore the ones a hand-written case list omits: threshold 0 passes
// everything, and a threshold above the representable maximum passes nothing.

template <typename WordType>
void testThresholdGEValues(const char* wordTypeName) {
    std::cout << "\n--- thresholdGE over every (value, threshold): " << wordTypeName << " ---\n";

    const WordType guardValue = static_cast<WordType>(~static_cast<WordType>(0));

    for (size_t nPlanes = 0; nPlanes <= MAX_PLANES; ++nPlanes) {
        const uint64_t values = UINT64_C(1) << nPlanes;   // nPlanes == 0 -> the value 0 alone

        // One word longer than the sweep needs, and the extra one is a guard:
        // thresholdGE reads `nPlanes` planes and packLanes writes `nPlanes` of
        // them, so a word past the count must come back untouched. (It also keeps
        // GCC's -Wstringop-overflow analysis from mis-bounding the enumeration
        // loop over an exactly-sized array, which is a false positive -- but a
        // guard word is worth having on its own terms, so it is checked rather
        // than left as an unexplained +1.)
        WordType planes[MAX_PLANES + 1];
        planes[MAX_PLANES] = guardValue;

        for (uint64_t base = 0; base < values; base += wordBits<WordType>()) {
            const size_t lanes = packLanes(planes, nPlanes, base, values);
            const bool guardIntact = (planes[MAX_PLANES] == guardValue);
            BITSLICE_EXPECT(guardIntact, "packLanes writes no word past its count",
                            std::string(wordTypeName) + " nPlanes=" + std::to_string(nPlanes));

            for (uint64_t threshold = 0; threshold <= values + 1; ++threshold) {
                const WordType mask =
                    thresholdGE(planes, nPlanes, static_cast<unsigned>(threshold));

                for (size_t lane = 0; lane < lanes; ++lane) {
                    const uint64_t value = base + lane;
                    const bool expected = (value >= threshold);
                    const bool ok = (bitAt(mask, lane) == expected);
                    BITSLICE_EXPECT(
                        ok, "thresholdGE matches value >= threshold",
                        std::string(wordTypeName) + " nPlanes=" + std::to_string(nPlanes) +
                            " value=" + std::to_string(value) + " threshold=" +
                            std::to_string(threshold) + ": got " +
                            std::to_string(bitAt(mask, lane) ? 1 : 0));
                }
            }
        }
    }
}

// ===========================================================================
// 4. The composition the MVP actually calls: sum, then threshold
// ===========================================================================
//
// Every threshold from 0 (everything passes) to k+1 (nothing passes), over every
// pattern, for the counts the MVP uses. names both ends explicitly, and they
// are exactly where a comparison built from `>` rather than `>=` survives a
// mid-range test.

template <typename WordType>
void sweepComposed(const char* wordTypeName, size_t k, uint64_t patternStep) {
    const size_t nPlanes = bitSlicedSumPlanes(k);
    const uint64_t total = UINT64_C(1) << k;

    WordType inputs[MAX_INPUTS];
    WordType planes[MAX_PLANES];

    for (uint64_t base = 0; base < total; base += patternStep * wordBits<WordType>()) {
        // With patternStep > 1 the chunk is still contiguous; the step skips whole
        // chunks rather than lanes, so every lane it does visit is still a
        // distinct pattern and the lane-independence property is unaffected.
        const size_t lanes = packLanes(inputs, k, base, total);
        bitSlicedSum(inputs, k, planes);

        for (unsigned threshold = 0; threshold <= static_cast<unsigned>(k) + 1; ++threshold) {
            const WordType mask = thresholdGE(planes, nPlanes, threshold);
            for (size_t lane = 0; lane < lanes; ++lane) {
                const bool expected = (refCount(inputs, k, lane) >= threshold);
                const bool ok = (bitAt(mask, lane) == expected);
                BITSLICE_EXPECT(ok, "thresholded count matches the per-pixel count",
                                patternText(wordTypeName, k, base + lane) + " threshold=" +
                                    std::to_string(threshold) + ": got " +
                                    std::to_string(bitAt(mask, lane) ? 1 : 0) + ", expected " +
                                    std::to_string(expected ? 1 : 0));
            }
        }
    }
}

template <typename WordType>
void testComposed(const char* wordTypeName) {
    std::cout << "\n--- bitSlicedSum + thresholdGE, every threshold: " << wordTypeName << " ---\n";
    for (size_t k : {size_t(1), size_t(2), size_t(3), size_t(4), size_t(9)}) {
        sweepComposed<WordType>(wordTypeName, k, 1);
    }
    // k = 16 at every threshold over every pattern would be 4.7 million checks
    // for a function whose own input space (5 planes) is already enumerated
    // completely above. Every 97th chunk keeps the composition covered at the
    // wide count without paying for a second exhaustive sweep of thresholdGE.
    sweepComposed<WordType>(wordTypeName, 16, 97);
}

// ===========================================================================
// 5. Degenerate arguments, and the write extent
// ===========================================================================

template <typename WordType>
void testDegenerate(const char* wordTypeName) {
    std::cout << "\n--- degenerate arguments: " << wordTypeName << " ---\n";

    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));
    const std::string label(wordTypeName);

    // k == 0: no inputs, no planes, nothing written. The sentinel array is what
    // says "nothing written" rather than "wrote zeros".
    {
        WordType out[2] = {allOnes, allOnes};
        WordType inputs[1] = {allOnes};
        bitSlicedSum<WordType>(inputs, 0, out);
        const bool ok = (out[0] == allOnes && out[1] == allOnes);
        BITSLICE_EXPECT(ok, "bitSlicedSum with k == 0 writes nothing", label);
    }

    // A value of no planes is zero, so it clears every threshold but 0.
    {
        const WordType* none = nullptr;
        const bool okZero = (thresholdGE(none, 0, 0u) == allOnes);
        BITSLICE_EXPECT(okZero, "thresholdGE(0 planes, 0) passes every lane", label);
        const bool okOne = (thresholdGE(none, 0, 1u) == 0);
        BITSLICE_EXPECT(okOne, "thresholdGE(0 planes, 1) passes no lane", label);
    }

    // A threshold no value of nPlanes bits can reach, from just past the top to
    // the largest unsigned there is. The second one is the case where a
    // `threshold >> nPlanes` guard written as a signed shift goes wrong.
    {
        WordType planes[3];
        for (size_t p = 0; p < 3; ++p) planes[p] = allOnes;   // every lane holds 7
        const bool okTop = (thresholdGE(planes, 3, 7u) == allOnes);
        BITSLICE_EXPECT(okTop, "thresholdGE at the representable maximum passes every lane", label);
        const bool okPast = (thresholdGE(planes, 3, 8u) == 0);
        BITSLICE_EXPECT(okPast, "thresholdGE one past the maximum passes no lane", label);
        const bool okHuge = (thresholdGE(planes, 3, ~0u) == 0);
        BITSLICE_EXPECT(okHuge, "thresholdGE at UINT_MAX passes no lane", label);
    }

    // bitSlicedSum writes exactly bitSlicedSumPlanes(k) words and not one more.
    // A ripple that ran to a fixed plane count would corrupt the caller's next
    // stack slot, which no value comparison in this file would see.
    {
        for (size_t k : {size_t(1), size_t(2), size_t(3), size_t(4), size_t(9), size_t(16)}) {
            WordType inputs[MAX_INPUTS];
            WordType guarded[MAX_PLANES + 2];
            for (size_t j = 0; j < MAX_INPUTS; ++j) inputs[j] = allOnes;
            for (size_t p = 0; p < MAX_PLANES + 2; ++p) guarded[p] = allOnes;

            bitSlicedSum(inputs, k, guarded);

            bool intact = true;
            for (size_t p = bitSlicedSumPlanes(k); p < MAX_PLANES + 2; ++p) {
                if (guarded[p] != allOnes) intact = false;
            }
            BITSLICE_EXPECT(intact, "bitSlicedSum writes no plane past its plane count",
                            label + " k=" + std::to_string(k));

            //... and the count it did write is k in every lane, which is the
            // all-ones input's answer and the one the ripple has to carry
            // furthest.
            const bool okValue =
                (laneValue(guarded, bitSlicedSumPlanes(k), 0) == static_cast<unsigned>(k));
            BITSLICE_EXPECT(okValue, "bitSlicedSum of all-ones inputs is k",
                            label + " k=" + std::to_string(k));
        }
    }
}

// ===========================================================================
// 6. The view-level kernel: majority3
// ===========================================================================
//
// Everything below tests the kernel's PLUMBING -- strides, row loops, the
// trailing word, aliasing -- rather than the majority itself, which the
// exhaustive word-level sweep above has already settled at every input.

// Content generation: the same SplitMix64 draw as tests/equivalence.hpp, minus
// OpenCV, duplicated for the reason gives (a harness and the suite it judges
// must not share a generator, or a fault in the shared part cancels).

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
    return UINT64_C(0xB175E11CE0000001) + static_cast<uint64_t>(width) * UINT64_C(1000003) +
           static_cast<uint64_t>(height) * UINT64_C(10007) + static_cast<uint64_t>(index);
}

/// @brief Set bits across the whole STRIDE, padding included.
/// @note Deliberately not a library operation -- binCV exposes no per-word
/// popcount. Compared against countNonZero's per-pixel loop it is
/// how a padding-bit violation becomes visible: the two agree only when
/// every bit past `width` is zero.
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

/// @brief Pixels on which majority3's result differs from the per-pixel median.
template <typename WordType>
int disagreements(const bincv::BinMat<WordType>& a, const bincv::BinMat<WordType>& b,
                  const bincv::BinMat<WordType>& c, const bincv::BinMat<WordType>& dst) {
    int differing = 0;
    for (int y = 0; y < dst.rows(); ++y) {
        for (int x = 0; x < dst.cols(); ++x) {
            if (dst.at(y, x) != refMedian3(a.at(y, x), b.at(y, x), c.at(y, x))) ++differing;
        }
    }
    return differing;
}

std::string sizeLabel(const char* wordTypeName, int width, int height, const char* extra) {
    return std::string(wordTypeName) + " " + std::to_string(width) + "x" +
           std::to_string(height) + " " + extra;
}

// The widths, plus 128 -- an exact multiple of every supported word width,
// which is the only shape that reaches majority3's single-contiguous-run path.
const int WIDTHS[] = {1, 7, 31, 33, 40, 63, 65, 70, 128, 640};
const int HEIGHTS[] = {1, 2, 3, 17};
const float FILLS[] = {0.0f, 0.01f, 0.5f, 0.99f, 1.0f};

// An over-aligned row stride (the design rule makes alignment a per-object choice).
constexpr size_t PADDED_ALIGNMENT = 32;

template <typename WordType>
void testMajority3Reference(const char* wordTypeName) {
    std::cout << "\n--- majority3 vs a per-pixel median: " << wordTypeName << " ---\n";

    for (int width : WIDTHS) {
        for (int height : HEIGHTS) {
            for (size_t f = 0; f < sizeof(FILLS) / sizeof(FILLS[0]); ++f) {
                const uint64_t seed = caseSeed(width, height, f);
                bincv::BinMat<WordType> a(width, height);
                bincv::BinMat<WordType> b(width, height);
                bincv::BinMat<WordType> c(width, height);
                fillRandom(a, FILLS[f], seed);
                fillRandom(b, FILLS[f], seed ^ UINT64_C(0xDEADBEEF));
                fillRandom(c, FILLS[f], seed ^ UINT64_C(0x5EED5EED));

                bincv::BinMat<WordType> dst(width, height);
                majority3(a.constView(), b.constView(), c.constView(), dst.view());

                const bool okPixels = (disagreements(a, b, c, dst) == 0);
                BITSLICE_EXPECT(okPixels, "majority3 matches the per-pixel median",
                                sizeLabel(wordTypeName, width, height, "reference"));

                // The invariant a pixel comparison is blind to: bits past `width`
                // must be zero, or the first word-wise reduction built on this
                // result over-counts.
                const bool okPadding = (bitsAcrossStride(dst) == dst.countNonZero());
                BITSLICE_EXPECT(okPadding, "majority3 leaves the padding bits zero",
                                sizeLabel(wordTypeName, width, height, "padding"));
            }
        }
    }
}

// Three stride flavours per argument, as in tests/test_logic.cpp:
// tight stride == ceil(width / WordBits) the default
// padded stride from a 32-byte row alignment the opt-in
// odd stride == tight + 3 a wrapped buffer

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
    std::cout << "\n--- majority3 across differing strides: " << wordTypeName << " ---\n";

    // Sixteen combinations rather than all 81, and the first nine are the ones
    // that matter: all-tight (the dense path), then EXACTLY ONE argument
    // non-tight, in each of the four positions, at both non-tight flavours.
    //
    // That structure is a review finding turned into cases. The list was six
    // mixed combinations, none of which had a, b and dst tight with only `c`
    // over-strided -- so deleting `c.stride == words` from the dense-path
    // condition left all 348061 checks green in every configuration. The kernel
    // then walked `c` as one contiguous run at the wrong stride whenever the
    // other three happened to be dense, which is what a caller gets by mixing an
    // over-aligned frame with tightly-packed ones. A guard that only fires when
    // exactly one argument differs needs a case where exactly one argument
    // differs.
    const Stride combos[][4] = {
        {Stride::Tight,  Stride::Tight,  Stride::Tight,  Stride::Tight},
        {Stride::Padded, Stride::Tight,  Stride::Tight,  Stride::Tight},
        {Stride::Tight,  Stride::Padded, Stride::Tight,  Stride::Tight},
        {Stride::Tight,  Stride::Tight,  Stride::Padded, Stride::Tight},
        {Stride::Tight,  Stride::Tight,  Stride::Tight,  Stride::Padded},
        {Stride::Odd,    Stride::Tight,  Stride::Tight,  Stride::Tight},
        {Stride::Tight,  Stride::Odd,    Stride::Tight,  Stride::Tight},
        {Stride::Tight,  Stride::Tight,  Stride::Odd,    Stride::Tight},
        {Stride::Tight,  Stride::Tight,  Stride::Tight,  Stride::Odd},
        {Stride::Tight,  Stride::Padded, Stride::Odd,    Stride::Tight},
        {Stride::Padded, Stride::Tight,  Stride::Tight,  Stride::Odd},
        {Stride::Odd,    Stride::Odd,    Stride::Padded, Stride::Padded},
        {Stride::Padded, Stride::Odd,    Stride::Tight,  Stride::Padded},
        {Stride::Odd,    Stride::Tight,  Stride::Padded, Stride::Tight},
        {Stride::Tight,  Stride::Odd,    Stride::Odd,    Stride::Odd},
        {Stride::Padded, Stride::Padded, Stride::Odd,    Stride::Tight},
    };

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            for (const auto& combo : combos) {
                StridedMat<WordType> a, b, c, dst;
                makeStrided(a, combo[0], width, height);
                makeStrided(b, combo[1], width, height);
                makeStrided(c, combo[2], width, height);
                makeStrided(dst, combo[3], width, height);

                const uint64_t seed = caseSeed(width, height, 900);
                fillRandom(a.mat, 0.5f, seed);
                fillRandom(b.mat, 0.5f, seed ^ UINT64_C(0xDEADBEEF));
                fillRandom(c.mat, 0.5f, seed ^ UINT64_C(0x5EED5EED));

                majority3(a.mat.constView(), b.mat.constView(), c.mat.constView(),
                          dst.mat.view());

                const std::string label =
                    sizeLabel(wordTypeName, width, height, strideName(combo[0])) + "/" +
                    strideName(combo[1]) + "/" + strideName(combo[2]) + "/" +
                    strideName(combo[3]);

                const bool okPixels = (disagreements(a.mat, b.mat, c.mat, dst.mat) == 0);
                BITSLICE_EXPECT(okPixels, "majority3 matches the reference across strides", label);
                const bool okPadding = (bitsAcrossStride(dst.mat) == dst.mat.countNonZero());
                BITSLICE_EXPECT(okPadding, "majority3 leaves the padding bits zero", label);
            }
        }
    }
}

/// @brief dst may BE one of the sources: the kernel is pointwise in the word index.
template <typename WordType>
void testInPlace(const char* wordTypeName) {
    std::cout << "\n--- majority3 in place: " << wordTypeName << " ---\n";

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            for (int which = 0; which < 3; ++which) {
                const uint64_t seed = caseSeed(width, height, 950 + static_cast<size_t>(which));
                bincv::BinMat<WordType> a(width, height);
                bincv::BinMat<WordType> b(width, height);
                bincv::BinMat<WordType> c(width, height);
                fillRandom(a, 0.5f, seed);
                fillRandom(b, 0.5f, seed ^ UINT64_C(0xDEADBEEF));
                fillRandom(c, 0.5f, seed ^ UINT64_C(0x5EED5EED));

                // The expected answer, computed out of place before anything is
                // overwritten.
                bincv::BinMat<WordType> expected(width, height);
                majority3(a.constView(), b.constView(), c.constView(), expected.view());

                bincv::BinMat<WordType>& target = (which == 0) ? a : (which == 1) ? b : c;
                majority3(a.constView(), b.constView(), c.constView(), target.view());

                int differing = 0;
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        if (target.at(y, x) != expected.at(y, x)) ++differing;
                    }
                }
                const std::string label = sizeLabel(wordTypeName, width, height, "in place #") +
                                          std::to_string(which);
                BITSLICE_EXPECT(differing == 0, "in-place majority3 matches the out-of-place one",
                                label);
                const bool okPadding = (bitsAcrossStride(target) == target.countNonZero());
                BITSLICE_EXPECT(okPadding, "in-place majority3 leaves the padding bits zero",
                                label);
            }
        }
    }
}

/// @brief An empty view is a no-op, not an error.
template <typename WordType>
void testDegenerateViews(const char* wordTypeName) {
    std::cout << "\n--- majority3 on degenerate views: " << wordTypeName << " ---\n";

    const int shapes[][2] = {{0, 0}, {0, 5}, {5, 0}};
    for (const auto& shape : shapes) {
        bincv::BinMat<WordType> a(shape[0], shape[1]);
        bincv::BinMat<WordType> b(shape[0], shape[1]);
        bincv::BinMat<WordType> c(shape[0], shape[1]);
        bincv::BinMat<WordType> dst(shape[0], shape[1]);
        majority3(a.constView(), b.constView(), c.constView(), dst.view());

        const std::string label = sizeLabel(wordTypeName, shape[0], shape[1], "degenerate");
        BITSLICE_EXPECT(dst.countNonZero() == 0, "an empty majority3 writes nothing", label);
    }
}

/// @brief Sources whose padding bits are ALREADY SET, which is a legal
/// construction (BinMat's wrap constructor: a wrapped buffer's padding
/// belongs to its caller).
/// @note Without the trailing-word mask in the kernel the majority of three dirty
/// padding words is itself dirty, and the destination leaves phantom pixels
/// behind for the next reduction to count. The pixel comparison alone would
/// not see it.
template <typename WordType>
void makeDirtyPadded(StridedMat<WordType>& out, int width, int height, float fillRatio,
                     uint64_t seed) {
    constexpr size_t bits = bincv::BinMat<WordType>::WordBits;
    const size_t minWords = (static_cast<size_t>(width) + bits - 1) / bits;
    const size_t stride = minWords + 1;
    out.buffer.assign(stride * static_cast<size_t>(height),
                      static_cast<WordType>(~static_cast<WordType>(0)));
    out.mat = bincv::BinMat<WordType>(out.buffer.data(), width, height, stride);

    // Every bit is set, padding included; now write the pixels, leaving the bits
    // past `width` exactly as dirty as they started.
    uint64_t state = seed;
    const uint32_t threshold = fillThreshold(fillRatio);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            out.mat.set(y, x, static_cast<uint32_t>(nextRandom(state) >> 40) < threshold);
        }
    }
}

template <typename WordType>
void testDirtySources(const char* wordTypeName) {
    std::cout << "\n--- majority3 with dirty source padding: " << wordTypeName << " ---\n";

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            const uint64_t seed = caseSeed(width, height, 1000);
            StridedMat<WordType> a, b, c;
            makeDirtyPadded(a, width, height, 0.5f, seed);
            makeDirtyPadded(b, width, height, 0.5f, seed ^ UINT64_C(0xDEADBEEF));
            makeDirtyPadded(c, width, height, 0.5f, seed ^ UINT64_C(0x5EED5EED));

            bincv::BinMat<WordType> dst(width, height);
            majority3(a.mat.constView(), b.mat.constView(), c.mat.constView(), dst.view());

            const std::string label = sizeLabel(wordTypeName, width, height, "dirty sources");
            const bool okPixels = (disagreements(a.mat, b.mat, c.mat, dst) == 0);
            BITSLICE_EXPECT(okPixels, "majority3 ignores the sources' padding bits", label);
            const bool okPadding = (bitsAcrossStride(dst) == dst.countNonZero());
            BITSLICE_EXPECT(okPadding, "majority3 leaves the destination's padding zero", label);
        }
    }
}

// ===========================================================================
// 7. The word-level padding contract: thresholdGE answers WHOLE WORDS
// ===========================================================================
//
// REVIEW FINDING, turned into cases. thresholdGE returns a full word and has no
// notion of `width`: every lane is answered, including the lanes past a row's
// last pixel, and at `threshold == 0` every lane is answered *yes* whatever the
// planes hold -- which is precisely the value a caller sweeping thresholds from 0
// reaches by arithmetic rather than by choice (that work’s requantization does exactly
// that sweep). A caller that stores such a word into a row's trailing word
// without masking leaves padding bits set past `width`, and the next word-wise
// reduction over that image over-counts: the failure, in a place no
// -Werror, no assert and no pixel comparison can see.
//
// majority3 masks internally because it owns its destination. thresholdGE returns
// a word and cannot, so the contract lives in its docstring -- and here, where it
// is executable. Two halves, and BOTH are the point:
//
// 1. the RAW result really does carry set bits past `width` (if a later change
// makes thresholdGE mask internally, this fails and the docstring is what
// has to change), and
// 2. the documented remedy -- AND with impl::rowTailMask<W>(width) -- really
// does leave the padding zero while leaving every live pixel's answer
// intact, checked against the same per-pixel count reference section 2 uses.
//
// The shape under test is the one the finding named: a 2x2 box sum (k = 4) over
// four sources whose padding is already dirty, then a threshold sweep, then the
// masked store -- i.e. what/this will do word by word.

template <typename WordType>
void testThresholdPaddingContract(const char* wordTypeName) {
    std::cout << "\n--- thresholdGE answers whole words, not rows: " << wordTypeName << " ---\n";

    constexpr size_t bits = wordBits<WordType>();
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));

    for (int width : WIDTHS) {
        const size_t w = static_cast<size_t>(width);
        const WordType tailMask = bincv::impl::rowTailMask<WordType>(w);
        const size_t livePixelsInTail = ((w - 1) % bits) + 1;
        const std::string label = sizeLabel(wordTypeName, width, 1, "tail word");

        // Four sources, each a single row whose trailing word is dirty past
        // `width`: a wrapped buffer's padding belongs to its caller (BinMat's
        // wrap constructor), so a kernel may not assume it is clean.
        StridedMat<WordType> src[4];
        for (size_t j = 0; j < 4; ++j) {
            makeDirtyPadded(src[j], width, 1, 0.5f, caseSeed(width, 1, 2000 + j));
        }

        // The trailing word of each source -- the only word where `width` and the
        // word boundary differ.
        const size_t last = bincv::impl::minRowWords<WordType>(w) - 1;
        WordType quad[4];
        for (size_t j = 0; j < 4; ++j) quad[j] = src[j].mat.ptr(0)[last];

        WordType planes[bitSlicedSumPlanes(4)];
        bitSlicedSum<WordType>(quad, 4, planes);

        for (unsigned t = 0; t <= 5u; ++t) {
            const WordType raw = thresholdGE<WordType>(planes, bitSlicedSumPlanes(4), t);
            const WordType stored = static_cast<WordType>(raw & tailMask);

            // (1) The raw word is not row-aware. At threshold 0 that is total and
            // unconditional -- all ones, however dirty or clean the planes were.
            if (t == 0u) {
                BITSLICE_EXPECT(raw == allOnes,
                                "thresholdGE at threshold 0 answers every lane, padding included",
                                label);
            }
            //... and whenever the row does not end on a word boundary, that
            // answer includes bits the row does not own. This is the assertion a
            // future "fix" that masks inside thresholdGE would break, on purpose.
            if (tailMask != allOnes && t == 0u) {
                const WordType past = static_cast<WordType>(raw & static_cast<WordType>(~tailMask));
                BITSLICE_EXPECT(past != 0,
                                "thresholdGE leaves the masking to its caller (bits past width set)",
                                label);
            }

            // (2) The documented remedy leaves the padding zero...
            const WordType padding = static_cast<WordType>(stored & static_cast<WordType>(~tailMask));
            BITSLICE_EXPECT(padding == 0,
                            "rowTailMask on the result leaves no bit past width",
                            label + " t=" + std::to_string(t));

            //... and costs no live pixel its answer. The reference is the count
            // of set pixels per lane, done per pixel, exactly as in section 2.
            bool okPixels = true;
            for (size_t lane = 0; lane < livePixelsInTail; ++lane) {
                const int x = static_cast<int>(last * bits + lane);
                unsigned count = 0;
                for (size_t j = 0; j < 4; ++j) {
                    if (src[j].mat.at(0, x)) ++count;
                }
                if (bitAt(stored, lane) != (count >= t)) okPixels = false;
            }
            BITSLICE_EXPECT(okPixels, "the masked result still answers every live pixel",
                            label + " t=" + std::to_string(t));
        }
    }

    // The finding's own trigger, spelled out: an all-zero 2x2 box over a row
    // narrower than one word still answers `>= 0` in every lane, so the word that
    // reaches a 5-pixel row's storage is 0xff raw and 0x1f masked at uint8_t.
    {
        const size_t narrow = 5;
        WordType quad[4] = {0, 0, 0, 0};
        WordType planes[bitSlicedSumPlanes(4)];
        bitSlicedSum<WordType>(quad, 4, planes);
        const WordType raw = thresholdGE<WordType>(planes, bitSlicedSumPlanes(4), 0u);
        const WordType masked =
            static_cast<WordType>(raw & bincv::impl::rowTailMask<WordType>(narrow));
        const std::string label = sizeLabel(wordTypeName, static_cast<int>(narrow), 1, "all-zero row");
        BITSLICE_EXPECT(raw == allOnes,
                        "an all-zero box sum still clears threshold 0 in every lane", label);
        const bool okMasked = (masked == static_cast<WordType>(0x1f));
        BITSLICE_EXPECT(okMasked, "the caller's mask cuts it back to the row's five pixels", label);
    }
}

// ===========================================================================
// 8. The OpenCV half: the reference pipeline's median, run as OpenCV calls
// ===========================================================================

#ifdef BINCV_WITH_OPENCV

/// @brief max(min(a, b), min(max(a, b), c)) over CV_8U -- the reference frontend's three-pixel
/// median, unchanged.
/// @note NOT a Tier 1 claim: OpenCV has no pointwise median of three images, and
/// bit-sliced arithmetic is Tier 3 (the design notes). This is a second,
/// independent reference for the operation has to reproduce.
cv::Mat openCvMedian3(const cv::Mat& a, const cv::Mat& b, const cv::Mat& c) {
    cv::Mat minAB, maxAB, minMaxC, out;
    cv::min(a, b, minAB);
    cv::max(a, b, maxAB);
    cv::min(maxAB, c, minMaxC);
    cv::max(minAB, minMaxC, out);
    return out;
}

template <typename WordType>
void testOpenCvMedian(const char* wordTypeName) {
    std::cout << "\n--- majority3 vs the reference median over CV_8U: " << wordTypeName
              << " ---\n";

    for (int width : bincv::test::equivalenceWidths()) {
        for (int height : {1, 3, 17}) {
            for (size_t f = 0; f < bincv::test::equivalenceFillRatios().size(); ++f) {
                const float fill = bincv::test::equivalenceFillRatios()[f];
                const uint64_t seed = caseSeed(width, height, f + 1100);

                bincv::BinMat<WordType> a =
                    bincv::test::randomBinary<WordType>(width, height, fill, seed);
                bincv::BinMat<WordType> b = bincv::test::randomBinary<WordType>(
                    width, height, fill, seed ^ UINT64_C(0xDEADBEEF));
                bincv::BinMat<WordType> c = bincv::test::randomBinary<WordType>(
                    width, height, fill, seed ^ UINT64_C(0x5EED5EED));

                // The harness's SECOND generator, which never touches the packing
                // or the unpacking path -- so the two sides of the comparison do
                // not share a conversion that could cancel.
                const cv::Mat cvA = bincv::test::randomCvMask(width, height, fill, seed);
                const cv::Mat cvB = bincv::test::randomCvMask(width, height, fill,
                                                              seed ^ UINT64_C(0xDEADBEEF));
                const cv::Mat cvC = bincv::test::randomCvMask(width, height, fill,
                                                              seed ^ UINT64_C(0x5EED5EED));

                bincv::BinMat<WordType> dst(width, height);
                majority3(a.constView(), b.constView(), c.constView(), dst.view());

                BINCV_EXPECT_BIT_EXACT(dst.constView(), openCvMedian3(cvA, cvB, cvC),
                                       bincv::test::caseLabel(wordTypeName, width, height, fill) +
                                           " [majority3]");
            }
        }
    }
}

#endif // BINCV_WITH_OPENCV

} // namespace

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

BINCV_TEST(BitSlice, Maj3_uint8_t)  { testMaj3<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, Maj3_uint16_t) { testMaj3<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, Maj3_uint32_t) { testMaj3<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, Maj3_uint64_t) { testMaj3<uint64_t>("uint64_t"); }

BINCV_TEST(BitSlice, Sum_uint8_t)  { testSumSmall<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, Sum_uint16_t) { testSumSmall<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, Sum_uint32_t) { testSumSmall<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, Sum_uint64_t) { testSumSmall<uint64_t>("uint64_t"); }

BINCV_TEST(BitSlice, SumWide_uint8_t)  { testSumWide<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, SumWide_uint16_t) { testSumWide<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, SumWide_uint32_t) { testSumWide<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, SumWide_uint64_t) { testSumWide<uint64_t>("uint64_t"); }

BINCV_TEST(BitSlice, Threshold_uint8_t)  { testThresholdGEValues<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, Threshold_uint16_t) { testThresholdGEValues<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, Threshold_uint32_t) { testThresholdGEValues<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, Threshold_uint64_t) { testThresholdGEValues<uint64_t>("uint64_t"); }

BINCV_TEST(BitSlice, Composed_uint8_t)  { testComposed<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, Composed_uint16_t) { testComposed<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, Composed_uint32_t) { testComposed<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, Composed_uint64_t) { testComposed<uint64_t>("uint64_t"); }

BINCV_TEST(BitSlice, Degenerate_uint8_t)  { testDegenerate<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, Degenerate_uint16_t) { testDegenerate<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, Degenerate_uint32_t) { testDegenerate<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, Degenerate_uint64_t) { testDegenerate<uint64_t>("uint64_t"); }

BINCV_TEST(BitSlice, ThresholdPadding_uint8_t)  { testThresholdPaddingContract<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, ThresholdPadding_uint16_t) { testThresholdPaddingContract<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, ThresholdPadding_uint32_t) { testThresholdPaddingContract<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, ThresholdPadding_uint64_t) { testThresholdPaddingContract<uint64_t>("uint64_t"); }

BINCV_TEST(BitSlice, Majority3_uint8_t)  { testMajority3Reference<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, Majority3_uint16_t) { testMajority3Reference<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, Majority3_uint32_t) { testMajority3Reference<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, Majority3_uint64_t) { testMajority3Reference<uint64_t>("uint64_t"); }

BINCV_TEST(BitSlice, Strides_uint8_t)  { testDifferingStrides<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, Strides_uint16_t) { testDifferingStrides<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, Strides_uint32_t) { testDifferingStrides<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, Strides_uint64_t) { testDifferingStrides<uint64_t>("uint64_t"); }

BINCV_TEST(BitSlice, InPlace_uint8_t)  { testInPlace<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, InPlace_uint16_t) { testInPlace<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, InPlace_uint32_t) { testInPlace<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, InPlace_uint64_t) { testInPlace<uint64_t>("uint64_t"); }

BINCV_TEST(BitSlice, DegenerateViews_uint8_t)  { testDegenerateViews<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, DegenerateViews_uint16_t) { testDegenerateViews<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, DegenerateViews_uint32_t) { testDegenerateViews<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, DegenerateViews_uint64_t) { testDegenerateViews<uint64_t>("uint64_t"); }

BINCV_TEST(BitSlice, DirtySources_uint8_t)  { testDirtySources<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, DirtySources_uint16_t) { testDirtySources<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, DirtySources_uint32_t) { testDirtySources<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, DirtySources_uint64_t) { testDirtySources<uint64_t>("uint64_t"); }

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(BitSlice, OpenCv_uint8_t)  { testOpenCvMedian<uint8_t>("uint8_t"); }
BINCV_TEST(BitSlice, OpenCv_uint16_t) { testOpenCvMedian<uint16_t>("uint16_t"); }
BINCV_TEST(BitSlice, OpenCv_uint32_t) { testOpenCvMedian<uint32_t>("uint32_t"); }
BINCV_TEST(BitSlice, OpenCv_uint64_t) { testOpenCvMedian<uint64_t>("uint64_t"); }
#endif

BINCV_TEST_MAIN("test_bitslice")
