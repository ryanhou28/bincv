// The LK gradient covariance -- ops/covariance.hpp.
//
// THIS SUITE IS WHAT STANDS BEHIND THE PROJECT'S CENTRAL TECHNICAL CLAIM.
// the design notes asserts that the whole 2x2 Lucas-Kanade gradient covariance
// reduces to masked population counts over sign-magnitude ternary planes. If that
// identity is wrong, bit-parallel software cannot do this job and the project's
// premise is wrong -- so the identity is not assumed here, it is PROVEN against a
// reference that knows nothing about bits.
//
// HOW IT IS PROVEN, AND WHY THE ORACLE IS IN FLOAT
//
// The reference is a per-pixel float accumulation: it reads each ternary value as
// a float in {-1.0f, 0.0f, +1.0f} and forms `xx += a*a; yy += b*b; xy += a*b` --
// the textbook covariance, with multiplies, one pixel at a time. It shares no
// code, no clipping ladder and no word arithmetic with the library. Writing it in
// float rather than in integers is deliberate: it is the formulation the identity
// CLAIMS to replace, so agreement is evidence about the identity rather than about
// two spellings of the same popcount.
//
// **Agreement is required to be EXACT, and the comparison is made twice.** Every
// entry of the matrix is an integer -- ternary products are in {-1, 0, +1} and
// there are at most a window's worth of them -- so a tolerance would not be a
// rounding allowance, it would be a place for a real disagreement to hide. Each
// comparison checks integer equality AND float equality of the same pair, which is
// what makes "the float oracle is integral" a checked property rather than an
// argument.
//
// THE SWEEP, AND WHY IT IS A SWEEP AND NOT A SAMPLE
//
// Window positions are swept from a FULL WINDOW WIDTH OUTSIDE every edge to a full
// window width past it, in both axes, at W = 7, 15 and 31, at all four word types,
// over two independently built frames. That covers, without anyone choosing them:
// wholly-outside windows, windows clipped on one/two/three edges, windows whose
// first and last words are the same word, windows that begin and end at every
// residue mod WordBits, and the interior. The total number of positions compared
// is printed and is itself CHECKED against the arithmetic that defines it, so a
// sweep that silently got smaller fails rather than passing faster.
//
// **THE FRAME IS TALLER THAN THE LARGEST WINDOW, and that is load-bearing.** It
// was not: at SWEEP_HEIGHT = 11 every swept position of a 15x15 or 31x31 window
// was clipped to 11 rows, so nothing in this file ever reduced a window taller
// than 11 image rows and the 31x31 reference LK window was only ever evaluated
// clipped. A mutant returning junk for any window taller than 11 rows passed the
// whole suite. The frame is now 35 rows, two static_asserts pin the relationship,
// and sweepAgainstOracle COUNTS the fully-interior positions at each window size
// and requires them -- clipped coverage and interior coverage are different
// properties and this suite needs both.
//
// WHY THERE IS NO OPENCV HALF, AND WHY THIS SUITE IS STILL CORE
//
// The covariance is API TIER 3: cv:: has no operation with these semantics, so
// there is no bit-exactness promise to a cv:: call and no Tier 1 denominator to
// compare against (cornerEigenValsAndVecs computes something else, from byte
// images, with a Sobel and a box filter). What stands behind this operation is the
// float oracle, the full-frame sweeps, and the invariance checks below -- none of
// which needs OpenCV, and all of which have to run where the embedded claim does:
// core-only, -fno-exceptions, and Debug, the only configuration in which the
// kernels' BINCV_ASSERT preconditions are live.
//
// WHAT ELSE IS PINNED HERE
//
// * NO SCRATCH AND NO HEAP. `operator new` is counted across the calls -- the
// plain AND the C++17 over-aligned forms, since only the plain pair was
// replaced at first and an over-aligned scratch buffer was therefore invisible
// to it -- and must be zero. The counter is itself exercised on one allocation
// of each kind, so the zero is a reading and not a blind spot. This is not
// decoration: the four-argument selector form was chosen over the 11-14%
// FASTER precomputed-plane form precisely because it needs no plane (
// axis 3,, CLAUDE.md's memory tiebreak). A covariance that quietly
// allocated would have discarded the speed and bought nothing.
// * THE SIGN PLANE IS READ ONLY WHERE BOTH MAGNITUDES ARE SET. Dirtying every
// sign bit over a zero magnitude -- which the canonical-zero rule says carries
// no information -- must not move any of the three numbers.
// * PADDING IS NEVER COUNTED, including when the source's padding bits
// are all ones, and including when the "padding" is a neighbour's live pixels
// because the view windows a wider frame.
// * The container spelling and the view spelling agree, and both agree with the
// fused entry point they are a naming of.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/covariance.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/reduce.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "test_util.hpp"

// ---------------------------------------------------------------------------
// The allocation counter, in the idiom tests/test_storage.cpp established and
// tests/test_pyramid.cpp reuses.
//
// ops/covariance.hpp promise 3 says "no allocation, no throw, no scratch". The
// no-scratch half is the REASON this operation calls the four-argument selector
// form rather than the faster plane form, so it is measured rather than asserted.
//
// **THE OVER-ALIGNED FORMS ARE REPLACED TOO, and that is not decoration.** C++17
// routes `new` for an over-aligned type to a DIFFERENT pair of functions --
// `operator new(size_t, align_val_t)` -- and a counter that replaces only the
// plain ones does not see them. Measured: with only the plain forms replaced,
// `new char[16]` counts 1 while `new` of an `alignas(64)` type counts 0. Scratch
// for a vectorized or cache-line-aligned kernel is exactly the kind that would be
// over-aligned, so a covariance that allocated one would have passed the check
// below unchanged. The counter has to cover the allocation the property is about.
// ---------------------------------------------------------------------------
namespace {
std::size_t g_newCount = 0;

void* countedAllocate(std::size_t bytes) {
    ++g_newCount;
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    // Cannot throw std::bad_alloc: this file also builds with -fno-exceptions.
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) std::abort();
    return p;
}

/// @brief The over-aligned allocation path, counted on the same counter.
/// @note std::aligned_alloc requires the size to be a multiple of the alignment,
/// which `operator new` does not promise, so it is rounded up here. The
/// alignment is already a power of two -- the language guarantees it.
void* countedAllocateAligned(std::size_t bytes, std::size_t alignment) {
    ++g_newCount;
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    const std::size_t wanted = (bytes == 0) ? 1 : bytes;
    const std::size_t rounded = ((wanted + alignment - 1) / alignment) * alignment;
    void* p = std::aligned_alloc(alignment, rounded);
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

void* operator new(std::size_t bytes, std::align_val_t a) {
    return countedAllocateAligned(bytes, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t bytes, std::align_val_t a) {
    return countedAllocateAligned(bytes, static_cast<std::size_t>(a));
}

void operator delete(void* p, std::align_val_t) noexcept                { countedFree(p); }
void operator delete[](void* p, std::align_val_t) noexcept              { countedFree(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept   { countedFree(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { countedFree(p); }

namespace {

using bincv::BinMatConstView;
using bincv::GradientCovariance;
using bincv::Rect;
using bincv::TernaryMat;
using bincv::gradientCovariance;

#define COV_EXPECT(ok, what, detailExpr) \
    ::bincv::test::reportCheck((ok), (what), __FILE__, __LINE__, \
                               (ok) ? std::string() : (detailExpr))

// The three window sizes this is specified over. 31 is the reference pipeline's
// LK window; 7 and 15 are the smaller ones a pyramid level uses.
const int WINDOW_SIZES[] = {7, 15, 31};

// The sweep frame. Wide enough that every word type has SEVERAL words per row and
// a partial trailing one (70 pixels is 9 words at uint8_t, 5 at uint16_t, 3 at
// uint32_t and 2 at uint64_t, with 2, 10, 26 and 58 padding bits respectively).
// Width 70 is deliberately not a multiple of any word width: a region's word
// geometry depends on (x0 mod WordBits, x1 mod WordBits), and the sweep's 2W+1
// origins per row cover every residue at every word type.
//
// **THE HEIGHT MUST EXCEED THE LARGEST WINDOW, and that is a correctness property
// of this suite rather than a sizing preference.** It was 11 -- smaller than two
// of the three window sizes -- and at that height EVERY swept position of a 15x15
// or 31x31 window was vertically clipped to 11 rows, so no value check anywhere in
// the file ever reduced a window taller than 11 image rows. The 31x31 reference LK
// window was therefore only ever evaluated clipped, and a whole class of
// row-accumulation bugs was invisible. Measured: injecting
// `if (clippedHeight >= 12) return {12345, 6789, -42};` into the view form of
// gradientCovariance left this suite at 2372/2372 checks passed, exit 0. At 35 the
// same mutant fails 402 checks. The floor below is what keeps it that way.
const int SWEEP_WIDTH = 70;
const int SWEEP_HEIGHT = 35;

// The largest window swept. Kept as its own constant so the relationship above can
// be a compile-time property: a future edit that shrinks the frame, or adds a
// window size larger than the frame, is a build failure rather than a suite that
// is still green while proving less. checkWindowSizes below pins that this is
// really the maximum of WINDOW_SIZES, since a static_assert cannot read the array.
constexpr int MAX_WINDOW_SIZE = 31;
static_assert(SWEEP_HEIGHT > MAX_WINDOW_SIZE,
              "the sweep frame must be TALLER than the largest window, or no window position "
              "is ever fully interior in y and every reduction is clipped -- see the note above");
static_assert(SWEEP_WIDTH > MAX_WINDOW_SIZE,
              "the sweep frame must be WIDER than the largest window, for the same reason");

// Total positions compared against the float oracle, across every case in this
// binary. Printed at the end of each case and checked against the arithmetic that
// defines it, so a sweep that got smaller fails rather than passing faster.
size_t g_oraclePositions = 0;

// Window positions compared kernel-against-kernel rather than against the oracle:
// the invariance sweeps (dirty signs, dirty padding), the container-versus-view
// sweep and the aliasing sweep. Counted separately, because only the oracle
// comparisons are evidence about the IDENTITY -- these are evidence about the
// contracts around it, and adding the two together would overstate the first.
size_t g_invariancePositions = 0;

// ---------------------------------------------------------------------------
// Content
// ---------------------------------------------------------------------------

// splitmix64, so a failure reproduces exactly. Deliberately this file's own copy:
// the generator that builds an input and the reference that judges the output must
// not share machinery, or a fault in the shared part cancels (that work’s rule).
uint64_t nextRandom(uint64_t& state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

/// @brief Fills a ternary matrix with values in {-1, 0, +1} through set, so its
/// padding bits stay clear and its zeros stay canonical.
/// @param zeroBias Out of 4 draws, how many are forced to 0. Higher means sparser
/// gradients, which is what a real binarized derivative looks like.
template <typename WordType>
void fillRandomTernary(TernaryMat<WordType>& m, uint64_t seed, int zeroBias) {
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            const uint64_t r = nextRandom(state);
            const int bucket = static_cast<int>(r & 3u);
            int value = 0;
            if (bucket >= zeroBias) value = ((r >> 8) & 1u) ? 1 : -1;
            m.set(y, x, value);
        }
    }
}

/// @brief Fills a binary matrix, for the case whose ternary planes come out of the
/// real derivative rather than out of a generator.
template <typename WordType>
void fillRandomBinary(bincv::BinMat<WordType>& m, uint64_t seed) {
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            if ((nextRandom(state) & 1u) != 0) m.set(y, x, true);
        }
    }
}

/// @brief Sets one bit of a view directly, bypassing set.
/// @note Used to dirty a sign bit where the magnitude is zero -- which set
/// correctly refuses to do (the canonical-zero rule) -- and to dirty padding
/// bits, which no public writer will touch either.
template <typename WordType>
void setBit(bincv::BinMatView<WordType> v, size_t y, size_t x) {
    constexpr size_t bits = bincv::BinMatView<WordType>::WordBits;
    WordType* row = v.row(y);
    row[x / bits] = static_cast<WordType>(row[x / bits] |
                                          static_cast<WordType>(static_cast<WordType>(1)
                                                                << (x % bits)));
}

/// @brief Reads one pixel out of a view. The bit convention, and nothing else.
template <typename WordType>
bool bitAt(const BinMatConstView<WordType>& v, int y, int x) {
    constexpr size_t bits = BinMatConstView<WordType>::WordBits;
    const size_t ux = static_cast<size_t>(x);
    const WordType mask = static_cast<WordType>(static_cast<WordType>(1) << (ux % bits));
    return (v.row(static_cast<size_t>(y))[ux / bits] & mask) != 0;
}

// ---------------------------------------------------------------------------
// THE ORACLE: a per-pixel float covariance, written before the kernel it judges
// ---------------------------------------------------------------------------
//
// It knows nothing about words, masks or popcounts. It reads a ternary pixel as a
// float, multiplies, and accumulates -- the formulation the design notes claims
// the masked popcounts are equal to. That is the whole point: two spellings of the
// same popcount agreeing would prove nothing about the identity.

/// @brief A frame of ternary values as plain floats, indexed [y * width + x].
/// @note The conversion applies the canonical-zero rule INDEPENDENTLY of the
/// library: magnitude clear means 0.0f whatever the sign bit holds. So a
/// frame whose sign plane is dirty over zero magnitudes produces the same
/// oracle frame as a clean one, which is what makes that invariance a real
/// comparison rather than a tautology.
struct FloatFrame {
    int width = 0;
    int height = 0;
    std::vector<float> dx;
    std::vector<float> dy;
};

template <typename WordType>
FloatFrame toFloatFrame(const BinMatConstView<WordType>& magX,
                        const BinMatConstView<WordType>& magY,
                        const BinMatConstView<WordType>& signX,
                        const BinMatConstView<WordType>& signY) {
    FloatFrame f;
    f.width = static_cast<int>(magX.width);
    f.height = static_cast<int>(magX.height);
    f.dx.assign(static_cast<size_t>(f.width) * static_cast<size_t>(f.height), 0.0f);
    f.dy.assign(static_cast<size_t>(f.width) * static_cast<size_t>(f.height), 0.0f);
    for (int y = 0; y < f.height; ++y) {
        for (int x = 0; x < f.width; ++x) {
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(f.width) +
                             static_cast<size_t>(x);
            if (bitAt(magX, y, x)) f.dx[i] = bitAt(signX, y, x) ? -1.0f : 1.0f;
            if (bitAt(magY, y, x)) f.dy[i] = bitAt(signY, y, x) ? -1.0f : 1.0f;
        }
    }
    return f;
}

/// @brief The 2x2 covariance over a window, in float, one pixel at a time.
/// @note The clip is written independently of impl::clipRegion -- min/max against
/// the extents rather than the library's early-exit ladder -- so the region
/// contract has two implementations to disagree.
struct FloatCovariance {
    float xx = 0.0f;
    float yy = 0.0f;
    float xy = 0.0f;
};

FloatCovariance refCovariance(const FloatFrame& f, const Rect& w) {
    FloatCovariance out;
    if (w.width <= 0 || w.height <= 0) return out;
    const long long x0 = std::max<long long>(static_cast<long long>(w.x), 0);
    const long long y0 = std::max<long long>(static_cast<long long>(w.y), 0);
    const long long x1 = std::min<long long>(
        static_cast<long long>(w.x) + static_cast<long long>(w.width), f.width);
    const long long y1 = std::min<long long>(
        static_cast<long long>(w.y) + static_cast<long long>(w.height), f.height);
    for (long long y = y0; y < y1; ++y) {
        for (long long x = x0; x < x1; ++x) {
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(f.width) +
                             static_cast<size_t>(x);
            const float a = f.dx[i];
            const float b = f.dy[i];
            out.xx += a * a;
            out.yy += b * b;
            out.xy += a * b;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

std::string rectText(const Rect& r) {
    return "[x=" + std::to_string(r.x) + " y=" + std::to_string(r.y) +
           " w=" + std::to_string(r.width) + " h=" + std::to_string(r.height) + "]";
}

std::string covText(const GradientCovariance& c) {
    return "{xx=" + std::to_string(c.sumXX) + " yy=" + std::to_string(c.sumYY) +
           " xy=" + std::to_string(c.sumXY) + "}";
}

std::string floatCovText(const FloatCovariance& c) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "{xx=%.1f yy=%.1f xy=%.1f}", static_cast<double>(c.xx),
                  static_cast<double>(c.yy), static_cast<double>(c.xy));
    return std::string(buf);
}

/// @brief Exact agreement with the oracle -- integer AND float, no tolerance.
/// @note Both spellings of the same equality, on purpose. The integer comparison
/// is the operation's contract; the float one additionally pins that the
/// oracle's accumulation is integral, which is the premise that makes "no
/// tolerance" a legitimate demand rather than a strict one.
bool agrees(const GradientCovariance& k, const FloatCovariance& r) {
    return k.sumXX == static_cast<int64_t>(r.xx) && k.sumYY == static_cast<int64_t>(r.yy) &&
           k.sumXY == static_cast<int64_t>(r.xy) && static_cast<float>(k.sumXX) == r.xx &&
           static_cast<float>(k.sumYY) == r.yy && static_cast<float>(k.sumXY) == r.xy;
}

bool same(const GradientCovariance& a, const GradientCovariance& b) {
    return a.sumXX == b.sumXX && a.sumYY == b.sumYY && a.sumXY == b.sumXY;
}

// ---------------------------------------------------------------------------
// 1. THE IDENTITY, at every window position of a whole frame
// ---------------------------------------------------------------------------

/// @brief Sweeps every window position from a full window outside the frame to a
/// full window past it, against the float oracle.
/// @param magX, magY, signX, signY The four planes, as views -- so the same sweep
/// serves a container, a dirtied container and a window onto a wider frame.
/// @return Positions compared.
/// @note ONE CHECK PER (window size, row of positions), not per position: a
/// per-position check would put ~50000 entries in the CHECKS column for one
/// property, and a per-case check would make the column blind to a sweep
/// that lost its margin. The row is the granularity at which a failure still
/// names where it happened -- the message carries the first bad window and
/// the mismatch count for the row.
template <typename WordType>
size_t sweepAgainstOracle(const BinMatConstView<WordType>& magX,
                          const BinMatConstView<WordType>& magY,
                          const BinMatConstView<WordType>& signX,
                          const BinMatConstView<WordType>& signY, const std::string& label) {
    const FloatFrame frame = toFloatFrame(magX, magY, signX, signY);
    const int width = frame.width;
    const int height = frame.height;
    size_t positions = 0;

    for (int windowSize : WINDOW_SIZES) {
        // Positions at which the window lies WHOLLY INSIDE the frame, so the
        // reduction runs over all windowSize rows and all windowSize columns with
        // nothing clipped away. Counted rather than assumed: when the frame was
        // 11 rows tall this was ZERO at W = 15 and W = 31, and the suite passed a
        // mutant that returned junk for any window taller than 11 rows.
        size_t interior = 0;
        for (int y = -windowSize; y <= height + windowSize; ++y) {
            size_t bad = 0;
            std::string firstBad;
            for (int x = -windowSize; x <= width + windowSize; ++x) {
                const Rect w(x, y, windowSize, windowSize);
                const GradientCovariance got =
                    gradientCovariance<WordType>(magX, magY, signX, signY, w);
                const FloatCovariance want = refCovariance(frame, w);
                ++positions;
                if (x >= 0 && y >= 0 && x + windowSize <= width && y + windowSize <= height) {
                    ++interior;
                }
                if (!agrees(got, want)) {
                    ++bad;
                    if (firstBad.empty()) {
                        firstBad = rectText(w) + " got " + covText(got) + " oracle " +
                                   floatCovText(want);
                    }
                }
            }
            COV_EXPECT(bad == 0,
                       "the popcount identity equals the per-pixel float covariance at every "
                       "window position of the row",
                       label + " W=" + std::to_string(windowSize) + " y=" + std::to_string(y) +
                           " mismatches=" + std::to_string(bad) + " first: " + firstBad);
        }
        // THE REGRESSION CHECK for the frame that was too short. Clipped positions
        // are the interesting ones for earlier work, but a suite made only of them never
        // reduces a full-height window at all, and the 31x31 window of
        // the design notes is the one the operation exists for.
        const size_t expectedInterior = static_cast<size_t>(width - windowSize + 1) *
                                        static_cast<size_t>(height - windowSize + 1);
        COV_EXPECT(interior == expectedInterior && interior > 0,
                   "this window size has FULLY INTERIOR positions -- the frame is larger than "
                   "the window, so the identity is checked unclipped and not only at edges",
                   label + " W=" + std::to_string(windowSize) + " interior=" +
                       std::to_string(interior) + ", expected " +
                       std::to_string(expectedInterior) + " on a " + std::to_string(width) + "x" +
                       std::to_string(height) + " frame");
    }
    return positions;
}

/// @brief MAX_WINDOW_SIZE really is the largest entry of WINDOW_SIZES.
/// @note The two static_asserts at the top of this file guard the frame against
/// the windows, but a static_assert cannot read a runtime array. Without
/// this, adding a fourth window size larger than the frame would restore
/// exactly the gap those asserts exist to close, silently.
void checkWindowSizes() {
    int largest = 0;
    for (int windowSize : WINDOW_SIZES) {
        if (windowSize > largest) largest = windowSize;
    }
    COV_EXPECT(largest == MAX_WINDOW_SIZE,
               "MAX_WINDOW_SIZE is the largest window actually swept, so the frame-size "
               "static_asserts guard every window size",
               "largest swept = " + std::to_string(largest) + ", MAX_WINDOW_SIZE = " +
                   std::to_string(MAX_WINDOW_SIZE));
}

/// @brief Positions sweepAgainstOracle must have compared, from the geometry.
/// @note The sweep's own arithmetic, written out separately, so that a margin that
/// shrank from `windowSize` to something smaller fails a check instead of
/// quietly testing less. This is the one property in the file whose failure
/// mode is "still green, but no longer proving anything".
size_t expectedSweepPositions(int width, int height) {
    size_t total = 0;
    for (int windowSize : WINDOW_SIZES) {
        const size_t xs = static_cast<size_t>(width + 2 * windowSize + 1);
        const size_t ys = static_cast<size_t>(height + 2 * windowSize + 1);
        total += xs * ys;
    }
    return total;
}

// ---------------------------------------------------------------------------
// The cases
// ---------------------------------------------------------------------------

/// @brief Case 1: a generated ternary pair, swept whole.
template <typename WordType>
void testGeneratedFrame(const char* wordTypeName) {
    const std::string label = std::string(wordTypeName) + " [generated]";
    TernaryMat<WordType> dx(SWEEP_WIDTH, SWEEP_HEIGHT);
    TernaryMat<WordType> dy(SWEEP_WIDTH, SWEEP_HEIGHT);
    fillRandomTernary(dx, UINT64_C(0x5EED0C0FFEE00001), 1);
    fillRandomTernary(dy, UINT64_C(0x5EED0C0FFEE00002), 1);

    const size_t positions =
        sweepAgainstOracle<WordType>(dx.constMagnitude(0), dy.constMagnitude(0), dx.constSign(),
                                     dy.constSign(), label);
    g_oraclePositions += positions;
    COV_EXPECT(positions == expectedSweepPositions(SWEEP_WIDTH, SWEEP_HEIGHT),
               "the sweep visited every position its geometry defines",
               label + " compared " + std::to_string(positions) + ", expected " +
                   std::to_string(expectedSweepPositions(SWEEP_WIDTH, SWEEP_HEIGHT)));

    // The container spelling is the one specifies, and it must be the same
    // three numbers as the view spelling it forwards to -- at every position of
    // one window size, not at a sample.
    {
        size_t bad = 0;
        for (int y = -15; y <= SWEEP_HEIGHT + 15; ++y) {
            for (int x = -15; x <= SWEEP_WIDTH + 15; ++x) {
                const Rect w(x, y, 15, 15);
                const GradientCovariance viaContainer = gradientCovariance(dx, dy, w);
                const GradientCovariance viaViews = gradientCovariance<WordType>(
                    dx.constMagnitude(0), dy.constMagnitude(0), dx.constSign(), dy.constSign(), w);
                ++g_invariancePositions;
                if (!same(viaContainer, viaViews)) ++bad;
            }
        }
        COV_EXPECT(bad == 0, "the container spelling equals the view spelling everywhere",
                   label + " mismatches=" + std::to_string(bad));
    }

    // Aliasing is unrestricted (ops/reduce.hpp promise 3): dx against itself is
    // the degenerate case a caller reaches by accident, and it has a closed form
    // -- a plane never disagrees in sign with itself, so the cross term is ΣIx².
    {
        size_t bad = 0;
        for (int y = -7; y <= SWEEP_HEIGHT + 7; ++y) {
            for (int x = -7; x <= SWEEP_WIDTH + 7; ++x) {
                const Rect w(x, y, 7, 7);
                const GradientCovariance self = gradientCovariance(dx, dx, w);
                ++g_invariancePositions;
                if (self.sumXX != self.sumYY || self.sumXY != self.sumXX) ++bad;
            }
        }
        COV_EXPECT(bad == 0,
                   "gradientCovariance(dx, dx, w) gives xx == yy == xy: a plane never "
                   "disagrees in sign with itself",
                   label + " mismatches=" + std::to_string(bad));
    }
}

/// @brief Case 2: the planes the REAL pipeline produces -- that work’s derivative.
/// @note The generated case above draws dx and dy independently, so its cross term
/// hovers around zero. A real binarized derivative pair is correlated, sparse
/// and structured, and its sign planes are the borrow bits ops/derivative.hpp
/// computes rather than random draws. This is the input this is FOR, and it
/// is swept against the same oracle.
template <typename WordType>
void testDerivativeFrame(const char* wordTypeName) {
    const std::string label = std::string(wordTypeName) + " [from derivative]";
    bincv::BinMat<WordType> src(SWEEP_WIDTH, SWEEP_HEIGHT);
    fillRandomBinary(src, UINT64_C(0x5EED0C0FFEE00011));

    TernaryMat<WordType> dx(SWEEP_WIDTH, SWEEP_HEIGHT);
    TernaryMat<WordType> dy(SWEEP_WIDTH, SWEEP_HEIGHT);
    bincv::derivativeX(src, dx);
    bincv::derivativeY(src, dy);

    const size_t positions =
        sweepAgainstOracle<WordType>(dx.constMagnitude(0), dy.constMagnitude(0), dx.constSign(),
                                     dy.constSign(), label);
    g_oraclePositions += positions;
    COV_EXPECT(positions == expectedSweepPositions(SWEEP_WIDTH, SWEEP_HEIGHT),
               "the sweep visited every position its geometry defines",
               label + " compared " + std::to_string(positions));

    // The cross term must actually go both ways somewhere in the frame, or the
    // sweep above proved the identity only on the half of its range that never
    // exercises the subtraction. A binarized derivative pair is correlated, so
    // both signs occur -- and if a future change made them not, this says so
    // rather than leaving a passing suite that tests less than it reads as.
    bool sawNegative = false;
    bool sawPositive = false;
    for (int y = 0; y + 7 <= SWEEP_HEIGHT; ++y) {
        for (int x = 0; x + 7 <= SWEEP_WIDTH; ++x) {
            const GradientCovariance c = gradientCovariance(dx, dy, Rect(x, y, 7, 7));
            if (c.sumXY < 0) sawNegative = true;
            if (c.sumXY > 0) sawPositive = true;
        }
    }
    COV_EXPECT(sawNegative && sawPositive,
               "the swept frame produces cross terms of BOTH signs, so the signed "
               "subtraction is exercised in both directions",
               label + " negative=" + std::to_string(sawNegative ? 1 : 0) +
                   " positive=" + std::to_string(sawPositive ? 1 : 0));
}

/// @brief Case 3: sign bits dirtied wherever the magnitude is zero.
/// @note The canonical-zero rule (quantMat.hpp) says such a bit carries no
/// information. ops/covariance.hpp promise 5 turns that into a property of
/// this operation: the sign planes are read only where BOTH magnitudes are
/// set, so dirtying them cannot move a number. Checked by comparing a clean
/// frame against a dirtied copy at every position, and by running the
/// dirtied frame against the oracle -- which applies the same rule
/// independently.
template <typename WordType>
void testDirtySigns(const char* wordTypeName) {
    const std::string label = std::string(wordTypeName) + " [dirty signs]";
    TernaryMat<WordType> dx(SWEEP_WIDTH, SWEEP_HEIGHT);
    TernaryMat<WordType> dy(SWEEP_WIDTH, SWEEP_HEIGHT);
    fillRandomTernary(dx, UINT64_C(0x5EED0C0FFEE00021), 2);
    fillRandomTernary(dy, UINT64_C(0x5EED0C0FFEE00022), 2);

    TernaryMat<WordType> dirtyX(dx);  // deep copy
    TernaryMat<WordType> dirtyY(dy);
    size_t dirtied = 0;
    for (int y = 0; y < SWEEP_HEIGHT; ++y) {
        for (int x = 0; x < SWEEP_WIDTH; ++x) {
            if (dirtyX.magnitudeAt(y, x) == 0) {
                setBit(dirtyX.sign(), static_cast<size_t>(y), static_cast<size_t>(x));
                ++dirtied;
            }
            if (dirtyY.magnitudeAt(y, x) == 0) {
                setBit(dirtyY.sign(), static_cast<size_t>(y), static_cast<size_t>(x));
                ++dirtied;
            }
        }
    }
    COV_EXPECT(dirtied > 0, "the dirty-sign case actually dirtied something",
               label + " dirtied=" + std::to_string(dirtied));

    size_t bad = 0;
    std::string firstBad;
    size_t positions = 0;
    for (int windowSize : WINDOW_SIZES) {
        for (int y = -windowSize; y <= SWEEP_HEIGHT + windowSize; ++y) {
            for (int x = -windowSize; x <= SWEEP_WIDTH + windowSize; ++x) {
                const Rect w(x, y, windowSize, windowSize);
                const GradientCovariance clean = gradientCovariance(dx, dy, w);
                const GradientCovariance dirty = gradientCovariance(dirtyX, dirtyY, w);
                ++positions;
                ++g_invariancePositions;
                if (!same(clean, dirty)) {
                    ++bad;
                    if (firstBad.empty()) {
                        firstBad = rectText(w) + " clean " + covText(clean) + " dirty " +
                                   covText(dirty);
                    }
                }
            }
        }
    }
    COV_EXPECT(bad == 0,
               "a sign bit over a zero magnitude changes nothing, at every window position",
               label + " positions=" + std::to_string(positions) +
                   " mismatches=" + std::to_string(bad) + " first: " + firstBad);

    // And against the oracle, which applies the canonical-zero rule its own way.
    g_oraclePositions += sweepAgainstOracle<WordType>(dirtyX.constMagnitude(0),
                                                      dirtyY.constMagnitude(0), dirtyX.constSign(),
                                                      dirtyY.constSign(), label);
}

/// @brief Case 4: every padding bit set, in all four planes.
/// @note A wrapped buffer's padding belongs to its caller -- sensor DMA, a
/// sub-region of a larger frame -- so dirty padding is a SUPPORTED
/// construction and a covariance that over-counts on one returns a wrong
/// answer from a legal input. Width 70 leaves 2 padding bits at uint8_t, 10
/// at uint16_t, 26 at uint32_t and 58 at uint64_t, so every word type has
/// some.
template <typename WordType>
void testDirtyPadding(const char* wordTypeName) {
    const std::string label = std::string(wordTypeName) + " [dirty padding]";
    TernaryMat<WordType> dx(SWEEP_WIDTH, SWEEP_HEIGHT);
    TernaryMat<WordType> dy(SWEEP_WIDTH, SWEEP_HEIGHT);
    fillRandomTernary(dx, UINT64_C(0x5EED0C0FFEE00031), 1);
    fillRandomTernary(dy, UINT64_C(0x5EED0C0FFEE00032), 1);

    TernaryMat<WordType> paddedX(dx);
    TernaryMat<WordType> paddedY(dy);
    // getAlignedWidth is the row stride in WORDS, so the padding runs from the
    // last real column to the end of the last word -- not to getAlignedWidth.
    // The first version of this loop conflated the two and dirtied nothing at all,
    // which is exactly why the "actually dirtied something" check below exists.
    const size_t paddingEnd = paddedX.getAlignedWidth() * TernaryMat<WordType>::WordBits;
    size_t dirtied = 0;
    for (size_t plane = 0; plane < TernaryMat<WordType>::Planes; ++plane) {
        bincv::BinMatView<WordType> px = paddedX.planes().plane(plane);
        bincv::BinMatView<WordType> py = paddedY.planes().plane(plane);
        for (size_t y = 0; y < static_cast<size_t>(SWEEP_HEIGHT); ++y) {
            for (size_t x = static_cast<size_t>(SWEEP_WIDTH); x < paddingEnd; ++x) {
                setBit(px, y, x);
                setBit(py, y, x);
                dirtied += 2;
            }
        }
    }
    COV_EXPECT(dirtied > 0, "the dirty-padding case actually dirtied something",
               label + " padding bits per row=" + std::to_string(paddingEnd - SWEEP_WIDTH) +
                   " dirtied=" + std::to_string(dirtied));

    size_t bad = 0;
    std::string firstBad;
    for (int windowSize : WINDOW_SIZES) {
        for (int y = -windowSize; y <= SWEEP_HEIGHT + windowSize; ++y) {
            for (int x = -windowSize; x <= SWEEP_WIDTH + windowSize; ++x) {
                const Rect w(x, y, windowSize, windowSize);
                const GradientCovariance clean = gradientCovariance(dx, dy, w);
                const GradientCovariance dirty = gradientCovariance(paddedX, paddedY, w);
                ++g_invariancePositions;
                if (!same(clean, dirty)) {
                    ++bad;
                    if (firstBad.empty()) {
                        firstBad = rectText(w) + " clean " + covText(clean) + " padded " +
                                   covText(dirty);
                    }
                }
            }
        }
    }
    COV_EXPECT(bad == 0, "a bit at or past width is never counted, at every window position",
               label + " mismatches=" + std::to_string(bad) + " first: " + firstBad);
}

/// @brief Case 5: a view that WINDOWS A WIDER FRAME (the design rule’s second half).
/// @note The pixels past the view's width are not padding, they are a neighbour's
/// live pixels -- and they are set, since the wider frame is dense. One
/// sentence of earlier work covers both, and this is the half no container can
/// express: the views are built by hand over the wide frame's planes with a
/// narrower width and the wide frame's stride.
template <typename WordType>
void testWindowOntoWiderFrame(const char* wordTypeName) {
    const std::string label = std::string(wordTypeName) + " [view onto a wider frame]";
    const int wideWidth = SWEEP_WIDTH + 29;  // 29 is not a multiple of any word width
    TernaryMat<WordType> dx(wideWidth, SWEEP_HEIGHT);
    TernaryMat<WordType> dy(wideWidth, SWEEP_HEIGHT);
    // zeroBias 0: every pixel of the wide frame is +/-1, so anything counted past
    // the narrow width would be counted, loudly.
    fillRandomTernary(dx, UINT64_C(0x5EED0C0FFEE00041), 0);
    fillRandomTernary(dy, UINT64_C(0x5EED0C0FFEE00042), 0);

    const auto narrow = [](BinMatConstView<WordType> v) {
        BinMatConstView<WordType> out = v;
        out.width = static_cast<size_t>(SWEEP_WIDTH);
        return out;
    };
    g_oraclePositions += sweepAgainstOracle<WordType>(
        narrow(dx.constMagnitude(0)), narrow(dy.constMagnitude(0)), narrow(dx.constSign()),
        narrow(dy.constSign()), label);
}

/// @brief Case 6: windows that are outside, empty, or degenerate.
/// @note The sweeps above already cover wholly-outside windows at a window's
/// distance; these are the far ones and the ill-formed ones, where the
/// arithmetic that clips could overflow rather than merely clip.
template <typename WordType>
void testDegenerate(const char* wordTypeName) {
    const std::string label = std::string(wordTypeName) + " [degenerate]";
    TernaryMat<WordType> dx(SWEEP_WIDTH, SWEEP_HEIGHT);
    TernaryMat<WordType> dy(SWEEP_WIDTH, SWEEP_HEIGHT);
    fillRandomTernary(dx, UINT64_C(0x5EED0C0FFEE00051), 0);
    fillRandomTernary(dy, UINT64_C(0x5EED0C0FFEE00052), 0);

    const Rect outside[] = {
        Rect(-1000, 0, 31, 31),        // far left
        Rect(1000, 0, 31, 31),         // far right
        Rect(0, -1000, 31, 31),        // far above
        Rect(0, 1000, 31, 31),         // far below
        Rect(-31, 0, 31, 31),          // exactly one window left of column 0
        Rect(SWEEP_WIDTH, 0, 31, 31),  // exactly one pixel right of the last column
        Rect(0, SWEEP_HEIGHT, 31, 31), // exactly one row below the last
        Rect(0, 0, 0, 31),             // zero width
        Rect(0, 0, 31, 0),             // zero height
        Rect(0, 0, -5, -5),            // negative extents
        Rect(2147483600, 2147483600, 31, 31),  // origin near INT_MAX: x + width overflows int
    };
    for (const Rect& w : outside) {
        const GradientCovariance c = gradientCovariance(dx, dy, w);
        COV_EXPECT(c.sumXX == 0 && c.sumYY == 0 && c.sumXY == 0,
                   "a window with no pixels inside the image gives {0, 0, 0}",
                   label + " " + rectText(w) + " -> " + covText(c));
    }

    // An empty container, and null views: nothing is dereferenced and the answer
    // is three zeros rather than an error.
    const TernaryMat<WordType> empty;
    const GradientCovariance fromEmpty = gradientCovariance(empty, empty, Rect(0, 0, 31, 31));
    COV_EXPECT(fromEmpty.sumXX == 0 && fromEmpty.sumYY == 0 && fromEmpty.sumXY == 0,
               "an empty ternary pair gives {0, 0, 0}", label + " " + covText(fromEmpty));

    const BinMatConstView<WordType> nullView{};
    const GradientCovariance fromNull =
        gradientCovariance<WordType>(nullView, nullView, nullView, nullView, Rect(0, 0, 31, 31));
    COV_EXPECT(fromNull.sumXX == 0 && fromNull.sumYY == 0 && fromNull.sumXY == 0,
               "null views give {0, 0, 0}", label + " " + covText(fromNull));

    // A 1x1 window and a 1-pixel-wide image are the extreme of the head/tail mask
    // arithmetic: firstWord == lastWord, and both masks apply.
    {
        TernaryMat<WordType> one(1, 1);
        one.set(0, 0, -1);
        const GradientCovariance c = gradientCovariance(one, one, Rect(0, 0, 1, 1));
        COV_EXPECT(c.sumXX == 1 && c.sumYY == 1 && c.sumXY == 1,
                   "a 1x1 frame of -1 has xx = yy = xy = 1: (-1)*(-1) is +1",
                   label + " " + covText(c));
    }
    {
        TernaryMat<WordType> a(1, 1);
        TernaryMat<WordType> b(1, 1);
        a.set(0, 0, -1);
        b.set(0, 0, 1);
        const GradientCovariance c = gradientCovariance(a, b, Rect(0, 0, 1, 1));
        COV_EXPECT(c.sumXX == 1 && c.sumYY == 1 && c.sumXY == -1,
                   "opposing signs give a NEGATIVE cross term -- the one number in the "
                   "matrix that can be, and the reason the fields are signed",
                   label + " " + covText(c));
    }
}

/// @brief Case 7: WHAT A TAP-ORDER INVERSION DOES TO THIS MATRIX, and what it
/// does not.
/// @note the design notes, ops/derivative.hpp and tests/test_derivative.cpp all
/// used to say that a cv::filter2D correlate-vs-convolve mix-up would leave
/// sumXX and sumYY correct "while silently negating the cross term", so that
/// this covariance was a tripwire for it. **It is not, and the arithmetic
/// says so: a tap-order inversion negates BOTH derivatives, and
/// (-Ix)(-Iy) = IxIy.** The entire 2x2 matrix, cross term included, is
/// INVARIANT under a global negation -- so the direction of the taps is
/// guarded by Derivative.OpenCvFilter2D_Direction alone, and nothing here
/// can see it. That correction is pinned here rather than only written down,
/// because it is the kind of sentence that gets restored by someone
/// reasoning from "the cross term reads the sign planes".
/// @note The half that IS true is also pinned: negating ONE derivative negates the
/// cross term and leaves the diagonal alone. That is a real property of the
/// identity, and it is what the old sentence was probably reaching for.
template <typename WordType>
void testNegationInvariance(const char* wordTypeName) {
    const std::string label = std::string(wordTypeName) + " [negation]";
    bincv::BinMat<WordType> src(SWEEP_WIDTH, SWEEP_HEIGHT);
    fillRandomBinary(src, UINT64_C(0x5EED0C0FFEE00071));

    TernaryMat<WordType> dx(SWEEP_WIDTH, SWEEP_HEIGHT);
    TernaryMat<WordType> dy(SWEEP_WIDTH, SWEEP_HEIGHT);
    bincv::derivativeX(src, dx);
    bincv::derivativeY(src, dy);

    // Reversing a 1x3 kernel's taps computes src(x-1) - src(x+1) instead of
    // src(x+1) - src(x-1): same magnitude everywhere, sign flipped wherever the
    // magnitude is set. Both kernels are reversed together, because the mix-up
    // being described is filter2D's convention and it applies to both calls.
    TernaryMat<WordType> negX(dx);  // deep copy
    TernaryMat<WordType> negY(dy);
    size_t flipped = 0;
    for (int y = 0; y < SWEEP_HEIGHT; ++y) {
        for (int x = 0; x < SWEEP_WIDTH; ++x) {
            const int a = negX.at(y, x);
            const int b = negY.at(y, x);
            if (a != 0) { negX.set(y, x, -a); ++flipped; }
            if (b != 0) { negY.set(y, x, -b); ++flipped; }
        }
    }
    COV_EXPECT(flipped > 0, "the negation case actually negated something",
               label + " flipped=" + std::to_string(flipped));

    size_t bothBad = 0;
    size_t oneBad = 0;
    std::string firstBothBad;
    for (int windowSize : WINDOW_SIZES) {
        for (int y = -windowSize; y <= SWEEP_HEIGHT + windowSize; ++y) {
            for (int x = -windowSize; x <= SWEEP_WIDTH + windowSize; ++x) {
                const Rect w(x, y, windowSize, windowSize);
                const GradientCovariance base = gradientCovariance(dx, dy, w);
                const GradientCovariance both = gradientCovariance(negX, negY, w);
                const GradientCovariance one = gradientCovariance(negX, dy, w);
                g_invariancePositions += 2;
                if (!same(base, both)) {
                    ++bothBad;
                    if (firstBothBad.empty()) {
                        firstBothBad = rectText(w) + " base " + covText(base) + " negated " +
                                       covText(both);
                    }
                }
                if (one.sumXX != base.sumXX || one.sumYY != base.sumYY ||
                    one.sumXY != -base.sumXY) {
                    ++oneBad;
                }
            }
        }
    }
    COV_EXPECT(bothBad == 0,
               "negating BOTH derivatives -- what a filter2D tap-order inversion does -- "
               "leaves the WHOLE matrix unchanged, cross term included: this covariance is "
               "NOT a tripwire for that bug",
               label + " mismatches=" + std::to_string(bothBad) + " first: " + firstBothBad);
    COV_EXPECT(oneBad == 0,
               "negating ONE derivative negates the cross term and leaves the diagonal "
               "alone -- the half of the old claim that is true",
               label + " mismatches=" + std::to_string(oneBad));
}

/// @brief Case 8: what SlidingWindowCount can and cannot do for a column sweep.
/// @note ops/covariance.hpp points a column-sweeping caller at
/// SlidingWindowCount, and used to point it there for the whole operation.
/// SlidingWindowCount slides ONE plane's popcount, so it delivers sumXX and
/// sumYY and cannot deliver sumXY -- there is no sliding form of the
/// `magX & magY` split anywhere in ops/reduce.hpp. Both halves are pinned:
/// the two that slide must agree with this operation position for position
/// down a column, and the reason the third does not is that no such class
/// exists to compare against.
template <typename WordType>
void testColumnSweepAgreement(const char* wordTypeName) {
    const std::string label = std::string(wordTypeName) + " [column sweep]";
    TernaryMat<WordType> dx(SWEEP_WIDTH, SWEEP_HEIGHT);
    TernaryMat<WordType> dy(SWEEP_WIDTH, SWEEP_HEIGHT);
    fillRandomTernary(dx, UINT64_C(0x5EED0C0FFEE00081), 1);
    fillRandomTernary(dy, UINT64_C(0x5EED0C0FFEE00082), 1);

    size_t bad = 0;
    size_t positions = 0;
    std::string firstBad;
    for (int windowSize : WINDOW_SIZES) {
        for (int x = -windowSize; x <= SWEEP_WIDTH + windowSize; x += 3) {
            const Rect first(x, -windowSize, windowSize, windowSize);
            bincv::SlidingWindowCount<WordType> slideX(dx.constMagnitude(0), first);
            bincv::SlidingWindowCount<WordType> slideY(dy.constMagnitude(0), first);
            for (int y = -windowSize; y <= SWEEP_HEIGHT + windowSize; ++y) {
                const Rect w(x, y, windowSize, windowSize);
                const GradientCovariance c = gradientCovariance(dx, dy, w);
                ++positions;
                ++g_invariancePositions;
                if (c.sumXX != static_cast<int64_t>(slideX.count()) ||
                    c.sumYY != static_cast<int64_t>(slideY.count())) {
                    ++bad;
                    if (firstBad.empty()) {
                        firstBad = rectText(w) + " covariance " + covText(c) + " sliding xx=" +
                                   std::to_string(slideX.count()) + " yy=" +
                                   std::to_string(slideY.count());
                    }
                }
                slideX.slideDown();
                slideY.slideDown();
            }
        }
    }
    COV_EXPECT(bad == 0,
               "down a column, SlidingWindowCount reproduces sumXX and sumYY exactly -- the "
               "two of the three numbers that DO slide, which is what the docstring now says",
               label + " positions=" + std::to_string(positions) + " mismatches=" +
                   std::to_string(bad) + " first: " + firstBad);
}

/// @brief Case 9: WHICH SPELLING ACCEPTS WHAT, checked rather than asserted.
/// @note shipped with `SignedQuantMat<N, W>` for N > 1 matching NO overload,
/// and this case pinned that. ** reverses it deliberately**: a measurement found
/// the tracker's accuracy failure IS the 1-bit pyramid, so the frontend needs
/// N-bit levels and the covariance has to form at N > 1. The container
/// spelling now dispatches on the plane count -- ternary to the
/// single-popcount kernel, N-bit to the bit-sliced one -- and what this case
/// pins is that BOTH still exist and that the five-argument view form has not
/// silently grown an N it cannot check.
/// @note The one claim that has NOT changed: a `BinMatConstView` carries no plane
/// count, so the five-argument view form still cannot tell a ternary level
/// from an N-bit level's LSB plane. The N-bit VIEW form takes plane ARRAYS
/// precisely so that N is in the type there -- and the last two traits below
/// check that a loose view, and an array of the wrong length, do not compile.
template <typename WordType, typename = void>
struct ContainerCallable : std::false_type {};

template <typename WordType>
struct ContainerCallable<
    WordType,
    decltype(void(gradientCovariance(std::declval<const bincv::SignedQuantMat<3, WordType>&>(),
                                     std::declval<const bincv::SignedQuantMat<3, WordType>&>(),
                                     std::declval<Rect>())))> : std::true_type {};

template <typename WordType, typename = void>
struct TernaryCallable : std::false_type {};

template <typename WordType>
struct TernaryCallable<
    WordType, decltype(void(gradientCovariance(std::declval<const TernaryMat<WordType>&>(),
                                               std::declval<const TernaryMat<WordType>&>(),
                                               std::declval<Rect>())))> : std::true_type {};

/// @brief Is the N-bit VIEW form callable with loose views instead of plane arrays?
/// @note It must not be. This is the difference between the two view spellings:
/// the five-argument one cannot know N and says so, the array one knows N by
/// construction.
template <typename WordType, typename = void>
struct LooseViewCallableAsNBit : std::false_type {};

template <typename WordType>
struct LooseViewCallableAsNBit<
    WordType, decltype(void(gradientCovariance<3, WordType>(
                  std::declval<const BinMatConstView<WordType>&>(),
                  std::declval<const BinMatConstView<WordType>&>(),
                  std::declval<BinMatConstView<WordType>>(),
                  std::declval<BinMatConstView<WordType>>(), std::declval<Rect>())))>
    : std::true_type {};

/// @brief Does a plane array of the WRONG length reach the N-bit view form?
template <typename WordType, typename = void>
struct MismatchedArrayCallable : std::false_type {};

template <typename WordType>
struct MismatchedArrayCallable<
    WordType, decltype(void(gradientCovariance<3, WordType>(
                  std::declval<const BinMatConstView<WordType> (&)[3]>(),
                  std::declval<const BinMatConstView<WordType> (&)[2]>(),
                  std::declval<BinMatConstView<WordType>>(),
                  std::declval<BinMatConstView<WordType>>(), std::declval<Rect>())))>
    : std::true_type {};

template <typename WordType>
void testNBitDispatch(const char* wordTypeName) {
    const std::string label = std::string(wordTypeName) + " [dispatch]";
    COV_EXPECT(ContainerCallable<WordType>::value,
               "the CONTAINER spelling ACCEPTS SignedQuantMat<N> for N > 1 -- added "
               "the bit-sliced kernel this made a precondition, and the compile-time "
               "refusal is deliberately gone",
               label);
    COV_EXPECT(TernaryCallable<WordType>::value,
               "...and it still accepts TernaryMat, so the ternary kernel has not been "
               "absorbed into the generic one",
               label);
    COV_EXPECT(!LooseViewCallableAsNBit<WordType>::value,
               "the N-bit VIEW spelling does not accept loose views: N is in the plane-array "
               "type, which is the promise the five-argument view form cannot make",
               label);
    COV_EXPECT(!MismatchedArrayCallable<WordType>::value,
               "...and it does not accept two plane arrays of different lengths, so the x "
               "and y levels cannot silently differ in bit depth",
               label);
}

/// @brief Case 10: no heap, no scratch (ops/covariance.hpp promise 3).
/// @note This is the check that keeps the the axis-3 trade honest. The
/// four-argument selector form was taken over the 11-14% faster plane form
/// for one reason: it needs no plane. An implementation that allocated a
/// window buffer, or built a selector plane internally, would have given up
/// the speed and kept the memory -- and every value test in this file would
/// still pass.
template <typename WordType>
void testNoScratch(const char* wordTypeName) {
    const std::string label = std::string(wordTypeName) + " [no scratch]";
    TernaryMat<WordType> dx(64, 33);
    TernaryMat<WordType> dy(64, 33);
    fillRandomTernary(dx, UINT64_C(0x5EED0C0FFEE00061), 1);
    fillRandomTernary(dy, UINT64_C(0x5EED0C0FFEE00062), 1);

    // Arm the counter AFTER the containers exist -- their storage is a legitimate
    // allocation, and it is the caller's, not the kernel's.
    volatile int64_t sink = 0;
    const size_t before = g_newCount;
    for (int windowSize : WINDOW_SIZES) {
        for (int y = -windowSize; y <= 33; y += 3) {
            for (int x = -windowSize; x <= 64; x += 3) {
                const Rect w(x, y, windowSize, windowSize);
                const GradientCovariance viaContainer = gradientCovariance(dx, dy, w);
                const GradientCovariance viaViews = gradientCovariance<WordType>(
                    dx.constMagnitude(0), dy.constMagnitude(0), dx.constSign(), dy.constSign(), w);
                sink += viaContainer.sumXX + viaContainer.sumYY + viaContainer.sumXY +
                        viaViews.sumXX + viaViews.sumYY + viaViews.sumXY;
            }
        }
    }
    const size_t allocations = g_newCount - before;
    COV_EXPECT(allocations == 0,
               "gradientCovariance allocates nothing, in either spelling -- the no-scratch "
               "property the four-argument selector form was chosen FOR",
               label + " operator new called " + std::to_string(allocations) + " time(s)");
    (void)sink;

    // THE COUNTER'S OWN TEETH. A zero from an instrument that cannot register a
    // reading is not evidence, and this counter had exactly that hole: replacing
    // only the plain operator new left C++17's OVER-ALIGNED path -- the path a
    // vectorized or cache-line-aligned scratch buffer takes -- uncounted, so the
    // check above would have read 0 for a kernel that allocated one. Both arms are
    // exercised here, immediately after the measurement they qualify.
    {
        const size_t plainBefore = g_newCount;
        char* plain = new char[16];
        const size_t plainSeen = g_newCount - plainBefore;
        delete[] plain;

        struct alignas(64) OverAligned {
            char bytes[64];
        };
        const size_t alignedBefore = g_newCount;
        OverAligned* over = new OverAligned;
        const size_t alignedSeen = g_newCount - alignedBefore;
        delete over;

        COV_EXPECT(plainSeen == 1 && alignedSeen == 1,
                   "the allocation counter registers BOTH a plain and an over-aligned new, so "
                   "the zero above is a measurement rather than a blind spot",
                   label + " plain new counted " + std::to_string(plainSeen) +
                       ", over-aligned new counted " + std::to_string(alignedSeen));
    }
}

// ---------------------------------------------------------------------------
// THE N-BIT ORACLE, AND WHY IT IS A SECOND ORACLE RATHER THAN THE FIRST
// ONE WIDENED
// ---------------------------------------------------------------------------
//
// The float oracle above reads a TERNARY pixel. An N-bit level's pixel is
// `+/- SUM_i 2^i * m[i]`, which is a different reading of a different number of
// planes, so widening the existing one would mean the ternary sweeps and the
// N-bit sweeps shared a reconstruction -- and a fault in that reconstruction
// would cancel against itself in both. This is written from the plane bits
// upward, in INTEGERS, with multiplies:
//
// Ix = magnitude 0 ? 0 : (sign ? -SUM 2^i m[i] : +SUM 2^i m[i])
// xx += Ix*Ix; yy += Iy*Iy; xy += Ix*Iy
//
// It shares no code, no clipping ladder and no word arithmetic with
// ops/covariance.hpp -- it does not know that a plane pair or a popcount exists.
// That is the whole point: the library's claim is that N^2 weighted masked
// popcounts EQUAL this loop, so the oracle has to be this loop.
//
// **The float column is kept, and it is a real check at N-bit widths too.** The
// largest value any of the three entries can take is (2^N - 1)^2 * P, which at
// N = 4 and a 31x31 window is 225 * 961 = 216 225 -- comfortably inside the
// 2^24 = 16 777 216 that float represents exactly, and every partial sum along
// the way is an integer below the same bound. So float equality is exact here
// for the same reason it is exact at N = 1, and a static_assert below pins the
// arithmetic rather than leaving it in this comment.

/// @brief The largest bit depth swept. Every N-bit sweep runs at 1..MAX_BIT_DEPTH,
/// DRIVEN BY THIS CONSTANT rather than hand-unrolled, so the depths swept
/// and the exactness guard below cannot drift apart.
/// @note **7, and the justification is that measurement’s, not that measurement’s.** An earlier version of
/// this comment stopped at 4 and cited that measurement’s "1/3/4/5 bits". superseded
/// that premise (the design notes: "'1/3/4/5' was the sample, not the
/// requirement"): the reachable alphabet an uncapped 2x2 mean produces is
/// 1/3/5/7 bits, because a four-input sum of N-bit values needs N + 2. So
/// N = 5 and N = 7 are the depths this will actually run, and stopping at 4
/// left them uninstantiated. 7 is also SignedQuantMat's own limit, so this
/// sweeps the whole range a derivative can be built at.
constexpr size_t MAX_BIT_DEPTH = 7;

/// The value bound of an N-bit signed level, and the covariance bound it implies.
constexpr long long maxMagnitudeOf(size_t bits) { return (1LL << bits) - 1LL; }
constexpr long long maxCovarianceEntry(size_t bits, long long windowPixels) {
    return maxMagnitudeOf(bits) * maxMagnitudeOf(bits) * windowPixels;
}
static_assert(maxCovarianceEntry(MAX_BIT_DEPTH, MAX_WINDOW_SIZE * MAX_WINDOW_SIZE) < (1LL << 24),
              "the float half of the N-bit oracle comparison is only exact while every entry "
              "fits in float's 24-bit integer range -- widen the check, not the tolerance");
static_assert(MAX_BIT_DEPTH <= 7,
              "SignedQuantMat<N> is 1..7, so a sweep above 7 would not compile a container -- "
              "and at N = 8 the float half above stops being exact (255^2 * 961 > 2^24), which "
              "would silently become a tolerance rather than a check");

/// @brief A frame of N-bit signed values as plain ints, indexed [y * width + x].
/// @note Reconstructed from the PLANE VIEWS with this file's own bit reader, not
/// through SignedQuantMat::at, so the same oracle serves a container, a
/// dirtied container and a hand-built view onto a wider frame -- and so that
/// the container's own plane reassembly is not the thing judging the kernel.
/// @note The canonical-zero rule is applied here INDEPENDENTLY: magnitude zero
/// reads as 0 whatever the sign bit says. That is what makes the dirty-sign
/// invariance below a real comparison rather than a tautology.
struct IntFrame {
    int width = 0;
    int height = 0;
    std::vector<long long> dx;
    std::vector<long long> dy;
};

template <size_t N, typename WordType>
IntFrame toIntFrame(const BinMatConstView<WordType> (&magX)[N],
                    const BinMatConstView<WordType> (&magY)[N],
                    const BinMatConstView<WordType>& signX,
                    const BinMatConstView<WordType>& signY) {
    IntFrame f;
    f.width = static_cast<int>(magX[0].width);
    f.height = static_cast<int>(magX[0].height);
    const size_t pixels = static_cast<size_t>(f.width) * static_cast<size_t>(f.height);
    f.dx.assign(pixels, 0);
    f.dy.assign(pixels, 0);
    for (int y = 0; y < f.height; ++y) {
        for (int x = 0; x < f.width; ++x) {
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(f.width) +
                             static_cast<size_t>(x);
            long long mx = 0;
            long long my = 0;
            for (size_t p = 0; p < N; ++p) {
                if (bitAt(magX[p], y, x)) mx += (1LL << p);
                if (bitAt(magY[p], y, x)) my += (1LL << p);
            }
            if (mx != 0) f.dx[i] = bitAt(signX, y, x) ? -mx : mx;
            if (my != 0) f.dy[i] = bitAt(signY, y, x) ? -my : my;
        }
    }
    return f;
}

/// @brief The 2x2 covariance over a window, one pixel at a time, with multiplies.
/// @note Accumulated in `long long` AND in `float`, in the same loop. The integer
/// column is the contract; the float column additionally pins that the
/// accumulation is integral, which is what makes "no tolerance" a legitimate
/// demand at N-bit widths rather than merely a strict one.
/// @note The clip is min/max against the extents, written independently of
/// impl::clipRegion's early-exit ladder, so the region contract has two
/// implementations to disagree at every one of the swept positions.
struct IntCovariance {
    long long xx = 0;
    long long yy = 0;
    long long xy = 0;
    float fxx = 0.0f;
    float fyy = 0.0f;
    float fxy = 0.0f;
};

IntCovariance refCovarianceN(const IntFrame& f, const Rect& w) {
    IntCovariance out;
    if (w.width <= 0 || w.height <= 0) return out;
    const long long x0 = std::max<long long>(static_cast<long long>(w.x), 0);
    const long long y0 = std::max<long long>(static_cast<long long>(w.y), 0);
    const long long x1 = std::min<long long>(
        static_cast<long long>(w.x) + static_cast<long long>(w.width), f.width);
    const long long y1 = std::min<long long>(
        static_cast<long long>(w.y) + static_cast<long long>(w.height), f.height);
    for (long long y = y0; y < y1; ++y) {
        for (long long x = x0; x < x1; ++x) {
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(f.width) +
                             static_cast<size_t>(x);
            const long long a = f.dx[i];
            const long long b = f.dy[i];
            out.xx += a * a;
            out.yy += b * b;
            out.xy += a * b;
            const float fa = static_cast<float>(a);
            const float fb = static_cast<float>(b);
            out.fxx += fa * fa;
            out.fyy += fb * fb;
            out.fxy += fa * fb;
        }
    }
    return out;
}

std::string intCovText(const IntCovariance& c) {
    return "{xx=" + std::to_string(c.xx) + " yy=" + std::to_string(c.yy) + " xy=" +
           std::to_string(c.xy) + "}";
}

/// @brief Exact agreement with the N-bit oracle -- integer AND float, no tolerance.
bool agreesN(const GradientCovariance& k, const IntCovariance& r) {
    return k.sumXX == r.xx && k.sumYY == r.yy && k.sumXY == r.xy &&
           static_cast<float>(k.sumXX) == r.fxx && static_cast<float>(k.sumYY) == r.fyy &&
           static_cast<float>(k.sumXY) == r.fxy;
}

// Window positions compared against the N-bit oracle, and positions compared
// between the two kernels at N == 1. Separate counters, because they are evidence
// about different things: the first about the bit-sliced identity, the second
// about ternary being the N = 1 instance of it.
size_t g_bitSlicedPositions = 0;
size_t g_identityPositions = 0;

/// @brief Names a container's magnitude planes into the array the kernel takes.
template <size_t N, typename WordType>
void magnitudePlanesOf(const bincv::SignedQuantMat<N, WordType>& m,
                       BinMatConstView<WordType> (&out)[N]) {
    for (size_t p = 0; p < N; ++p) out[p] = m.constMagnitude(p);
}

/// @brief Fills an N-bit signed matrix through set, so padding stays clear and
/// zeros stay canonical.
/// @param zeroBias Out of 4 draws, how many are forced to 0 -- a real binarized
/// derivative is sparse, and a dense one never exercises the zero lanes.
template <size_t N, typename WordType>
void fillRandomSigned(bincv::SignedQuantMat<N, WordType>& m, uint64_t seed, int zeroBias) {
    const int maxMagnitude = static_cast<int>(bincv::SignedQuantMat<N, WordType>::MaxMagnitude);
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            const uint64_t r = nextRandom(state);
            int value = 0;
            if (static_cast<int>(r & 3u) >= zeroBias) {
                const int magnitude =
                    1 + static_cast<int>((r >> 8) % static_cast<uint64_t>(maxMagnitude));
                value = ((r >> 32) & 1u) ? magnitude : -magnitude;
            }
            m.set(y, x, value);
        }
    }
}

/// @brief Sweeps every window position from a full window outside the frame to a
/// full window past it, against the N-bit integer oracle.
/// @return Positions compared.
/// @note One check per (window size, row of positions), for sweepAgainstOracle's
/// reason: a per-position check would put ~24000 entries in the CHECKS column
/// for one property, and a per-case check would make the column blind to a
/// sweep that lost its margin.
template <size_t N, typename WordType>
size_t sweepAgainstIntOracle(const BinMatConstView<WordType> (&magX)[N],
                             const BinMatConstView<WordType> (&magY)[N],
                             const BinMatConstView<WordType>& signX,
                             const BinMatConstView<WordType>& signY, const std::string& label) {
    // Keyed to THIS sweep's N, not to MAX_BIT_DEPTH: the float half of agreesN is
    // exact only while every entry fits float's 24-bit integer range, and a sweep
    // instantiated at a depth the constant above does not describe would otherwise
    // turn that half into a silent tolerance.
    static_assert(maxCovarianceEntry(N, MAX_WINDOW_SIZE * MAX_WINDOW_SIZE) < (1LL << 24),
                  "this sweep's N puts the covariance outside float's exact integer range");

    const IntFrame frame = toIntFrame<N, WordType>(magX, magY, signX, signY);
    const int width = frame.width;
    const int height = frame.height;
    size_t positions = 0;

    for (int windowSize : WINDOW_SIZES) {
        size_t interior = 0;
        for (int y = -windowSize; y <= height + windowSize; ++y) {
            size_t bad = 0;
            std::string firstBad;
            for (int x = -windowSize; x <= width + windowSize; ++x) {
                const Rect w(x, y, windowSize, windowSize);
                const GradientCovariance got =
                    gradientCovariance<N, WordType>(magX, magY, signX, signY, w);
                const IntCovariance want = refCovarianceN(frame, w);
                ++positions;
                if (x >= 0 && y >= 0 && x + windowSize <= width && y + windowSize <= height) {
                    ++interior;
                }
                if (!agreesN(got, want)) {
                    ++bad;
                    if (firstBad.empty()) {
                        firstBad = rectText(w) + " got " + covText(got) + " oracle " +
                                   intCovText(want);
                    }
                }
            }
            COV_EXPECT(bad == 0,
                       "the bit-sliced weighted popcounts equal the per-pixel integer "
                       "covariance at every window position of the row",
                       label + " W=" + std::to_string(windowSize) + " y=" + std::to_string(y) +
                           " mismatches=" + std::to_string(bad) + " first: " + firstBad);
        }
        const size_t expectedInterior = static_cast<size_t>(width - windowSize + 1) *
                                        static_cast<size_t>(height - windowSize + 1);
        COV_EXPECT(interior == expectedInterior && interior > 0,
                   "this window size has FULLY INTERIOR positions, so the bit-sliced identity "
                   "is checked unclipped and not only at edges",
                   label + " W=" + std::to_string(windowSize) + " interior=" +
                       std::to_string(interior) + ", expected " +
                       std::to_string(expectedInterior));
    }
    return positions;
}

// ---------------------------------------------------------------------------
// the case A: a generated N-bit pair, swept whole, at N = 1..MAX_BIT_DEPTH
// ---------------------------------------------------------------------------

template <size_t N, typename WordType>
void testBitSlicedGeneratedFrame(const char* wordTypeName) {
    const std::string label =
        std::string(wordTypeName) + " [generated N=" + std::to_string(N) + "]";
    bincv::SignedQuantMat<N, WordType> dx(SWEEP_WIDTH, SWEEP_HEIGHT);
    bincv::SignedQuantMat<N, WordType> dy(SWEEP_WIDTH, SWEEP_HEIGHT);
    fillRandomSigned(dx, UINT64_C(0x5EED0C0FFEE00101) + N, 1);
    fillRandomSigned(dy, UINT64_C(0x5EED0C0FFEE00102) + N, 1);

    BinMatConstView<WordType> magX[N];
    BinMatConstView<WordType> magY[N];
    magnitudePlanesOf(dx, magX);
    magnitudePlanesOf(dy, magY);

    const size_t positions = sweepAgainstIntOracle<N, WordType>(magX, magY, dx.constSign(),
                                                                dy.constSign(), label);
    g_bitSlicedPositions += positions;
    COV_EXPECT(positions == expectedSweepPositions(SWEEP_WIDTH, SWEEP_HEIGHT),
               "the sweep visited every position its geometry defines",
               label + " compared " + std::to_string(positions) + ", expected " +
                   std::to_string(expectedSweepPositions(SWEEP_WIDTH, SWEEP_HEIGHT)));

    // The container spelling must be the same three numbers as the view spelling it
    // forwards to -- at every position of one window size, not at a sample.
    {
        size_t bad = 0;
        for (int y = -15; y <= SWEEP_HEIGHT + 15; ++y) {
            for (int x = -15; x <= SWEEP_WIDTH + 15; ++x) {
                const Rect w(x, y, 15, 15);
                const GradientCovariance viaContainer = gradientCovariance(dx, dy, w);
                const GradientCovariance viaViews =
                    gradientCovariance<N, WordType>(magX, magY, dx.constSign(), dy.constSign(), w);
                ++g_invariancePositions;
                if (!same(viaContainer, viaViews)) ++bad;
            }
        }
        COV_EXPECT(bad == 0,
                   "the N-bit container spelling equals the N-bit view spelling everywhere",
                   label + " mismatches=" + std::to_string(bad));
    }

    // Aliasing is unrestricted, and dx against itself has a closed form at every N:
    // a plane never disagrees in sign with itself, so the cross term is SumIx^2.
    {
        size_t bad = 0;
        for (int y = -7; y <= SWEEP_HEIGHT + 7; ++y) {
            for (int x = -7; x <= SWEEP_WIDTH + 7; ++x) {
                const Rect w(x, y, 7, 7);
                const GradientCovariance self = gradientCovariance(dx, dx, w);
                ++g_invariancePositions;
                if (self.sumXX != self.sumYY || self.sumXY != self.sumXX) ++bad;
            }
        }
        COV_EXPECT(bad == 0,
                   "gradientCovariance(dx, dx, w) gives xx == yy == xy at N bits too -- the "
                   "N^2 cross-term pairs must reduce to the N^2 diagonal pairs",
                   label + " mismatches=" + std::to_string(bad));
    }
}

// ---------------------------------------------------------------------------
// the case B: the planes the REAL N-bit pipeline produces -- that work’s derivative
// over a QuantMat<N> level
// ---------------------------------------------------------------------------

template <size_t N, typename WordType>
void testBitSlicedDerivativeFrame(const char* wordTypeName) {
    const std::string label =
        std::string(wordTypeName) + " [N=" + std::to_string(N) + " from derivative]";
    bincv::QuantMat<N, WordType> src(SWEEP_WIDTH, SWEEP_HEIGHT);
    uint64_t state = UINT64_C(0x5EED0C0FFEE00111) + N;
    for (int y = 0; y < SWEEP_HEIGHT; ++y) {
        for (int x = 0; x < SWEEP_WIDTH; ++x) {
            src.set(y, x, static_cast<unsigned>(nextRandom(state) &
                                                bincv::QuantMat<N, WordType>::MaxValue));
        }
    }

    bincv::SignedQuantMat<N, WordType> dx(SWEEP_WIDTH, SWEEP_HEIGHT);
    bincv::SignedQuantMat<N, WordType> dy(SWEEP_WIDTH, SWEEP_HEIGHT);
    bincv::derivativeX(src, dx);
    bincv::derivativeY(src, dy);

    BinMatConstView<WordType> magX[N];
    BinMatConstView<WordType> magY[N];
    magnitudePlanesOf(dx, magX);
    magnitudePlanesOf(dy, magY);

    g_bitSlicedPositions +=
        sweepAgainstIntOracle<N, WordType>(magX, magY, dx.constSign(), dy.constSign(), label);

    // The cross term must go both ways somewhere, or the sweep proved the identity
    // only on the half of its range that never exercises the signed subtraction.
    bool sawNegative = false;
    bool sawPositive = false;
    for (int y = 0; y + 7 <= SWEEP_HEIGHT; ++y) {
        for (int x = 0; x + 7 <= SWEEP_WIDTH; ++x) {
            const GradientCovariance c = gradientCovariance(dx, dy, Rect(x, y, 7, 7));
            if (c.sumXY < 0) sawNegative = true;
            if (c.sumXY > 0) sawPositive = true;
        }
    }
    COV_EXPECT(sawNegative && sawPositive,
               "the N-bit derivative frame produces cross terms of BOTH signs, so the "
               "per-pair signed subtraction is exercised in both directions",
               label + " negative=" + std::to_string(sawNegative ? 1 : 0) +
                   " positive=" + std::to_string(sawPositive ? 1 : 0));
}

// ---------------------------------------------------------------------------
// the case C: THE N = 1 IDENTITY. The single most important check in the file.
// ---------------------------------------------------------------------------
//
// Ternary is the N = 1 instance of the bit-sliced form, and this requires it as an
// equality between two entry points on the SAME data rather than as an algebraic
// argument. If they can differ at any window position, one of them is wrong.
//
// THE TWO CALLS ARE SPELLED SO THAT NEITHER CAN BE THE OTHER. The left-hand side
// is `gradientCovariance<WordType>(magX, magY, signX, signY, w)` -- five loose
// views, explicit single template argument, which only the ternary overload
// can match. The right-hand side is `gradientCovariance<1, WordType>(...)` on
// ARRAYS of one view, which only the overload can match. A future edit that
// deleted the ternary overload would make the left-hand call fail to compile
// rather than quietly turn this into the bit-sliced kernel compared against
// itself -- the "compares a route against itself" hazard named.

template <typename WordType>
void testTernaryIsTheNEqualsOneInstance(const char* wordTypeName) {
    const std::string label = std::string(wordTypeName) + " [N=1 identity]";

    // Two frames: a generated ternary pair, and the pair that work’s derivative writes.
    TernaryMat<WordType> genX(SWEEP_WIDTH, SWEEP_HEIGHT);
    TernaryMat<WordType> genY(SWEEP_WIDTH, SWEEP_HEIGHT);
    fillRandomTernary(genX, UINT64_C(0x5EED0C0FFEE00121), 1);
    fillRandomTernary(genY, UINT64_C(0x5EED0C0FFEE00122), 1);

    bincv::BinMat<WordType> src(SWEEP_WIDTH, SWEEP_HEIGHT);
    fillRandomBinary(src, UINT64_C(0x5EED0C0FFEE00123));
    TernaryMat<WordType> derX(SWEEP_WIDTH, SWEEP_HEIGHT);
    TernaryMat<WordType> derY(SWEEP_WIDTH, SWEEP_HEIGHT);
    bincv::derivativeX(src, derX);
    bincv::derivativeY(src, derY);

    const TernaryMat<WordType>* frames[2][2] = {{&genX, &genY}, {&derX, &derY}};
    const char* frameNames[2] = {"generated", "from derivative"};

    for (int frame = 0; frame < 2; ++frame) {
        const TernaryMat<WordType>& dx = *frames[frame][0];
        const TernaryMat<WordType>& dy = *frames[frame][1];

        BinMatConstView<WordType> magX[1] = {dx.constMagnitude(0)};
        BinMatConstView<WordType> magY[1] = {dy.constMagnitude(0)};

        size_t bad = 0;
        size_t positions = 0;
        size_t nonZero = 0;
        std::string firstBad;
        for (int windowSize : WINDOW_SIZES) {
            for (int y = -windowSize; y <= SWEEP_HEIGHT + windowSize; ++y) {
                for (int x = -windowSize; x <= SWEEP_WIDTH + windowSize; ++x) {
                    const Rect w(x, y, windowSize, windowSize);
                    const GradientCovariance ternary = gradientCovariance<WordType>(
                        dx.constMagnitude(0), dy.constMagnitude(0), dx.constSign(),
                        dy.constSign(), w);
                    const GradientCovariance bitSliced = gradientCovariance<1, WordType>(
                        magX, magY, dx.constSign(), dy.constSign(), w);
                    ++positions;
                    ++g_identityPositions;
                    if (ternary.sumXX != 0 || ternary.sumYY != 0 || ternary.sumXY != 0) {
                        ++nonZero;
                    }
                    if (!same(ternary, bitSliced)) {
                        ++bad;
                        if (firstBad.empty()) {
                            firstBad = rectText(w) + " ternary " + covText(ternary) +
                                       " bit-sliced " + covText(bitSliced);
                        }
                    }
                }
            }
        }
        COV_EXPECT(bad == 0,
                   "the bit-sliced kernel at N = 1 is BIT-IDENTICAL to the ternary "
                   "kernel at every window position -- ternary IS the N = 1 instance",
                   label + " [" + frameNames[frame] + "] positions=" + std::to_string(positions) +
                       " mismatches=" + std::to_string(bad) + " first: " + firstBad);
        // An identity between two functions that both return {0, 0, 0} everywhere
        // would also pass the check above. It does not, and this says so.
        COV_EXPECT(nonZero > positions / 4,
                   "...and the two agreed on NON-ZERO matrices at most positions, so the "
                   "identity is not two zeros agreeing",
                   label + " [" + frameNames[frame] + "] non-zero=" + std::to_string(nonZero) +
                       " of " + std::to_string(positions));

        // The container spellings must agree too, and they are DIFFERENT overloads:
        // TernaryMat is SignedQuantMat<1, W>, so the ternary overload wins partial
        // ordering and the N-bit container overload is never selected at N = 1.
        // Whichever is picked, the answer is the same -- that is what makes the
        // dispatch a detail rather than a contract.
        size_t containerBad = 0;
        for (int y = -15; y <= SWEEP_HEIGHT + 15; ++y) {
            for (int x = -15; x <= SWEEP_WIDTH + 15; ++x) {
                const Rect w(x, y, 15, 15);
                const GradientCovariance viaContainer = gradientCovariance(dx, dy, w);
                const GradientCovariance bitSliced = gradientCovariance<1, WordType>(
                    magX, magY, dx.constSign(), dy.constSign(), w);
                ++g_identityPositions;
                if (!same(viaContainer, bitSliced)) ++containerBad;
            }
        }
        COV_EXPECT(containerBad == 0,
                   "the TernaryMat container spelling equals the bit-sliced kernel at N = 1",
                   label + " [" + frameNames[frame] +
                       "] mismatches=" + std::to_string(containerBad));
    }
}

// ---------------------------------------------------------------------------
// the case D: the contracts, at N bits -- dirty signs, dirty padding, a view
// onto a wider frame, and the degenerate windows
// ---------------------------------------------------------------------------

template <size_t N, typename WordType>
void testBitSlicedContracts(const char* wordTypeName) {
    const std::string label =
        std::string(wordTypeName) + " [N=" + std::to_string(N) + " contracts]";

    bincv::SignedQuantMat<N, WordType> dx(SWEEP_WIDTH, SWEEP_HEIGHT);
    bincv::SignedQuantMat<N, WordType> dy(SWEEP_WIDTH, SWEEP_HEIGHT);
    fillRandomSigned(dx, UINT64_C(0x5EED0C0FFEE00131), 2);
    fillRandomSigned(dy, UINT64_C(0x5EED0C0FFEE00132), 2);

    // (i) A sign bit over a zero magnitude carries no information (promise 5), and
    // at N bits "zero magnitude" means ALL N planes clear.
    bincv::SignedQuantMat<N, WordType> dirtyX(dx);
    bincv::SignedQuantMat<N, WordType> dirtyY(dy);
    size_t dirtied = 0;
    for (int y = 0; y < SWEEP_HEIGHT; ++y) {
        for (int x = 0; x < SWEEP_WIDTH; ++x) {
            if (dirtyX.magnitudeAt(y, x) == 0) {
                setBit(dirtyX.sign(), static_cast<size_t>(y), static_cast<size_t>(x));
                ++dirtied;
            }
            if (dirtyY.magnitudeAt(y, x) == 0) {
                setBit(dirtyY.sign(), static_cast<size_t>(y), static_cast<size_t>(x));
                ++dirtied;
            }
        }
    }
    COV_EXPECT(dirtied > 0, "the N-bit dirty-sign case actually dirtied something",
               label + " dirtied=" + std::to_string(dirtied));

    // (ii) Every padding bit set, in all N + 1 planes of both derivatives.
    bincv::SignedQuantMat<N, WordType> paddedX(dx);
    bincv::SignedQuantMat<N, WordType> paddedY(dy);
    const size_t paddingEnd =
        paddedX.getAlignedWidth() * bincv::SignedQuantMat<N, WordType>::WordBits;
    size_t paddingDirtied = 0;
    for (size_t plane = 0; plane < bincv::SignedQuantMat<N, WordType>::Planes; ++plane) {
        bincv::BinMatView<WordType> px = paddedX.planes().plane(plane);
        bincv::BinMatView<WordType> py = paddedY.planes().plane(plane);
        for (size_t y = 0; y < static_cast<size_t>(SWEEP_HEIGHT); ++y) {
            for (size_t x = static_cast<size_t>(SWEEP_WIDTH); x < paddingEnd; ++x) {
                setBit(px, y, x);
                setBit(py, y, x);
                paddingDirtied += 2;
            }
        }
    }
    COV_EXPECT(paddingDirtied > 0, "the N-bit dirty-padding case actually dirtied something",
               label + " padding bits per row=" + std::to_string(paddingEnd - SWEEP_WIDTH) +
                   " dirtied=" + std::to_string(paddingDirtied));

    size_t signBad = 0;
    size_t padBad = 0;
    std::string firstBad;
    for (int windowSize : WINDOW_SIZES) {
        for (int y = -windowSize; y <= SWEEP_HEIGHT + windowSize; ++y) {
            for (int x = -windowSize; x <= SWEEP_WIDTH + windowSize; ++x) {
                const Rect w(x, y, windowSize, windowSize);
                const GradientCovariance clean = gradientCovariance(dx, dy, w);
                const GradientCovariance dirtySign = gradientCovariance(dirtyX, dirtyY, w);
                const GradientCovariance dirtyPad = gradientCovariance(paddedX, paddedY, w);
                g_invariancePositions += 2;
                if (!same(clean, dirtySign)) ++signBad;
                if (!same(clean, dirtyPad)) {
                    ++padBad;
                    if (firstBad.empty()) {
                        firstBad = rectText(w) + " clean " + covText(clean) + " padded " +
                                   covText(dirtyPad);
                    }
                }
            }
        }
    }
    COV_EXPECT(signBad == 0,
               "at N bits a sign bit over an all-planes-clear magnitude changes nothing, at "
               "every window position",
               label + " mismatches=" + std::to_string(signBad));
    COV_EXPECT(padBad == 0,
               "at N bits a bit at or past width is never counted, in any of the N + 1 "
               "planes, at every window position",
               label + " mismatches=" + std::to_string(padBad) + " first: " + firstBad);

    // (iii) The dirtied frames against the ORACLE, which applies the canonical-zero
    // rule and the width bound its own way rather than by comparison.
    {
        BinMatConstView<WordType> magX[N];
        BinMatConstView<WordType> magY[N];
        magnitudePlanesOf(dirtyX, magX);
        magnitudePlanesOf(dirtyY, magY);
        g_bitSlicedPositions += sweepAgainstIntOracle<N, WordType>(
            magX, magY, dirtyX.constSign(), dirtyY.constSign(), label + " [dirty signs]");
    }

    // (iv) A view that WINDOWS A WIDER FRAME: the bits past the view's width are a
    // neighbour's live pixels, not padding, and every one of them is set.
    {
        const int wideWidth = SWEEP_WIDTH + 29;  // not a multiple of any word width
        bincv::SignedQuantMat<N, WordType> wideX(wideWidth, SWEEP_HEIGHT);
        bincv::SignedQuantMat<N, WordType> wideY(wideWidth, SWEEP_HEIGHT);
        fillRandomSigned(wideX, UINT64_C(0x5EED0C0FFEE00141), 0);
        fillRandomSigned(wideY, UINT64_C(0x5EED0C0FFEE00142), 0);

        const auto narrow = [](BinMatConstView<WordType> v) {
            BinMatConstView<WordType> out = v;
            out.width = static_cast<size_t>(SWEEP_WIDTH);
            return out;
        };
        BinMatConstView<WordType> magX[N];
        BinMatConstView<WordType> magY[N];
        for (size_t p = 0; p < N; ++p) {
            magX[p] = narrow(wideX.constMagnitude(p));
            magY[p] = narrow(wideY.constMagnitude(p));
        }
        g_bitSlicedPositions += sweepAgainstIntOracle<N, WordType>(
            magX, magY, narrow(wideX.constSign()), narrow(wideY.constSign()),
            label + " [view onto a wider frame]");
    }

    // (v) Windows outside, empty, degenerate, and the overflow-bait origin.
    const Rect outside[] = {
        Rect(-1000, 0, 31, 31),
        Rect(1000, 0, 31, 31),
        Rect(0, -1000, 31, 31),
        Rect(0, 1000, 31, 31),
        Rect(-31, 0, 31, 31),
        Rect(SWEEP_WIDTH, 0, 31, 31),
        Rect(0, SWEEP_HEIGHT, 31, 31),
        Rect(0, 0, 0, 31),
        Rect(0, 0, 31, 0),
        Rect(0, 0, -5, -5),
        Rect(2147483600, 2147483600, 31, 31),
    };
    size_t outsideBad = 0;
    for (const Rect& w : outside) {
        const GradientCovariance c = gradientCovariance(dx, dy, w);
        if (c.sumXX != 0 || c.sumYY != 0 || c.sumXY != 0) ++outsideBad;
    }
    COV_EXPECT(outsideBad == 0,
               "at N bits a window with no pixels inside the image gives {0, 0, 0} -- "
               "clipped, not rejected",
               label + " mismatches=" + std::to_string(outsideBad) + " of " +
                   std::to_string(sizeof(outside) / sizeof(outside[0])));

    const bincv::SignedQuantMat<N, WordType> empty;
    const GradientCovariance fromEmpty = gradientCovariance(empty, empty, Rect(0, 0, 31, 31));
    COV_EXPECT(fromEmpty.sumXX == 0 && fromEmpty.sumYY == 0 && fromEmpty.sumXY == 0,
               "an empty N-bit pair gives {0, 0, 0}", label + " " + covText(fromEmpty));

    BinMatConstView<WordType> nullPlanes[N];
    const BinMatConstView<WordType> nullView{};
    const GradientCovariance fromNull = gradientCovariance<N, WordType>(
        nullPlanes, nullPlanes, nullView, nullView, Rect(0, 0, 31, 31));
    COV_EXPECT(fromNull.sumXX == 0 && fromNull.sumYY == 0 && fromNull.sumXY == 0,
               "null N-bit view arrays give {0, 0, 0}", label + " " + covText(fromNull));
}

// ---------------------------------------------------------------------------
// the case E: the weights, worked by hand
// ---------------------------------------------------------------------------
//
// Everything above compares two loops. This compares the kernel against arithmetic
// a reader can do on paper, which is the check that catches a weight that is
// consistently wrong in both the kernel and a reviewer's reading of the oracle.

template <typename WordType>
void testBitSlicedWeightsByHand(const char* wordTypeName) {
    const std::string label = std::string(wordTypeName) + " [weights by hand]";

    // One pixel, 4-bit: Ix = -3, Iy = +5. xx = 9, yy = 25, xy = -15.
    {
        bincv::SignedQuantMat<4, WordType> a(1, 1);
        bincv::SignedQuantMat<4, WordType> b(1, 1);
        a.set(0, 0, -3);
        b.set(0, 0, 5);
        const GradientCovariance c = gradientCovariance(a, b, Rect(0, 0, 1, 1));
        COV_EXPECT(c.sumXX == 9 && c.sumYY == 25 && c.sumXY == -15,
                   "a 1x1 4-bit frame with Ix = -3, Iy = +5 gives {9, 25, -15}: the plane "
                   "weights and the sign split, on one pixel",
                   label + " " + covText(c));
    }
    // Same magnitudes, agreeing signs: the cross term flips to +15 and the diagonal
    // does not move -- the sign really is read only by the cross term.
    {
        bincv::SignedQuantMat<4, WordType> a(1, 1);
        bincv::SignedQuantMat<4, WordType> b(1, 1);
        a.set(0, 0, -3);
        b.set(0, 0, -5);
        const GradientCovariance c = gradientCovariance(a, b, Rect(0, 0, 1, 1));
        COV_EXPECT(c.sumXX == 9 && c.sumYY == 25 && c.sumXY == 15,
                   "the same magnitudes with AGREEING signs give {9, 25, +15}",
                   label + " " + covText(c));
    }
    // THE SATURATED WINDOW -- the overflow bound at its documented worst case.
    // Every pixel of a 31x31 window at N = 4 holds the extreme value, so each entry
    // reaches (2^4 - 1)^2 * 961 = 216225, which is what ops/covariance.hpp's
    // accumulator note says int64_t must hold and float must represent exactly.
    {
        bincv::SignedQuantMat<4, WordType> a(MAX_WINDOW_SIZE, MAX_WINDOW_SIZE);
        bincv::SignedQuantMat<4, WordType> b(MAX_WINDOW_SIZE, MAX_WINDOW_SIZE);
        for (int y = 0; y < MAX_WINDOW_SIZE; ++y) {
            for (int x = 0; x < MAX_WINDOW_SIZE; ++x) {
                a.set(y, x, 15);
                b.set(y, x, -15);
            }
        }
        const GradientCovariance c =
            gradientCovariance(a, b, Rect(0, 0, MAX_WINDOW_SIZE, MAX_WINDOW_SIZE));
        const long long saturated =
            maxCovarianceEntry(4, MAX_WINDOW_SIZE * MAX_WINDOW_SIZE);
        COV_EXPECT(c.sumXX == saturated && c.sumYY == saturated && c.sumXY == -saturated,
                   "the SATURATED 31x31 window at N = 4 gives exactly (2^4-1)^2 * 961 = "
                   "216225 on the diagonal and its negation off it -- the documented "
                   "accumulator bound, reached rather than argued",
                   label + " " + covText(c) + " expected " + std::to_string(saturated));
    }
}

// ---------------------------------------------------------------------------
// the case F: no heap at N bits
// ---------------------------------------------------------------------------
//
// The ternary case took the SLOWER of two selector forms to avoid a frame-sized
// plane (the axis 3). The N-bit kernel has N^2 plane pairs and an obvious
// temptation to materialize something per pair; it does not, and this is the
// reading that says so rather than the docstring.

template <size_t N, typename WordType>
void testBitSlicedNoScratch(const char* wordTypeName) {
    const std::string label =
        std::string(wordTypeName) + " [N=" + std::to_string(N) + " no scratch]";
    bincv::SignedQuantMat<N, WordType> dx(64, 33);
    bincv::SignedQuantMat<N, WordType> dy(64, 33);
    fillRandomSigned(dx, UINT64_C(0x5EED0C0FFEE00151), 1);
    fillRandomSigned(dy, UINT64_C(0x5EED0C0FFEE00152), 1);

    BinMatConstView<WordType> magX[N];
    BinMatConstView<WordType> magY[N];
    magnitudePlanesOf(dx, magX);
    magnitudePlanesOf(dy, magY);

    volatile int64_t sink = 0;
    const size_t before = g_newCount;
    for (int windowSize : WINDOW_SIZES) {
        for (int y = -windowSize; y <= 33; y += 3) {
            for (int x = -windowSize; x <= 64; x += 3) {
                const Rect w(x, y, windowSize, windowSize);
                const GradientCovariance viaContainer = gradientCovariance(dx, dy, w);
                const GradientCovariance viaViews =
                    gradientCovariance<N, WordType>(magX, magY, dx.constSign(), dy.constSign(), w);
                sink += viaContainer.sumXX + viaContainer.sumYY + viaContainer.sumXY +
                        viaViews.sumXX + viaViews.sumYY + viaViews.sumXY;
            }
        }
    }
    const size_t allocations = g_newCount - before;
    COV_EXPECT(allocations == 0,
               "the N-bit covariance allocates nothing, in either spelling -- the N^2 "
               "plane-pair counters are automatic storage, not a buffer",
               label + " operator new called " + std::to_string(allocations) + " time(s)");
    (void)sink;
}

/// @brief Every case, for one word type. The word type is the axis the design rule makes
/// load-bearing: every mask and shift is compiled at 8, 16, 32 and 64 bits.
template <typename WordType>
void testWordType(const char* wordTypeName) {
    std::cout << "\n--- LK gradient covariance: " << wordTypeName << " ---\n";
    const size_t before = g_oraclePositions;
    const size_t beforeInvariance = g_invariancePositions;
    checkWindowSizes();
    testGeneratedFrame<WordType>(wordTypeName);
    testDerivativeFrame<WordType>(wordTypeName);
    testDirtySigns<WordType>(wordTypeName);
    testDirtyPadding<WordType>(wordTypeName);
    testWindowOntoWiderFrame<WordType>(wordTypeName);
    testDegenerate<WordType>(wordTypeName);
    testNegationInvariance<WordType>(wordTypeName);
    testColumnSweepAgreement<WordType>(wordTypeName);
    testNBitDispatch<WordType>(wordTypeName);
    testNoScratch<WordType>(wordTypeName);
    std::cout << " " << wordTypeName << ": " << (g_oraclePositions - before)
              << " window positions compared against the per-pixel float oracle, "
              << (g_invariancePositions - beforeInvariance)
              << " more against an invariant (running totals " << g_oraclePositions << " / "
              << g_invariancePositions << ")\n";
}

/// @brief Both whole-frame sweeps at EVERY bit depth 1..MAX_BIT_DEPTH.
/// @note A fold over an index sequence rather than a hand-written list of
/// instantiations. The depths swept are then a consequence of MAX_BIT_DEPTH,
/// which is also what the float-exactness guard is keyed to -- so raising one
/// cannot leave the other checking a depth nobody runs any more.
template <typename WordType, size_t... Is>
void sweepEveryBitDepth(const char* wordTypeName, std::index_sequence<Is...>) {
    (testBitSlicedGeneratedFrame<Is + 1, WordType>(wordTypeName), ...);
    (testBitSlicedDerivativeFrame<Is + 1, WordType>(wordTypeName), ...);
}

/// @brief The operator-new count at every bit depth 1..MAX_BIT_DEPTH.
template <typename WordType, size_t... Is>
void noScratchAtEveryBitDepth(const char* wordTypeName, std::index_sequence<Is...>) {
    (testBitSlicedNoScratch<Is + 1, WordType>(wordTypeName), ...);
}

/// @brief Every the case, for one word type, at N = 1..MAX_BIT_DEPTH.
/// @note **N and the word type are independent axes and both are swept whole.** N
/// decides how many plane pairs there are (the kernel's arithmetic); the word
/// type decides every mask and shift. A bug in the pair weighting shows
/// at one N and every word type; a bug in the head/tail masks shows at one
/// word type and every N. Neither axis alone would find both.
template <typename WordType>
void testBitSlicedWordType(const char* wordTypeName) {
    std::cout << "\n--- N-bit gradient covariance: " << wordTypeName << " ---\n";
    const size_t before = g_bitSlicedPositions;
    const size_t beforeIdentity = g_identityPositions;

    // THE N = 1 IDENTITY FIRST. If ternary is not the N = 1 instance, nothing else
    // in this case means what it says it means.
    testTernaryIsTheNEqualsOneInstance<WordType>(wordTypeName);

    sweepEveryBitDepth<WordType>(wordTypeName, std::make_index_sequence<MAX_BIT_DEPTH>{});

    // The contracts are checked at N = 3, the depth puts pyramid level 1 at:
    // it has more than one magnitude plane, so a plane the kernel forgets is
    // visible, and it is not the largest N so the arrays are not degenerate.
    testBitSlicedContracts<3, WordType>(wordTypeName);
    testBitSlicedWeightsByHand<WordType>(wordTypeName);
    // **Every depth, not only the deep ones.** The no-heap rule is the library's
    // central one and N = 1 and N = 2 are the depths the frontend's lowest levels
    // run at, so leaving them out left the promise unmeasured exactly where a
    // per-plane-pair temporary would be cheapest to introduce unnoticed.
    noScratchAtEveryBitDepth<WordType>(wordTypeName, std::make_index_sequence<MAX_BIT_DEPTH>{});

    std::cout << " " << wordTypeName << ": " << (g_bitSlicedPositions - before)
              << " window positions compared against the per-pixel INTEGER oracle at "
              << "N = 1.." << MAX_BIT_DEPTH << ", " << (g_identityPositions - beforeIdentity)
              << " more against the ternary kernel at N = 1 (running totals "
              << g_bitSlicedPositions << " / " << g_identityPositions << ")\n";
}

} // namespace

BINCV_TEST(Covariance, Identity_uint8_t)  { testWordType<uint8_t>("uint8_t"); }
BINCV_TEST(Covariance, Identity_uint16_t) { testWordType<uint16_t>("uint16_t"); }
BINCV_TEST(Covariance, Identity_uint32_t) { testWordType<uint32_t>("uint32_t"); }
BINCV_TEST(Covariance, Identity_uint64_t) { testWordType<uint64_t>("uint64_t"); }

BINCV_TEST(Covariance, BitSliced_uint8_t)  { testBitSlicedWordType<uint8_t>("uint8_t"); }
BINCV_TEST(Covariance, BitSliced_uint16_t) { testBitSlicedWordType<uint16_t>("uint16_t"); }
BINCV_TEST(Covariance, BitSliced_uint32_t) { testBitSlicedWordType<uint32_t>("uint32_t"); }
BINCV_TEST(Covariance, BitSliced_uint64_t) { testBitSlicedWordType<uint64_t>("uint64_t"); }

BINCV_TEST_MAIN("test_covariance")
