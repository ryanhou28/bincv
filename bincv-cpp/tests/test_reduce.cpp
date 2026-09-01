// Bulk reductions: countNonZero / countAnd / countAndSplit /
// countCovariance / SlidingWindowCount.
//
// added three entry points and one accumulator to what shipped,
// and every one of them computes something an older entry point already computes
// -- faster, by traversing less. So the suite's central question for them is not
// "is this number right" but "is this number THE SAME": Reduce.Sliding_* sweeps
// whole frames comparing the incremental accumulator against recompute position
// by position, and Reduce.Fused_* does the same for the fused covariance and the
// four-argument selector against the composition. Both also go through the
// per-pixel references, so two implementations that were wrong the same way could
// not agree their way past.
//
// TWO HALVES, and they answer different questions -- the same split as
// tests/test_logic.cpp and tests/test_shift.cpp.
//
// 1. The CORE half (everything up to the OpenCV guard) needs no OpenCV, so it
// runs in all four verification configurations -- including the Debug one,
// which is the only place the kernels' BINCV_ASSERT preconditions are live,
// and the -fno-exceptions one, which is the embedded claim. It checks every
// reduction against a per-pixel reference written before the kernels, over a
// region sweep that is deliberately hostile to word arithmetic.
//
// 2. The OPENCV half asserts what Tier 1 promises for countNonZero: equality
// with cv::countNonZero on the same binary content stored as CV_8U
// (the design notes, 10.3), across that work’s full size and fill matrix, at all
// four word widths, whole-image and by region.
//
// WHERE THE EXHAUSTIVE SWEEP IS, AND WHY IT IS NOT IN THE VALUE TESTS
//
// The word geometry of a region depends on (x0 mod WordBits, x1 mod WordBits) and
// on whether a region-row fits inside one word. It does NOT depend on the image
// content, the fill ratio or the height. So the exhaustive cross product of
// origins and extents runs against the GEOMETRY (Reduce.Geometry_*), where each
// entry costs a few word operations, and the value sweeps run a curated region
// list over the full size and fill matrix instead. Sweeping both exhaustively
// would multiply cost by content that cannot change the answer.
//
// ONE CHECK PER (CASE, REGION), NOT PER PIXEL AND NOT PER CASE
//
// Per pixel would drown the summary -- a 640x17 case is 10880 pixels and says one
// thing. Per case would make the CHECKS column blind to a region list that got
// shorter, which is the regression this suite is most likely to suffer: every
// interesting case here IS a region. So each region reports one check, and its
// message is built only when it fails (see expectCount).
//
// WHY THERE ARE NO BINCV_EQUIVALENCE_INJECT_FAULT TARGETS FOR THIS SUITE
//
// tests/CMakeLists.txt rebuilds test_logic under three conversion faults as
// WILL_FAIL targets, because its Tier 1 half reads binCV back through unpackTo8U
// and a fault there could cancel through a pointwise operation. This suite never
// calls unpackTo8U: its OpenCV half compares a binCV count against
// cv::countNonZero of content built by the harness's INDEPENDENT generator
// (randomCvMask), so there is no shared conversion to cancel through. Fault 4
// (dirty padding) is the interesting one, and it is covered in-process instead by
// Reduce.DirtyPadding_*: this file's padding contract says a dirty padding bit
// changes no count, so a WILL_FAIL target built under fault 4 would NOT fail --
// that is the contract, not a gap.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/logic.hpp"
#include "bincv-cpp/ops/reduce.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "test_util.hpp"

// OpenCV-only, and self-guarding: everything in it sits behind BINCV_WITH_OPENCV,
// so this include is a no-op in the three configurations that have no OpenCV.
#include "equivalence.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#endif

namespace {

using bincv::Rect;
using bincv::SplitCount;
using bincv::countAnd;
using bincv::CovarianceCount;
using bincv::SlidingWindowCount;
using bincv::countAndSplit;
using bincv::countCovariance;
using bincv::countNonZero;

// ---------------------------------------------------------------------------
// The size and fill matrix (the same one tests/test_logic.cpp sweeps)
// ---------------------------------------------------------------------------

const int WIDTHS[] = {1, 7, 31, 33, 40, 63, 65, 70, 128, 640};
const int HEIGHTS[] = {1, 2, 3, 17};
const float FILLS[] = {0.0f, 0.01f, 0.5f, 0.99f, 1.0f};

// An over-aligned row stride (the design rule makes alignment a per-object choice): 32 bytes
// is a whole number of 1-, 2-, 4- and 8-byte words, so every word type gets a
// stride strictly greater than ceil(width / WordBits) at most widths -- which is
// the one thing a reduction must not assume.
constexpr size_t PADDED_ALIGNMENT = 32;

// ---------------------------------------------------------------------------
// Content: the same generator as tests/equivalence.hpp, minus OpenCV
// ---------------------------------------------------------------------------
//
// Duplicated rather than shared, for the reason gives: the harness that
// judges a kernel and the kernel's own test must not share machinery, or a fault
// in the shared part cancels. tests/test_logic.cpp carries the same three
// functions for the same reason.

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

/// @brief Fills a matrix through set, so its padding bits stay clear on entry.
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
    return UINT64_C(0x5EED00C0FFEE0001) + static_cast<uint64_t>(width) * UINT64_C(1000003) +
           static_cast<uint64_t>(height) * UINT64_C(10007) + index * UINT64_C(31);
}

// ---------------------------------------------------------------------------
// The per-pixel references, written before the kernels they judge
// ---------------------------------------------------------------------------
//
// All of them take VIEWS rather than containers, so one reference serves a
// BinMat, an over-aligned one, a hand-built view over a caller's buffer, and a
// sub-width window onto a wider image -- four of the constructions swept below,
// and the last two have no container at all.
//
// Each encodes the padding contract from ops/reduce.hpp by construction: the
// column loop stops at `width`, so a bit at or past `width` is never read, let
// alone counted. If a kernel and a reference ever disagree on a view whose
// padding is dirty, that IS the contract failing.

/// @brief Reads one pixel out of a view. The bit convention, and nothing else.
template <typename WordType>
bool pixelAt(const bincv::BinMatConstView<WordType>& v, int y, int x) {
    constexpr size_t bits = bincv::BinMatConstView<WordType>::WordBits;
    const size_t ux = static_cast<size_t>(x);
    const WordType mask = static_cast<WordType>(static_cast<WordType>(1) << (ux % bits));
    return (v.row(static_cast<size_t>(y))[ux / bits] & mask) != 0;
}

/// @brief Sets one pixel of a view directly, bypassing set.
/// @note Used only to dirty a sign plane where the magnitude is zero, which set
/// correctly refuses to do (the canonical-zero rule).
template <typename WordType>
void setPixel(bincv::BinMatView<WordType> v, int y, int x) {
    constexpr size_t bits = bincv::BinMatView<WordType>::WordBits;
    const size_t ux = static_cast<size_t>(x);
    const WordType mask = static_cast<WordType>(static_cast<WordType>(1) << (ux % bits));
    WordType* row = v.row(static_cast<size_t>(y));
    row[ux / bits] = static_cast<WordType>(row[ux / bits] | mask);
}

/// @brief A rectangle already intersected with an image; half-open on both axes.
struct Clipped {
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    bool empty() const { return x0 >= x1 || y0 >= y1; }
};

/// @brief The region contract, written independently of impl::clipRegion.
/// @note Deliberately a different shape from the library's -- min/max on long long
/// against the extents, rather than the library's early-exit ladder. Two
/// implementations of one rule, so the core-only configurations still have
/// something for the library's version to disagree with.
Clipped clipReference(int width, int height, const Rect& r) {
    Clipped c;
    if (r.width <= 0 || r.height <= 0) return c;
    const long long x0 = std::max<long long>(static_cast<long long>(r.x), 0);
    const long long y0 = std::max<long long>(static_cast<long long>(r.y), 0);
    const long long x1 = std::min<long long>(
        static_cast<long long>(r.x) + static_cast<long long>(r.width), width);
    const long long y1 = std::min<long long>(
        static_cast<long long>(r.y) + static_cast<long long>(r.height), height);
    if (x0 >= x1 || y0 >= y1) return c;
    c.x0 = static_cast<int>(x0);
    c.y0 = static_cast<int>(y0);
    c.x1 = static_cast<int>(x1);
    c.y1 = static_cast<int>(y1);
    return c;
}

template <typename WordType>
size_t refCountNonZero(const bincv::BinMatConstView<WordType>& v, const Rect& r) {
    const Clipped c = clipReference(static_cast<int>(v.width), static_cast<int>(v.height), r);
    size_t n = 0;
    for (int y = c.y0; y < c.y1; ++y) {
        for (int x = c.x0; x < c.x1; ++x) {
            if (pixelAt(v, y, x)) ++n;
        }
    }
    return n;
}

template <typename WordType>
size_t refCountAnd(const bincv::BinMatConstView<WordType>& a,
                   const bincv::BinMatConstView<WordType>& b, const Rect& r) {
    const Clipped c = clipReference(static_cast<int>(a.width), static_cast<int>(a.height), r);
    size_t n = 0;
    for (int y = c.y0; y < c.y1; ++y) {
        for (int x = c.x0; x < c.x1; ++x) {
            if (pixelAt(a, y, x) && pixelAt(b, y, x)) ++n;
        }
    }
    return n;
}

template <typename WordType>
SplitCount refCountAndSplit(const bincv::BinMatConstView<WordType>& a,
                            const bincv::BinMatConstView<WordType>& b,
                            const bincv::BinMatConstView<WordType>& c, const Rect& r) {
    const Clipped clip = clipReference(static_cast<int>(a.width), static_cast<int>(a.height), r);
    SplitCount out;
    for (int y = clip.y0; y < clip.y1; ++y) {
        for (int x = clip.x0; x < clip.x1; ++x) {
            if (!pixelAt(a, y, x) || !pixelAt(b, y, x)) continue;
            if (pixelAt(c, y, x)) {
                ++out.whenSet;
            } else {
                ++out.whenClear;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

std::string rectText(const Rect& r) {
    return "[x=" + std::to_string(r.x) + " y=" + std::to_string(r.y) + " w=" +
           std::to_string(r.width) + " h=" + std::to_string(r.height) + "]";
}

std::string sizeLabel(const char* wordTypeName, int width, int height, float fill) {
    char fillText[16];
    std::snprintf(fillText, sizeof(fillText), "%.2f", static_cast<double>(fill));
    return std::string(wordTypeName) + " " + std::to_string(width) + "x" +
           std::to_string(height) + " fill=" + fillText;
}

std::string sizeLabel(const char* wordTypeName, int width, int height) {
    return std::string(wordTypeName) + " " + std::to_string(width) + "x" + std::to_string(height);
}

/// @brief One check for one (case, region) comparison.
/// @note The message is built ONLY when the check fails -- the ternary below is
/// what keeps a sweep of a hundred thousand regions from spending its time
/// in std::string. On success it constructs an empty one, which allocates
/// nothing.
void expectCount(size_t actual, size_t expected, const char* what, const Rect& region,
                 const std::string& label, const char* file, int line) {
    const bool ok = (actual == expected);
    ::bincv::test::reportCheck(
        ok, "count matches the reference", file, line,
        ok ? std::string()
           : label + " " + what + " " + rectText(region) + ": got " + std::to_string(actual) +
                 ", expected " + std::to_string(expected));
}

/// @brief One check for a property that is already a boolean.
void expectTrue(bool ok, const char* what, const std::string& detail, const std::string& label,
                const char* file, int line) {
    ::bincv::test::reportCheck(ok, what, file, line,
                               ok ? std::string() : label + " " + what + ": " + detail);
}

#define REDUCE_EXPECT_COUNT(actual, expected, what, region, label) \
    expectCount((actual), (expected), (what), (region), (label), __FILE__, __LINE__)

#define REDUCE_EXPECT_TRUE(ok, what, detail, label) \
    expectTrue((ok), (what), (detail), (label), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// The region lists
// ---------------------------------------------------------------------------

/// @brief The regions every value sweep runs. Curated, and each entry is a case
/// someone could get wrong rather than a number that looked plausible.
std::vector<Rect> valueRegions(int w, int h) {
    std::vector<Rect> out;

    // The whole image, spelled as a region -- must agree with the no-region
    // overload, which is a different path into the same geometry.
    out.push_back(Rect(0, 0, w, h));
    // Larger than the image on every side: the clipping contract at its extreme.
    out.push_back(Rect(-7, -5, w + 20, h + 11));

    // Single pixels, including both corners. Width 1 is the degenerate LK window
    // and the case where the head and tail masks are applied to the SAME word.
    out.push_back(Rect(0, 0, 1, 1));
    out.push_back(Rect(w - 1, h - 1, 1, 1));
    out.push_back(Rect(w / 2, h / 2, 1, 1));

    // Origins and extents that land on and off every word boundary the four
    // supported word widths have (8, 16, 32, 64), from both sides.
    const int xs[] = {0, 1, 7, 8, 31, 33};
    const int ws[] = {1, 7, 8, 32, 33, 65};
    for (int x0 : xs) {
        for (int ww : ws) {
            out.push_back(Rect(x0, 0, ww, h));
        }
    }

    // Regions exactly one word wide, aligned to a word boundary and not.
    out.push_back(Rect(0, 0, 8, h));
    out.push_back(Rect(0, 0, 16, h));
    out.push_back(Rect(0, 0, 64, h));
    out.push_back(Rect(3, 0, 32, h));
    out.push_back(Rect(3, 0, 64, h));

    // Row geometry: rows are whole strides, so these cannot produce a packing bug,
    // but they can produce a stride bug.
    const int ys[] = {0, 1, h - 1, h};
    const int hs[] = {1, 2, h};
    for (int y0 : ys) {
        for (int hh : hs) {
            out.push_back(Rect(0, y0, w, hh));
            out.push_back(Rect(1, y0, w, hh));  // and off the word boundary
        }
    }

    // Wholly outside on each side, and straddling each edge.
    out.push_back(Rect(-w - 5, 0, w, h));
    out.push_back(Rect(w, 0, 5, h));
    out.push_back(Rect(0, -h - 5, w, h));
    out.push_back(Rect(0, h, w, 5));
    out.push_back(Rect(-3, -3, 7, 7));
    out.push_back(Rect(w - 3, h - 3, 7, 7));

    // Empty by extent rather than by position. Not the same path through the clip.
    out.push_back(Rect(0, 0, 0, h));
    out.push_back(Rect(0, 0, w, 0));
    out.push_back(Rect(0, 0, -4, h));
    out.push_back(Rect(0, 0, w, -4));

    return out;
}

/// @brief The exhaustive region cross product the GEOMETRY sweep runs.
/// @note A few word operations per entry, so this one can afford to be a cross
/// product where the value sweeps cannot -- see the header comment.
std::vector<Rect> geometryRegions(int w, int h) {
    std::vector<Rect> out;
    const int xs[] = {-9, -1, 0, 1, 5, 7, 8, 15, 16, 31, 32, 33, 63, 64, w - 1, w, w + 3};
    const int ws[] = {0, 1, 2, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, w, w + 9};
    const int ys[] = {-2, 0, 1, h - 1, h};
    const int hs[] = {0, 1, 2, h, h + 3};
    for (int x0 : xs) {
        for (int ww : ws) {
            out.push_back(Rect(x0, 0, ww, h));
        }
    }
    for (int y0 : ys) {
        for (int hh : hs) {
            out.push_back(Rect(0, y0, w, hh));
            out.push_back(Rect(5, y0, 40, hh));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 1. countNonZero against the per-pixel reference
// ---------------------------------------------------------------------------

template <typename WordType>
void testCountNonZeroReference(const char* wordTypeName) {
    std::cout << "\n--- countNonZero vs per-pixel reference: " << wordTypeName << " ---\n";

    for (int width : WIDTHS) {
        for (int height : HEIGHTS) {
            for (size_t f = 0; f < sizeof(FILLS) / sizeof(FILLS[0]); ++f) {
                bincv::BinMat<WordType> m(width, height);
                fillRandom(m, FILLS[f], caseSeed(width, height, f));
                const bincv::BinMatConstView<WordType> v = m.constView();
                const std::string label = sizeLabel(wordTypeName, width, height, FILLS[f]);
                const Rect whole(0, 0, width, height);

                // The no-region overload, against the whole-image reference and
                // against the container's own per-pixel countNonZero.
                REDUCE_EXPECT_COUNT(countNonZero(v), refCountNonZero(v, whole), "whole image",
                                    whole, label);
                REDUCE_EXPECT_COUNT(countNonZero(v), static_cast<size_t>(m.countNonZero()),
                                    "vs BinMat::countNonZero", whole, label);

                for (const Rect& r : valueRegions(width, height)) {
                    REDUCE_EXPECT_COUNT(countNonZero(v, r), refCountNonZero(v, r), "countNonZero",
                                        r, label);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. countAnd and countAndSplit against the per-pixel references
// ---------------------------------------------------------------------------

template <typename WordType>
void testMaskedReference(const char* wordTypeName) {
    std::cout << "\n--- countAnd / countAndSplit vs per-pixel reference: " << wordTypeName
              << " ---\n";

    for (int width : WIDTHS) {
        for (int height : HEIGHTS) {
            for (size_t f = 0; f < sizeof(FILLS) / sizeof(FILLS[0]); ++f) {
                const uint64_t seed = caseSeed(width, height, f);
                bincv::BinMat<WordType> a(width, height);
                bincv::BinMat<WordType> b(width, height);
                bincv::BinMat<WordType> c(width, height);
                fillRandom(a, FILLS[f], seed);
                fillRandom(b, FILLS[f], seed ^ UINT64_C(0xDEADBEEF));
                // The selector is drawn at a fixed 0.5 whatever the sources' fill:
                // at fill 0.0 and 1.0 a selector that followed it would make one
                // half of every split identically zero and prove nothing.
                fillRandom(c, 0.5f, seed ^ UINT64_C(0x5A5A5A5A5A5A5A5A));

                const bincv::BinMatConstView<WordType> va = a.constView();
                const bincv::BinMatConstView<WordType> vb = b.constView();
                const bincv::BinMatConstView<WordType> vc = c.constView();
                const std::string label = sizeLabel(wordTypeName, width, height, FILLS[f]);

                for (const Rect& r : valueRegions(width, height)) {
                    const size_t andCount = countAnd(va, vb, r);
                    REDUCE_EXPECT_COUNT(andCount, refCountAnd(va, vb, r), "countAnd", r, label);

                    const SplitCount split = countAndSplit(va, vb, vc, r);
                    const SplitCount expected = refCountAndSplit(va, vb, vc, r);
                    REDUCE_EXPECT_COUNT(split.whenClear, expected.whenClear, "whenClear", r, label);
                    REDUCE_EXPECT_COUNT(split.whenSet, expected.whenSet, "whenSet", r, label);

                    // The two entry points must also agree with each other: a
                    // split that dropped or double-counted a word would otherwise
                    // have to be wrong in both halves the same way to stay
                    // consistent with the reference.
                    REDUCE_EXPECT_COUNT(split.whenClear + split.whenSet, andCount,
                                        "split sums to countAnd", r, label);
                }

                // Aliasing is unrestricted for a read-only kernel, and this is the
                // spelling that an alias predicate copied in from ops/logic.hpp
                // would reject: every argument the same view.
                const Rect whole(0, 0, width, height);
                REDUCE_EXPECT_COUNT(countAnd(va, va, whole), countNonZero(va), "countAnd(a, a)",
                                    whole, label);
                const SplitCount selfSplit = countAndSplit(va, va, va, whole);
                REDUCE_EXPECT_COUNT(selfSplit.whenSet, countNonZero(va),
                                    "countAndSplit(a, a, a) whenSet", whole, label);
                REDUCE_EXPECT_COUNT(selfSplit.whenClear, 0u, "countAndSplit(a, a, a) whenClear",
                                    whole, label);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Geometry: the clipping rule and the single-pass row skeleton
// ---------------------------------------------------------------------------
//
// This is the test that reaches into impl::, and it is worth doing because the
// two properties it pins are invisible in a value comparison:
//
// - impl::visitRowWords visits each word index of a region-row EXACTLY ONCE, in
// ascending order. That is what that work’s "single pass" means, and a skeleton
// that visited the head word twice would still produce the right count for
// every region whose head word happens to be empty.
// - the masks select EXACTLY the region's columns and nothing else -- in
// particular no column at or past `width`, which is the padding contract
// stated as a property of the geometry rather than of an image.
//
// tests/test_shift.cpp does the same to impl::borderIndex, and for the same
// reason: an image comparison reaches the border only at the border.

template <typename WordType>
void testGeometry(const char* wordTypeName) {
    std::cout << "\n--- region geometry and the single-pass skeleton: " << wordTypeName
              << " ---\n";
    constexpr size_t wordBits = bincv::BinMatConstView<WordType>::WordBits;

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            const size_t uw = static_cast<size_t>(width);
            const size_t uh = static_cast<size_t>(height);
            const std::string label = sizeLabel(wordTypeName, width, height);

            std::vector<size_t> visited;
            std::vector<bool> selected;

            for (const Rect& r : geometryRegions(width, height)) {
                const Clipped expected = clipReference(width, height, r);
                const bincv::impl::RegionWords<WordType> region =
                    bincv::impl::clipRegion<WordType>(uw, uh, r);

                if (region.isEmpty || expected.empty()) {
                    REDUCE_EXPECT_TRUE(region.isEmpty == expected.empty(), "clips to empty",
                                       rectText(r), label);
                    continue;
                }

                // Walk the row skeleton exactly as a kernel does, recording what
                // it hands back.
                visited.clear();
                selected.assign(uw, false);
                size_t outsideWidth = 0;
                bincv::impl::visitRowWords<WordType>(region, [&](size_t i, WordType mask) {
                    visited.push_back(i);
                    for (size_t bit = 0; bit < wordBits; ++bit) {
                        const WordType probe =
                            static_cast<WordType>(static_cast<WordType>(1) << bit);
                        if ((mask & probe) == 0) continue;
                        const size_t column = i * wordBits + bit;
                        if (column < uw) {
                            selected[column] = true;
                        } else {
                            ++outsideWidth;
                        }
                    }
                });

                bool singlePass = visited.size() == (region.lastWord - region.firstWord + 1);
                for (size_t k = 0; k < visited.size(); ++k) {
                    if (visited[k] != region.firstWord + k) singlePass = false;
                }

                size_t mismatchedColumns = 0;
                for (size_t x = 0; x < uw; ++x) {
                    const bool inside = x >= static_cast<size_t>(expected.x0) &&
                                        x < static_cast<size_t>(expected.x1);
                    if (selected[x] != inside) ++mismatchedColumns;
                }

                const bool rowsMatch = region.y0 == static_cast<size_t>(expected.y0) &&
                                       region.y1 == static_cast<size_t>(expected.y1);
                const bool ok = singlePass && rowsMatch && mismatchedColumns == 0 &&
                                outsideWidth == 0;
                REDUCE_EXPECT_TRUE(
                    ok, "geometry: one visit per word, masks select exactly the region",
                    rectText(r) + " rows=" + std::to_string(region.y0) + ".." +
                        std::to_string(region.y1) + " words=" + std::to_string(region.firstWord) +
                        ".." + std::to_string(region.lastWord) + " visits=" +
                        std::to_string(visited.size()) + " badColumns=" +
                        std::to_string(mismatchedColumns) + " pastWidth=" +
                        std::to_string(outsideWidth),
                    label);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. The padding-bit contract
// ---------------------------------------------------------------------------
//
// ops/reduce.hpp: "a reduction counts only bits inside the requested region
// intersected with the image; a bit at or past `width` is never counted, whatever
// it holds." Three constructions where such a bit is set, none of which may change
// an answer:
//
// a. a wrapped buffer of all ones -- every pixel AND every padding bit set;
// b. clean content copied into a buffer whose padding is then dirtied;
// c. a sub-width WINDOW onto a wider image, where the bits past the window's
// `width` are not padding at all but a neighbour's live pixels.
//
// (c) is the one that reasoning about padding alone would miss, and it is the
// construction the design notes needs: an LK window is a view onto a frame.

template <typename WordType>
void testDirtyPadding(const char* wordTypeName) {
    std::cout << "\n--- reductions ignore bits past width: " << wordTypeName << " ---\n";
    constexpr size_t wordBits = bincv::BinMatConstView<WordType>::WordBits;
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));
    const float dirtyFills[] = {0.01f, 0.5f, 1.0f};

    for (int width : WIDTHS) {
        for (int height : {1, 2, 17}) {
            const size_t uw = static_cast<size_t>(width);
            const size_t uh = static_cast<size_t>(height);
            const size_t minWords = (uw + wordBits - 1) / wordBits;
            // A stride wider than the row needs, so there are whole padding WORDS
            // as well as padding bits inside the trailing one.
            const size_t stride = minWords + 2;
            const std::string label = sizeLabel(wordTypeName, width, height);
            const Rect whole(0, 0, width, height);
            const std::vector<Rect> regions = valueRegions(width, height);

            // (a) everything set, pixels and padding alike.
            {
                std::vector<WordType> buffer(stride * uh, allOnes);
                const bincv::BinMatConstView<WordType> v{buffer.data(), uw, uh, stride};
                REDUCE_EXPECT_COUNT(countNonZero(v), uw * uh, "all-ones buffer, whole image",
                                    whole, label);
                REDUCE_EXPECT_COUNT(countAnd(v, v, whole), uw * uh, "all-ones buffer, countAnd",
                                    whole, label);
                for (const Rect& r : regions) {
                    REDUCE_EXPECT_COUNT(countNonZero(v, r), refCountNonZero(v, r),
                                        "all-ones buffer", r, label);
                }
            }

            // (b) clean content, dirtied padding: no count may move.
            for (float fill : dirtyFills) {
                bincv::BinMat<WordType> clean(width, height);
                fillRandom(clean, fill, caseSeed(width, height, 40));

                std::vector<WordType> buffer(stride * uh, allOnes);
                for (size_t y = 0; y < uh; ++y) {
                    const WordType* src = clean.constView().row(y);
                    for (size_t i = 0; i < minWords; ++i) buffer[y * stride + i] = src[i];
                    // Set every bit of the trailing word that is not a pixel; the
                    // words past minWords are already all ones from the fill.
                    const size_t tail = uw % wordBits;
                    if (tail != 0) {
                        const WordType keep =
                            static_cast<WordType>((static_cast<WordType>(1) << tail) - 1);
                        buffer[y * stride + minWords - 1] = static_cast<WordType>(
                            buffer[y * stride + minWords - 1] | static_cast<WordType>(~keep));
                    }
                }
                const bincv::BinMatConstView<WordType> dirty{buffer.data(), uw, uh, stride};
                const bincv::BinMatConstView<WordType> pristine = clean.constView();

                REDUCE_EXPECT_COUNT(countNonZero(dirty), countNonZero(pristine),
                                    "dirty padding, whole image", whole, label);
                for (const Rect& r : regions) {
                    const size_t expected = refCountNonZero(pristine, r);
                    REDUCE_EXPECT_COUNT(countNonZero(dirty, r), expected, "dirty padding", r,
                                        label);
                    REDUCE_EXPECT_COUNT(countAnd(dirty, dirty, r), expected,
                                        "dirty padding countAnd", r, label);
                    const SplitCount split = countAndSplit(dirty, dirty, dirty, r);
                    REDUCE_EXPECT_COUNT(split.whenSet, expected, "dirty padding split whenSet", r,
                                        label);
                    REDUCE_EXPECT_COUNT(split.whenClear, 0u, "dirty padding split whenClear", r,
                                        label);
                }
            }

            // (c) a sub-width window onto a wider image. The bits past the
            // window's `width` are the parent's live pixels and must not be
            // counted -- a reduction over a window is over the window.
            {
                const int parentWidth = 640;
                bincv::BinMat<WordType> parent(parentWidth, height);
                fillRandom(parent, 0.5f, caseSeed(parentWidth, height, 77));
                const size_t parentStride = parent.getAlignedWidth();

                // Two windows: one starting at word 0, one at word 1, so the
                // window's columns are and are not the parent's columns.
                for (size_t startWord : {size_t{0}, size_t{1}}) {
                    const int parentX0 = static_cast<int>(startWord * wordBits);
                    if (parentX0 + width > parentWidth) continue;
                    if (startWord + minWords > parentStride) continue;

                    const bincv::BinMatConstView<WordType> window{parent.data() + startWord, uw, uh,
                                                                  parentStride};
                    size_t expected = 0;
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            if (parent.at(y, parentX0 + x)) ++expected;
                        }
                    }
                    REDUCE_EXPECT_COUNT(countNonZero(window), expected,
                                        "sub-width window onto a 640-wide image", whole, label);
                    REDUCE_EXPECT_COUNT(countNonZero(window, whole), expected,
                                        "sub-width window, as a region", whole, label);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Strides: over-aligned, and differing between the arguments
// ---------------------------------------------------------------------------

template <typename WordType>
void testStrides(const char* wordTypeName) {
    std::cout << "\n--- differing strides: " << wordTypeName << " ---\n";
    constexpr size_t wordBits = bincv::BinMatConstView<WordType>::WordBits;

    for (int width : WIDTHS) {
        for (int height : {1, 3, 17}) {
            const uint64_t seed = caseSeed(width, height, 900);
            const std::string label = sizeLabel(wordTypeName, width, height);

            // Three sources, three different strides: word granularity (the design rule’s
            // default), 32-byte aligned, and a hand-built view with two spare
            // words per row. A reduction must read each one's own stride.
            bincv::BinMat<WordType> dense(width, height);
            bincv::BinMat<WordType> padded(width, height, PADDED_ALIGNMENT);
            fillRandom(dense, 0.5f, seed);
            fillRandom(padded, 0.37f, seed ^ UINT64_C(0x1234));

            const size_t uw = static_cast<size_t>(width);
            const size_t uh = static_cast<size_t>(height);
            const size_t minWords = (uw + wordBits - 1) / wordBits;
            const size_t wideStride = minWords + 2;
            std::vector<WordType> wideBuffer(wideStride * uh, 0);
            {
                bincv::BinMat<WordType> tmp(width, height);
                fillRandom(tmp, 0.71f, seed ^ UINT64_C(0x9876));
                for (size_t y = 0; y < uh; ++y) {
                    const WordType* src = tmp.constView().row(y);
                    for (size_t i = 0; i < minWords; ++i) wideBuffer[y * wideStride + i] = src[i];
                }
            }
            const bincv::BinMatConstView<WordType> wide{wideBuffer.data(), uw, uh, wideStride};
            const bincv::BinMatConstView<WordType> vDense = dense.constView();
            const bincv::BinMatConstView<WordType> vPadded = padded.constView();

            for (const Rect& r : valueRegions(width, height)) {
                REDUCE_EXPECT_COUNT(countNonZero(vPadded, r), refCountNonZero(vPadded, r),
                                    "over-aligned stride", r, label);
                REDUCE_EXPECT_COUNT(countNonZero(wide, r), refCountNonZero(wide, r), "wide stride",
                                    r, label);
                REDUCE_EXPECT_COUNT(countAnd(vDense, vPadded, r), refCountAnd(vDense, vPadded, r),
                                    "countAnd across strides", r, label);
                const SplitCount split = countAndSplit(vDense, vPadded, wide, r);
                const SplitCount expected = refCountAndSplit(vDense, vPadded, wide, r);
                REDUCE_EXPECT_COUNT(split.whenClear, expected.whenClear,
                                    "whenClear across strides", r, label);
                REDUCE_EXPECT_COUNT(split.whenSet, expected.whenSet, "whenSet across strides", r,
                                    label);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Degenerate shapes
// ---------------------------------------------------------------------------

template <typename WordType>
void testDegenerate(const char* wordTypeName) {
    std::cout << "\n--- degenerate shapes: " << wordTypeName << " ---\n";
    const std::string label = std::string(wordTypeName) + " [degenerate]";
    const Rect none(0, 0, 0, 0);

    // An empty view, in each of the ways one can be empty. A count is 0, not an
    // error -- there is no error for a reduction to return.
    const bincv::BinMatConstView<WordType> nullView{};
    REDUCE_EXPECT_COUNT(countNonZero(nullView), 0u, "default-constructed view", none, label);
    REDUCE_EXPECT_COUNT(countNonZero(nullView, Rect(0, 0, 10, 10)), 0u,
                        "default-constructed view, region", none, label);
    REDUCE_EXPECT_COUNT(countAnd(nullView, nullView, Rect(0, 0, 10, 10)), 0u, "empty countAnd",
                        none, label);
    REDUCE_EXPECT_COUNT(countAndSplit(nullView, nullView, nullView, Rect(0, 0, 10, 10)).whenSet, 0u,
                        "empty countAndSplit", none, label);

    WordType storage[4] = {0, 0, 0, 0};
    const bincv::BinMatConstView<WordType> zeroWidth{storage, 0, 4, 1};
    const bincv::BinMatConstView<WordType> zeroHeight{storage, 4, 0, 1};
    REDUCE_EXPECT_COUNT(countNonZero(zeroWidth), 0u, "zero width", none, label);
    REDUCE_EXPECT_COUNT(countNonZero(zeroHeight), 0u, "zero height", none, label);
    REDUCE_EXPECT_COUNT(countNonZero(zeroWidth, Rect(0, 0, 4, 4)), 0u, "zero width, region", none,
                        label);
    REDUCE_EXPECT_COUNT(countNonZero(zeroHeight, Rect(0, 0, 4, 4)), 0u, "zero height, region",
                        none, label);

    // 1x1, set and clear: the smallest image there is, and the one where the head
    // and tail masks are two masks over one word.
    for (int pixel = 0; pixel <= 1; ++pixel) {
        bincv::BinMat<WordType> one(1, 1);
        one.set(0, 0, pixel != 0);
        const bincv::BinMatConstView<WordType> v = one.constView();
        const size_t expected = static_cast<size_t>(pixel);
        REDUCE_EXPECT_COUNT(countNonZero(v), expected, "1x1 whole", none, label);
        REDUCE_EXPECT_COUNT(countNonZero(v, Rect(0, 0, 1, 1)), expected, "1x1 region", none, label);
        REDUCE_EXPECT_COUNT(countNonZero(v, Rect(0, 0, 100, 100)), expected, "1x1 oversized region",
                            none, label);
        REDUCE_EXPECT_COUNT(countNonZero(v, Rect(-5, -5, 100, 100)), expected,
                            "1x1 negative origin", none, label);
        REDUCE_EXPECT_COUNT(countNonZero(v, Rect(1, 0, 1, 1)), 0u, "1x1 past the right edge", none,
                            label);
        REDUCE_EXPECT_COUNT(countNonZero(v, Rect(0, 1, 1, 1)), 0u, "1x1 past the bottom edge", none,
                            label);
        REDUCE_EXPECT_COUNT(countNonZero(v, Rect(0, 0, 0, 1)), 0u, "1x1 zero-width region", none,
                            label);
        REDUCE_EXPECT_COUNT(countNonZero(v, Rect(0, 0, 1, 0)), 0u, "1x1 zero-height region", none,
                            label);
        REDUCE_EXPECT_COUNT(countAnd(v, v, Rect(0, 0, 1, 1)), expected, "1x1 countAnd", none,
                            label);
        const SplitCount split = countAndSplit(v, v, v, Rect(0, 0, 1, 1));
        REDUCE_EXPECT_COUNT(split.whenSet, expected, "1x1 split whenSet", none, label);
        REDUCE_EXPECT_COUNT(split.whenClear, 0u, "1x1 split whenClear", none, label);
    }

    // Regions whose extents would overflow int if the clipping were done in int.
    // impl::clipRegion runs in long long precisely so these stay clipped rather
    // than wrapping into a region that reads outside the image.
    {
        bincv::BinMat<WordType> m(70, 3);
        fillRandom(m, 1.0f, 12345);
        const bincv::BinMatConstView<WordType> v = m.constView();
        const int big = 2147483000;  // x + width overflows int
        REDUCE_EXPECT_COUNT(countNonZero(v, Rect(big, 0, big, 3)), 0u, "far right, no int overflow",
                            none, label);
        REDUCE_EXPECT_COUNT(countNonZero(v, Rect(-big, 0, big, 3)), 0u, "far left", none, label);
        REDUCE_EXPECT_COUNT(countNonZero(v, Rect(-big, -big, 2 * (big / 2) + 200,
                                                 2 * (big / 2) + 200)),
                            static_cast<size_t>(70 * 3), "huge extents containing the image", none,
                            label);
    }

    // Rect's own three predicates. They are public API on a type this task added,
    // and impl::clipRegion calls empty -- but a member whose body is wrong and
    // whose only caller is a fast path can still leave every count right, so they
    // are checked directly rather than inferred from the counts above. (This is
    // the same argument that put Reduce.PortablePopcount_* in this file: source
    // no configuration exercises is source no configuration verifies.)
    REDUCE_EXPECT_TRUE(Rect().empty(), "default Rect is empty", "Rect()", label);
    REDUCE_EXPECT_TRUE(Rect(0, 0, 0, 5).empty(), "zero width is empty", "Rect(0,0,0,5)", label);
    REDUCE_EXPECT_TRUE(Rect(0, 0, 5, 0).empty(), "zero height is empty", "Rect(0,0,5,0)", label);
    REDUCE_EXPECT_TRUE(Rect(0, 0, -1, 5).empty(), "negative width is empty", "Rect(0,0,-1,5)",
                       label);
    REDUCE_EXPECT_TRUE(Rect(0, 0, 5, -1).empty(), "negative height is empty", "Rect(0,0,5,-1)",
                       label);
    REDUCE_EXPECT_TRUE(!Rect(0, 0, 1, 1).empty(), "1x1 is not empty", "Rect(0,0,1,1)", label);
    // A rectangle wholly outside an image is NOT empty by this test: emptiness
    // after clipping is the operation's business, which is what the docstring on
    // Rect::empty says and what impl::clipRegion relies on.
    REDUCE_EXPECT_TRUE(!Rect(-500, -500, 31, 31).empty(), "outside but not empty",
                       "Rect(-500,-500,31,31)", label);

    REDUCE_EXPECT_TRUE(Rect(1, 2, 3, 4) == Rect(1, 2, 3, 4), "equal rectangles compare equal",
                       "Rect(1,2,3,4)", label);
    REDUCE_EXPECT_TRUE(!(Rect(1, 2, 3, 4) != Rect(1, 2, 3, 4)), "operator!= is operator=='s inverse",
                       "Rect(1,2,3,4)", label);
    // One field at a time, because a comparison that ignores a field is exactly
    // the wrong body that a same-rectangle check cannot see.
    REDUCE_EXPECT_TRUE(Rect(1, 2, 3, 4) != Rect(9, 2, 3, 4), "x is compared", "x", label);
    REDUCE_EXPECT_TRUE(Rect(1, 2, 3, 4) != Rect(1, 9, 3, 4), "y is compared", "y", label);
    REDUCE_EXPECT_TRUE(Rect(1, 2, 3, 4) != Rect(1, 2, 9, 4), "width is compared", "width", label);
    REDUCE_EXPECT_TRUE(Rect(1, 2, 3, 4) != Rect(1, 2, 3, 9), "height is compared", "height", label);
}

// ---------------------------------------------------------------------------
// 7. The builtin-free popcount, which no configuration here would otherwise run
// ---------------------------------------------------------------------------
//
// impl::popcountWordPortable is what ops/reduce.hpp uses on a toolchain without
// __builtin_popcountll -- MSVC, and the claim that binCV needs a C++17 compiler
// and nothing else. Every configuration this project verifies is GCC or clang, so
// nothing else in the suite executes a single instruction of it. It is compiled
// unconditionally (rather than behind the #if) precisely so that this case can
// call it, and this case is the only thing standing between it and a typo that
// ships to the one toolchain that needs it.

template <typename WordType>
void testPortablePopcount(const char* wordTypeName) {
    std::cout << "\n--- builtin-free popcount vs the builtin: " << wordTypeName << " ---\n";
    const std::string label = std::string(wordTypeName) + " [portable popcount]";
    constexpr size_t bits = sizeof(WordType) * 8;
    const Rect none(0, 0, 0, 0);

    // Every value at 8 bits; a strided sweep plus the structurally interesting
    // ones at the wider widths, where exhaustive is not an option.
    uint64_t state = 0x9E3779B97F4A7C15ull;
    size_t mismatches = 0;
    size_t checked = 0;
    for (size_t i = 0; i < 4096; ++i) {
        const WordType candidates[] = {
            static_cast<WordType>(i),
            static_cast<WordType>(~static_cast<WordType>(i)),
            static_cast<WordType>(nextRandom(state)),
            static_cast<WordType>(static_cast<WordType>(1) << (i % bits)),
            static_cast<WordType>(~static_cast<WordType>(static_cast<WordType>(1) << (i % bits))),
        };
        for (WordType w : candidates) {
            ++checked;
            if (bincv::impl::popcountWordPortable<WordType>(w) !=
                static_cast<size_t>(__builtin_popcountll(static_cast<unsigned long long>(w)))) {
                ++mismatches;
            }
        }
    }
    REDUCE_EXPECT_COUNT(mismatches, 0u, "portable popcount agrees with the builtin", none, label);
    REDUCE_EXPECT_COUNT(checked, 4096u * 5u, "values compared", none, label);

    // The two endpoints by name, since a strided sweep can miss both.
    REDUCE_EXPECT_COUNT(bincv::impl::popcountWordPortable<WordType>(0), 0u, "popcount(0)", none,
                        label);
    REDUCE_EXPECT_COUNT(
        bincv::impl::popcountWordPortable<WordType>(static_cast<WordType>(~static_cast<WordType>(0))),
        bits, "popcount(all ones)", none, label);
}

// ---------------------------------------------------------------------------
// 8. The LK gradient covariance identity (the design notes)
// ---------------------------------------------------------------------------
//
// This is the reason exists, so it is tested as the thing it is for: build a
// pair of ternary derivative images, compute the 2x2 covariance THROUGH these
// primitives, and compare against a per-pixel floating-point reference over the
// same window.
//
// sumXX = countNonZero(mag_x, window)
// sumYY = countNonZero(mag_y, window)
// sumXY = split.crossTerm, from ONE countAndSplit pass over
// (mag_x, mag_y, sign_x ^ sign_y)
//
// sumXY is taken through SplitCount::crossTerm and not through the fields,
// because that is the spelling this is meant to copy. `whenClear - whenSet`
// written on the two size_t fields is unsigned arithmetic and wraps for every
// negatively correlated window -- half of them -- with no warning from a build
// that has -Wconversion -Wsign-conversion -Werror on. The anti-correlated block
// at the end of this function pins that: it is the case where the two spellings
// differ by 2^64, so a crossTerm that lost its casts cannot pass.
//
// Every quantity is an integer, so the float reference must agree EXACTLY. An
// approximate comparison would accept an off-by-one in the split, which is
// precisely the bug this operation can have.
//
// The sign planes are deliberately DIRTIED where the magnitude is zero. that work’s
// canonical-zero rule says the sign bit carries no information there and set
// will not write one, but a caller that writes the sign plane directly -- which
// that work’s derivative does, `sign = neg` being a whole-plane assignment -- can leave
// sign bits standing over zero magnitudes. The identity survives only because the
// `a & b` factor removes them, and that is what the dirty-sign variant checks.

template <typename WordType>
void testCovarianceIdentity(const char* wordTypeName) {
    std::cout << "\n--- LK gradient covariance identity: " << wordTypeName << " ---\n";

    struct Extent {
        int width;
        int height;
    };
    const Extent sizes[] = {{1, 1}, {17, 13}, {33, 9}, {64, 8}, {70, 37}, {129, 31}};

    for (const Extent& size : sizes) {
        for (int dirtySigns = 0; dirtySigns <= 1; ++dirtySigns) {
            const int width = size.width;
            const int height = size.height;
            const std::string label = sizeLabel(wordTypeName, width, height) +
                                      (dirtySigns != 0 ? " [covariance, dirty signs]"
                                                       : " [covariance]");

            bincv::TernaryMat<WordType> dx(width, height);
            bincv::TernaryMat<WordType> dy(width, height);

            uint64_t state = caseSeed(width, height, static_cast<size_t>(dirtySigns) + 500);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    // {-1, 0, +1}, drawn independently per image so that sign
                    // agreement is not correlated by construction.
                    const int vx = static_cast<int>(nextRandom(state) % 3) - 1;
                    const int vy = static_cast<int>(nextRandom(state) % 3) - 1;
                    dx.set(y, x, vx);
                    dy.set(y, x, vy);
                }
            }

            if (dirtySigns != 0) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        if (dx.magnitudeAt(y, x) == 0) setPixel(dx.sign(), y, x);
                        if (dy.magnitudeAt(y, x) == 0 && ((x + y) % 2 == 0)) {
                            setPixel(dy.sign(), y, x);
                        }
                    }
                }
            }

            // The scratch plane this will take as a caller-provided buffer. No
            // allocation happens inside any kernel here.
            bincv::BinMat<WordType> signXor(width, height);
            bincv::bitwiseXor(dx.constSign(), dy.constSign(), signXor.view());

            const int windowSizes[] = {7, 15, 31};
            for (int k : windowSizes) {
                const int half = k / 2;
                // Centers at the corners, edges and middle, so every window here
                // is clipped on at least one side except the middle ones.
                const int cxs[] = {0, half, width / 2, width - 1, width + 2};
                const int cys[] = {0, half, height / 2, height - 1};
                for (int cx : cxs) {
                    for (int cy : cys) {
                        const Rect window(cx - half, cy - half, k, k);

                        const size_t sumXX = countNonZero(dx.constMagnitude(0), window);
                        const size_t sumYY = countNonZero(dy.constMagnitude(0), window);
                        const SplitCount split = countAndSplit(
                            dx.constMagnitude(0), dy.constMagnitude(0), signXor.constView(),
                            window);
                        const long long sumXY = split.crossTerm();

                        // The per-pixel float reference, over the same window.
                        double refXX = 0.0;
                        double refYY = 0.0;
                        double refXY = 0.0;
                        const Clipped clip = clipReference(width, height, window);
                        for (int y = clip.y0; y < clip.y1; ++y) {
                            for (int x = clip.x0; x < clip.x1; ++x) {
                                const double ix = static_cast<double>(dx.at(y, x));
                                const double iy = static_cast<double>(dy.at(y, x));
                                refXX += ix * ix;
                                refYY += iy * iy;
                                refXY += ix * iy;
                            }
                        }

                        // THE SHAPE CALLS ( items 2 and 3): all four
                        // numbers from one pass, and the selector formed in the
                        // word loop rather than in a plane. Checked against the
                        // SAME float reference as the composition above, in the
                        // same loop, so "the fused path agrees window for window"
                        // covers the identity and not only the counts -- and every
                        // window here is clipped on at least one side except the
                        // middle ones.
                        const CovarianceCount fused =
                            countCovariance(dx.constMagnitude(0), dy.constMagnitude(0),
                                            signXor.constView(), window);
                        const CovarianceCount fusedXor =
                            countCovariance(dx.constMagnitude(0), dy.constMagnitude(0),
                                            dx.constSign(), dy.constSign(), window);
                        const bool fusedExact =
                            static_cast<double>(fused.xx) == refXX &&
                            static_cast<double>(fused.yy) == refYY &&
                            static_cast<double>(fused.crossTerm()) == refXY &&
                            fused.xx == sumXX && fused.yy == sumYY &&
                            fused.xy.whenClear == split.whenClear &&
                            fused.xy.whenSet == split.whenSet &&
                            fusedXor.xx == fused.xx && fusedXor.yy == fused.yy &&
                            fusedXor.xy.whenClear == fused.xy.whenClear &&
                            fusedXor.xy.whenSet == fused.xy.whenSet;
                        REDUCE_EXPECT_TRUE(
                            fusedExact,
                            "countCovariance matches the composition and the float reference",
                            rectText(window) + " fused xx=" + std::to_string(fused.xx) + " yy=" +
                                std::to_string(fused.yy) + " crossTerm=" +
                                std::to_string(fused.crossTerm()) + " ref " +
                                std::to_string(refXX) + "/" + std::to_string(refYY) + "/" +
                                std::to_string(refXY),
                            label);

                        const bool exact = static_cast<double>(sumXX) == refXX &&
                                           static_cast<double>(sumYY) == refYY &&
                                           static_cast<double>(sumXY) == refXY;
                        REDUCE_EXPECT_TRUE(
                            exact, "covariance matches the per-pixel float reference exactly",
                            rectText(window) + " sumXX=" + std::to_string(sumXX) + "/" +
                                std::to_string(refXX) + " sumYY=" + std::to_string(sumYY) + "/" +
                                std::to_string(refYY) + " sumXY=" + std::to_string(sumXY) + "/" +
                                std::to_string(refXY),
                            label);

                        // And the primitives' own consistency: the cross term's
                        // two halves must sum to the pixels where both magnitudes
                        // are set.
                        REDUCE_EXPECT_COUNT(
                            split.whenClear + split.whenSet,
                            countAnd(dx.constMagnitude(0), dy.constMagnitude(0), window),
                            "split sums to countAnd", window, label);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // The cross term is SIGNED, and this is the case that can tell.
    //
    // Every window above draws its signs at random, so whenClear - whenSet lands
    // near zero and a wrapped result is only occasionally distinguishable. Here
    // the correlation is -1 everywhere by construction: dx = +1, dy = -1, so the
    // signs disagree at every pixel, whenClear is 0, and the cross term is
    // -(width * height) exactly. Written on the fields as `whenClear - whenSet`
    // that is 2^64 - width*height instead, which is the bug crossTerm exists to
    // make unwritable -- and it compiles clean under the project's -Werror set,
    // so no build configuration would have caught it.
    // -----------------------------------------------------------------------
    {
        const std::string label = std::string(wordTypeName) + " [cross term, anti-correlated]";
        const int width = 32;
        const int height = 32;

        bincv::TernaryMat<WordType> dx(width, height);
        bincv::TernaryMat<WordType> dy(width, height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                dx.set(y, x, +1);
                dy.set(y, x, -1);
            }
        }
        bincv::BinMat<WordType> signXor(width, height);
        bincv::bitwiseXor(dx.constSign(), dy.constSign(), signXor.view());

        const Rect whole(0, 0, width, height);
        const SplitCount split =
            countAndSplit(dx.constMagnitude(0), dy.constMagnitude(0), signXor.constView(), whole);

        REDUCE_EXPECT_COUNT(split.whenClear, 0u, "no window agrees in sign", whole, label);
        REDUCE_EXPECT_COUNT(split.whenSet, static_cast<size_t>(width * height),
                            "every pixel disagrees in sign", whole, label);
        REDUCE_EXPECT_TRUE(split.crossTerm() == -static_cast<long long>(width) * height,
                           "crossTerm() is negative and exact",
                           "crossTerm=" + std::to_string(split.crossTerm()) + " expected=" +
                               std::to_string(-static_cast<long long>(width) * height),
                           label);
        //... and the unsigned spelling on the same two fields is the enormous
        // positive number this accessor exists to keep out of earlier work. Asserted, not
        // narrated, so the hazard is a checked fact rather than a comment that
        // could quietly stop being true.
        REDUCE_EXPECT_TRUE((split.whenClear - split.whenSet) > (~size_t{0} / 2),
                           "the unsigned spelling really does wrap (hence crossTerm)",
                           std::to_string(split.whenClear - split.whenSet), label);
        REDUCE_EXPECT_TRUE(split.crossTerm() < 0, "a negative correlation stays negative",
                           std::to_string(split.crossTerm()), label);

        // The reverse pairing, so the two halves cannot be swapped and pass.
        bincv::TernaryMat<WordType> dz(width, height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) dz.set(y, x, +1);
        }
        bincv::BinMat<WordType> agreeXor(width, height);
        bincv::bitwiseXor(dx.constSign(), dz.constSign(), agreeXor.view());
        const SplitCount agree =
            countAndSplit(dx.constMagnitude(0), dz.constMagnitude(0), agreeXor.constView(), whole);
        REDUCE_EXPECT_TRUE(agree.crossTerm() == static_cast<long long>(width) * height,
                           "an agreeing pair gives the positive cross term",
                           std::to_string(agree.crossTerm()), label);

        // THE SPELLING the design notes AND TASKS PRINT, compiled from a
        // NON-const container -- which is what this will hold. This is a
        // compile-time regression as much as a value one: reduce.hpp's entry
        // points take BinMatConstView and template argument deduction does not
        // consider BinMatView's conversion to it, so `dx.magnitude(0)` is a
        // deduction failure while `dx.constMagnitude(0)` is not. The header
        // documents that; this pins the documented form to something that must
        // keep compiling.
        bincv::TernaryMat<WordType>& mutableDx = dx;  // deliberately non-const
        bincv::BinMat<WordType>& mutableXor = signXor;
        REDUCE_EXPECT_COUNT(countNonZero(mutableDx.constMagnitude(0), whole),
                            static_cast<size_t>(width * height),
                            "constMagnitude() spelling from a non-const container", whole, label);
        REDUCE_EXPECT_COUNT(countAnd(mutableDx.constMagnitude(0), dy.constMagnitude(0), whole),
                            static_cast<size_t>(width * height),
                            "countAnd through const accessors", whole, label);
        REDUCE_EXPECT_TRUE(countAndSplit(mutableDx.constMagnitude(0), dy.constMagnitude(0),
                                         mutableXor.constView(), whole)
                                   .crossTerm() == -static_cast<long long>(width) * height,
                           "countAndSplit through const accessors", "crossTerm", label);
    }
}

// ---------------------------------------------------------------------------
// 9. SlidingWindowCount agrees with recompute WINDOW FOR WINDOW ( item 1)
// ---------------------------------------------------------------------------
//
// The whole risk of an incremental accumulator is that it is right for a while.
// A sum that gains a row and loses a row can be off by one row's popcount at a
// border, or can drift after many slides, and either bug leaves the first few
// windows of every sweep correct -- which is exactly the sample a spot check
// takes. So this sweeps WHOLE FRAMES: every x offset from one window-width left
// of the image to one past its right edge, and within each, every y position from
// one window-height above the image to one past its bottom. Every position is
// compared against countNonZero(src, window) -- the recompute path, unchanged by
// -- and against the rectangle the accumulator says it is on.
//
// The y range is deliberately wider than the image on BOTH sides. Above it, the
// window is clipped from the top and the "outgoing" row does not exist; below it,
// the count has to walk back down to zero and STAY there, which is the case a
// signed/unsigned slip in the subtraction would fail.
//
// Reporting is one check per (case, window size, x column) -- one per COLUMN of
// positions rather than one per position, for the reason the file header gives:
// per position would drown the summary, and per case would make the CHECKS column
// blind to a shortened sweep. The message names the first disagreeing position.

/// @brief One column of window positions, checked at every y. Returns the number
/// of positions that disagreed, and the first one's description.
/// @note TWO accumulators per position, not one, and the second is not redundant.
/// `slid` is constructed once above the frame and walked down, so it tests
/// the incremental path; `fresh` is constructed AT each position, so it
/// tests the constructor's initial sum over an arbitrary clipped row range.
/// Measured: with only the slid one, a constructor that dropped the last row
/// of its initial sum passed the whole suite -- because a column that starts
/// above the image starts with an empty row range and an initial sum of
/// zero, so the initialization loop never ran with anything to do.
template <typename WordType>
size_t slideColumnMismatches(const bincv::BinMatConstView<WordType>& v, int x, int windowW,
                             int windowH, std::string& firstBad) {
    const int yFirst = -windowH - 1;
    const int yLast = static_cast<int>(v.height) + windowH + 1;
    SlidingWindowCount<WordType> slid(v, Rect(x, yFirst, windowW, windowH));

    size_t mismatches = 0;
    for (int y = yFirst; y <= yLast; ++y) {
        const Rect w(x, y, windowW, windowH);
        const size_t got = slid.count();
        const size_t want = refCountNonZero(v, w);
        const size_t recompute = countNonZero(v, w);
        const size_t constructed = SlidingWindowCount<WordType>(v, w).count();
        const bool rectOk = slid.window() == w;
        if (got != want || got != recompute || constructed != want || !rectOk) {
            ++mismatches;
            if (firstBad.empty()) {
                firstBad = rectText(w) + " slid=" + std::to_string(got) + " constructed=" +
                           std::to_string(constructed) + " recompute=" +
                           std::to_string(recompute) + " reference=" + std::to_string(want) +
                           (rectOk ? "" : " (window() disagreed: " + rectText(slid.window()) + ")");
            }
        }
        slid.slideDown();
    }
    return mismatches;
}

/// @brief Sweeps every window position of one view and reports per x column.
template <typename WordType>
void sweepSlidingWindow(const bincv::BinMatConstView<WordType>& v, int windowW, int windowH,
                        const std::string& label) {
    const int width = static_cast<int>(v.width);
    for (int x = -windowW - 1; x <= width + 1; ++x) {
        std::string firstBad;
        const size_t bad = slideColumnMismatches<WordType>(v, x, windowW, windowH, firstBad);
        REDUCE_EXPECT_TRUE(bad == 0,
                           "sliding window agrees with recompute at every position of the column",
                           "W=" + std::to_string(windowW) + "x" + std::to_string(windowH) +
                               " x=" + std::to_string(x) + " mismatches=" + std::to_string(bad) +
                               " first: " + firstBad,
                           label);
    }
}

template <typename WordType>
void testSlidingWindow(const char* wordTypeName) {
    std::cout << "\n--- sliding window count vs recompute, window for window: " << wordTypeName
              << " ---\n";
    constexpr size_t wordBits = bincv::BinMatConstView<WordType>::WordBits;

    struct Frame {
        int width;
        int height;
        float fill;
    };
    // Widths on and off every word boundary the four supported widths have, and
    // one full frame row count so the sweep is over a realistic image rather than
    // only over small ones.
    const Frame frames[] = {
        {1, 1, 1.0f},   {7, 5, 0.5f},    {33, 9, 0.5f},   {64, 8, 0.99f},
        {70, 37, 0.5f}, {129, 21, 0.01f}, {640, 17, 0.5f},
    };
    const int squareWindows[] = {1, 7, 15, 31};

    for (const Frame& f : frames) {
        bincv::BinMat<WordType> m(f.width, f.height);
        fillRandom(m, f.fill, caseSeed(f.width, f.height, 1100));
        const bincv::BinMatConstView<WordType> v = m.constView();
        const std::string label = sizeLabel(wordTypeName, f.width, f.height, f.fill) + " [sliding]";
        for (int W : squareWindows) {
            sweepSlidingWindow<WordType>(v, W, W, label);
        }
    }

    // Non-square windows. The accumulator's x extent and y extent are independent
    // -- the columns are clipped once and the rows per position -- so a window
    // that is wider than it is tall, or a single column of pixels, has to work.
    {
        const int width = 70;
        const int height = 37;
        bincv::BinMat<WordType> m(width, height);
        fillRandom(m, 0.5f, caseSeed(width, height, 1101));
        const bincv::BinMatConstView<WordType> v = m.constView();
        const std::string label =
            sizeLabel(wordTypeName, width, height) + " [sliding, non-square]";
        struct Shape { int w; int h; };
        const Shape shapes[] = {{7, 15}, {15, 7}, {31, 3}, {1, 31}, {31, 1}, {64, 2}, {2, 64}};
        for (const Shape& sh : shapes) {
            sweepSlidingWindow<WordType>(v, sh.w, sh.h, label);
        }
    }

    // The same sweep over the view constructions a reduction must not assume away:
    // an over-aligned stride, a hand-built wide stride, and -- the one that can
    // silently over-count -- a buffer whose padding bits are all ones.
    {
        const int width = 33;
        const int height = 9;
        const size_t uw = static_cast<size_t>(width);
        const size_t uh = static_cast<size_t>(height);
        const size_t minWords = (uw + wordBits - 1) / wordBits;
        const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));

        bincv::BinMat<WordType> padded(width, height, PADDED_ALIGNMENT);
        fillRandom(padded, 0.5f, caseSeed(width, height, 1102));

        bincv::BinMat<WordType> clean(width, height);
        fillRandom(clean, 0.5f, caseSeed(width, height, 1103));
        const size_t wideStride = minWords + 2;
        std::vector<WordType> wideBuffer(wideStride * uh, 0);
        std::vector<WordType> dirtyBuffer(wideStride * uh, allOnes);
        for (size_t y = 0; y < uh; ++y) {
            const WordType* src = clean.constView().row(y);
            for (size_t i = 0; i < minWords; ++i) {
                wideBuffer[y * wideStride + i] = src[i];
                dirtyBuffer[y * wideStride + i] = src[i];
            }
            const size_t tail = uw % wordBits;
            if (tail != 0) {
                const WordType keep =
                    static_cast<WordType>((static_cast<WordType>(1) << tail) - 1);
                dirtyBuffer[y * wideStride + minWords - 1] = static_cast<WordType>(
                    dirtyBuffer[y * wideStride + minWords - 1] | static_cast<WordType>(~keep));
            }
        }
        const bincv::BinMatConstView<WordType> wide{wideBuffer.data(), uw, uh, wideStride};
        const bincv::BinMatConstView<WordType> dirty{dirtyBuffer.data(), uw, uh, wideStride};

        for (int W : {7, 31}) {
            sweepSlidingWindow<WordType>(padded.constView(), W, W,
                                         sizeLabel(wordTypeName, width, height) +
                                             " [sliding, over-aligned stride]");
            sweepSlidingWindow<WordType>(wide, W, W,
                                         sizeLabel(wordTypeName, width, height) +
                                             " [sliding, wide stride]");
            sweepSlidingWindow<WordType>(dirty, W, W,
                                         sizeLabel(wordTypeName, width, height) +
                                             " [sliding, dirty padding]");
        }
    }

    // Degenerate columns: nothing here may read a word, and every position counts
    // zero however many times it is slid.
    {
        const std::string label = std::string(wordTypeName) + " [sliding, degenerate]";
        const Rect none(0, 0, 0, 0);

        bincv::BinMat<WordType> m(40, 6);
        fillRandom(m, 1.0f, caseSeed(40, 6, 1104));
        const bincv::BinMatConstView<WordType> v = m.constView();

        const Rect degenerate[] = {
            Rect(0, 0, 0, 7),        // zero width
            Rect(0, 0, 7, 0),        // zero height
            Rect(0, 0, -3, 7),       // negative width
            Rect(0, 0, 7, -3),       // negative height
            Rect(-100, 0, 7, 7),     // wholly left
            Rect(40, 0, 7, 7),       // wholly right, touching the edge
            Rect(2147483000, 0, 2147483000, 7),   // x + width overflows int
            Rect(-2147483000, 0, 2000000, 7),     // far left, no wrap into the image
        };
        for (const Rect& r : degenerate) {
            SlidingWindowCount<WordType> acc(v, r);
            size_t nonZero = 0;
            for (int step = 0; step < 20; ++step) {
                if (acc.count() != 0) ++nonZero;
                acc.slideDown();
            }
            REDUCE_EXPECT_COUNT(nonZero, 0u, "degenerate column counts zero at every position", r,
                                label);
        }

        // A default-constructed view has no pixels and no pointer; the accumulator
        // must not dereference it.
        const bincv::BinMatConstView<WordType> nullView{};
        SlidingWindowCount<WordType> nullAcc(nullView, Rect(0, 0, 31, 31));
        REDUCE_EXPECT_COUNT(nullAcc.count(), 0u, "empty view, first window", none, label);
        nullAcc.slideDown();
        REDUCE_EXPECT_COUNT(nullAcc.count(), 0u, "empty view, after a slide", none, label);

        // window tracks y and nothing else, including through positions whose
        // count is zero.
        SlidingWindowCount<WordType> tracked(v, Rect(-3, -5, 9, 4));
        bool rectsOk = true;
        for (int step = 0; step < 25; ++step) {
            if (tracked.window() != Rect(-3, -5 + step, 9, 4)) rectsOk = false;
            tracked.slideDown();
        }
        REDUCE_EXPECT_TRUE(rectsOk, "window() advances by exactly one row per slide", "25 slides",
                           label);
    }
}

// ---------------------------------------------------------------------------
// 10. countCovariance and the four-argument split ( items 2 and 3)
// ---------------------------------------------------------------------------
//
// Two properties, and they are different claims:
//
// * the FUSED pass returns what the three-call COMPOSITION returns -- for every
// region in the curated list, and window for window over a whole frame at the
// three LK window sizes, edge-clipped positions included;
// * the FOUR-ARGUMENT selector returns what the precomputed XOR plane returns.
// `c0 ^ c1` is formed a word at a time inside the loop there, so the padding
// hazard is not the same one: a trailing word's padding bits of c0 and c1 are
// both the caller's, and their XOR is whatever it is. It may not reach a count.
//
// Both are also checked against the per-pixel references written before the
// kernels, so a fused pass and a composition that were wrong the same way could
// not agree their way past this.

template <typename WordType>
bool covarianceAgrees(const CovarianceCount& fused, const bincv::BinMatConstView<WordType>& a,
                      const bincv::BinMatConstView<WordType>& b,
                      const bincv::BinMatConstView<WordType>& c, const Rect& r) {
    const size_t xx = countNonZero(a, r);
    const size_t yy = countNonZero(b, r);
    const SplitCount xy = countAndSplit(a, b, c, r);
    const SplitCount ref = refCountAndSplit(a, b, c, r);
    return fused.xx == xx && fused.yy == yy && fused.xy.whenClear == xy.whenClear &&
           fused.xy.whenSet == xy.whenSet && fused.xx == refCountNonZero(a, r) &&
           fused.yy == refCountNonZero(b, r) && fused.xy.whenClear == ref.whenClear &&
           fused.xy.whenSet == ref.whenSet && fused.crossTerm() == xy.crossTerm();
}

template <typename WordType>
std::string covarianceText(const CovarianceCount& f) {
    return "xx=" + std::to_string(f.xx) + " yy=" + std::to_string(f.yy) + " whenClear=" +
           std::to_string(f.xy.whenClear) + " whenSet=" + std::to_string(f.xy.whenSet);
}

template <typename WordType>
void testFusedCovariance(const char* wordTypeName) {
    std::cout << "\n--- fused covariance and the four-argument split: " << wordTypeName << " ---\n";

    for (int width : {1, 7, 33, 64, 70, 129}) {
        for (int height : {1, 3, 13}) {
            const uint64_t seed = caseSeed(width, height, 1200);
            bincv::BinMat<WordType> a(width, height);
            bincv::BinMat<WordType> b(width, height);
            bincv::BinMat<WordType> c0(width, height);
            bincv::BinMat<WordType> c1(width, height);
            fillRandom(a, 0.5f, seed);
            fillRandom(b, 0.5f, seed ^ UINT64_C(0xDEADBEEF));
            fillRandom(c0, 0.5f, seed ^ UINT64_C(0x1111));
            fillRandom(c1, 0.5f, seed ^ UINT64_C(0x2222));
            bincv::BinMat<WordType> sel(width, height);
            bincv::bitwiseXor(c0.constView(), c1.constView(), sel.view());

            const bincv::BinMatConstView<WordType> va = a.constView();
            const bincv::BinMatConstView<WordType> vb = b.constView();
            const bincv::BinMatConstView<WordType> v0 = c0.constView();
            const bincv::BinMatConstView<WordType> v1 = c1.constView();
            const bincv::BinMatConstView<WordType> vs = sel.constView();
            const std::string label = sizeLabel(wordTypeName, width, height) + " [fused]";

            for (const Rect& r : valueRegions(width, height)) {
                const CovarianceCount fused = countCovariance(va, vb, vs, r);
                REDUCE_EXPECT_TRUE(covarianceAgrees<WordType>(fused, va, vb, vs, r),
                                   "fused covariance equals the composition and the reference",
                                   rectText(r) + " " + covarianceText<WordType>(fused), label);

                // The four-argument forms: identical to the plane forms, with no
                // plane in existence.
                const CovarianceCount xored = countCovariance(va, vb, v0, v1, r);
                REDUCE_EXPECT_TRUE(xored.xx == fused.xx && xored.yy == fused.yy &&
                                       xored.xy.whenClear == fused.xy.whenClear &&
                                       xored.xy.whenSet == fused.xy.whenSet,
                                   "four-argument covariance equals the plane form",
                                   rectText(r) + " " + covarianceText<WordType>(xored), label);

                const SplitCount planeSplit = countAndSplit(va, vb, vs, r);
                const SplitCount xorSplit = countAndSplit(va, vb, v0, v1, r);
                REDUCE_EXPECT_COUNT(xorSplit.whenClear, planeSplit.whenClear,
                                    "four-argument split whenClear", r, label);
                REDUCE_EXPECT_COUNT(xorSplit.whenSet, planeSplit.whenSet,
                                    "four-argument split whenSet", r, label);
            }

            // Aliasing is unrestricted (promise 3): every argument the same view.
            const Rect whole(0, 0, width, height);
            const CovarianceCount self = countCovariance(va, va, va, va, whole);
            REDUCE_EXPECT_COUNT(self.xy.whenClear, countNonZero(va),
                                "countCovariance(a, a, a, a): a ^ a is clear everywhere", whole,
                                label);
            REDUCE_EXPECT_COUNT(self.xy.whenSet, 0u, "countCovariance(a, a, a, a) whenSet", whole,
                                label);
            REDUCE_EXPECT_COUNT(self.xx, self.yy, "countCovariance(a, a, ...) xx equals yy", whole,
                                label);
        }
    }

    // Window for window over a whole frame, at the three LK window sizes, with the
    // centers placed so that every window on the border is clipped. This is the
    // half of the requirement the curated region list does not cover: the region
    // list is chosen, and a sweep is not.
    {
        const int width = 70;
        const int height = 23;
        const uint64_t seed = caseSeed(width, height, 1201);
        bincv::BinMat<WordType> a(width, height);
        bincv::BinMat<WordType> b(width, height);
        bincv::BinMat<WordType> c0(width, height);
        bincv::BinMat<WordType> c1(width, height);
        fillRandom(a, 0.5f, seed);
        fillRandom(b, 0.37f, seed ^ UINT64_C(0xABCD));
        fillRandom(c0, 0.5f, seed ^ UINT64_C(0x1111));
        fillRandom(c1, 0.5f, seed ^ UINT64_C(0x2222));
        bincv::BinMat<WordType> sel(width, height);
        bincv::bitwiseXor(c0.constView(), c1.constView(), sel.view());

        const bincv::BinMatConstView<WordType> va = a.constView();
        const bincv::BinMatConstView<WordType> vb = b.constView();
        const bincv::BinMatConstView<WordType> v0 = c0.constView();
        const bincv::BinMatConstView<WordType> v1 = c1.constView();
        const bincv::BinMatConstView<WordType> vs = sel.constView();
        const std::string label =
            sizeLabel(wordTypeName, width, height) + " [fused, full-frame sweep]";

        for (int W : {7, 15, 31}) {
            for (int y = -W; y <= height; ++y) {
                size_t bad = 0;
                std::string firstBad;
                for (int x = -W; x <= width; ++x) {
                    const Rect w(x, y, W, W);
                    const CovarianceCount fused = countCovariance(va, vb, vs, w);
                    const CovarianceCount xored = countCovariance(va, vb, v0, v1, w);
                    const SplitCount xorSplit = countAndSplit(va, vb, v0, v1, w);
                    const bool ok = covarianceAgrees<WordType>(fused, va, vb, vs, w) &&
                                    xored.xx == fused.xx && xored.yy == fused.yy &&
                                    xored.xy.whenClear == fused.xy.whenClear &&
                                    xored.xy.whenSet == fused.xy.whenSet &&
                                    xorSplit.whenClear == fused.xy.whenClear &&
                                    xorSplit.whenSet == fused.xy.whenSet;
                    if (!ok) {
                        ++bad;
                        if (firstBad.empty()) {
                            firstBad = rectText(w) + " fused " + covarianceText<WordType>(fused) +
                                       " four-arg " + covarianceText<WordType>(xored);
                        }
                    }
                }
                REDUCE_EXPECT_TRUE(bad == 0,
                                   "fused and four-argument forms agree with the composition at "
                                   "every window of the row",
                                   "W=" + std::to_string(W) + " y=" + std::to_string(y) +
                                       " mismatches=" + std::to_string(bad) + " first: " + firstBad,
                                   label);
            }
        }
    }

    // Degenerate: an empty view and an empty region both give four zeros, and
    // nothing is dereferenced.
    {
        const std::string label = std::string(wordTypeName) + " [fused, degenerate]";
        const Rect none(0, 0, 0, 0);
        const bincv::BinMatConstView<WordType> nullView{};
        const CovarianceCount empty = countCovariance(nullView, nullView, nullView, Rect(0, 0, 9, 9));
        REDUCE_EXPECT_COUNT(empty.xx + empty.yy + empty.xy.whenClear + empty.xy.whenSet, 0u,
                            "empty view gives four zeros", none, label);
        const CovarianceCount empty4 =
            countCovariance(nullView, nullView, nullView, nullView, Rect(0, 0, 9, 9));
        REDUCE_EXPECT_COUNT(empty4.xx + empty4.yy + empty4.xy.whenClear + empty4.xy.whenSet, 0u,
                            "empty view gives four zeros, four-argument form", none, label);
        REDUCE_EXPECT_TRUE(empty.crossTerm() == 0, "an empty cross term is zero", "0", label);
    }
}

// ---------------------------------------------------------------------------
// The OpenCV half: Tier 1 for countNonZero
// ---------------------------------------------------------------------------

#ifdef BINCV_WITH_OPENCV

/// @brief countNonZero against cv::countNonZero over the full the matrix.
/// @note OpenCV's side is built by randomCvMask -- the harness's INDEPENDENT
/// generator, which never constructs a BinMat and never runs the unpacking
/// path. That is property 2 applied here: if both sides went through
/// toCvMask, a fault in the conversion would land on both and cancel, and
/// the comparison would pass while proving nothing.
template <typename WordType>
void testOpenCvEquivalence(const char* wordTypeName) {
    std::cout << "\n--- countNonZero vs cv::countNonZero: " << wordTypeName << " ---\n";

    for (int width : bincv::test::equivalenceWidths()) {
        for (int height : bincv::test::equivalenceHeights()) {
            for (float fill : bincv::test::equivalenceFillRatios()) {
                const uint64_t seed = caseSeed(width, height, static_cast<size_t>(fill * 100.0f));
                const bincv::BinMat<WordType> m =
                    bincv::test::randomBinary<WordType>(width, height, fill, seed);
                const cv::Mat mask = bincv::test::randomCvMask(width, height, fill, seed);
                const bincv::BinMatConstView<WordType> v = m.constView();
                const std::string label = bincv::test::caseLabel(wordTypeName, width, height, fill);
                const Rect whole(0, 0, width, height);

                REDUCE_EXPECT_COUNT(countNonZero(v), static_cast<size_t>(cv::countNonZero(mask)),
                                    "whole image vs cv::countNonZero", whole, label);

                // The region overload against the expression a user writes today:
                // cv::countNonZero over a submatrix.
                for (const Rect& r : valueRegions(width, height)) {
                    const Clipped c = clipReference(width, height, r);
                    size_t expected = 0;
                    if (!c.empty()) {
                        const cv::Mat roi = mask(cv::Rect(c.x0, c.y0, c.x1 - c.x0, c.y1 - c.y0));
                        expected = static_cast<size_t>(cv::countNonZero(roi));
                    }
                    REDUCE_EXPECT_COUNT(countNonZero(v, r), expected, "vs cv::countNonZero(roi)", r,
                                        label);
                }
            }
        }
    }
}

/// @brief The masked reductions against the OpenCV composite they replace.
/// @note countAnd and countAndSplit are Tier 3 -- OpenCV has no equivalent -- but
/// the composite it would take to get the same number does exist:
/// cv::bitwise_and into a temporary, then cv::countNonZero. This is not a
/// Tier 1 promise; it is a second independent reference built out of code
/// that shares nothing with the word arithmetic under test.
template <typename WordType>
void testOpenCvMaskedComposite(const char* wordTypeName) {
    std::cout << "\n--- countAnd / countAndSplit vs an OpenCV composite: " << wordTypeName
              << " ---\n";

    for (int width : bincv::test::equivalenceWidths()) {
        for (int height : {1, 3, 17}) {
            for (float fill : {0.01f, 0.5f, 0.99f}) {
                const uint64_t seed = caseSeed(width, height, 300);
                const bincv::BinMat<WordType> a =
                    bincv::test::randomBinary<WordType>(width, height, fill, seed);
                const bincv::BinMat<WordType> b = bincv::test::randomBinary<WordType>(
                    width, height, fill, seed ^ UINT64_C(0xDEADBEEF));
                const bincv::BinMat<WordType> c = bincv::test::randomBinary<WordType>(
                    width, height, 0.5f, seed ^ UINT64_C(0x5A5A));
                const cv::Mat cvA = bincv::test::randomCvMask(width, height, fill, seed);
                const cv::Mat cvB =
                    bincv::test::randomCvMask(width, height, fill, seed ^ UINT64_C(0xDEADBEEF));
                const cv::Mat cvC =
                    bincv::test::randomCvMask(width, height, 0.5f, seed ^ UINT64_C(0x5A5A));

                cv::Mat both;
                cv::bitwise_and(cvA, cvB, both);
                cv::Mat bothAndC;
                cv::bitwise_and(both, cvC, bothAndC);

                const std::string label = bincv::test::caseLabel(wordTypeName, width, height, fill);
                for (const Rect& r : valueRegions(width, height)) {
                    const Clipped clip = clipReference(width, height, r);
                    size_t expectedAnd = 0;
                    size_t expectedSet = 0;
                    if (!clip.empty()) {
                        const cv::Rect roi(clip.x0, clip.y0, clip.x1 - clip.x0, clip.y1 - clip.y0);
                        expectedAnd = static_cast<size_t>(cv::countNonZero(both(roi)));
                        expectedSet = static_cast<size_t>(cv::countNonZero(bothAndC(roi)));
                    }
                    REDUCE_EXPECT_COUNT(countAnd(a.constView(), b.constView(), r), expectedAnd,
                                        "countAnd vs cv composite", r, label);
                    const SplitCount split =
                        countAndSplit(a.constView(), b.constView(), c.constView(), r);
                    REDUCE_EXPECT_COUNT(split.whenSet, expectedSet, "whenSet vs cv composite", r,
                                        label);
                    REDUCE_EXPECT_COUNT(split.whenClear, expectedAnd - expectedSet,
                                        "whenClear vs cv composite", r, label);
                }
            }
        }
    }
}

#endif // BINCV_WITH_OPENCV

} // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

BINCV_TEST(Reduce, Reference_uint8_t)  { testCountNonZeroReference<uint8_t>("uint8_t"); }
BINCV_TEST(Reduce, Reference_uint16_t) { testCountNonZeroReference<uint16_t>("uint16_t"); }
BINCV_TEST(Reduce, Reference_uint32_t) { testCountNonZeroReference<uint32_t>("uint32_t"); }
BINCV_TEST(Reduce, Reference_uint64_t) { testCountNonZeroReference<uint64_t>("uint64_t"); }

BINCV_TEST(Reduce, Masked_uint8_t)  { testMaskedReference<uint8_t>("uint8_t"); }
BINCV_TEST(Reduce, Masked_uint16_t) { testMaskedReference<uint16_t>("uint16_t"); }
BINCV_TEST(Reduce, Masked_uint32_t) { testMaskedReference<uint32_t>("uint32_t"); }
BINCV_TEST(Reduce, Masked_uint64_t) { testMaskedReference<uint64_t>("uint64_t"); }

BINCV_TEST(Reduce, Geometry_uint8_t)  { testGeometry<uint8_t>("uint8_t"); }
BINCV_TEST(Reduce, Geometry_uint16_t) { testGeometry<uint16_t>("uint16_t"); }
BINCV_TEST(Reduce, Geometry_uint32_t) { testGeometry<uint32_t>("uint32_t"); }
BINCV_TEST(Reduce, Geometry_uint64_t) { testGeometry<uint64_t>("uint64_t"); }

BINCV_TEST(Reduce, DirtyPadding_uint8_t)  { testDirtyPadding<uint8_t>("uint8_t"); }
BINCV_TEST(Reduce, DirtyPadding_uint16_t) { testDirtyPadding<uint16_t>("uint16_t"); }
BINCV_TEST(Reduce, DirtyPadding_uint32_t) { testDirtyPadding<uint32_t>("uint32_t"); }
BINCV_TEST(Reduce, DirtyPadding_uint64_t) { testDirtyPadding<uint64_t>("uint64_t"); }

BINCV_TEST(Reduce, Strides_uint8_t)  { testStrides<uint8_t>("uint8_t"); }
BINCV_TEST(Reduce, Strides_uint16_t) { testStrides<uint16_t>("uint16_t"); }
BINCV_TEST(Reduce, Strides_uint32_t) { testStrides<uint32_t>("uint32_t"); }
BINCV_TEST(Reduce, Strides_uint64_t) { testStrides<uint64_t>("uint64_t"); }

BINCV_TEST(Reduce, Degenerate_uint8_t)  { testDegenerate<uint8_t>("uint8_t"); }
BINCV_TEST(Reduce, Degenerate_uint16_t) { testDegenerate<uint16_t>("uint16_t"); }
BINCV_TEST(Reduce, Degenerate_uint32_t) { testDegenerate<uint32_t>("uint32_t"); }
BINCV_TEST(Reduce, Degenerate_uint64_t) { testDegenerate<uint64_t>("uint64_t"); }

BINCV_TEST(Reduce, PortablePopcount_uint8_t)  { testPortablePopcount<uint8_t>("uint8_t"); }
BINCV_TEST(Reduce, PortablePopcount_uint16_t) { testPortablePopcount<uint16_t>("uint16_t"); }
BINCV_TEST(Reduce, PortablePopcount_uint32_t) { testPortablePopcount<uint32_t>("uint32_t"); }
BINCV_TEST(Reduce, PortablePopcount_uint64_t) { testPortablePopcount<uint64_t>("uint64_t"); }

BINCV_TEST(Reduce, Covariance_uint8_t)  { testCovarianceIdentity<uint8_t>("uint8_t"); }
BINCV_TEST(Reduce, Covariance_uint16_t) { testCovarianceIdentity<uint16_t>("uint16_t"); }
BINCV_TEST(Reduce, Covariance_uint32_t) { testCovarianceIdentity<uint32_t>("uint32_t"); }
BINCV_TEST(Reduce, Covariance_uint64_t) { testCovarianceIdentity<uint64_t>("uint64_t"); }

BINCV_TEST(Reduce, Sliding_uint8_t)  { testSlidingWindow<uint8_t>("uint8_t"); }
BINCV_TEST(Reduce, Sliding_uint16_t) { testSlidingWindow<uint16_t>("uint16_t"); }
BINCV_TEST(Reduce, Sliding_uint32_t) { testSlidingWindow<uint32_t>("uint32_t"); }
BINCV_TEST(Reduce, Sliding_uint64_t) { testSlidingWindow<uint64_t>("uint64_t"); }

BINCV_TEST(Reduce, Fused_uint8_t)  { testFusedCovariance<uint8_t>("uint8_t"); }
BINCV_TEST(Reduce, Fused_uint16_t) { testFusedCovariance<uint16_t>("uint16_t"); }
BINCV_TEST(Reduce, Fused_uint32_t) { testFusedCovariance<uint32_t>("uint32_t"); }
BINCV_TEST(Reduce, Fused_uint64_t) { testFusedCovariance<uint64_t>("uint64_t"); }

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Reduce, OpenCv_uint8_t)  { testOpenCvEquivalence<uint8_t>("uint8_t"); }
BINCV_TEST(Reduce, OpenCv_uint16_t) { testOpenCvEquivalence<uint16_t>("uint16_t"); }
BINCV_TEST(Reduce, OpenCv_uint32_t) { testOpenCvEquivalence<uint32_t>("uint32_t"); }
BINCV_TEST(Reduce, OpenCv_uint64_t) { testOpenCvEquivalence<uint64_t>("uint64_t"); }

BINCV_TEST(Reduce, OpenCvComposite_uint8_t)  { testOpenCvMaskedComposite<uint8_t>("uint8_t"); }
BINCV_TEST(Reduce, OpenCvComposite_uint16_t) { testOpenCvMaskedComposite<uint16_t>("uint16_t"); }
BINCV_TEST(Reduce, OpenCvComposite_uint32_t) { testOpenCvMaskedComposite<uint32_t>("uint32_t"); }
BINCV_TEST(Reduce, OpenCvComposite_uint64_t) { testOpenCvMaskedComposite<uint64_t>("uint64_t"); }
#endif

BINCV_TEST_MAIN("BinMat reduction kernel tests")
