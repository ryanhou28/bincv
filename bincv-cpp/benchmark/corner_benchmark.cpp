// -- the corner response map: THE SLIDING FORM AGAINST A COVARIANCE CALL PER
// POSITION.
//
// WHY THIS FILE EXISTS WHEN ALREADY MEASURED "incremental against recompute"
//
// the axis 1 measured that question on ops/reduce.hpp's own entry points: a
// `SlidingWindowCount` sweep against a `countNonZero` per position, one plane, one
// number. It reported 15.9x on a dense scan at 31x31 and that number is quoted in
// three docstrings. this is the first CALLER of that shape in the MVP, and what it
// sweeps is NOT one plane's popcount -- it is a 2x2 covariance of which only TWO
// of the three numbers have an incremental form. `sumXX` and `sumYY` slide;
// `sumXY` needs `magX & magY` split by `signX ^ signY`, nothing in ops/reduce.hpp
// slides a split, and making it slide would cost two frame-sized planes per
// pyramid level (the axis 3 already declined one). So the cross term is
// recomputed per position on BOTH sides of this comparison, and the saving is
// bounded by the share of the work the other two numbers represent.
//
// "ON BOTH SIDES" IS TRUE OF THE WORK, NOT OF THE ENTRY POINT, and the difference
// is what the table's incremental column actually measures:
//
// sliding two slid row counts (`SlidingWindowCount`, one plane each) PLUS
// `countAndSplit(magX, magY, signX, signY, window)` -- TWO popcounts
// per word (ops/reduce.hpp's crossing table).
// recompute `gradientCovariance` -> `countCovariance(magX, magY, signX, signY,
// window)` -- FOUR popcounts per word, producing xx, yy AND the
// split in ONE pass. The cross term is never issued separately there;
// it is fused into the pass that also produces xx and yy.
//
// So the recompute side is not "the sliding side plus two extra row counts", and
// saying it that way would misread the column. It is a different reduction with a
// different per-word cost, and the incremental column prices exactly that delta:
// two slid row counts and a 2-popcount pass against one 4-popcount pass.
//
// That bound is the whole point. A reader who takes 15.9x from ops/reduce.hpp and
// expects it here would be wrong, and the only way to say so honestly is to
// measure the ratio at THIS level, on the entry point actually ships.
//
// THE RULE, WRITTEN BEFORE MEASURING (CLAUDE.md: "write the decision rule before
// measuring"):
//
// * SLIDING FASTER THAN RECOMPUTE AT EVERY BLOCK SIZE -> the axis 1's advantage
// survives being embedded in a caller that can only slide two thirds of its
// state. ops/corner.hpp's "this is the sliding form" note stands, and the
// magnitude recorded here -- not 15.9x -- is what a caller should plan with.
// * SLIDING WITHIN THE MEASURED SPREAD OF RECOMPUTE, OR SLOWER -> that
// CONTRADICTS a documented claim (, ops/reduce.hpp's "WHICH SHAPE TO REACH
// FOR" table, ops/covariance.hpp's docstring, and that work’s own spec, all of
// which point a dense sweep at the incremental form). CLAUDE.md's rule for
// that case is explicit: report it, do not adjust the code to fit the doc. The
// conclusion would be that the sliding form is not worth its complexity in
// THIS caller, and ops/corner.hpp would need re-deciding rather than
// re-measuring.
// * A RATIO NEAR 15.9x WOULD ALSO BE A SURPRISE and would mean the cross term is
// not the dominant cost the argument above assumes. It is written down here so
// that it cannot be quietly welcomed as a good result.
//
// No threshold is attached, deliberately: this is confirming the direction of an
// interface already selected, so the question is direction and magnitude against
// the measured spread, not a gate.
//
// WHAT IS MEASURED
//
// SLIDING ops/corner.hpp's cornerMinEigenVal -- WHAT SHIPS. Per column,
// two SlidingWindowCounts carry sumXX and sumYY; per position, one
// four-argument countAndSplit for the cross term.
// RECOMPUTE the same map, with `gradientCovariance(dx, dy, window)` called per
// position -- that work’s own entry point, which is the obvious way to
// write this operation and is what ops/covariance.hpp's docstring
// tells a dense caller NOT to do. All three numbers are recomputed
// over the whole window at every pixel, ROW-MAJOR, which is how
// anyone would write it.
// RECOMPUTE-COL
// the same recomputation swept COLUMN-MAJOR. It exists because
// `sliding` differs from `recompute` in TWO ways at once -- the
// incremental state AND the traversal order the accumulator forces
// -- and those pull in opposite directions. Without this control the
// table cannot say which effect it measured, and on a 32 KiB L1 the
// traversal effect is not small.
//
// All three produce the same map, and the maps are compared BIT FOR BIT on EVERY
// ONE OF THE FOUR ROTATING INPUTS before anything is timed -- the timed bodies
// rotate, so a guard on one frame would not cover what the table reports. A faster
// wrong answer is not a result.
//
// MEMORY IS REPORTED BESIDE SPEED, and here it is the flat half of the table --
// which is itself the finding. Both variants read the same four one-bit planes and
// write the same float map; the sliding form's entire state is two accumulator
// objects on the stack. So this is speed against speed at EQUAL footprint, with
// nothing traded, which is the cleanest form CLAUDE.md's "report both" can take.
// The `operator new` count is measured on the same binary rather than asserted.
//
// Validity: measure_util.hpp's protocol -- volatile sink, calibrated batches,
// interleaved variants, spread reported next to the median.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/corner.hpp"
#include "bincv-cpp/ops/covariance.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "measure_util.hpp"

// ---------------------------------------------------------------------------
// The allocation counter, so the memory column is a READING on the same binary
// that produced the speed column. Includes the C++17 over-aligned forms -- scratch
// for a vectorized kernel takes exactly that path, and a counter replacing only
// the plain pair cannot see it (tests/test_covariance.cpp measured that).
// ---------------------------------------------------------------------------
namespace {
std::size_t g_newCount = 0;

void* benchAllocate(std::size_t bytes) {
    ++g_newCount;
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    // Cannot throw std::bad_alloc: this file also builds with -fno-exceptions.
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) std::abort();
    return p;
}

void* benchAllocateAligned(std::size_t bytes, std::size_t alignment) {
    ++g_newCount;
    // The bound is not defensive decoration: without it gcc's
    // -Walloc-size-larger-than proves `rounded` can reach SIZE_MAX and FAILS the
    // -fno-exceptions configuration, which is the one build that inlines far
    // enough to see it. tests/test_covariance.cpp carries the same line.
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    const std::size_t wanted = (bytes == 0) ? 1 : bytes;
    const std::size_t rounded = ((wanted + alignment - 1) / alignment) * alignment;
    void* p = std::aligned_alloc(alignment, rounded);
    if (p == nullptr) std::abort();
    return p;
}

void benchFree(void* p) noexcept { std::free(p); }
} // namespace

void* operator new(std::size_t bytes)   { return benchAllocate(bytes); }
void* operator new[](std::size_t bytes) { return benchAllocate(bytes); }
void operator delete(void* p) noexcept                { benchFree(p); }
void operator delete[](void* p) noexcept              { benchFree(p); }
void operator delete(void* p, std::size_t) noexcept   { benchFree(p); }
void operator delete[](void* p, std::size_t) noexcept { benchFree(p); }

void* operator new(std::size_t bytes, std::align_val_t a) {
    return benchAllocateAligned(bytes, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t bytes, std::align_val_t a) {
    return benchAllocateAligned(bytes, static_cast<std::size_t>(a));
}
void operator delete(void* p, std::align_val_t) noexcept                { benchFree(p); }
void operator delete[](void* p, std::align_val_t) noexcept              { benchFree(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept   { benchFree(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { benchFree(p); }

namespace {

using Word = uint32_t;  // the design rule’s default, and what a VIO frontend would run

constexpr int kWidth = 640;
constexpr int kHeight = 480;
const int kBlockSizes[] = {3, 7, 15, 31};

/// @brief THE AVOIDABLE COST: the same response map, with that work’s covariance called
/// once per pixel instead of two accumulators slid down each column.
///
/// This is not a straw man -- it is the natural way to write the operation once
/// ops/covariance.hpp exists, and it is exactly what that file's docstring tells a
/// dense caller not to do.
///
/// It does NOT call the same reduction the sliding form calls. `gradientCovariance`
/// goes to `countCovariance` -- four popcounts per word, all three numbers in one
/// pass -- where the sliding form issues `countAndSplit` (two popcounts per word)
/// plus two slid row counts. That pair is the delta the "incremental effect" column
/// isolates, and it is stated here rather than glossed as "the same call" so the
/// comparison is auditable.
void responseMapByRecompute(const bincv::TernaryMat<Word>& dx, const bincv::TernaryMat<Word>& dy,
                            int blockSize, bincv::ResponseMap dst) {
    const int width = static_cast<int>(dx.cols());
    const int height = static_cast<int>(dx.rows());
    const int off = blockSize / 2;
    for (int y = 0; y < height; ++y) {
        float* row = dst.row(static_cast<size_t>(y));
        for (int x = 0; x < width; ++x) {
            const bincv::GradientCovariance c =
                bincv::gradientCovariance(dx, dy, bincv::Rect(x - off, y - off, blockSize,
                                                              blockSize));
            row[static_cast<size_t>(x)] =
                bincv::impl::minEigenValue(c.sumXX, c.sumYY, c.sumXY);
        }
    }
}

/// @brief The same recomputation, swept COLUMN-MAJOR -- the traversal order the
/// sliding form is forced into, because SlidingWindowCount only slides
/// downward.
///
/// WITHOUT THIS VARIANT THE TABLE MEASURES TWO THINGS AT ONCE and cannot say
/// which. `sliding` differs from `recompute` in BOTH the incremental state and
/// the traversal order, and those pull in opposite directions: sliding saves
/// popcounts, column-major costs locality (the four planes are 153 600 B against a
/// 32 KiB L1 on the reference device, and a column sweep re-walks all of them per
/// column). Three variants make the two separable:
///
/// sliding / recompute-col the INCREMENTAL effect, traversal held equal
/// recompute-col / recompute the TRAVERSAL effect, arithmetic held equal
/// sliding / recompute what a caller actually chooses between
void responseMapByRecomputeColumnMajor(const bincv::TernaryMat<Word>& dx,
                                       const bincv::TernaryMat<Word>& dy, int blockSize,
                                       bincv::ResponseMap dst) {
    const int width = static_cast<int>(dx.cols());
    const int height = static_cast<int>(dx.rows());
    const int off = blockSize / 2;
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            const bincv::GradientCovariance c =
                bincv::gradientCovariance(dx, dy, bincv::Rect(x - off, y - off, blockSize,
                                                              blockSize));
            dst.row(static_cast<size_t>(y))[static_cast<size_t>(x)] =
                bincv::impl::minEigenValue(c.sumXX, c.sumYY, c.sumXY);
        }
    }
}

/// @brief A frame with real corner structure rather than salt-and-pepper noise:
/// overlapping rectangles and a diagonal, so the map has edges (rank one,
/// response 0), corners (rank two) and flat regions.
bincv::BinMat<Word> makeFrame(uint64_t seed) {
    bincv::BinMat<Word> src(kWidth, kHeight);
    uint64_t state = seed;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const unsigned block = static_cast<unsigned>((x / 37) + (y / 29)) % 2u;
            const unsigned diag = (x > y + 40) ? 1u : 0u;
            unsigned v = block ^ diag;
            if ((measure::nextRandom(state) & 63ULL) == 0ULL) v ^= 1u;  // sparse texture
            src.set(y, x, v);
        }
    }
    return src;
}

size_t planeBytes(const bincv::TernaryMat<Word>& t) {
    // Two planes per ternary image (one magnitude, one sign), each stride x height
    // words. Read out of the view rather than recomputed from the dimensions, so
    // row alignment is included.
    const bincv::BinMatConstView<Word> p = t.constMagnitude(0);
    return 2 * p.stride * p.height * sizeof(Word);
}

} // namespace

int main() {
    std::printf("binCV -- corner response: sliding against a covariance call per position\n");
    std::printf("frame %dx%d, word uint32_t, one column of sliding state, no caller scratch\n\n",
                kWidth, kHeight);

    // Four rotating inputs -- validity hazard 2 in measure_util.hpp.
    constexpr int kInputs = 4;
    std::vector<bincv::BinMat<Word>> sources;
    std::vector<bincv::TernaryMat<Word>> dxs, dys;
    sources.reserve(kInputs);
    dxs.reserve(kInputs);
    dys.reserve(kInputs);
    for (int i = 0; i < kInputs; ++i) {
        sources.push_back(makeFrame(uint64_t{0x9E3779B9} + static_cast<uint64_t>(i) * uint64_t{7919}));
        dxs.emplace_back(kWidth, kHeight);
        dys.emplace_back(kWidth, kHeight);
        bincv::derivativeX(sources.back(), dxs.back());
        bincv::derivativeY(sources.back(), dys.back());
    }

    std::vector<float> mapA(static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight), 0.0f);
    std::vector<float> mapB(mapA.size(), 0.0f);
    std::vector<float> mapC(mapA.size(), 0.0f);
    bincv::ResponseMap viewA{mapA.data(), static_cast<size_t>(kWidth),
                             static_cast<size_t>(kHeight), static_cast<size_t>(kWidth)};
    bincv::ResponseMap viewB{mapB.data(), static_cast<size_t>(kWidth),
                             static_cast<size_t>(kHeight), static_cast<size_t>(kWidth)};
    bincv::ResponseMap viewC{mapC.data(), static_cast<size_t>(kWidth),
                             static_cast<size_t>(kHeight), static_cast<size_t>(kWidth)};

    const size_t pixels = static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight);
    const size_t planes = planeBytes(dxs[0]) + planeBytes(dys[0]);
    const size_t mapBytes = pixels * sizeof(float);

    std::printf(" working set, identical for both variants:\n");
    std::printf(" four one-bit derivative planes %9zu B\n", planes);
    std::printf(" float response map %9zu B\n", mapBytes);
    std::printf(" caller scratch 0 B (two stack accumulators)\n");
    std::printf(" TOTAL %9zu B\n\n", planes + mapBytes);

    std::printf(" %-6s %-14s %10s %10s %8s %8s %s\n", "block", "variant", "ns/frame",
                "ns/pixel", "spread", "vs.slide", "allocs");

    for (int blockSize : kBlockSizes) {
        // Correctness first: a faster wrong answer is not a result. EVERY INPUT
        // THAT IS TIMED IS ALSO CHECKED -- the timed bodies rotate over all four
        // frames (`i % kInputs`), so gating on frame 0 alone would let a variant
        // that disagrees only on frames 1..3 through the guard and into the table.
        // The two counters are separate so the message names which variant diverged.
        size_t differingB = 0, differingC = 0;
        for (int k = 0; k < kInputs; ++k) {
            const size_t ki = static_cast<size_t>(k);
            bincv::cornerMinEigenVal(dxs[ki], dys[ki], blockSize, viewA);
            responseMapByRecompute(dxs[ki], dys[ki], blockSize, viewB);
            responseMapByRecomputeColumnMajor(dxs[ki], dys[ki], blockSize, viewC);
            for (size_t i = 0; i < mapA.size(); ++i) {
                if (mapA[i] != mapB[i]) ++differingB;
                if (mapA[i] != mapC[i]) ++differingC;
            }
        }
        if (differingB != 0 || differingC != 0) {
            std::printf(" MISMATCH at block %d over %d inputs: recompute %zu, recompute-col %zu "
                        "of %zu positions differ -- not timing\n",
                        blockSize, kInputs, differingB, differingC, mapA.size() * kInputs);
            return 1;
        }

        // The allocation reading is taken around ONE call of each variant, before
        // the harness's own vectors and std::functions are built -- those allocate
        // (measured: 10 per block size) and would otherwise be charged to the
        // kernel, which is precisely the kind of number that gets quoted later.
        std::size_t slidingAllocs = 0, recomputeAllocs = 0, recomputeColAllocs = 0;
        {
            const std::size_t before = g_newCount;
            bincv::cornerMinEigenVal(dxs[0], dys[0], blockSize, viewA);
            slidingAllocs = g_newCount - before;
        }
        {
            const std::size_t before = g_newCount;
            responseMapByRecompute(dxs[0], dys[0], blockSize, viewB);
            recomputeAllocs = g_newCount - before;
        }
        {
            const std::size_t before = g_newCount;
            responseMapByRecomputeColumnMajor(dxs[0], dys[0], blockSize, viewC);
            recomputeColAllocs = g_newCount - before;
        }

        std::vector<measure::Bench> benches;
        benches.push_back({"sliding", [&](int i) {
                               const int k = i % kInputs;
                               bincv::cornerMinEigenVal(dxs[static_cast<size_t>(k)],
                                                        dys[static_cast<size_t>(k)], blockSize,
                                                        viewA);
                               measure::g_sink += static_cast<size_t>(mapA[0]);
                           }});
        benches.push_back({"recompute", [&](int i) {
                               const int k = i % kInputs;
                               responseMapByRecompute(dxs[static_cast<size_t>(k)],
                                                      dys[static_cast<size_t>(k)], blockSize,
                                                      viewB);
                               measure::g_sink += static_cast<size_t>(mapB[0]);
                           }});
        benches.push_back({"recompute-col", [&](int i) {
                               const int k = i % kInputs;
                               responseMapByRecomputeColumnMajor(dxs[static_cast<size_t>(k)],
                                                                 dys[static_cast<size_t>(k)],
                                                                 blockSize, viewC);
                               measure::g_sink += static_cast<size_t>(mapC[0]);
                           }});

        const std::vector<measure::Timing> t = measure::measureInterleaved(benches, 5, 200.0);

        const double slidingNs = t[0].medianNs;
        const double recomputeNs = t[1].medianNs;
        const double recomputeColNs = t[2].medianNs;
        std::printf(" %-6d %-14s %10.0f %10.3f %7.2f%% %8s %6zu\n", blockSize, "sliding",
                    slidingNs, slidingNs / static_cast<double>(pixels), t[0].spreadPct(), "-",
                    slidingAllocs);
        std::printf(" %-6d %-14s %10.0f %10.3f %7.2f%% %7.2fx %6zu\n", blockSize, "recompute",
                    recomputeNs, recomputeNs / static_cast<double>(pixels), t[1].spreadPct(),
                    recomputeNs / slidingNs, recomputeAllocs);
        std::printf(" %-6d %-14s %10.0f %10.3f %7.2f%% %7.2fx %6zu\n", blockSize,
                    "recompute-col", recomputeColNs,
                    recomputeColNs / static_cast<double>(pixels), t[2].spreadPct(),
                    recomputeColNs / slidingNs, recomputeColAllocs);
        std::printf(" -> incremental effect (sliding vs recompute-col) %6.2fx, "
                    "traversal effect (recompute vs recompute-col) %6.2fx\n",
                    recomputeColNs / slidingNs, recomputeColNs / recomputeNs);
    }

    std::printf("\n the `allocs` column is `operator new` calls -- plain AND C++17 "
                "over-aligned --\n");
    std::printf(" counted around ONE call of each variant. Zero is what no scratch means.\n");
    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
