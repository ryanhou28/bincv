// T3.11 / E-10 -- EXPERIMENTS.md X-23: the rolling response ring against the
// frame-sized response map.
//
// WHAT THIS MEASURES AND WHY THE RULE WAS WRITTEN FIRST
//
// X-20 measured the VIO frontend's peak working set at 640x480 as 1 721 568 B, of
// which T3.7's `float` response map is 1 228 800 B -- 71.4%, more than everything
// else combined, at 4 BYTES per pixel where every other plane is one or two BITS.
// Replacing it with a three-row ring is a speed/footprint conflict with no prior
// decision, so it follows CLAUDE.md's experiment loop: the decision rule is
// committed BEFORE the streaming form exists (X-23, commit 79db8f8), both sides
// are measured, and only then is the choice taken.
//
// THREE ARMS, EACH IN ITS OWN TRANSLATION UNIT. See corner_streaming_arms.hpp --
// the layout hazard is measured twice in this repository (~10% between two
// instantiations in one TU; 1.46x between binaries) and an A/B taken inside one
// object file would be an artefact of it. This file only drives them.
//
// WHAT IT PRINTS, IN THE ORDER X-23 ASKS FOR
//
//   1. CORRECTNESS FIRST, AND IT IS A PRECONDITION RATHER THAN A COLUMN. Every
//      arm's answer is compared against the control's element for element -- the
//      whole ranked prefix, coordinates and exact float bits, plus all three
//      CornerResult fields -- for every frame, block size and word type that is
//      subsequently timed. A mismatch stops the run; it does not print a slower
//      table with a footnote.
//   2. THE TRUE PEAK of each form: the frame map, or the ring PLUS everything
//      carried to preserve the selection's global properties. Counted the way
//      X-20 counted it, with the measurement-only buffers scoped and destroyed
//      before the accounting point.
//   3. ns/pixel and ms/frame for the WHOLE detector and for the response stage
//      alone, medians of 11 interleaved batches with the spread beside them.
//   4. The decision cell re-run with the arm ORDER SWAPPED. A verdict that moves
//      when the order moves is layout, not a result.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "corner_streaming_arms.hpp"
#include "measure_util.hpp"

// ---------------------------------------------------------------------------
// The allocation counter, so the memory column is a READING on the same binary
// that produced the speed column. Includes the C++17 over-aligned forms.
// ---------------------------------------------------------------------------
namespace {
std::size_t g_newCount = 0;

void* benchAllocate(std::size_t bytes) {
    ++g_newCount;
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) std::abort();
    return p;
}

void* benchAllocateAligned(std::size_t bytes, std::size_t alignment) {
    ++g_newCount;
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    const std::size_t wanted = (bytes == 0) ? 1 : bytes;
    const std::size_t rounded = ((wanted + alignment - 1) / alignment) * alignment;
    void* p = std::aligned_alloc(alignment, rounded);
    if (p == nullptr) std::abort();
    return p;
}

void benchFree(void* p) noexcept { std::free(p); }
}  // namespace

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

using bincv::Corner;
using bincv::CornerResult;
using bincv::GoodFeaturesParams;
using bincv::ResponseMap;

constexpr int kInputs = 4;
const int kBlockSizes[] = {3, 7, 15, 31};

// X-20's non-corner frontend stages at 640x480: denoise 76 800 + pyramid 102 240
// + derivative 204 480 + track 4 200. Fixed by the frame size, so the frontend
// total under either corner shape is this plus the corner stage measured here.
// It is a QUOTED constant and labelled as one; the end-to-end re-measurement of
// the whole frontend lives in tests/test_opticalflow.cpp, which owns the
// frontend, and this line only lets the table be read against X-20's row for row.
constexpr std::size_t kNonCornerFrontend640 = 387720;

/// @brief A frame with real corner structure rather than salt-and-pepper noise --
///        corner_benchmark.cpp's frame, so the two experiments share content.
template <typename W>
bincv::BinMat<W> makeFrame(int width, int height, uint64_t seed) {
    bincv::BinMat<W> src(width, height);
    uint64_t state = seed;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const unsigned block = static_cast<unsigned>((x / 37) + (y / 29)) % 2u;
            const unsigned diag = (x > y + 40) ? 1u : 0u;
            unsigned v = block ^ diag;
            if ((measure::nextRandom(state) & 63ULL) == 0ULL) v ^= 1u;  // sparse texture
            src.set(y, x, v);
        }
    }
    return src;
}

struct Answer {
    CornerResult result;
    std::vector<Corner> corners;
};

bool sameAnswer(const Answer& a, const Answer& b) {
    if (a.result.count != b.result.count) return false;
    if (a.result.candidatesRanked != b.result.candidatesRanked) return false;
    if (a.result.candidatesTruncated != b.result.candidatesTruncated) return false;
    for (std::size_t i = 0; i < a.result.candidatesRanked; ++i) {
        if (a.corners[i].x != b.corners[i].x) return false;
        if (a.corners[i].y != b.corners[i].y) return false;
        if (a.corners[i].response != b.corners[i].response) return false;
    }
    return true;
}

/// @brief Everything one word type needs at one frame size, built once.
template <typename W>
struct Fixture {
    int width = 0;
    int height = 0;
    std::vector<bincv::TernaryMat<W>> dx;
    std::vector<bincv::TernaryMat<W>> dy;
    std::vector<float> frameMapStorage;
    std::vector<float> ringStorage;
    ResponseMap frameMap;
    ResponseMap ring;

    Fixture(int w, int h) : width(w), height(h) {
        dx.reserve(kInputs);
        dy.reserve(kInputs);
        for (int i = 0; i < kInputs; ++i) {
            const bincv::BinMat<W> src =
                makeFrame<W>(w, h, uint64_t{0x9E3779B9} + static_cast<uint64_t>(i) * 7919u);
            dx.emplace_back(w, h);
            dy.emplace_back(w, h);
            bincv::derivativeX(src, dx.back());
            bincv::derivativeY(src, dy.back());
        }
        frameMapStorage.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0.0f);
        ringStorage.assign(bincv::kResponseRingRows * static_cast<std::size_t>(w), 0.0f);
        frameMap = ResponseMap{frameMapStorage.data(), static_cast<std::size_t>(w),
                               static_cast<std::size_t>(h), static_cast<std::size_t>(w)};
        ring = ResponseMap{ringStorage.data(), static_cast<std::size_t>(w),
                           bincv::kResponseRingRows, static_cast<std::size_t>(w)};
    }

    t311::Planes<W> planes(int k) const {
        return t311::planesOf(dx[static_cast<std::size_t>(k)], dy[static_cast<std::size_t>(k)]);
    }
};

/// @brief The survivor count for one input, measured with a buffer that cannot
///        truncate. THE MEASURING BUFFER IS SCOPED AND GONE before anything is
///        added up -- X-20 caught itself reporting a "peak" that excluded a live
///        `W*H` buffer 2.1x the total, and this is that lesson applied.
template <typename W>
std::size_t survivorCount(const Fixture<W>& f, const GoodFeaturesParams& params, int k) {
    std::vector<Corner> probe(static_cast<std::size_t>(f.width) *
                              static_cast<std::size_t>(f.height));
    const CornerResult r = t311::frameDetect<W>(f.planes(k), params, f.frameMap, probe.data(),
                                                probe.size());
    if (r.candidatesTruncated) {
        std::printf("  the probe buffer truncated -- the survivor count is not a reading\n");
        std::abort();
    }
    return r.candidatesRanked;
}

/// @brief Run all three arms over one input and compare their answers.
/// @return false if any arm disagreed with the control.
template <typename W>
bool answersAgree(const Fixture<W>& f, const GoodFeaturesParams& params, int k,
                  std::size_t capacity, const char* wordName) {
    Answer af, a2, a1;
    af.corners.assign(capacity + 1, Corner{});
    a2.corners.assign(capacity + 1, Corner{});
    a1.corners.assign(capacity + 1, Corner{});
    af.result = t311::frameDetect<W>(f.planes(k), params, f.frameMap, af.corners.data(), capacity);
    a2.result = t311::stream2Detect<W>(f.planes(k), params, f.ring, a2.corners.data(), capacity);
    a1.result = t311::stream1Detect<W>(f.planes(k), params, f.ring, a1.corners.data(), capacity);

    const bool ok2 = sameAnswer(af, a2);
    const bool ok1 = sameAnswer(af, a1);
    if (!ok2 || !ok1) {
        std::printf("  MISMATCH %s %dx%d block %d input %d: F {%zu, %zu, %d}"
                    "  S2 {%zu, %zu, %d}%s  S1 {%zu, %zu, %d}%s\n",
                    wordName, f.width, f.height, params.blockSize, k, af.result.count,
                    af.result.candidatesRanked, af.result.candidatesTruncated ? 1 : 0,
                    a2.result.count, a2.result.candidatesRanked,
                    a2.result.candidatesTruncated ? 1 : 0, ok2 ? "" : "  <-- DIFFERS",
                    a1.result.count, a1.result.candidatesRanked,
                    a1.result.candidatesTruncated ? 1 : 0, ok1 ? "" : "  <-- DIFFERS");
    }
    return ok2 && ok1;
}

struct CellRow {
    const char* arm;
    double detectNs;
    double detectSpread;
    double respondNs;
    double respondSpread;
    std::size_t allocs;
};

/// @brief One (word type, frame size, block size) cell: equality, then timing.
/// @param swapped Runs the arms in the reverse order inside the interleaved
///        batch. measure_util interleaves round-robin in vector order, so this is
///        the control for a verdict that is really a batch-position effect.
template <typename W>
void runCell(Fixture<W>& f, int blockSize, const char* wordName, bool swapped,
             double* tDetectOut, double* tRespondOut) {
    GoodFeaturesParams params;
    params.blockSize = blockSize;

    // ---- equality, over EVERY input the timed bodies rotate through ----------
    for (int k = 0; k < kInputs; ++k) {
        const std::size_t survivors = survivorCount<W>(f, params, k);
        // Two capacities: the measured survivor count (the frontend's sizing, no
        // truncation) and half of it (truncation on both sides, which is where
        // the two forms' mechanisms differ most).
        if (!answersAgree<W>(f, params, k, survivors, wordName) ||
            !answersAgree<W>(f, params, k, survivors / 2, wordName)) {
            std::printf("  not timing a form that returns different corners.\n");
            std::exit(1);
        }
    }

    // ---- the candidate buffers, sized as X-20 sizes them --------------------
    const std::size_t capacity = survivorCount<W>(f, params, 0);
    std::vector<Corner> cornersF(capacity), cornersS2(capacity), cornersS1(capacity);

    // ---- allocations, around ONE call of each arm, before the harness's own
    //      vectors and std::functions exist (those allocate, and would otherwise
    //      be charged to the kernel) --------------------------------------------
    std::size_t allocF = 0, allocS2 = 0, allocS1 = 0;
    {
        const std::size_t before = g_newCount;
        (void)t311::frameDetect<W>(f.planes(0), params, f.frameMap, cornersF.data(), capacity);
        allocF = g_newCount - before;
    }
    {
        const std::size_t before = g_newCount;
        (void)t311::stream2Detect<W>(f.planes(0), params, f.ring, cornersS2.data(), capacity);
        allocS2 = g_newCount - before;
    }
    {
        const std::size_t before = g_newCount;
        (void)t311::stream1Detect<W>(f.planes(0), params, f.ring, cornersS1.data(), capacity);
        allocS1 = g_newCount - before;
    }

    std::vector<measure::Bench> benches;
    auto addF = [&]() {
        benches.push_back({"F  frame-map", [&](int i) {
                               const int k = i % kInputs;
                               const CornerResult r = t311::frameDetect<W>(
                                   f.planes(k), params, f.frameMap, cornersF.data(), capacity);
                               measure::g_sink += r.count;
                           }});
        benches.push_back({"F  respond", [&](int i) {
                               const int k = i % kInputs;
                               measure::g_sink += static_cast<std::size_t>(
                                   t311::frameRespond<W>(f.planes(k), blockSize, f.frameMap));
                           }});
    };
    auto addS2 = [&]() {
        benches.push_back({"S2 two-pass", [&](int i) {
                               const int k = i % kInputs;
                               const CornerResult r = t311::stream2Detect<W>(
                                   f.planes(k), params, f.ring, cornersS2.data(), capacity);
                               measure::g_sink += r.count;
                           }});
        benches.push_back({"S2 respond", [&](int i) {
                               const int k = i % kInputs;
                               measure::g_sink += static_cast<std::size_t>(
                                   t311::stream2Respond<W>(f.planes(k), blockSize, f.ring));
                           }});
    };
    auto addS1 = [&]() {
        benches.push_back({"S1 one-pass", [&](int i) {
                               const int k = i % kInputs;
                               const CornerResult r = t311::stream1Detect<W>(
                                   f.planes(k), params, f.ring, cornersS1.data(), capacity);
                               measure::g_sink += r.count;
                           }});
        benches.push_back({"S1 respond", [&](int i) {
                               const int k = i % kInputs;
                               measure::g_sink += static_cast<std::size_t>(
                                   t311::stream1Respond<W>(f.planes(k), blockSize, f.ring));
                           }});
    };
    if (swapped) { addS1(); addS2(); addF(); } else { addF(); addS2(); addS1(); }

    const std::vector<measure::Timing> t = measure::measureInterleaved(benches, 11, 200.0);

    // Recover the arms by name, so the swapped run reads out in the same order.
    const std::size_t iF = swapped ? 4u : 0u;
    const std::size_t iS2 = 2u;
    const std::size_t iS1 = swapped ? 0u : 4u;
    const CellRow rows[3] = {
        {"F  frame-map", t[iF].medianNs, t[iF].spreadPct(), t[iF + 1].medianNs,
         t[iF + 1].spreadPct(), allocF},
        {"S2 two-pass", t[iS2].medianNs, t[iS2].spreadPct(), t[iS2 + 1].medianNs,
         t[iS2 + 1].spreadPct(), allocS2},
        {"S1 one-pass", t[iS1].medianNs, t[iS1].spreadPct(), t[iS1 + 1].medianNs,
         t[iS1 + 1].spreadPct(), allocS1},
    };

    const double pixels =
        static_cast<double>(f.width) * static_cast<double>(f.height);
    for (const CellRow& r : rows) {
        std::printf("  %-6d %-13s %9.3f %8.2f%% %8.2f %9.3f %8.2f%% %7.2fx %5zu\n", blockSize,
                    r.arm, r.detectNs / pixels, r.detectSpread, r.detectNs / 1e6,
                    r.respondNs / pixels, r.respondSpread, r.detectNs / rows[0].detectNs,
                    r.allocs);
    }
    std::printf("         -> T(S1) %.3fx   T(S2) %.3fx   (whole detector, medians)   "
                "candidates %zu\n",
                rows[2].detectNs / rows[0].detectNs, rows[1].detectNs / rows[0].detectNs,
                capacity);
    if (tDetectOut != nullptr) *tDetectOut = rows[2].detectNs / rows[0].detectNs;
    if (tRespondOut != nullptr) *tRespondOut = rows[2].respondNs / rows[0].respondNs;
}

void printCellHeader() {
    std::printf("  %-6s %-13s %9s %9s %8s %9s %9s %8s %5s\n", "block", "arm", "det ns/px",
                "spread", "det ms/f", "rsp ns/px", "spread", "vs F", "allc");
}

/// @brief The true peak of each form, as X-23 defines it.
template <typename W>
void printFootprint(const Fixture<W>& f, const char* wordName) {
    GoodFeaturesParams params;  // blockSize 3, the decision point
    const std::size_t capacity = survivorCount<W>(f, params, 0);

    const std::size_t pixels =
        static_cast<std::size_t>(f.width) * static_cast<std::size_t>(f.height);
    const std::size_t mapBytes = pixels * sizeof(float);
    const std::size_t ringBytes =
        bincv::kResponseRingRows * static_cast<std::size_t>(f.width) * sizeof(float);
    const std::size_t candidateBytes = capacity * sizeof(Corner);
    // The streaming form's ENTIRE carry beyond the ring and the candidate array:
    // the running maximum, the running count, and the strongest discarded
    // response. Three scalars, and they are counted rather than waved away
    // because X-23 said every byte of carry comes off the saving.
    const std::size_t streamCarry = sizeof(float) * 2 + sizeof(std::size_t);
    const std::size_t peakFrame = mapBytes + candidateBytes;
    const std::size_t peakStream = ringBytes + candidateBytes + streamCarry;

    std::printf("\n  TRUE PEAK, %s %dx%d, blockSize 3, candidate array sized to the MEASURED\n"
                "  survivor count (%zu). The probe buffer that measured it is destroyed before\n"
                "  these numbers are taken.\n", wordName, f.width, f.height, capacity);
    std::printf("    %-34s %11s %11s\n", "term", "frame map", "streaming");
    std::printf("    %-34s %9zu B %9zu B\n", "response storage", mapBytes, ringBytes);
    std::printf("    %-34s %9zu B %9zu B\n", "candidate array (also the output)", candidateBytes,
                candidateBytes);
    std::printf("    %-34s %9d B %9zu B\n", "carry for the GLOBAL properties", 0, streamCarry);
    std::printf("    %-34s %9zu B %9zu B   R = %.2fx\n", "TRUE PEAK", peakFrame, peakStream,
                static_cast<double>(peakFrame) / static_cast<double>(peakStream));
    std::printf("    %-34s %9zu B %9zu B\n", "structural worst case (w-2)(h-2)",
                mapBytes + static_cast<std::size_t>(f.width - 2) *
                               static_cast<std::size_t>(f.height - 2) * sizeof(Corner),
                ringBytes + static_cast<std::size_t>(f.width - 2) *
                                static_cast<std::size_t>(f.height - 2) * sizeof(Corner) +
                    streamCarry);
    // THE CANDIDATE ROW IS THIS FRAME'S OWN READING AND NOTHING ELSE. Every
    // other row is fixed by the frame size; this one is fixed by how many NMS
    // survivors THIS synthetic frame happens to have, and it differs from X-20's
    // frontend content. Both counts are printed so the two tables can be read
    // against each other instead of looking like a contradiction.
    if (f.width == 640 && f.height == 480) {
        const std::size_t x20Candidates = 8754 * sizeof(Corner);
        const std::size_t frontFrame = kNonCornerFrontend640 + mapBytes + x20Candidates;
        const std::size_t frontStream =
            kNonCornerFrontend640 + ringBytes + x20Candidates + streamCarry;
        std::printf("    the candidate row is THIS FRAME's reading (%zu survivors). At X-20's own\n"
                    "    frontend content (8 754 survivors, %zu B) the same arithmetic gives, with\n"
                    "    X-20's other four stages = %zu B:\n"
                    "      frontend  frame map %zu B   streaming %zu B   R = %.2fx\n",
                    capacity, x20Candidates, kNonCornerFrontend640, frontFrame, frontStream,
                    static_cast<double>(frontFrame) / static_cast<double>(frontStream));
        std::printf("      X-23's saving gate is P_stream <= 750 000 B: %s\n",
                    (frontStream <= 750000) ? "PASSES" : "FAILS");
        std::printf("      THE END-TO-END RE-MEASUREMENT IS NOT THIS LINE: it is\n"
                    "      Flow.FrontendFootprint_640x480 in tests/test_opticalflow.cpp, which owns\n"
                    "      the frontend, counts `operator new` through all five stages and asserts\n"
                    "      the two shapes return identical corners. This line only lets the\n"
                    "      benchmark's own peaks be read in X-20's units.\n");
    }
}

}  // namespace

int main() {
    std::printf("binCV T3.11 / E-10 (X-23) -- the rolling response ring against the frame map\n");
    std::printf("three arms, one translation unit each; every answer checked before anything "
                "is timed\n");
    std::printf("selection parameters are GoodFeaturesParams' defaults = seal_params.yaml "
                "verbatim\n\n");

    {
        Fixture<uint32_t> f(640, 480);
        printFootprint<uint32_t>(f, "uint32_t");
        std::printf("\n  640x480, uint32_t (D-14's default) -- THE DECISION CELL IS blockSize 3\n");
        printCellHeader();
        double t3 = 0.0, r3 = 0.0;
        for (int blockSize : kBlockSizes) {
            runCell<uint32_t>(f, blockSize, "uint32_t", false,
                              blockSize == 3 ? &t3 : nullptr, blockSize == 3 ? &r3 : nullptr);
        }

        std::printf("\n  THE SAME CELLS WITH THE ARM ORDER SWAPPED IN THE BATCH (S1, S2, F).\n"
                    "  A verdict that moves here is layout, not a result.\n");
        printCellHeader();
        double t3s = 0.0, r3s = 0.0;
        for (int blockSize : kBlockSizes) {
            runCell<uint32_t>(f, blockSize, "uint32_t", true,
                              blockSize == 3 ? &t3s : nullptr, blockSize == 3 ? &r3s : nullptr);
        }
        std::printf("\n  blockSize 3 decision cell: T = %.3fx in order (F, S2, S1) and %.3fx in "
                    "order (S1, S2, F);\n  response stage %.3fx and %.3fx.\n", t3, t3s, r3, r3s);
    }

    {
        Fixture<uint64_t> f(640, 480);
        printFootprint<uint64_t>(f, "uint64_t");
        std::printf("\n  640x480, uint64_t\n");
        printCellHeader();
        for (int blockSize : kBlockSizes) {
            runCell<uint64_t>(f, blockSize, "uint64_t", false, nullptr, nullptr);
        }
    }

    {
        // The repository's real frame size. X-20 measured 9 774 survivors here
        // against 8 754 at 640x480, which is why the candidate row is a per-frame
        // reading and not a bound.
        Fixture<uint32_t> f32(752, 480);
        printFootprint<uint32_t>(f32, "uint32_t");
        std::printf("\n  752x480, uint32_t\n");
        printCellHeader();
        runCell<uint32_t>(f32, 3, "uint32_t", false, nullptr, nullptr);
    }
    {
        Fixture<uint64_t> f64(752, 480);
        std::printf("\n  752x480, uint64_t\n");
        printCellHeader();
        runCell<uint64_t>(f64, 3, "uint64_t", false, nullptr, nullptr);
    }

    std::printf("\n  the `allc` column is `operator new` calls -- plain AND C++17 over-aligned "
                "--\n  counted around ONE call of each arm. Zero is what \"no heap in the "
                "kernel\" means.\n");
    std::printf("  sink %zu\n", static_cast<std::size_t>(measure::g_sink));
    return 0;
}
