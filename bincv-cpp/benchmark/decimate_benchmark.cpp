// E-8 / X-14 -- horizontal decimation by two: which route, and at what footprint?
//
// The decision rule this benchmark feeds was committed BEFORE it ran; it is in
// EXPERIMENTS.md as X-14, and the short form is:
//
//   1. The frame-masked route (C) ships only if it beats the better word-local
//      route by >= 1.5x with non-overlapping spreads at both word types on
//      640x480 -> 320x240. It is the only one that costs bytes, and below that
//      bar memory wins (CLAUDE.md's tiebreak).
//   2. Between the two zero-byte routes (A, B) speed alone decides, and a
//      difference inside the larger spread is a null result that takes the
//      simpler one -- the gather loop.
//
// NO OPENCV. All three variants are binCV, so ARCHITECTURE 10.3's denominator
// does not apply and this builds in the reference device's DEFAULT core-only
// build. (cv::resize would not be that denominator anyway: it resamples both axes
// on a byte image and rounds, rather than keeping the even columns.)
//
// VARIANTS   impl::decimateColumnsBy2Gather       per-pixel gather loop, 0 B aux
//            impl::decimateColumnsBy2Unshuffle    word-local Morton deinterleave, 0 B aux
//            impl::decimateColumnsBy2FrameMasked  big-integer masked unshuffle,
//                                                 mask table + scratch row
// WORKLOAD   the pyramid ladder T3.4 will call this with -- 640x480, 320x240,
//            160x120 and 94x60 sources -- at ~50% fill, four rotated inputs, at
//            uint32_t (D-14) and uint64_t.
// METRIC     ns per destination pixel, min/median/max over interleaved batches,
//            beside the auxiliary bytes each route needs. Speed and memory in one
//            table, because rule 1 weighs the pair.
//
// VALIDITY (EXPERIMENTS.md "Verify the benchmark measures something"):
//   * measure::g_sink consumes a destination word from every timed call;
//   * four distinct random sources rotate, so nothing constant-folds;
//   * all three variants are compared against a per-pixel reference AND against
//     each other, on every case, before anything is timed -- a benchmark between
//     a right answer and a wrong one is not a measurement;
//   * every row prints effective bytes/s next to it. At 640x480/uint32_t a call
//     touches 28.1 KiB, which is L1-resident on a Cortex-A72, so DRAM bandwidth
//     is NOT the bound -- L1 load throughput (~12-24 GB/s at 1.5 GHz) is, and a
//     row above that is a dead-code measurement rather than a fast kernel.
//
// The vertical half of a 2x2 subsample is free and is NOT what is being compared:
// all three variants read the same rowsDecimatedBy2() view, so the difference
// between them is the horizontal half alone.
//
// On x86_64 this is INDICATIVE ONLY (EXPERIMENTS.md, "Measurement platforms").
// The authoritative run is
//
//   ./scripts/run_on_pi.sh pi4 './benchmark/decimate_benchmark'

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/resample.hpp"
#include "measure_util.hpp"

namespace {

using bincv::BinMat;
using bincv::BinMatConstView;
using bincv::decimatedWidth;
using bincv::rowsDecimatedBy2;

constexpr int kInputs = 4;
constexpr int kRepeats = 9;
constexpr double kTargetMs = 40.0;

struct Case {
    const char* name;
    size_t width;
    size_t height;
};

// The pyramid ladder, plus the small frame X-10 used. 640 columns is 20 words at
// uint32_t and 10 at uint64_t -- neither a power of two, which is the case that
// makes variant C pad its row.
const Case kCases[] = {
    {"640x480", 640, 480},
    {"320x240", 320, 240},
    {"160x120", 160, 120},
    {"94x60", 94, 60},
};

uint64_t nextRandom(uint64_t& state) { return measure::nextRandom(state); }

template <typename Word>
void fillRandom(BinMat<Word>& m, uint64_t seed) {
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) m.set(y, x, (nextRandom(state) & 1u) != 0);
    }
}

// ---------------------------------------------------------------------------
// One (case, word type) fixture: four sources, one destination per variant, and
// variant C's plan built once -- which is the most favourable accounting it can
// have, since a pyramid reuses a plan across every frame at that width.
// ---------------------------------------------------------------------------
template <typename Word>
struct Fixture {
    std::vector<BinMat<Word>> src;
    BinMat<Word> dstA, dstB, dstC;
    std::vector<Word> masks;
    std::vector<Word> scratch;
    size_t dstPixels = 0;
    size_t auxBytes = 0;
    size_t touchedBytes = 0;  // source rows actually read + destination written

    Fixture(const Case& c, uint64_t seed)
        : dstA(static_cast<int>(decimatedWidth(c.width)),
               static_cast<int>(decimatedWidth(c.height))),
          dstB(static_cast<int>(decimatedWidth(c.width)),
               static_cast<int>(decimatedWidth(c.height))),
          dstC(static_cast<int>(decimatedWidth(c.width)),
               static_cast<int>(decimatedWidth(c.height))),
          masks(bincv::impl::frameMaskedPlanWords<Word>(c.width), Word{0}),
          scratch(bincv::impl::frameMaskedRowWords<Word>(c.width), Word{0}) {
        src.reserve(kInputs);
        for (int i = 0; i < kInputs; ++i) {
            src.emplace_back(static_cast<int>(c.width), static_cast<int>(c.height));
            fillRandom(src.back(), seed + static_cast<uint64_t>(i) * 7919u);
        }
        bincv::impl::buildFrameMaskedPlan<Word>(c.width, masks.data());

        dstPixels = decimatedWidth(c.width) * decimatedWidth(c.height);
        auxBytes = (masks.size() + scratch.size()) * sizeof(Word);
        const size_t srcRowBytes = src[0].getAlignedWidth() * sizeof(Word);
        const size_t dstBytes = dstA.getAlignedWidth() * sizeof(Word) * dstA.getHeight();
        touchedBytes = srcRowBytes * decimatedWidth(c.height) + dstBytes;
    }

    BinMatConstView<Word> source(int i) const {
        return rowsDecimatedBy2(src[static_cast<size_t>(i % kInputs)].constView());
    }
};

// ---------------------------------------------------------------------------
// Hazard 4: agreement before timing
// ---------------------------------------------------------------------------

/// Per-pixel reference, sharing no expression with any of the three kernels.
template <typename Word>
bool matchesReference(const BinMat<Word>& src, const BinMat<Word>& dst, const char* what) {
    for (size_t y = 0; y < dst.getHeight(); ++y) {
        for (size_t x = 0; x < dst.getWidth(); ++x) {
            const bool want = src.at(static_cast<int>(2 * y), static_cast<int>(2 * x));
            if (dst.at(static_cast<int>(y), static_cast<int>(x)) != want) {
                std::printf("  DISAGREEMENT: %s at (%zu,%zu)\n", what, y, x);
                return false;
            }
        }
    }
    return true;
}

template <typename Word>
bool agree(Fixture<Word>& f, const char* wordName) {
    for (int i = 0; i < kInputs; ++i) {
        bincv::impl::decimateColumnsBy2Gather(f.source(i), f.dstA.view());
        bincv::impl::decimateColumnsBy2Unshuffle(f.source(i), f.dstB.view());
        bincv::impl::decimateColumnsBy2FrameMasked(f.source(i), f.dstC.view(), f.masks.data(),
                                                   f.scratch.data());
        const BinMat<Word>& s = f.src[static_cast<size_t>(i)];
        if (!matchesReference(s, f.dstA, wordName) || !matchesReference(s, f.dstB, wordName) ||
            !matchesReference(s, f.dstC, wordName)) {
            return false;
        }
        // Word for word, not pixel for pixel: this is also the padding-bit check,
        // since a variant that left dirt past the width would differ here.
        for (size_t w = 0; w < f.dstA.sizeInWords(); ++w) {
            if (f.dstA.data()[w] != f.dstB.data()[w] || f.dstA.data()[w] != f.dstC.data()[w]) {
                std::printf("  DISAGREEMENT: %s variants differ at word %zu\n", wordName, w);
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

struct Row {
    std::string label;
    measure::Timing timing;
    size_t auxBytes;
    size_t dstPixels;
    size_t touchedBytes;
};

void printRows(const std::vector<Row>& rows) {
    std::printf("\n  %-22s %26s %9s %9s %10s\n", "variant",
                "ns/dst-pixel min/med/max", "spread", "aux B", "GB/s");
    double best = 0.0;
    for (const Row& r : rows) {
        if (best == 0.0 || r.timing.medianNs < best) best = r.timing.medianNs;
    }
    for (const Row& r : rows) {
        const double px = static_cast<double>(r.dstPixels);
        const double gbs = static_cast<double>(r.touchedBytes) / r.timing.medianNs;  // B/ns == GB/s
        std::printf("  %-22s %7.4f/%7.4f/%7.4f %8.1f%% %9zu %10.2f\n", r.label.c_str(),
                    r.timing.minNs / px, r.timing.medianNs / px, r.timing.maxNs / px,
                    r.timing.spreadPct(), r.auxBytes, gbs);
    }
    std::printf("\n  %-22s %12s %12s\n", "variant", "x slower", "vs fastest");
    for (const Row& r : rows) {
        std::printf("  %-22s %11.2fx %12s\n", r.label.c_str(), r.timing.medianNs / best,
                    (r.timing.medianNs == best) ? "<- fastest" : "");
    }
}

template <typename Word>
bool addCase(const Case& c, const char* wordName, std::vector<measure::Bench>& benches,
             std::vector<Row>& rows, Fixture<Word>& f) {
    if (!agree(f, wordName)) return false;

    Fixture<Word>* p = &f;
    const std::string suffix = std::string(" ") + wordName;

    benches.push_back({std::string("gather") + suffix, [p](int i) {
                           bincv::impl::decimateColumnsBy2Gather(p->source(i), p->dstA.view());
                           measure::g_sink += p->dstA.data()[0];
                       }});
    benches.push_back({std::string("unshuffle") + suffix, [p](int i) {
                           bincv::impl::decimateColumnsBy2Unshuffle(p->source(i), p->dstB.view());
                           measure::g_sink += p->dstB.data()[0];
                       }});
    benches.push_back({std::string("frame-masked") + suffix, [p](int i) {
                           bincv::impl::decimateColumnsBy2FrameMasked(
                               p->source(i), p->dstC.view(), p->masks.data(), p->scratch.data());
                           measure::g_sink += p->dstC.data()[0];
                       }});

    for (int v = 0; v < 3; ++v) {
        Row r;
        r.auxBytes = (v == 2) ? f.auxBytes : 0;
        r.dstPixels = f.dstPixels;
        r.touchedBytes = f.touchedBytes;
        rows.push_back(r);
    }
    static_cast<void>(c);
    return true;
}

bool runCase(const Case& c) {
    std::printf("\n=== %s -> %zux%zu ===\n", c.name, decimatedWidth(c.width),
                decimatedWidth(c.height));

    Fixture<uint32_t> f32(c, 0xC0FFEEu);
    Fixture<uint64_t> f64(c, 0xC0FFEEu);

    std::vector<measure::Bench> benches;
    std::vector<Row> rows;
    if (!addCase<uint32_t>(c, "uint32_t", benches, rows, f32)) return false;
    if (!addCase<uint64_t>(c, "uint64_t", benches, rows, f64)) return false;
    std::printf("  all three variants match the per-pixel reference and each other\n");

    std::printf("  working set of one call: %zu B read + written at uint32_t, %zu B at "
                "uint64_t\n",
                f32.touchedBytes, f64.touchedBytes);

    const std::vector<measure::Timing> timings =
        measure::measureInterleaved(benches, kRepeats, kTargetMs);
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i].label = benches[i].name;
        rows[i].timing = timings[i];
    }
    printRows(rows);
    return true;
}

}  // namespace

int main() {
    std::printf("binCV -- E-8 / X-14: horizontal decimation by two\n");
    std::printf("three routes to the same destination; %d inputs, %d interleaved batches, "
                "%.0f ms budget\n",
                kInputs, kRepeats, kTargetMs);
    std::printf("aux B is the mask table plus the scratch row -- built ONCE per width, which "
                "is\nthe most favourable accounting variant C can have\n");

    bool ok = true;
    for (const Case& c : kCases) ok = runCase(c) && ok;

    std::printf("\nsink=%llu\n", static_cast<unsigned long long>(measure::g_sink));
    if (!ok) {
        std::printf("\nAT LEAST ONE CASE DISAGREED -- no number above is a measurement\n");
        return 1;
    }
    return 0;
}
