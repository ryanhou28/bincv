// E-2 / T2.9 -- what should BinMat's default word type be?
//
// The default template argument is uint32_t today and every kernel in the library
// inherits it. Nothing has measured the alternative on the target.
//
// DECISION RULE -- copied verbatim from TASKS.md T2.9, and recorded in
// EXPERIMENTS.md X-10 before this file was written:
//
//   * uint64_t wins by > 10% on bulk kernels AND does not increase footprint at
//     representative widths -> change the default
//   * Within 10%, or footprint increases at small pyramid levels -> keep uint32_t
//     (memory wins ties)
//
// And the trap the task states explicitly, which this file exists to not fall
// into: "wider words round row strides up more coarsely, so the footprint effect
// is worst exactly at upper pyramid levels. Measure footprint at 94x60, not only
// at 640x480, or this experiment will reach the wrong conclusion." Both sizes are
// measured below, and a whole pyramid ladder is tabulated on top of them -- that
// half is exact integer arithmetic and is architecture-independent, so it closes
// anywhere. The SPEED half closes only on the reference device.
//
// VARIANTS   uint8_t, uint16_t, uint32_t, uint64_t, each at its own word
//            granularity (D-4's default alignment -- the alignment axis is E-1's)
// WORKLOAD   bitwiseAnd (T2.2) and countNonZero (T2.5), whole image,
//            640x480 and 94x60
// METRIC     ns/pixel and allocated bytes at both resolutions
//
// WHY THIS IS THE MOST 32-BIT-SENSITIVE OF THE THREE EXPERIMENTS. On armv7l every
// uint64_t operation is synthesised from 32-bit pairs, so the answer would
// describe the compiler rather than the hardware. scripts/run_on_pi.sh refuses to
// run on anything but aarch64 for exactly this reason, and the target is printed
// below so a recorded log carries the evidence rather than the assumption.
//
// X-7 CAVEAT, at its sharpest on this axis. binCV builds with no -march flags, so
// __builtin_popcountll lowers to `call __popcountdi2@PLT` on x86_64 -- one library
// call PER WORD, whatever the word is. That makes narrow words look artificially
// good on x86 (a uint8_t image pays 8x the calls of a uint64_t one but each call
// is the same price, so the ranking can inverts against the target), while on
// aarch64 it is fmov/cnt/uaddlv/fmov. x86 numbers from this file cannot rank these
// variants at all. No -march flag is added: that is a dispatch decision
// (ROADMAP 2.3) that no experiment has settled.
//
// VALIDITY: measure::g_sink consumes every result; four distinct random images
// rotate through each timed body, on a call counter that runs on across batches so
// the rotation does not degenerate when a batch is one call long; batches are
// calibrated, interleaved across variants and repeated with the spread reported;
// and every width is checked to produce the same answers as uint8_t before
// anything is timed. The printed spread bounds within-run noise only.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/logic.hpp"
#include "bincv-cpp/ops/reduce.hpp"
#include "measure_util.hpp"

namespace {

using bincv::BinMat;

constexpr int kInputs = 4;
constexpr int kRepeats = 9;
constexpr double kTargetMs = 40.0;

struct Case {
    const char* name;
    int width;
    int height;
};

const Case kCases[] = {
    {"640x480 frame", 640, 480},
    {"94x60 pyramid L3", 94, 60},
};

/// @brief Row stride in BYTES at word granularity for a given word size. Exact
///        integer arithmetic -- this is the footprint half, and it needs no device.
size_t rowBytes(int width, size_t wordBytes) {
    const size_t bits = wordBytes * 8;
    const size_t words = (static_cast<size_t>(width) + bits - 1) / bits;
    return words * wordBytes;
}

/// @brief One word width's images, plus the geometry that width produced.
template <typename Word>
struct Fixture {
    std::vector<BinMat<Word>> a;
    std::vector<BinMat<Word>> b;
    BinMat<Word> dst;
    size_t strideWords = 0;
    size_t bytesPerImage = 0;

    Fixture(const Case& c) {
        a.reserve(static_cast<size_t>(kInputs));
        b.reserve(static_cast<size_t>(kInputs));
        for (int i = 0; i < kInputs; ++i) {
            BinMat<Word> ma(c.width, c.height);
            BinMat<Word> mb(c.width, c.height);
            fill(ma, UINT64_C(0xA11A) + static_cast<uint64_t>(i) * UINT64_C(7919));
            fill(mb, UINT64_C(0xB22B) + static_cast<uint64_t>(i) * UINT64_C(104729));
            a.push_back(std::move(ma));
            b.push_back(std::move(mb));
        }
        dst = BinMat<Word>(c.width, c.height);
        strideWords = a[0].getAlignedWidth();
        bytesPerImage = a[0].sizeInWords() * sizeof(Word);
    }

    /// @note The SAME seeds as every other width, so the four fixtures hold the
    ///       same four images and the agreement check below is meaningful.
    static void fill(BinMat<Word>& m, uint64_t seed) {
        uint64_t state = seed;
        for (int y = 0; y < m.rows(); ++y) {
            for (int x = 0; x < m.cols(); ++x) {
                m.set(y, x, (measure::nextRandom(state) >> 63) != 0);
            }
        }
    }
};

/// @brief Runs both kernels once per image and returns a flat digest of the
///        results, so that four different word types can be compared for
///        agreement without four different comparison functions.
template <typename Word>
std::vector<uint8_t> digest(Fixture<Word>& f, const Case& c, std::vector<size_t>& counts) {
    std::vector<uint8_t> flat;
    flat.reserve(static_cast<size_t>(kInputs) * static_cast<size_t>(c.width) *
                 static_cast<size_t>(c.height));
    counts.clear();
    for (int i = 0; i < kInputs; ++i) {
        const size_t k = static_cast<size_t>(i);
        counts.push_back(bincv::countNonZero(f.a[k].constView()));
        bincv::bitwiseAnd(f.a[k].constView(), f.b[k].constView(), f.dst.view());
        for (int y = 0; y < c.height; ++y) {
            for (int x = 0; x < c.width; ++x) {
                flat.push_back(f.dst.at(y, x) ? uint8_t(1) : uint8_t(0));
            }
        }
    }
    return flat;
}

template <typename Word>
void addBenches(Fixture<Word>& f, const char* name, std::vector<measure::Bench>& andB,
                std::vector<measure::Bench>& cnzB) {
    Fixture<Word>* p = &f;
    andB.push_back({name, [p](int i) {
                        const size_t k = static_cast<size_t>(i % kInputs);
                        bincv::bitwiseAnd(p->a[k].constView(), p->b[k].constView(),
                                          p->dst.view());
                        measure::g_sink += p->dst.data()[k];
                    }});
    cnzB.push_back({name, [p](int i) {
                        const size_t k = static_cast<size_t>(i % kInputs);
                        measure::g_sink += bincv::countNonZero(p->a[k].constView());
                    }});
}

bool runCase(const Case& c) {
    const double pixels = static_cast<double>(c.width) * static_cast<double>(c.height);
    std::printf("\n=== %s, word granularity (D-4 default alignment) ===\n", c.name);

    Fixture<uint8_t> f8(c);
    Fixture<uint16_t> f16(c);
    Fixture<uint32_t> f32(c);
    Fixture<uint64_t> f64(c);

    // Hazard 4: agreement before timing. Every width must produce the same counts
    // and the same AND image as uint8_t.
    std::vector<size_t> c8, c16, c32, c64;
    const std::vector<uint8_t> d8 = digest(f8, c, c8);
    const std::vector<uint8_t> d16 = digest(f16, c, c16);
    const std::vector<uint8_t> d32 = digest(f32, c, c32);
    const std::vector<uint8_t> d64 = digest(f64, c, c64);
    if (d16 != d8 || d32 != d8 || d64 != d8 || c16 != c8 || c32 != c8 || c64 != c8) {
        std::printf("  DISAGREEMENT between word widths -- not timing anything\n");
        return false;
    }
    std::printf("  all four widths agree on countNonZero and on every AND pixel "
                "(%zu set in image 0)\n",
                c8[0]);

    std::vector<measure::Bench> andB;
    std::vector<measure::Bench> cnzB;
    addBenches(f8, "uint8_t", andB, cnzB);
    addBenches(f16, "uint16_t", andB, cnzB);
    addBenches(f32, "uint32_t", andB, cnzB);
    addBenches(f64, "uint64_t", andB, cnzB);

    const std::vector<measure::Timing> andT =
        measure::measureInterleaved(andB, kRepeats, kTargetMs);
    const std::vector<measure::Timing> cnzT =
        measure::measureInterleaved(cnzB, kRepeats, kTargetMs);

    const size_t strides[4] = {f8.strideWords, f16.strideWords, f32.strideWords,
                               f64.strideWords};
    const size_t bytes[4] = {f8.bytesPerImage, f16.bytesPerImage, f32.bytesPerImage,
                             f64.bytesPerImage};
    const size_t ideal = (static_cast<size_t>(c.width) * static_cast<size_t>(c.height) + 7) / 8;

    std::printf("\n  FOOTPRINT (allocated bytes per image, padding included)\n");
    std::printf("  %-10s %8s %11s %12s %12s\n", "word", "stride", "bytes/img",
                "vs uint32", "vs ideal");
    for (size_t v = 0; v < 4; ++v) {
        std::printf("  %-10s %6zu w %11zu %11.1f%% %11.1f%%\n", andB[v].name.c_str(),
                    strides[v], bytes[v],
                    (static_cast<double>(bytes[v]) / static_cast<double>(bytes[2]) - 1.0) * 100.0,
                    (static_cast<double>(bytes[v]) / static_cast<double>(ideal) - 1.0) * 100.0);
    }
    std::printf("  (ideal = %zu B, the information-theoretic minimum at 1 bit per "
                "pixel)\n",
                ideal);

    std::printf("\n  SPEED (ns/pixel, min / median / max over %d interleaved batches)\n",
                kRepeats);
    std::printf("  %-10s %-26s %-26s\n", "word", "bitwiseAnd", "countNonZero");
    for (size_t v = 0; v < 4; ++v) {
        std::printf("  %-10s %7.4f/%7.4f/%7.4f %7.4f/%7.4f/%7.4f\n", andB[v].name.c_str(),
                    andT[v].minNs / pixels, andT[v].medianNs / pixels, andT[v].maxNs / pixels,
                    cnzT[v].minNs / pixels, cnzT[v].medianNs / pixels, cnzT[v].maxNs / pixels);
    }

    std::printf("\n  SPEEDUP vs uint32_t (>1.00 = faster than the current default; the "
                "rule reads uint64_t's row)\n");
    std::printf("  %-10s %12s %12s %12s %12s\n", "word", "bitwiseAnd", "countNonZero",
                "AND spread", "cnz spread");
    for (size_t v = 0; v < 4; ++v) {
        std::printf("  %-10s %11.3fx %11.3fx %11.1f%% %11.1f%%\n", andB[v].name.c_str(),
                    andT[2].medianNs / andT[v].medianNs, cnzT[2].medianNs / cnzT[v].medianNs,
                    andT[v].spreadPct(), cnzT[v].spreadPct());
    }

    // Physical sanity. TRAFFIC (three images per bitwiseAnd call) is the numerator
    // of the GB/s figure; RESIDENT is what a timed batch keeps live, which is the
    // number the cache tier must be read off. A batch rotates over kInputs input
    // pairs plus a destination -- 2*kInputs + 1 images -- and all four widths are
    // interleaved in one round-robin round. Classifying the tier from TRAFFIC, as
    // an earlier version did, understated the batch by 3x and the round by 12x.
    const double andTraffic = static_cast<double>(bytes[2]) * 3.0;
    const double perCall = static_cast<double>(2 * kInputs + 1);
    const double residentOne = static_cast<double>(bytes[2]) * perCall;
    const double residentAll = (static_cast<double>(bytes[0]) + static_cast<double>(bytes[1]) +
                                static_cast<double>(bytes[2]) + static_cast<double>(bytes[3])) *
                               perCall;
    std::printf("\n  sanity: bitwiseAnd (uint32_t) moves %.0f B per call in %.0f ns = "
                "%.2f GB/s\n",
                andTraffic, andT[2].medianNs, andTraffic / andT[2].medianNs);
    std::printf("          resident during its batch %.1f KiB (%d images) -- %s;\n"
                "          all four widths interleaved: %.1f KiB, %s the 1 MiB L2\n",
                residentOne / 1024.0, 2 * kInputs + 1,
                residentOne <= 32.0 * 1024.0
                    ? "inside the reference device's 32 KiB L1D"
                    : (residentOne <= 1024.0 * 1024.0 ? "inside its 1 MiB L2, so rates above "
                                                        "DRAM's ~4-6 GB/s are expected"
                                                      : "beyond L2 -- DRAM-bound"),
                residentAll / 1024.0, residentAll <= 1024.0 * 1024.0 ? "inside" : "beyond");
    return true;
}

/// @brief The footprint half of E-2, as exact arithmetic over a pyramid ladder.
/// @note This is the half the task warns about. It needs no device and no timing:
///       a row stride is ceil(width / wordBits) words, so a wider word rounds up
///       more coarsely, and the penalty grows as the level shrinks.
void printPyramidFootprint() {
    const Case ladder[] = {
        {"640x480  (L0, frame)", 640, 480}, {"320x240  (L1)", 320, 240},
        {"160x120  (L2)", 160, 120},        {"752x480  (L0, wide frame)", 752, 480},
        {"188x120  (L2 of 752)", 188, 120}, {"94x60    (L3 of 752)", 94, 60},
        {"47x30    (L4 of 752)", 47, 30},
    };
    const size_t wordBytes[4] = {1, 2, 4, 8};
    const char* names[4] = {"uint8", "uint16", "uint32", "uint64"};

    std::printf("\n=== FOOTPRINT LADDER -- exact, architecture-independent ===\n");
    std::printf("  Bytes for one plane at word granularity. The last column is what "
                "T2.9 warns about:\n");
    std::printf("  uint64_t's penalty against the current default, worst at the upper "
                "levels LK touches every frame.\n\n");
    std::printf("  %-26s %10s %10s %10s %10s %14s\n", "size", names[0], names[1], names[2],
                names[3], "u64 vs u32");
    for (const Case& c : ladder) {
        size_t b[4];
        for (size_t w = 0; w < 4; ++w) {
            b[w] = rowBytes(c.width, wordBytes[w]) * static_cast<size_t>(c.height);
        }
        std::printf("  %-26s %10zu %10zu %10zu %10zu %13.1f%%\n", c.name, b[0], b[1], b[2],
                    b[3], (static_cast<double>(b[3]) / static_cast<double>(b[2]) - 1.0) * 100.0);
    }
}

}  // namespace

int main() {
    std::printf("=== E-2 / T2.9: default word width ===\n");
#if defined(__aarch64__)
    std::printf("target: aarch64 -- AUTHORITATIVE for the speed half\n");
#else
    std::printf("target: not aarch64 -- speed numbers are INDICATIVE ONLY (see the X-7 "
                "caveat in this file's header)\n");
#endif
    std::printf("sizeof(void*) = %zu; a 32-bit host would synthesise every uint64_t "
                "operation and answer a different question.\n",
                sizeof(void*));
    std::printf("The decision rule is in this file's header, written before measuring "
                "(EXPERIMENTS.md X-10).\n");

    printPyramidFootprint();

    bool ok = true;
    for (const Case& c : kCases) ok = runCase(c) && ok;

    std::printf("\nsink: %zu (printed so nothing above can be optimized away)\n",
                static_cast<size_t>(measure::g_sink));
    return ok ? 0 : 1;
}
