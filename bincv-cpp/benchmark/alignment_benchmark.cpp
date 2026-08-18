// E-1 / T2.8 -- does row alignment beyond word granularity earn its memory?
//
// This is the BENEFIT side of X-1, which measured only the cost. D-4 (word
// granularity by default) is the project's only PROVISIONAL decision, and it is
// provisional precisely because nobody had measured whether a wider row stride
// buys any kernel anything.
//
// DECISION RULE -- copied verbatim from TASKS.md T2.8, and recorded in
// EXPERIMENTS.md X-9 before this file was written:
//
//   * Speedup < 5% on all kernels -> D-4 confirmed, close E-1, do not build a
//     profile system
//   * 5-20% -> D-4 stands as default; larger alignment stays opt-in and is
//     documented as worth it for specific kernels
//   * > 20% on a kernel the frontend calls per frame -> reopen D-4, report
//     before changing anything
//
// "Speedup" is read the way the rule is written: variant faster than the word
// granularity baseline. A variant that is SLOWER lands in the first band -- there
// is no speedup to justify the bytes -- and the size of the slowdown is reported
// rather than clamped, because it is the interesting part of the answer.
//
// VARIANTS   rowAlignment in {sizeof(WordType) == 4, 16, 32, 64} bytes
// WORKLOAD   bitwiseAnd (T2.2) and countNonZero (T2.5), whole image,
//            640x480 and 94x60 -- the two extremes X-1 measured
// METRIC     ns/pixel AND allocated bytes, both, per the protocol
// WORD TYPE  uint32_t, the shipped default. The word-width axis is E-2's, and
//            mixing them would leave neither answerable.
//
// WHAT ALIGNMENT ACTUALLY CHANGES HERE, so the result can be read honestly:
//
//   a) The row STRIDE, which is what rowAlignment is documented to control. Rows
//      become congruent modulo the alignment, and the padding words between them
//      are allocated, touched by nothing, and paid for in cache lines and TLB
//      reach whenever a kernel walks row to row.
//   b) NOT the base pointer. binMat allocates with new[], whose guarantee is
//      __STDCPP_DEFAULT_NEW_ALIGNMENT__ (16 bytes on this target), so
//      "rowAlignment = 64" does not make row 0 64-byte aligned -- it makes every
//      row the same offset into a 64-byte block as row 0. This benchmark PRINTS
//      the measured base alignment rather than assuming, because a reader would
//      otherwise credit the result to an alignment the allocation never provided.
//   c) For bitwiseAnd, whether the ops/logic.hpp contiguous fast path is taken at
//      all: that path requires every stride to equal the words a row needs, so any
//      over-alignment disables it. That is not a confound to subtract out -- it is
//      a consequence of choosing the alignment, and the caller pays it -- but it
//      is printed per variant so the mechanism is visible. countNonZero has no
//      such path (it walks rows unconditionally), so its column isolates the
//      alignment effect on its own.
//
// X-7 CAVEAT. binCV builds with no -march flags, so __builtin_popcountll lowers
// to `call __popcountdi2@PLT` on x86_64 and to fmov/cnt/uaddlv/fmov on aarch64.
// The countNonZero rows below therefore measure that lowering as much as they
// measure alignment, and x86 numbers from this file are signal only. No -march
// flag is added: that is a dispatch decision (ROADMAP 2.3) no experiment has
// settled, and changing it mid-experiment would confound this comparison.
// This experiment closes on the reference device (scripts/run_on_pi.sh).
//
// VALIDITY: measure::g_sink consumes every result; four distinct random images
// rotate through each timed body; batches are calibrated and repeated with the
// spread reported; and every variant is checked to produce the SAME answers as
// the baseline before anything is timed.

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
using Word = uint32_t;

constexpr int kInputs = 4;
constexpr int kRepeats = 9;
constexpr double kTargetMs = 40.0;

// The two extremes from X-1: a full VIO frame, and pyramid level 3, where X-1
// measured 172% overhead for 32-byte alignment.
struct Case {
    const char* name;
    int width;
    int height;
};

const Case kCases[] = {
    {"640x480 frame", 640, 480},
    {"94x60 pyramid L3", 94, 60},
};

const size_t kAlignments[] = {sizeof(Word), 16, 32, 64};
constexpr size_t kVariants = sizeof(kAlignments) / sizeof(kAlignments[0]);

/// @brief Deterministic random fill; rotating four of these is validity hazard 2.
void fillRandom(BinMat<Word>& m, uint64_t seed) {
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            m.set(y, x, (measure::nextRandom(state) >> 63) != 0);
        }
    }
}

/// @brief Bytes this matrix actually allocated, stride padding included.
size_t allocatedBytes(const BinMat<Word>& m) { return m.sizeInWords() * sizeof(Word); }

/// @brief What the base pointer is really aligned to, up to 64 bytes.
size_t measuredBaseAlignment(const BinMat<Word>& m) {
    const uintptr_t p = reinterpret_cast<uintptr_t>(m.data());
    for (size_t a = 64; a > 1; a >>= 1) {
        if ((p & static_cast<uintptr_t>(a - 1)) == 0) return a;
    }
    return 1;
}

/// @brief Words a row inherently needs, i.e. the stride at word granularity.
size_t minRowWords(int width) {
    return (static_cast<size_t>(width) + sizeof(Word) * 8 - 1) / (sizeof(Word) * 8);
}

/// @brief One alignment's images and the geometry that alignment produced.
struct Variant {
    size_t alignment = 0;
    size_t strideWords = 0;
    size_t bytesPerImage = 0;
    size_t baseAlign = 0;
    bool contiguousFastPath = false;
    std::vector<BinMat<Word>> a;
    std::vector<BinMat<Word>> b;
    BinMat<Word> dst;
};

bool runCase(const Case& c) {
    const double pixels = static_cast<double>(c.width) * static_cast<double>(c.height);
    const size_t tight = minRowWords(c.width);
    const bool widthIsWholeWords = (static_cast<size_t>(c.width) % (sizeof(Word) * 8)) == 0;

    std::printf("\n=== %s, uint32_t words ===\n", c.name);
    std::printf("  a row inherently needs %zu words (%zu B); width is %sa whole number "
                "of words\n",
                tight, tight * sizeof(Word), widthIsWholeWords ? "" : "NOT ");

    // Every variant is built first and none is timed until all of them agree, so
    // that no timed body is ever the first thing to touch a buffer.
    std::vector<Variant> variants(kVariants);
    for (size_t v = 0; v < kVariants; ++v) {
        Variant& var = variants[v];
        var.alignment = kAlignments[v];
        var.a.reserve(static_cast<size_t>(kInputs));
        var.b.reserve(static_cast<size_t>(kInputs));
        for (int i = 0; i < kInputs; ++i) {
            BinMat<Word> a(c.width, c.height, var.alignment);
            BinMat<Word> b(c.width, c.height, var.alignment);
            fillRandom(a, UINT64_C(0xA11A) + static_cast<uint64_t>(i) * UINT64_C(7919));
            fillRandom(b, UINT64_C(0xB22B) + static_cast<uint64_t>(i) * UINT64_C(104729));
            var.a.push_back(std::move(a));
            var.b.push_back(std::move(b));
        }
        var.dst = BinMat<Word>(c.width, c.height, var.alignment);
        var.strideWords = var.a[0].getAlignedWidth();
        var.bytesPerImage = allocatedBytes(var.a[0]);
        var.baseAlign = measuredBaseAlignment(var.a[0]);
        var.contiguousFastPath = widthIsWholeWords && var.strideWords == tight;
    }

    // Hazard 4: agreement before timing -- the reduction and every pixel of the
    // AND, against the word-granularity variant.
    std::vector<size_t> referenceCounts(static_cast<size_t>(kInputs), 0);
    std::vector<std::vector<uint8_t>> referenceAnd(static_cast<size_t>(kInputs));
    for (size_t v = 0; v < kVariants; ++v) {
        Variant& var = variants[v];
        for (int i = 0; i < kInputs; ++i) {
            const size_t k = static_cast<size_t>(i);
            const size_t count = bincv::countNonZero(var.a[k].constView());
            bincv::bitwiseAnd(var.a[k].constView(), var.b[k].constView(), var.dst.view());

            std::vector<uint8_t> flat;
            flat.reserve(static_cast<size_t>(c.width) * static_cast<size_t>(c.height));
            for (int y = 0; y < c.height; ++y) {
                for (int x = 0; x < c.width; ++x) {
                    flat.push_back(var.dst.at(y, x) ? uint8_t(1) : uint8_t(0));
                }
            }

            if (v == 0) {
                referenceCounts[k] = count;
                referenceAnd[k] = flat;
            } else if (count != referenceCounts[k] || flat != referenceAnd[k]) {
                std::printf("  DISAGREEMENT at alignment %zu on image %d\n", var.alignment, i);
                return false;
            }
        }
    }
    std::printf("  all %zu alignment variants agree with the word-granularity baseline "
                "on all %d images\n",
                kVariants, kInputs);

    std::vector<measure::Bench> andBenches;
    std::vector<measure::Bench> countBenches;
    for (size_t v = 0; v < kVariants; ++v) {
        Variant* var = &variants[v];
        andBenches.push_back({std::to_string(var->alignment), [var](int i) {
                                  const size_t k = static_cast<size_t>(i % kInputs);
                                  bincv::bitwiseAnd(var->a[k].constView(),
                                                    var->b[k].constView(), var->dst.view());
                                  measure::g_sink += var->dst.data()[k];
                              }});
        countBenches.push_back({std::to_string(var->alignment), [var](int i) {
                                    const size_t k = static_cast<size_t>(i % kInputs);
                                    measure::g_sink +=
                                        bincv::countNonZero(var->a[k].constView());
                                }});
    }

    const std::vector<measure::Timing> andT =
        measure::measureInterleaved(andBenches, kRepeats, kTargetMs);
    const std::vector<measure::Timing> cnzT =
        measure::measureInterleaved(countBenches, kRepeats, kTargetMs);

    // ---- footprint ----------------------------------------------------------
    std::printf("\n  FOOTPRINT (allocated bytes, one image; overhead vs word "
                "granularity)\n");
    std::printf("  %-8s %8s %11s %10s %11s %11s\n", "align", "stride", "bytes/img",
                "overhead", "AND set", "base align");
    for (size_t v = 0; v < kVariants; ++v) {
        const Variant& var = variants[v];
        const double overhead = (static_cast<double>(var.bytesPerImage) /
                                     static_cast<double>(variants[0].bytesPerImage) -
                                 1.0) * 100.0;
        std::printf("  %-8zu %6zu w %11zu %9.1f%% %11zu %9zu B\n", var.alignment,
                    var.strideWords, var.bytesPerImage, overhead, var.bytesPerImage * 3,
                    var.baseAlign);
    }
    std::printf("  (AND set = two inputs plus the destination -- that kernel's peak "
                "working set)\n");

    // ---- speed --------------------------------------------------------------
    std::printf("\n  SPEED (ns/pixel, min / median / max over %d interleaved batches)\n",
                kRepeats);
    std::printf("  %-8s %-26s %-26s %s\n", "align", "bitwiseAnd", "countNonZero",
                "AND path");
    for (size_t v = 0; v < kVariants; ++v) {
        std::printf("  %-8zu %7.4f/%7.4f/%7.4f %7.4f/%7.4f/%7.4f %s\n", variants[v].alignment,
                    andT[v].minNs / pixels, andT[v].medianNs / pixels, andT[v].maxNs / pixels,
                    cnzT[v].minNs / pixels, cnzT[v].medianNs / pixels, cnzT[v].maxNs / pixels,
                    variants[v].contiguousFastPath ? "contiguous" : "row loop");
    }

    std::printf("\n  SPEEDUP vs word granularity (>1.00 = wider alignment is FASTER; "
                "the rule reads this column)\n");
    std::printf("  %-8s %12s %12s %12s %12s\n", "align", "bitwiseAnd", "countNonZero",
                "AND spread", "cnz spread");
    for (size_t v = 0; v < kVariants; ++v) {
        std::printf("  %-8zu %11.3fx %11.3fx %11.1f%% %11.1f%%\n", variants[v].alignment,
                    andT[0].medianNs / andT[v].medianNs, cnzT[0].medianNs / cnzT[v].medianNs,
                    andT[v].spreadPct(), cnzT[v].spreadPct());
    }

    // ---- physical sanity ----------------------------------------------------
    // bitwiseAnd touches three images per call. Reporting the implied bandwidth is
    // how a reader checks the number against the machine rather than trusting it:
    // on the reference device DRAM is roughly 4-6 GB/s, L2 is 1 MiB and L1D is
    // 32 KiB, so a working set inside those may legitimately exceed DRAM rates and
    // one outside them may not.
    const double andBytes = static_cast<double>(variants[0].bytesPerImage) * 3.0;
    const double gbs = andBytes / andT[0].medianNs;  // B/ns == GB/s
    std::printf("\n  sanity: bitwiseAnd at word granularity moves %.0f B per call in "
                "%.0f ns = %.2f GB/s\n",
                andBytes, andT[0].medianNs, gbs);
    std::printf("          working set %.1f KiB -- %s\n", andBytes / 1024.0,
                andBytes <= 32.0 * 1024.0
                    ? "L1-resident on the reference device (32 KiB L1D)"
                    : (andBytes <= 1024.0 * 1024.0
                           ? "L2-resident (1 MiB L2), so rates above DRAM's ~4-6 GB/s are expected"
                           : "beyond L2 -- DRAM-bound, so ~4-6 GB/s is the ceiling"));
    return true;
}

}  // namespace

int main() {
    std::printf("=== E-1 / T2.8: does row alignment earn its memory? ===\n");
#if defined(__aarch64__)
    std::printf("target: aarch64 -- AUTHORITATIVE (the reference device closes E-1)\n");
#else
    std::printf("target: not aarch64 -- INDICATIVE ONLY; E-1 closes on the reference "
                "device\n");
#endif
    std::printf("The decision rule is in this file's header, written before measuring "
                "(EXPERIMENTS.md X-9).\n");
    std::printf("new[] guarantees %zu-byte alignment here, so rowAlignment aligns the "
                "STRIDE, not the base pointer.\n",
                static_cast<size_t>(__STDCPP_DEFAULT_NEW_ALIGNMENT__));

    bool ok = true;
    for (const Case& c : kCases) ok = runCase(c) && ok;

    std::printf("\nsink: %zu (printed so nothing above can be optimized away)\n",
                static_cast<size_t>(measure::g_sink));
    return ok ? 0 : 1;
}
