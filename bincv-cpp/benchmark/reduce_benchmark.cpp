// bulk reductions versus OpenCV, and versus the per-pixel loop.
//
// THE DENOMINATOR (the design notes, CLAUDE.md): OpenCV performing the SAME
// semantic operation on the SAME binary content stored as CV_8U -- what a user
// does today without binCV. For countNonZero that denominator is exact: OpenCV
// has the identical function, and this is a Tier 1 operation. For countAnd it is
// the composite a user would have to write instead -- cv::bitwise_and into a
// temporary, then cv::countNonZero -- and the temporary is part of the cost,
// which is the point of having a masked reduction at all.
//
// A THIRD ROW, because asks for it: BinMat::countNonZero, the per-pixel
// loop the container has carried since before this task. It is the "before" this
// work is supposed to improve on, and it is measured rather than assumed.
//
// ---------------------------------------------------------------------------
// MEASUREMENT VALIDITY -- the same four hazards benchmark/logic_benchmark.cpp
// enumerates, answered here in the same way:
//
// 1. DEAD CODE. A reduction whose result is unused is deleted outright, and it
// is a more inviting target than a kernel that writes to memory. Every count
// is folded into a `volatile` sink AND printed, so a loop that computed
// nothing would be visible twice.
// 2. CONSTANT FOLDING. The inputs rotate through four distinct random images,
// so no iteration can reuse the previous one's answer.
// 3. CALIBRATED BATCHES. A fixed iteration count measures clock resolution at
// 94x60; every case runs enough iterations to fill a target millisecond
// budget, and the reported figure is the minimum over several batches.
// 4. THE SIDES MUST AGREE. Every implementation's count is compared before
// anything is timed, and a disagreement skips the size and exits non-zero
// rather than printing a table under a warning.
//
// NOT reproduced here: logic_benchmark's physical-bound probe. That probe bounds
// a kernel that reads and writes; a reduction only reads, and this file does not
// have a validated ceiling to compare against. Rather than invent one, the
// absolute GB/s figures are reported as read bandwidth and no plausibility
// threshold is claimed for them.
//
// The timing helpers below are duplicated from logic_benchmark.cpp rather than
// shared through bench_util.hpp. That file's copy is measurement code behind a
// published result ( / results/logic_benchmark.log); putting a
// refactor underneath it would mean re-validating numbers that are already
// recorded, to save forty lines.
//
// ---------------------------------------------------------------------------
// WHAT THIS FILE IS NOT
//
// It is NOT or. It reports word types side by side and windows at
// three sizes because those are the workloads- need, but a number
// measured here is x86_64 and therefore NON-AUTHORITATIVE for every one of those
// questions (EXPERIMENTS.md, "Measurement platforms"). They close on the
// reference device through scripts/run_on_pi.sh, and this file is the code they
// run, not the answer they produce.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "bench_util.hpp"
#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/reduce.hpp"

namespace {

constexpr int kInputs = 4;      // distinct images, rotated through
constexpr int kRepeats = 5;     // batches per case; the minimum is reported
constexpr double kTargetMs = 40.0;

volatile size_t g_sink = 0;

// ---------------------------------------------------------------------------
// Content
// ---------------------------------------------------------------------------

uint64_t nextRandom(uint64_t& state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

/// @brief One image in every representation under test, from ONE draw per pixel.
/// @note The four packed matrices and the CV_8U mask are not merely statistically
/// similar -- they are the same picture, which is what makes the comparison
/// like for like.
struct Image {
    bincv::BinMat<uint8_t> packed8;
    bincv::BinMat<uint16_t> packed16;
    bincv::BinMat<uint32_t> packed32;
    bincv::BinMat<uint64_t> packed64;
    cv::Mat mask;
};

void makeImage(Image& out, int width, int height, uint64_t seed) {
    out.packed8 = bincv::BinMat<uint8_t>(width, height);
    out.packed16 = bincv::BinMat<uint16_t>(width, height);
    out.packed32 = bincv::BinMat<uint32_t>(width, height);
    out.packed64 = bincv::BinMat<uint64_t>(width, height);
    out.mask = cv::Mat::zeros(height, width, CV_8U);

    uint64_t state = seed;
    for (int y = 0; y < height; ++y) {
        uint8_t* row = out.mask.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            if ((nextRandom(state) >> 40) < (UINT64_C(1) << 23)) {  // ~50% fill
                out.packed8.set(y, x, true);
                out.packed16.set(y, x, true);
                out.packed32.set(y, x, true);
                out.packed64.set(y, x, true);
                row[x] = 255;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Timing (see the note at the top on why this is duplicated)
// ---------------------------------------------------------------------------

template <typename Body>
int calibrate(Body body, double targetMs) {
    int iterations = 8;
    for (int attempt = 0; attempt < 24; ++attempt) {
        const auto start = bench::Clock::now();
        for (int i = 0; i < iterations; ++i) body(i);
        const double ms =
            std::chrono::duration<double, std::milli>(bench::Clock::now() - start).count();
        if (ms >= targetMs || iterations >= (1 << 22)) return iterations;
        const double scale = (ms > 0.0) ? (targetMs / ms) : 8.0;
        const double next = static_cast<double>(iterations) * std::min(scale * 1.3, 16.0);
        iterations = static_cast<int>(next) + 1;
    }
    return iterations;
}

template <typename Body>
double measureNs(Body body, int repeats, double targetMs) {
    const int iterations = calibrate(body, targetMs);
    double bestNs = -1.0;
    for (int r = 0; r < repeats; ++r) {
        const auto start = bench::Clock::now();
        for (int i = 0; i < iterations; ++i) body(i);
        const double ns =
            std::chrono::duration<double, std::nano>(bench::Clock::now() - start).count() /
            static_cast<double>(iterations);
        if (bestNs < 0.0 || ns < bestNs) bestNs = ns;
    }
    return bestNs;
}

void report(const char* op, const char* impl, double nsPerCall, double pixels, double readBytes,
            double bufferBytes) {
    std::printf("[BENCH] %-24s %-18s %10.5f ns/px %8.2f GB/s read %10.0f B/buffer\n", op, impl,
                nsPerCall / pixels, (nsPerCall > 0.0) ? readBytes / nsPerCall : 0.0, bufferBytes);
}

// ---------------------------------------------------------------------------
// One size
// ---------------------------------------------------------------------------

bool runSize(int width, int height) {
    const double pixels = static_cast<double>(width) * static_cast<double>(height);
    std::vector<Image> images(kInputs);
    for (int i = 0; i < kInputs; ++i) {
        makeImage(images[static_cast<size_t>(i)], width, height,
                  UINT64_C(0xB1C0DE) + static_cast<uint64_t>(i) * UINT64_C(7919));
    }

    std::printf("\n--- %d x %d (%.0f pixels) ---\n", width, height, pixels);

    // --- hazard 4: the sides must agree before anything is timed -------------
    for (const Image& img : images) {
        const size_t expected = static_cast<size_t>(cv::countNonZero(img.mask));
        const size_t counts[] = {bincv::countNonZero(img.packed8.constView()),
                                 bincv::countNonZero(img.packed16.constView()),
                                 bincv::countNonZero(img.packed32.constView()),
                                 bincv::countNonZero(img.packed64.constView()),
                                 static_cast<size_t>(img.packed32.countNonZero()),
                                 bincv::countNonZero(img.packed32.constView(),
                                                     bincv::Rect(0, 0, width, height))};
        for (size_t c : counts) {
            if (c != expected) {
                std::printf(" DISAGREEMENT: %zu vs cv::countNonZero %zu -- size skipped\n", c,
                            expected);
                return false;
            }
        }
    }
    std::printf(" all implementations agree (%d set pixels in image 0)\n",
                cv::countNonZero(images[0].mask));

    const double packedBytes32 =
        static_cast<double>(images[0].packed32.getAlignedWidth() * sizeof(uint32_t)) *
        static_cast<double>(height);
    const double maskBytes = pixels;

    // --- countNonZero --------------------------------------------------------
    const double ns8 = measureNs(
        [&](int i) { g_sink += bincv::countNonZero(images[static_cast<size_t>(i % kInputs)].packed8.constView()); },
        kRepeats, kTargetMs);
    report("countNonZero", "binCV uint8", ns8, pixels, pixels / 8.0, pixels / 8.0);

    const double ns16 = measureNs(
        [&](int i) { g_sink += bincv::countNonZero(images[static_cast<size_t>(i % kInputs)].packed16.constView()); },
        kRepeats, kTargetMs);
    report("countNonZero", "binCV uint16", ns16, pixels, pixels / 8.0, pixels / 8.0);

    const double ns32 = measureNs(
        [&](int i) { g_sink += bincv::countNonZero(images[static_cast<size_t>(i % kInputs)].packed32.constView()); },
        kRepeats, kTargetMs);
    report("countNonZero", "binCV uint32", ns32, pixels, packedBytes32, packedBytes32);

    const double ns64 = measureNs(
        [&](int i) { g_sink += bincv::countNonZero(images[static_cast<size_t>(i % kInputs)].packed64.constView()); },
        kRepeats, kTargetMs);
    report("countNonZero", "binCV uint64", ns64, pixels, pixels / 8.0, pixels / 8.0);

    const double nsPerPixelLoop = measureNs(
        [&](int i) {
            g_sink += static_cast<size_t>(images[static_cast<size_t>(i % kInputs)].packed32.countNonZero());
        },
        kRepeats, kTargetMs);
    report("countNonZero", "binCV per-pixel", nsPerPixelLoop, pixels, packedBytes32, packedBytes32);

    const double nsCv = measureNs(
        [&](int i) {
            g_sink += static_cast<size_t>(cv::countNonZero(images[static_cast<size_t>(i % kInputs)].mask));
        },
        kRepeats, kTargetMs);
    report("countNonZero", "OpenCV CV_8U", nsCv, pixels, maskBytes, maskBytes);

    std::printf(" ratio vs OpenCV: uint32 %.2fx, uint64 %.2fx | vs the per-pixel loop: %.1fx\n",
                nsCv / ns32, nsCv / ns64, nsPerPixelLoop / ns32);

    // --- countAnd, against the composite a user would have to write ----------
    const bincv::Rect whole(0, 0, width, height);
    const double nsAnd = measureNs(
        [&](int i) {
            const Image& a = images[static_cast<size_t>(i % kInputs)];
            const Image& b = images[static_cast<size_t>((i + 1) % kInputs)];
            g_sink += bincv::countAnd(a.packed32.constView(), b.packed32.constView(), whole);
        },
        kRepeats, kTargetMs);
    report("countAnd", "binCV uint32", nsAnd, pixels, 2.0 * packedBytes32, packedBytes32);

    cv::Mat scratch;
    const double nsCvAnd = measureNs(
        [&](int i) {
            const Image& a = images[static_cast<size_t>(i % kInputs)];
            const Image& b = images[static_cast<size_t>((i + 1) % kInputs)];
            cv::bitwise_and(a.mask, b.mask, scratch);
            g_sink += static_cast<size_t>(cv::countNonZero(scratch));
        },
        kRepeats, kTargetMs);
    report("countAnd", "OpenCV composite", nsCvAnd, pixels, 3.0 * maskBytes, 2.0 * maskBytes);
    std::printf(" ratio vs the OpenCV composite: %.2fx (and binCV allocates no temporary)\n",
                nsCvAnd / nsAnd);

    // --- countAndSplit over LK-sized windows ---------------------------------
    //
    // Context for earlier work, not an answer to it: the MVP recomputes per
    // window, windows overlap heavily, and this is what recomputation costs. The
    // incremental alternative is deliberately not implemented here -- measuring
    // one option is not an experiment.
    for (int windowSize : {7, 15, 31}) {
        const int stepX = std::max(1, width / 20);
        const int stepY = std::max(1, height / 10);
        int windows = 0;
        for (int cy = 0; cy < height; cy += stepY) {
            for (int cx = 0; cx < width; cx += stepX) ++windows;
        }
        const double nsSweep = measureNs(
            [&](int i) {
                const Image& a = images[static_cast<size_t>(i % kInputs)];
                const Image& b = images[static_cast<size_t>((i + 1) % kInputs)];
                const Image& c = images[static_cast<size_t>((i + 2) % kInputs)];
                size_t acc = 0;
                for (int cy = 0; cy < height; cy += stepY) {
                    for (int cx = 0; cx < width; cx += stepX) {
                        const bincv::SplitCount split = bincv::countAndSplit(
                            a.packed32.constView(), b.packed32.constView(), c.packed32.constView(),
                            bincv::Rect(cx - windowSize / 2, cy - windowSize / 2, windowSize,
                                        windowSize));
                        acc += split.whenClear + split.whenSet;
                    }
                }
                g_sink += acc;
            },
            kRepeats, kTargetMs);
        std::printf("[BENCH] %-24s %-18s %10.1f ns/window over %d windows (%dx%d)\n",
                    "countAndSplit", "binCV uint32", nsSweep / windows, windows, windowSize,
                    windowSize);
    }

    return true;
}

}  // namespace

/// @brief How impl::popcountWord actually lowers in THIS build.
/// @note Printed because it changes the answer by roughly an order of magnitude
/// and is invisible in the table otherwise. binCV applies no -march flags
/// (the top-level CMakeLists detects AVX2/AVX-512 and deliberately does not
/// enable them until runtime dispatch lands), so on the portable x86-64
/// baseline __builtin_popcountll is a CALL to libgcc's __popcountdi2 -- one
/// function call per word. Verified by reading `g++ -O2 -S` output for a
/// countNonZero caller: four `call __popcountdi2@PLT` in the row loop, and
/// `popcntq` in its place under -mpopcnt.
const char* popcountLowering() {
#if defined(__aarch64__)
    // Per ENTRY POINT, not one sequence for all of them: the interior loops emit 1
    // GPR<->NEON crossing per word for countNonZero (the inbound fmov is elided --
    // `ldr d31, [x2], 8` -- and the horizontal add is `addv b31`, not `uaddlv h0`),
    // 2 for countAnd, and 4 for countAndSplit. Read off `g++ -O2 -DNDEBUG -S` on
    // the reference device; measured cost in.
    return "aarch64: cnt + addv per word, accumulator in a GPR -- 1/2/4 domain "
           "crossings per word for countNonZero/countAnd/countAndSplit";
#elif defined(__POPCNT__)
    return "x86-64 with POPCNT: popcntq per word";
#elif defined(__x86_64__) || defined(__i386__)
    return "x86-64 baseline WITHOUT POPCNT: a call to libgcc __popcountdi2 per word";
#else
    return "unknown target";
#endif
}

int main() {
    // binCV is single-threaded here, so OpenCV is held to one thread too. Left at its
    // default a multi-core box turns the ratio into a measurement of parallelism.
    cv::setNumThreads(1);

    std::printf("=== binCV reduction benchmark ===\n");
    std::printf("OpenCV %s, cv::getNumThreads() = %d; binCV is single-threaded\n",
                CV_VERSION, cv::getNumThreads());
    std::printf("Denominator: cv::countNonZero on the same content as CV_8U (ARCHITECTURE 10.3)\n");
    std::printf("popcount lowering: %s\n", popcountLowering());
    std::printf("x86_64 numbers are INDICATIVE. close on the reference device.\n");

    struct Extent {
        int width;
        int height;
    };
    // 640x480 and 94x60 are the two extremes name (a frame and pyramid
    // level 3); 1024x1024 is the cache-resident middle; 8192x4096 is the only one
    // where the CV_8U side does not fit in last-level cache and binCV does.
    const Extent sizes[] = {{94, 60}, {640, 480}, {1024, 1024}, {8192, 4096}};

    bool ok = true;
    for (const Extent& s : sizes) ok = runSize(s.width, s.height) && ok;

    std::printf("\nsink: %zu (printed so no reduction above can be optimized away)\n",
                static_cast<size_t>(g_sink));
    return ok ? 0 : 1;
}
