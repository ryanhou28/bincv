// logic kernels versus OpenCV, on the same binary content.
//
// THE DENOMINATOR (the design notes, CLAUDE.md): OpenCV performing the SAME
// semantic operation on the SAME binary content stored as CV_8U -- because that
// is exactly what a user does today without binCV. Not grayscale (different
// information content), not a hand-written per-pixel strawman. cv::bitwise_and on
// a {0,255} mask is the function that call site really calls.
//
// This is the first operation in the project where the thesis is testable: the
// two sides compute the same answer over the same pixels, and binCV moves an
// eighth of the bytes to do it. Whether that turns into speed is a measurement,
// not an argument -- so the numbers this prints are the claim, including when
// they are unflattering.
//
// ---------------------------------------------------------------------------
// MEASUREMENT VALIDITY (EXPERIMENTS.md, "Verify the benchmark measures
// something") -- four hazards, each answered here rather than assumed away:
//
// 1. DEAD CODE. A loop whose result is unused is deleted, and the number looks
// spectacular -- the first platform comparison in this repo was eliminated on
// both platforms and reported 524288 GB/s. Every iteration folds words from
// its destination into a `volatile` sink, and each destination's popcount is
// printed, so a kernel that computed nothing would be visible twice.
//
// 2. CONSTANT FOLDING / CACHED RESULTS. The inputs rotate through four distinct
// random pairs, so no iteration can reuse the previous one's answer and no
// input is a compile-time constant.
//
// 3. A PHYSICAL BOUND. These kernels are memory-bound, so their throughput
// cannot exceed what the machine can move. A one-read-one-write word copy at
// each footprint measures that ceiling in the same harness, and a kernel
// reporting more than 1.5x the probe is flagged IMPLAUSIBLE.
//
// THE PROBE IS NOISY AND SAYS SO. Measured across three runs of the shipped
// benchmark at ONE footprint (307200 B): 122.61, 102.20 and 74.08 GB/s -- a
// 1.65x spread, the same magnitude as the effect being bounded. The probe
// therefore runs more batches than the kernels do and prints its SLOWEST as
// well as its fastest batch, so a reader can see how much the ceiling itself
// moved. A bound whose own spread is unreported is not a bound.
//
// WHY THE THRESHOLD IS 1.5x AND NOT 4x: the probe is one load and one store
// per word; a two-input bitwise op is two loads and one store. A core with
// two load ports and one store port retires somewhat MORE bytes per cycle on
// the kernel than on the copy, so a kernel a little above the probe is
// expected rather than suspicious -- but only a little. The port-count
// argument supports roughly 1.5x. It supported nothing like the 4x that was
// here before, which with a 100-140 GB/s probe needed 400-560 GB/s to fire:
// a kernel measured three times too fast passed it silently, so the check
// could only ever have caught total elimination.
//
// std::memcpy is measured too, and is NOT the bound. It is not usable as one
// here: at 38400 B it reports ~130 GB/s and at 307200 B ~8 GB/s, and at
// 33 MB it comes back UP to ~25 GB/s. Whatever produces that shape -- store
// strategy switching with size is the usual explanation, but the numbers
// above are not monotonic in either direction and this file does not claim
// to have identified the cause -- a primitive that swings 16x across the
// sizes under test cannot bound them. It is printed because the swing itself
// is worth seeing next to any absolute number here.
//
// 4. THE TWO SIDES MUST AGREE. Before timing anything, each operation is run
// once through both implementations and the set-pixel counts compared. A
// "faster" kernel computing a different answer is not a result -- so a
// disagreement SKIPS the timing for that size entirely and makes the process
// exit non-zero. It used to print a warning and then print the full ratio
// table anyway, which is a benchmark publishing numbers it has itself
// declared meaningless.
//
// 5. FOOTPRINT IS PART OF THE RESULT, not a detail of the setup. binCV's
// buffers are an eighth the size of OpenCV's, so at most sizes the two sides
// sit at DIFFERENT levels of the memory hierarchy and the ratio is set by
// which caches each one fits in rather than by the 8x traffic ratio. At
// 640x480 and 1024x1024 both sides are cache-resident and the ratio lands
// near 8x; at 8192x4096 -- binCV 12.6 MB, OpenCV 100 MB on a 32 MB L3 -- it
// is 13-20x, because only one side went to DRAM. All three are measured, and
// the reported GB/s is cache bandwidth at the first two.
//
// Timing is the MINIMUM over several batches rather than the mean: the minimum is
// the run least disturbed by the scheduler, and a mean over a machine with other
// work on it measures the other work. Batch length is calibrated per case so every
// measurement covers a comparable interval regardless of image size.
//
// Release only (CLAUDE.md). Run:
//./build/benchmark/logic_benchmark
//./build/benchmark/logic_benchmark --width 752 --height 480
//
// Exit status is 0 only if every size timed cleanly.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "bench_util.hpp"
#include "bincv-cpp/ops/logic.hpp"

namespace {

// The sink every timed iteration reports into. `volatile` is what makes the
// stores to it observable, so nothing upstream of them can be deleted.
volatile uint64_t g_sink = 0;

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

/// @brief Builds the SAME content twice: packed at 1 bit per pixel, and as the
/// CV_8U {0,255} mask a user would have without binCV.
/// @note One draw per pixel, written into both representations from the same
/// draw, so the two sides are not merely statistically similar -- they are
/// the same image, which is what makes the comparison a like-for-like one.
void makePair(int width, int height, uint64_t seed, bincv::BinMat<uint32_t>& packed32,
              bincv::BinMat<uint64_t>& packed64, cv::Mat& mask) {
    packed32 = bincv::BinMat<uint32_t>(width, height);
    packed64 = bincv::BinMat<uint64_t>(width, height);
    mask = cv::Mat::zeros(height, width, CV_8U);

    uint64_t state = seed;
    for (int y = 0; y < height; ++y) {
        uint8_t* row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            const bool bit = (nextRandom(state) >> 40) < (UINT64_C(1) << 23);   // ~50% fill
            if (bit) {
                packed32.set(y, x, true);
                packed64.set(y, x, true);
                row[x] = 255;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// @brief Iterations that make one batch last about `targetMs`.
/// @note Fixed iteration counts measure clock resolution at small sizes and
/// nothing else; this keeps every case's batch comparable.
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

/// @brief Best and worst nanoseconds per iteration over `repeats` batches.
/// @note Both, not just the best. The reported figure is the minimum -- the batch
/// least disturbed by the scheduler -- but the spread between the two is
/// what says whether that minimum means anything, and for the physical-bound
/// probe it is the difference between a ceiling and a number.
struct Timing {
    double bestNs = -1.0;
    double worstNs = -1.0;
};

template <typename Body>
Timing measureTiming(Body body, int repeats, double targetMs) {
    const int iterations = calibrate(body, targetMs);
    Timing t;
    for (int r = 0; r < repeats; ++r) {
        const auto start = bench::Clock::now();
        for (int i = 0; i < iterations; ++i) body(i);
        const double ns =
            std::chrono::duration<double, std::nano>(bench::Clock::now() - start).count() /
            static_cast<double>(iterations);
        if (t.bestNs < 0.0 || ns < t.bestNs) t.bestNs = ns;
        if (t.worstNs < 0.0 || ns > t.worstNs) t.worstNs = ns;
    }
    return t;
}

/// @brief Nanoseconds per iteration: the minimum over `repeats` batches.
template <typename Body>
double measureNs(Body body, int repeats, double targetMs) {
    return measureTiming(body, repeats, targetMs).bestNs;
}

double gigabytesPerSecond(double bytesPerCall, double nsPerCall) {
    if (nsPerCall <= 0.0) return 0.0;
    return bytesPerCall / nsPerCall;   // bytes/ns == GB/s (10^9 bytes per second)
}

// ---------------------------------------------------------------------------
// One reported row
// ---------------------------------------------------------------------------

struct Row {
    std::string op;
    std::string impl;
    double nsPerPixel = 0.0;
    double gbPerSecond = 0.0;
    double bytesPerBuffer = 0.0;
};

std::vector<Row> g_rows;

void report(const std::string& op, const std::string& impl, double nsPerCall, double pixels,
            double trafficBytes, double bufferBytes) {
    Row row;
    row.op = op;
    row.impl = impl;
    row.nsPerPixel = nsPerCall / pixels;
    row.gbPerSecond = gigabytesPerSecond(trafficBytes, nsPerCall);
    row.bytesPerBuffer = bufferBytes;
    g_rows.push_back(row);
    std::printf("[BENCH] %-22s %-16s %10.5f ns/px %8.2f GB/s\n", op.c_str(), impl.c_str(),
                row.nsPerPixel, row.gbPerSecond);
}

enum class Op { And, Or, Xor, Not };

const char* opName(Op op) {
    switch (op) {
        case Op::And: return "bitwiseAnd";
        case Op::Or:  return "bitwiseOr";
        case Op::Xor: return "bitwiseXor";
        case Op::Not: return "bitwiseNot";
    }
    return "?";
}

/// @brief Bytes of memory traffic one call moves: two reads and a write, or one
/// read and a write for the unary operation.
double trafficFactor(Op op) { return op == Op::Not ? 2.0 : 3.0; }

template <typename WordType>
void runPacked(Op op, const std::vector<bincv::BinMat<WordType>>& a,
               const std::vector<bincv::BinMat<WordType>>& b, bincv::BinMat<WordType>& dst,
               const std::string& implName, double pixels, int repeats, double targetMs) {
    const size_t words = dst.sizeInWords();
    const double bufferBytes = static_cast<double>(words * sizeof(WordType));

    auto body = [&](int i) {
        const size_t k = static_cast<size_t>(i) & 3u;
        switch (op) {
            case Op::And: bincv::bitwiseAnd(a[k].constView(), b[k].constView(), dst.view()); break;
            case Op::Or:  bincv::bitwiseOr(a[k].constView(), b[k].constView(), dst.view()); break;
            case Op::Xor: bincv::bitwiseXor(a[k].constView(), b[k].constView(), dst.view()); break;
            case Op::Not: bincv::bitwiseNot(a[k].constView(), dst.view()); break;
        }
        // Consume the result: without this the whole kernel is deletable.
        g_sink += static_cast<uint64_t>(dst.data()[0]) +
                  static_cast<uint64_t>(dst.data()[words - 1]);
    };

    const double ns = measureNs(body, repeats, targetMs);
    report(opName(op), implName, ns, pixels, trafficFactor(op) * bufferBytes, bufferBytes);
}

void runOpenCv(Op op, const std::vector<cv::Mat>& a, const std::vector<cv::Mat>& b, cv::Mat& dst,
               double pixels, int repeats, double targetMs) {
    const double bufferBytes = pixels;   // CV_8U: one byte per pixel

    auto body = [&](int i) {
        const size_t k = static_cast<size_t>(i) & 3u;
        switch (op) {
            case Op::And: cv::bitwise_and(a[k], b[k], dst); break;
            case Op::Or:  cv::bitwise_or(a[k], b[k], dst); break;
            case Op::Xor: cv::bitwise_xor(a[k], b[k], dst); break;
            case Op::Not: cv::bitwise_not(a[k], dst); break;
        }
        g_sink += static_cast<uint64_t>(dst.data[0]) +
                  static_cast<uint64_t>(dst.data[dst.total() - 1]);
    };

    const double ns = measureNs(body, repeats, targetMs);
    report(opName(op), "OpenCV CV_8U", ns, pixels, trafficFactor(op) * bufferBytes, bufferBytes);
}

/// @brief The physical bound: one read and one write per word, at this footprint.
/// @note Deliberately a plain loop rather than std::memcpy -- see hazard 3 at the
/// top of this file for the measurement that rules memcpy out as a ceiling.
/// @note Reports its own SPREAD. The fastest batch is the ceiling used by the
/// plausibility check; the slowest is printed alongside so a reader can see
/// how stable the ceiling was. Measured across three runs of the previous
/// version at one footprint: 122.61, 102.20, 74.08 GB/s.
struct Bound {
    double bestGbPerSecond = 0.0;
    double worstGbPerSecond = 0.0;
};

Bound copyLoopBound(size_t bytes, int repeats, double targetMs) {
    const size_t words = bytes / sizeof(uint64_t) + 1;
    std::vector<uint64_t> source(words, UINT64_C(0x5A5A5A5A5A5A5A5A));
    std::vector<uint64_t> destination(words, 0);
    auto body = [&](int i) {
        for (size_t k = 0; k < words; ++k) destination[k] = source[k];
        g_sink += destination[static_cast<size_t>(i) & 15u];
    };
    // More batches than the kernels get: this number divides every plausibility
    // verdict below, so it is the one measurement that must not be a single
    // unlucky sample.
    const Timing t = measureTiming(body, 3 * repeats, targetMs);
    const double traffic = 2.0 * static_cast<double>(words * sizeof(uint64_t));
    Bound b;
    b.bestGbPerSecond = gigabytesPerSecond(traffic, t.bestNs);
    b.worstGbPerSecond = gigabytesPerSecond(traffic, t.worstNs);
    return b;
}

/// @brief The same footprint through std::memcpy, reported for context only.
double memcpyGbPerSecond(size_t bytes, int repeats, double targetMs) {
    std::vector<uint8_t> source(bytes, 0x5A);
    std::vector<uint8_t> destination(bytes, 0);
    auto body = [&](int i) {
        std::memcpy(destination.data(), source.data(), bytes);
        g_sink += destination[static_cast<size_t>(i) & 15u];
    };
    const double ns = measureNs(body, repeats, targetMs);
    return gigabytesPerSecond(2.0 * static_cast<double>(bytes), ns);
}

/// @brief The two implementations must compute the same image before either is
/// timed. Counts, not a full comparison: tests/test_logic.cpp already
/// proves bit-exactness over the whole the matrix, and this is the guard
/// against a benchmark that drifted out of step with it.
bool resultsAgree(const bincv::BinMat<uint32_t>& a32, const bincv::BinMat<uint32_t>& b32,
                  const cv::Mat& a8, const cv::Mat& b8, int width, int height) {
    const Op ops[] = {Op::And, Op::Or, Op::Xor, Op::Not};
    bool agree = true;
    for (Op op : ops) {
        bincv::BinMat<uint32_t> packedDst(width, height);
        cv::Mat maskDst;
        switch (op) {
            case Op::And:
                bincv::bitwiseAnd(a32.constView(), b32.constView(), packedDst.view());
                cv::bitwise_and(a8, b8, maskDst);
                break;
            case Op::Or:
                bincv::bitwiseOr(a32.constView(), b32.constView(), packedDst.view());
                cv::bitwise_or(a8, b8, maskDst);
                break;
            case Op::Xor:
                bincv::bitwiseXor(a32.constView(), b32.constView(), packedDst.view());
                cv::bitwise_xor(a8, b8, maskDst);
                break;
            case Op::Not:
                bincv::bitwiseNot(a32.constView(), packedDst.view());
                cv::bitwise_not(a8, maskDst);
                break;
        }
        const int packedCount = packedDst.countNonZero();
        const int maskCount = cv::countNonZero(maskDst);
        std::printf(" %-12s set pixels: binCV %8d OpenCV %8d %s\n", opName(op), packedCount,
                    maskCount, packedCount == maskCount ? "agree" : "DISAGREE");
        if (packedCount != maskCount) agree = false;
    }
    return agree;
}

/// @brief The denominator's own build configuration, which the version string
/// does not carry.
/// @note MEASURED, and the reason this is printed at all: two OpenCV 4.x installs
/// on ONE machine, both configured Release with the same dispatch list, ran
/// cv::bitwise_not 3.2-4.5x apart (0.019 ns/px against 0.062-0.085 ns/px)
/// while cv::bitwise_and / _or / _xor matched within run-to-run noise. A
/// published "18x faster than OpenCV" for the unary operation was therefore
/// a property of one local build, not of binCV. A ratio is only as
/// reportable as its denominator is identifiable, so the denominator now
/// identifies itself in the output.
void printOpenCvDispatch() {
    const std::string info = cv::getBuildInformation();
    const char* wanted[] = {"Baseline:", "Dispatched code generation:", "Configuration:"};
    size_t pos = 0;
    while (pos < info.size()) {
        size_t end = info.find('\n', pos);
        if (end == std::string::npos) end = info.size();
        const std::string line = info.substr(pos, end - pos);
        for (const char* w : wanted) {
            if (line.find(w) != std::string::npos) {
                std::printf(" OpenCV build:%s\n", line.c_str());
                break;
            }
        }
        pos = end + 1;
    }
}

/// @brief One size, timed. Returns false if the size was not measured cleanly.
bool runSize(int width, int height, int repeats, double targetMs) {
    const double pixels = static_cast<double>(width) * static_cast<double>(height);

    std::printf("\n================ %d x %d (%.0f pixels) ================\n", width, height,
                pixels);

    // Four distinct input pairs, rotated through by every timed loop.
    std::vector<bincv::BinMat<uint32_t>> a32(4), b32(4);
    std::vector<bincv::BinMat<uint64_t>> a64(4), b64(4);
    std::vector<cv::Mat> a8(4), b8(4);
    for (size_t k = 0; k < 4; ++k) {
        makePair(width, height, UINT64_C(0xB1CF00D) + k * 977u, a32[k], a64[k], a8[k]);
        makePair(width, height, UINT64_C(0xB1CF00D) + k * 977u + 1u, b32[k], b64[k], b8[k]);
    }

    bincv::BinMat<uint32_t> dst32(width, height);
    bincv::BinMat<uint64_t> dst64(width, height);
    cv::Mat dst8(height, width, CV_8U);

    const double packedBytes32 = static_cast<double>(dst32.sizeInWords() * sizeof(uint32_t));
    const double packedBytes64 = static_cast<double>(dst64.sizeInWords() * sizeof(uint64_t));
    std::printf(" footprint per buffer: binCV uint32 %.0f B, binCV uint64 %.0f B, "
                "OpenCV CV_8U %.0f B (%.1fx)\n",
                packedBytes32, packedBytes64, pixels, pixels / packedBytes32);

    // Hazard 4: a disagreement is fatal to this size, not a footnote above the
    // table. Nothing below is printed, so no scraper of the [BENCH] lines and no
    // CI wrapper can consume numbers this benchmark has declared meaningless.
    if (!resultsAgree(a32[0], b32[0], a8[0], b8[0], width, height)) {
        std::printf(" RESULTS DISAGREE -- SKIPPING THIS SIZE. The two implementations do not\n"
                    " compute the same image, so no timing of them is a comparison.\n");
        return false;
    }

    const Bound boundPacked =
        copyLoopBound(static_cast<size_t>(packedBytes32), repeats, targetMs);
    const Bound boundBytes = copyLoopBound(static_cast<size_t>(pixels), repeats, targetMs);
    std::printf(" physical bound (1 read + 1 write word copy, fastest batch; worst in "
                "parentheses):\n %.0f B buffer %.2f GB/s (%.2f) %.0f B buffer %.2f GB/s "
                "(%.2f)\n",
                packedBytes32, boundPacked.bestGbPerSecond, boundPacked.worstGbPerSecond, pixels,
                boundBytes.bestGbPerSecond, boundBytes.worstGbPerSecond);
    std::printf(" std::memcpy at the same footprints (context only, NOT the bound, and not "
                "monotonic in size): %.2f GB/s, %.2f GB/s\n",
                memcpyGbPerSecond(static_cast<size_t>(packedBytes32), repeats, targetMs),
                memcpyGbPerSecond(static_cast<size_t>(pixels), repeats, targetMs));
    std::printf(" working set per call: binCV %.0f B, OpenCV %.0f B (three buffers)\n",
                3.0 * packedBytes32, 3.0 * pixels);

    const size_t firstRow = g_rows.size();
    const Op ops[] = {Op::And, Op::Or, Op::Xor, Op::Not};
    for (Op op : ops) {
        runPacked(op, a32, b32, dst32, "binCV uint32", pixels, repeats, targetMs);
        runPacked(op, a64, b64, dst64, "binCV uint64", pixels, repeats, targetMs);
        runOpenCv(op, a8, b8, dst8, pixels, repeats, targetMs);
    }

    // --- the summary table, and the ratio the task is actually about ---------
    bool plausible = true;
    std::printf("\n %-12s %-16s %12s %10s %12s\n", "OP", "IMPLEMENTATION", "ns/pixel", "GB/s",
                "vs OpenCV");
    std::printf(" ---------------------------------------------------------------------\n");
    double openCvNs = 0.0;
    for (size_t i = firstRow; i < g_rows.size(); ++i) {
        const Row& row = g_rows[i];
        if (row.impl == "OpenCV CV_8U") openCvNs = row.nsPerPixel;
    }
    for (size_t i = firstRow; i < g_rows.size(); ++i) {
        const Row& row = g_rows[i];
        // The denominator for the ratio column is the OpenCV row of the SAME
        // operation, which is the row three back at most.
        double reference = 0.0;
        for (size_t j = firstRow; j < g_rows.size(); ++j) {
            if (g_rows[j].op == row.op && g_rows[j].impl == "OpenCV CV_8U") {
                reference = g_rows[j].nsPerPixel;
            }
        }
        if (reference <= 0.0) reference = openCvNs;
        std::printf(" %-12s %-16s %12.5f %10.2f %11.2fx\n", row.op.c_str(), row.impl.c_str(),
                    row.nsPerPixel, row.gbPerSecond,
                    row.nsPerPixel > 0.0 ? reference / row.nsPerPixel : 0.0);

        // The physical-bound check (EXPERIMENTS.md). A memory-bound kernel that
        // reports more than the machine can move is a broken measurement, not a
        // fast kernel. 1.5x, not 4x: see hazard 3 for why 4x could not fire.
        const double bound = (row.impl == "OpenCV CV_8U") ? boundBytes.bestGbPerSecond
                                                          : boundPacked.bestGbPerSecond;
        if (row.gbPerSecond > 1.5 * bound) {
            std::printf(" IMPLAUSIBLE: %.2f GB/s is more than 1.5x the %.2f GB/s copy-loop "
                        "bound at this footprint -- treat this row as a broken measurement\n",
                        row.gbPerSecond, bound);
            plausible = false;
        }
    }
    std::printf(" ratio > 1 means binCV is faster. sink=%llu\n",
                static_cast<unsigned long long>(g_sink));
    return plausible;
}

}  // namespace

int main(int argc, char* argv[]) {
    bench::Config cfg = bench::parseArgs(argc, argv);

    std::printf("\nT2.2 logic kernels vs OpenCV on the same binary content as CV_8U\n");
    std::printf("(ARCHITECTURE 10.3: the denominator is what a user runs today without binCV)\n");
    // Reported rather than pinned: a user gets whatever OpenCV does by default, so
    // that is what the denominator should be -- but a reader cannot interpret the
    // ratio without knowing whether the other side was allowed more than one core.
    // binCV is single-threaded here and has no threading of any kind today.
    //
    // NOTE that taskset -c <one core> leaves cv::getNumThreads at 1, whatever
    // the default would have been unpinned. The published log once recorded the
    // unpinned value in its header and the pinned value in every raw run.
    std::printf("OpenCV %s, cv::getNumThreads() = %d; binCV is single-threaded\n",
                cv::getVersionString().c_str(), cv::getNumThreads());
    printOpenCvDispatch();

    const int repeats = 9;
    const double targetMs = 20.0;

    // The two sizes asks for, a third that is DRAM-resident on one side and
    // not the other, plus whatever was passed on the command line (the shared
    // runner always passes 640x480, which is already here).
    //
    // 8192x4096 is not decoration. At the first two sizes BOTH implementations fit
    // in cache, so the ratio is close to the 8x traffic ratio; at this one OpenCV's
    // 100 MB working set does not fit a 32 MB L3 and binCV's 12.6 MB does, and the
    // ratio roughly doubles. A benchmark that reported only the cache-resident
    // sizes would be describing one regime and implying the other.
    int sizes[][2] = {{640, 480}, {1024, 1024}, {8192, 4096}, {cfg.width, cfg.height}};
    bool allClean = true;
    for (const auto& size : sizes) {
        bool duplicate = false;
        for (const auto& earlier : sizes) {
            if (&earlier == &size) break;
            if (earlier[0] == size[0] && earlier[1] == size[1]) duplicate = true;
        }
        if (duplicate || size[0] <= 0 || size[1] <= 0) continue;
        if (!runSize(size[0], size[1], repeats, targetMs)) allClean = false;
    }

    if (!allClean) {
        std::printf("\nAT LEAST ONE SIZE DID NOT MEASURE CLEANLY -- exiting non-zero. Do not\n"
                    "quote any number above without reading why.\n");
    }
    return allClean ? 0 : 1;
}
