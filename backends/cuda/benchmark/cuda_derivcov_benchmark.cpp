// The ternary/N-bit derivative and the gradient covariance, priced at birth.
//
// EVERY ARM HERE RUNS ON ONE EXPLICIT STREAM, both sides of every pair, the
// launch floor included. That is a protocol decision with a measurement behind
// it (docs/reports/cuda.md, and cuda_role_benchmark.cpp's note): OpenCV's
// cudafilters calls cudaDeviceSynchronize() inside apply() when and only when
// it is handed the default stream, so a default-stream comparison prices an arm
// nobody would use -- measured elsewhere in this backend at up to 7.18x against
// binCV controls at 1.03x. The legacy default stream also implicitly
// synchronizes with every blocking stream, so timing one arm there and the
// other on an explicit stream makes each arm's event bracket contain part of
// the other's work.
//
// THE DECISION RULE WAS WRITTEN BEFORE ANY OF THESE NUMBERS EXISTED and is
// printed by this binary, so a reader can reject the bar rather than inherit
// it. Two of its magnitudes are deliberately ABSENT: where no magnitude is
// derivable, this file says so and leaves the call to the owner. It does not
// fill one in.
//
// ONE RUN OF THIS BINARY IS NOT A NUMBER. Small kernels on this host sit near a
// 8.7-14.4 us launch floor with 24-338% spread. What is quotable is the median
// across at least seven independent PROCESS runs of the per-round interleaved
// medians below, and for every ratio, whether the two arms' sample RANGES
// overlap -- which printPaired states at each pair rather than leaving to the
// reader.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/core/types.hpp"
#include "bincv/cuda/covariance.hpp"
#include "bincv/cuda/derivative.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/covariance.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/quantMat.hpp"
#include "cuda_bench_util.hpp"

#if BINCV_CUDA_DERIVCOV_OPENCV
#  include <opencv2/core.hpp>
#  include <opencv2/core/cuda.hpp>
#  include <opencv2/core/cuda_stream_accessor.hpp>
#  include <opencv2/cudafilters.hpp>
#endif

using namespace cudabench;
using bincv::BinMatConstView;
using bincv::BinMatView;
using bincv::GradientCovariance;
using bincv::QuantMat;
using bincv::Rect;
using bincv::SignedQuantMat;
using bincv::cuda::DeviceBinMat;
using bincv::cuda::DeviceGradientCovariance;
using bincv::cuda::DevicePlaneBlockConstView;

namespace {

constexpr size_t kW = 752;   // the reference frame
constexpr size_t kH = 480;
// THE SECOND SIZE EXISTS BECAUSE OF CASE A3, NOT FOR VARIETY. At 752x480 both
// axes at N = 1 move 225,600 B, which is ~0.4 us at this device's bandwidth
// against a launch floor near 8 us -- so every derivative arm at the reference
// size IS the floor, and a ratio between two of them is a ratio between two
// launches. The rule says raise the size until the comparison is decidable
// rather than record a launch-overhead ratio as a kernel result. 3840x2160
// moves 5.9 MB per fused call, comfortably clear of it.
constexpr size_t kW2 = 3840;
constexpr size_t kH2 = 2160;
constexpr size_t kWindows = 200;  // the tracker's keypoint set
constexpr int kWin = 31;          // the tracker's window
// Rounds and batch size per paired comparison, INSIDE one process. Both are
// larger than the other family benchmarks use, for a reason this run measured:
// the first pass at 50 iterations and 15 rounds put the two ~1.00x CONTROLS at
// 0.89x and 1.17x with 100-400% spreads. A control that cannot resolve identity
// cannot certify anything else, so the batch is deep enough for the control to
// be readable. Both arms of a pair always use the same batch, so the comparison
// stays internally consistent even though a deeper batch pipelines.
constexpr int kRounds = 21;
constexpr int kIters = 300;
// Working sets per memory reading. This driver reserves in 2 MB units, which is
// larger than ONE of these working sets -- so a small replica count reads the
// allocator's leftover slack rather than the footprint. Measured here at 16
// replicas: 131,072 B/set against a predicted 230,400 B, the whole shortfall
// being under one unit. The reading is quoted as an INTERVAL of width
// step/kReplicas for exactly that reason.
constexpr int kReplicas = 64;

cudaStream_t gStream = nullptr;

uint64_t rngState = 0x9E3779B97F4A7C15ULL;
uint8_t nextByte() {
    rngState = rngState * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint8_t>(rngState >> 40);
}

/// A frame with structure in it, so a derivative has something to do other than
/// produce zeros: pure noise makes every pixel an edge and a flat frame makes
/// none, and both make a gradient covariance degenerate.
std::vector<uint8_t> makeFrame(size_t w, size_t h) {
    std::vector<uint8_t> img(w * h);
    for (auto& v : img) v = nextByte();
    std::vector<uint8_t> tmp(img);
    for (size_t y = 1; y + 1 < h; ++y)
        for (size_t x = 1; x + 1 < w; ++x)
            tmp[y * w + x] = static_cast<uint8_t>(
                (img[y * w + x - 1] + 2u * img[y * w + x] + img[y * w + x + 1]) / 4u);
    for (size_t y = 1; y + 1 < h; ++y)
        for (size_t x = 1; x + 1 < w; ++x)
            img[y * w + x] = static_cast<uint8_t>(
                (tmp[(y - 1) * w + x] + 2u * tmp[y * w + x] + tmp[(y + 1) * w + x]) / 4u);
    return img;
}

template <size_t N>
void fillQuant(QuantMat<N, uint32_t>& m, const std::vector<uint8_t>& frame) {
    BinMatView<uint32_t> planes[N];
    for (size_t p = 0; p < N; ++p) planes[p] = m.plane(p);
    bincv::packQuant<bincv::QuantRule::Scale, N, uint8_t, uint32_t>(
        frame.data(), m.getWidth(), m.getHeight(), m.getWidth(), planes);
}

template <size_t N>
BinMatConstView<uint32_t> stackOf(const QuantMat<N, uint32_t>& m) {
    return BinMatConstView<uint32_t>(m.data(), m.getWidth(), N * m.getHeight(),
                                     m.getAlignedWidth());
}

std::vector<Rect> makeWindows(size_t count, int win, size_t w, size_t h) {
    std::vector<Rect> r;
    r.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const int x = static_cast<int>((i * 37u) % (w - static_cast<size_t>(win)));
        const int y = static_cast<int>((i * 53u) % (h - static_cast<size_t>(win)));
        r.emplace_back(x, y, win, win);
    }
    return r;
}

void rule() {
    std::printf(
        "\n=====================================================================\n"
        " THE DECISION RULE -- WRITTEN BEFORE ANY MEASUREMENT\n"
        "=====================================================================\n"
        " Four cases. Two magnitudes are deliberately ABSENT: where none is\n"
        " derivable, this says so and leaves the call to the owner.\n"
        "\n"
        " CASE A -- the ternary/N-bit derivative.\n"
        "   Baseline: cv::cuda::createDerivFilter(CV_8UC1, CV_16SC1, dx, dy,\n"
        "   ksize=1, normalize=false, scale=1, BORDER_REFLECT_101) from\n"
        "   cudafilters. cv::cuda::createLinearFilter is NOT a second candidate:\n"
        "   probed against this very OpenCV, it asserts dstType == srcType\n"
        "   (cudafilters/src/filtering.cpp:237), so its 8U->16S form does not\n"
        "   construct, and an 8U->8U form cannot represent a signed derivative\n"
        "   at all -- it saturates the negative half to zero, which is a\n"
        "   different answer and not a slower arm. So there is ONE denominator\n"
        "   and the 'whichever is faster' clause is struck.\n"
        "   A1 MEMORY: report BOTH sides in ABSOLUTE BYTES, cudaMemGetInfo delta,\n"
        "     taken identically. The GATE IS ARITHMETIC AGREEMENT with each\n"
        "     side's predicted geometry, not a ratio; the ratio is a REPORTED\n"
        "     number. If the owner wants a memory floor below which the op does\n"
        "     not ship, that floor is theirs -- it is not derivable from the\n"
        "     compulsory traffic, and an earlier draft's '>= 4x' was 8x halved\n"
        "     by an unstated safety factor.\n"
        "   A2 SPEED: strictly faster than that denominator by more than the\n"
        "     LARGER of the two arms' printed spreads. No percentage is invented.\n"
        "   A3 VALIDITY, AND IT RUNS FIRST: the empty-launch floor. If binCV's\n"
        "     arm is within spread of the floor, A2 is NOT DECIDABLE at this size\n"
        "     and is recorded so -- never as a kernel result.\n"
        "\n"
        " CASE B -- does the fused derivativeXY arm exist at all.\n"
        "   Baseline: the two single-axis kernels, same binary, same stream,\n"
        "   through the runtime switch. Metric: byte-identical output AND faster\n"
        "   than two launches by more than both arms' printed per-round ranges.\n"
        "   THIS IS A SIGNATURE RESULT, NOT A KERNEL RESULT: the predicted win is\n"
        "   one launch out of two, so it is bounded by this platform's launch\n"
        "   overhead and decays toward 1.00x as the frame grows into\n"
        "   traffic-bound territory. Failure action: DELETE the arm and its\n"
        "   switch -- a dead arm is a second thing to keep bit-exact forever.\n"
        "\n"
        " CASE C -- the N-bit bit-sliced gradient covariance.\n"
        "   C1 CORRECTNESS, a gate before any timing, needing no threshold: the\n"
        "     device kernel equals the host N-bit kernel at N = 1..4, equals the\n"
        "     host FIVE-ARGUMENT ternary kernel at N = 1, equals this backend's\n"
        "     countCovarianceBatchAsync recombined at N = 1, and all three batch\n"
        "     arms equal the single-region form. tests/test_cuda_derivcov.cu.\n"
        "   C2 MEMORY, pass/fail: ZERO scratch -- 0 B beyond the two blocks read,\n"
        "     the window array and the result array -- no allocation inside any\n"
        "     kernel, <= 24 B per window. Meter: allocation sum, binCV to binCV\n"
        "     only, plus a cudaMemGetInfo delta ACROSS the call that must be 0.\n"
        "   C3 SPEED: **STOP AND ASK. NO MAGNITUDE IS SET HERE.** Neither\n"
        "     cv::cuda nor cv:: computes this quantity at any API level, so\n"
        "     ruling R2 applies: it ships on correctness + memory + the host\n"
        "     comparison with the speed verdict recorded OUTSTANDING. The host\n"
        "     ratio below is printed as CONTEXT and is NOT a ship gate. An\n"
        "     earlier draft derived '> 5x at N = 2' by multiplying two numbers\n"
        "     measured on different machines with different kernels and then\n"
        "     discounting the product; that is deleted, not re-derived.\n"
        "\n"
        " CASE D -- VALIDITY, not performance: the arms timed are the arms that\n"
        "   ran. Each optimized arm prints its on/off ratio in ONE binary, and\n"
        "   each prints a case ITS OWN GATE EXCLUDES, which must read ~1.00x:\n"
        "     * derivativeX timed with the FUSED switch flipped -- that switch\n"
        "       cannot reach a single-axis call.\n"
        "     * the single-region covariance timed with the THREAD-ARM switch\n"
        "       flipped -- that switch cannot reach a one-traversal entry point.\n"
        "     * the batch at a 40-pixel-wide window with the ALIGNED-RUN switch\n"
        "       flipped -- one funnel shift reaches 32 bits, so the arm's own\n"
        "       gate excludes any run wider than that.\n"
        "   Both families also get an empty-launch FLOOR arm, and any A/B inside\n"
        "   the floor's spread is reported NOT DECIDED rather than 'no win'.\n"
        "\n"
        " HYGIENE: one meter per comparison, named at the number; cudaMemGetInfo\n"
        " on BOTH sides of anything crossing to OpenCV and allocation sums for\n"
        " binCV against binCV, never crossed. A microbenchmark ratio is not an\n"
        " end-to-end result and is labelled at every printing.\n");
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return 77;
    }
    bool quiet = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--no-rule") == 0) quiet = true;

    std::printf("=====================================================================\n"
                " binCV CUDA -- the ternary derivative and the gradient covariance\n"
                "=====================================================================\n");
    printDevice();
    std::printf(" frame %zux%zu, %zu windows of %dx%d (the tracker's shape)\n", kW, kH,
                kWindows, kWin, kWin);
    std::printf("\n **THE GPU MAY BE SHARED WHILE THIS RUNS. Every number below is\n"
                " INDICATIVE until a serial pass re-takes it.** One run is not a\n"
                " number: the quotable figure is the median of at least seven\n"
                " independent process runs of these per-round medians.\n");
    if (!quiet) rule();

    cudaStreamCreate(&gStream);

    // A3/D: THE FLOOR RUNS FIRST, so that "is this a kernel result at all" is
    // answered before any ratio is read rather than after one is missed.
    const Timing floor = measureLaunchFloor(dim3(1), dim3(32), 100, 25, 250.0, gStream);
    std::printf("\n");
    printLaunchFloor(floor);

    // ------------------------------------------------------------------
    // The inputs, on both sides.
    // ------------------------------------------------------------------
    const std::vector<uint8_t> frame = makeFrame(kW, kH);
    QuantMat<1, uint32_t> src1(static_cast<int>(kW), static_cast<int>(kH));
    QuantMat<2, uint32_t> src2(static_cast<int>(kW), static_cast<int>(kH));
    fillQuant<1>(src1, frame);
    fillQuant<2>(src2, frame);

    DeviceBinMat dSrc1(static_cast<int>(kW), static_cast<int>(kH));
    DeviceBinMat dSrc2(static_cast<int>(kW), static_cast<int>(2 * kH));
    bincv::cuda::upload(stackOf(src1), dSrc1.view(), gStream);
    bincv::cuda::upload(stackOf(src2), dSrc2.view(), gStream);

    DeviceBinMat dDx1(static_cast<int>(kW), static_cast<int>(2 * kH));
    DeviceBinMat dDy1(static_cast<int>(kW), static_cast<int>(2 * kH));
    DeviceBinMat dDx2(static_cast<int>(kW), static_cast<int>(3 * kH));
    DeviceBinMat dDy2(static_cast<int>(kW), static_cast<int>(3 * kH));
    cudaStreamSynchronize(gStream);

    // The second size, for Case A3: the same kernels where the traffic, not the
    // launch, is the cost.
    const std::vector<uint8_t> big = makeFrame(kW2, kH2);
    QuantMat<1, uint32_t> bigSrc(static_cast<int>(kW2), static_cast<int>(kH2));
    fillQuant<1>(bigSrc, big);
    DeviceBinMat dBigSrc(static_cast<int>(kW2), static_cast<int>(kH2));
    DeviceBinMat dBigDx(static_cast<int>(kW2), static_cast<int>(2 * kH2));
    DeviceBinMat dBigDy(static_cast<int>(kW2), static_cast<int>(2 * kH2));
    bincv::cuda::upload(stackOf(bigSrc), dBigSrc.view(), gStream);
    cudaStreamSynchronize(gStream);
    const auto bs1 = bincv::cuda::planeBlock(dBigSrc.constView(), 1);
    const auto bx1 = bincv::cuda::planeBlock(dBigDx.view(), 2);
    const auto by1 = bincv::cuda::planeBlock(dBigDy.view(), 2);

    const auto s1 = bincv::cuda::planeBlock(dSrc1.constView(), 1);
    const auto s2 = bincv::cuda::planeBlock(dSrc2.constView(), 2);
    const auto x1 = bincv::cuda::planeBlock(dDx1.view(), 2);
    const auto y1 = bincv::cuda::planeBlock(dDy1.view(), 2);
    const auto x2 = bincv::cuda::planeBlock(dDx2.view(), 3);
    const auto y2 = bincv::cuda::planeBlock(dDy2.view(), 3);

    // ==================================================================
    // 1. THE DERIVATIVE -- binCV's own arms, against the floor
    // ==================================================================
    std::printf("\n=====================================================================\n"
                " 1. THE DERIVATIVE -- binCV arms, kernel-resident, one explicit stream\n"
                "=====================================================================\n");
    {
        const Timing tx1 = timeKernel([&] { bincv::cuda::derivativeX(s1, x1, bincv::BORDER_REFLECT_101, false, gStream); },
                                      50, kRounds, gStream);
        const Timing ty1 = timeKernel([&] { bincv::cuda::derivativeY(s1, y1, bincv::BORDER_REFLECT_101, false, gStream); },
                                      50, kRounds, gStream);
        const Timing txy1 = timeKernel([&] { bincv::cuda::derivativeXY(s1, x1, y1, bincv::BORDER_REFLECT_101, false, gStream); },
                                       50, kRounds, gStream);
        const Timing tx2 = timeKernel([&] { bincv::cuda::derivativeX(s2, x2, bincv::BORDER_REFLECT_101, false, gStream); },
                                      50, kRounds, gStream);
        const Timing txy2 = timeKernel([&] { bincv::cuda::derivativeXY(s2, x2, y2, bincv::BORDER_REFLECT_101, false, gStream); },
                                       50, kRounds, gStream);
        printArmVsFloor("derivativeX  N=1 (ternary)", tx1, floor, "kernel");
        printArmVsFloor("derivativeY  N=1 (ternary)", ty1, floor, "kernel");
        printArmVsFloor("derivativeXY N=1, fused", txy1, floor, "kernel");
        printArmVsFloor("derivativeX  N=2", tx2, floor, "kernel");
        printArmVsFloor("derivativeXY N=2, fused", txy2, floor, "kernel");
        std::printf("\n COMPULSORY TRAFFIC, for reading those against the floor: both axes\n"
                    " at N = 1 move 1 bit in and 2x2 bits out per pixel = %zu B, which at\n"
                    " this device's peak bandwidth is well under a microsecond. If the arms\n"
                    " above sit on the floor, that is the reason, and a RATIO between two\n"
                    " such arms reports almost none of either kernel.\n",
                    kW * kH / 8 + 2 * 2 * kW * kH / 8);

        // A3 CONTINUED: the same kernels at a size where they are not the floor.
        const Timing bx = timeKernel([&] { bincv::cuda::derivativeX(bs1, bx1, bincv::BORDER_REFLECT_101, false, gStream); },
                                     kIters / 10, kRounds, gStream);
        const Timing bxy = timeKernel([&] { bincv::cuda::derivativeXY(bs1, bx1, by1, bincv::BORDER_REFLECT_101, false, gStream); },
                                      kIters / 10, kRounds, gStream);
        std::printf("\n THE SAME KERNELS AT %zux%zu, where the traffic (%zu B for both\n"
                    " axes) is ~%.1f us at this device's peak and the launch is not the\n"
                    " cost. This is the size at which a derivative ratio is a KERNEL\n"
                    " result; the rows above are a LAUNCH result and are labelled so.\n",
                    kW2, kH2, kW2 * kH2 / 8 + 2 * 2 * kW2 * kH2 / 8,
                    static_cast<double>(kW2 * kH2 / 8 + 2 * 2 * kW2 * kH2 / 8) / 608.0e3);
        printArmVsFloor("derivativeX  N=1 @ 3840x2160", bx, floor, "kernel");
        printArmVsFloor("derivativeXY N=1 @ 3840x2160, fused", bxy, floor, "kernel");
    }

    // ==================================================================
    // 2. CASE B -- the fused arm, on and off, in ONE binary
    // ==================================================================
    std::printf("\n=====================================================================\n"
                " 2. CASE B -- derivativeXY fused, ON vs OFF (the runtime switch)\n"
                "=====================================================================\n"
                " Arm A is the fused arm ON (one launch); arm B is the SAME CALL with\n"
                " impl::derivativeFusedArmEnabled() false, which issues the two\n"
                " single-axis launches. B/A above 1.00x means the fused arm is ahead.\n"
                " A SIGNATURE RESULT: the saving is one launch out of two, so it is\n"
                " bounded by launch overhead and decays toward 1.00x as the frame\n"
                " grows into traffic-bound territory. It is NOT a kernel result.\n");
    {
        const auto fused = [&] {
            bincv::cuda::impl::derivativeFusedArmEnabled() = true;
            bincv::cuda::derivativeXY(s1, x1, y1, bincv::BORDER_REFLECT_101, false, gStream);
        };
        const auto split = [&] {
            bincv::cuda::impl::derivativeFusedArmEnabled() = false;
            bincv::cuda::derivativeXY(s1, x1, y1, bincv::BORDER_REFLECT_101, false, gStream);
        };
        const PairedTiming p = timeKernelPaired(fused, split, kIters, kIters, kRounds, gStream);
        printPaired("derivativeXY N=1, fused arm ON", "derivativeXY N=1, fused arm OFF", p,
                    "kernel");

        const auto fused2 = [&] {
            bincv::cuda::impl::derivativeFusedArmEnabled() = true;
            bincv::cuda::derivativeXY(s2, x2, y2, bincv::BORDER_REFLECT_101, false, gStream);
        };
        const auto split2 = [&] {
            bincv::cuda::impl::derivativeFusedArmEnabled() = false;
            bincv::cuda::derivativeXY(s2, x2, y2, bincv::BORDER_REFLECT_101, false, gStream);
        };
        const PairedTiming p2 = timeKernelPaired(fused2, split2, kIters, kIters, kRounds, gStream);
        printPaired("derivativeXY N=2, fused arm ON", "derivativeXY N=2, fused arm OFF", p2,
                    "kernel");

        // THE DECAY THE RULE PREDICTED, MEASURED RATHER THAN ASSERTED. At
        // 3840x2160 the same pair moves 26x the bytes behind the same two
        // launches, so if the win really is the launch it has to shrink toward
        // 1.00x here. That is what makes this a SIGNATURE result: its value is
        // bounded by the platform's launch overhead, not by the kernel.
        const auto fusedBig = [&] {
            bincv::cuda::impl::derivativeFusedArmEnabled() = true;
            bincv::cuda::derivativeXY(bs1, bx1, by1, bincv::BORDER_REFLECT_101, false, gStream);
        };
        const auto splitBig = [&] {
            bincv::cuda::impl::derivativeFusedArmEnabled() = false;
            bincv::cuda::derivativeXY(bs1, bx1, by1, bincv::BORDER_REFLECT_101, false, gStream);
        };
        const PairedTiming pb = timeKernelPaired(fusedBig, splitBig, kIters / 10, kIters / 10,
                                                 kRounds, gStream);
        printPaired("derivativeXY N=1 @ 3840x2160, fused ON",
                    "derivativeXY N=1 @ 3840x2160, fused OFF", pb, "kernel");
        std::printf("   Read the two sizes TOGETHER: a 4K ratio materially closer to\n"
                    "   1.00x than the 752x480 one says the win IS the launch.\n");
        bincv::cuda::impl::derivativeFusedArmEnabled() = true;
    }

    // ==================================================================
    // 3. CASE D control -- the fused switch cannot reach a single-axis call
    // ==================================================================
    std::printf("\n=====================================================================\n"
                " 3. CASE D control -- derivativeX with the FUSED switch flipped\n"
                "=====================================================================\n"
                " The fused arm's gate excludes this call by construction: derivativeX\n"
                " is one axis and the fused switch selects nothing in it. So this pair\n"
                " MUST read ~1.00x. If it does not, the switch is reaching code it has\n"
                " no business reaching, and no other ratio in this file is trustworthy.\n");
    {
        const auto on = [&] {
            bincv::cuda::impl::derivativeFusedArmEnabled() = true;
            bincv::cuda::derivativeX(s1, x1, bincv::BORDER_REFLECT_101, false, gStream);
        };
        const auto off = [&] {
            bincv::cuda::impl::derivativeFusedArmEnabled() = false;
            bincv::cuda::derivativeX(s1, x1, bincv::BORDER_REFLECT_101, false, gStream);
        };
        const PairedTiming p = timeKernelPaired(on, off, kIters, kIters, kRounds, gStream);
        printPaired("derivativeX, fused switch ON", "derivativeX, fused switch OFF", p,
                    "kernel", /*expect1x=*/true);
        bincv::cuda::impl::derivativeFusedArmEnabled() = true;
    }

    // ==================================================================
    // 4. THE COVARIANCE -- the batch, its two arms, and its controls
    // ==================================================================
    std::printf("\n=====================================================================\n"
                " 4. THE COVARIANCE -- %zu windows of %dx%d, ONE launch\n"
                "=====================================================================\n",
                kWindows, kWin, kWin);
    // Real derivative planes, produced on the device, are what the covariance
    // reads -- not random words, whose sign structure would be uniform.
    bincv::cuda::derivativeXY(s1, x1, y1, bincv::BORDER_REFLECT_101, false, gStream);
    bincv::cuda::derivativeXY(s2, x2, y2, bincv::BORDER_REFLECT_101, false, gStream);
    cudaStreamSynchronize(gStream);
    const DevicePlaneBlockConstView cx1 = bincv::cuda::planeBlock(dDx1.constView(), 2);
    const DevicePlaneBlockConstView cy1 = bincv::cuda::planeBlock(dDy1.constView(), 2);
    const DevicePlaneBlockConstView cx2 = bincv::cuda::planeBlock(dDx2.constView(), 3);
    const DevicePlaneBlockConstView cy2 = bincv::cuda::planeBlock(dDy2.constView(), 3);

    const std::vector<Rect> wins = makeWindows(kWindows, kWin, kW, kH);
    const std::vector<Rect> wide = makeWindows(kWindows, 40, kW, kH);
    bincv::cuda::DeviceArray<Rect> dWins(kWindows);
    bincv::cuda::DeviceArray<Rect> dWide(kWindows);
    bincv::cuda::DeviceArray<DeviceGradientCovariance> dOut(kWindows);
    cudaMemcpy(dWins.data(), wins.data(), kWindows * sizeof(Rect), cudaMemcpyHostToDevice);
    cudaMemcpy(dWide.data(), wide.data(), kWindows * sizeof(Rect), cudaMemcpyHostToDevice);

    const auto batch = [&](DevicePlaneBlockConstView a, DevicePlaneBlockConstView b,
                           const Rect* w) {
        bincv::cuda::gradientCovarianceBatchAsync(a, b, w, kWindows, dOut.data(), gStream);
    };
    {
        const Timing t1 = timeKernel([&] { batch(cx1, cy1, dWins.data()); }, kIters, kRounds, gStream);
        const Timing t2 = timeKernel([&] { batch(cx2, cy2, dWins.data()); }, kIters, kRounds, gStream);
        printArmVsFloor("gradientCovarianceBatch N=1 (ternary)", t1, floor, "kernel");
        printArmVsFloor("gradientCovarianceBatch N=2", t2, floor, "kernel");
        std::printf("\n Per word this issues 3N^2 + N popcounts against 2N + 2 loads: 4 on 4\n"
                    " at N = 1 (traffic-shaped) and 14 on 6 at N = 2, the shipped ladder's\n"
                    " operating point. __popc is ONE instruction here; on aarch64 the same\n"
                    " count costs two register-domain crossings per word.\n");
    }

    std::printf("\n --- the ALIGNED-RUN (funnel shift) arm ---\n"
                " Arm A divides (row, word) PAIRS across the block and masks each word;\n"
                " arm B extracts each row's whole run with ONE __funnelshift_r and\n"
                " divides ROWS. A 31-pixel run straddles two words on every row, so each\n"
                " masked __popc in arm A sees ~15.5 useful bits of 32; aligned, the 62\n"
                " units of work at 31x31 become 31. B/A under 1.00x means the funnel is\n"
                " ahead. Not gated on N: ptxas reports 0 bytes spilled at every N in\n"
                " [1, 4], so no cutoff is derivable and none is invented.\n"
                "\n **THIS ARM DEFAULTS OFF: IT IS A MEASURED REGRESSION.** It is timed\n"
                " here so the number that rejected it can be re-taken on a quiet device,\n"
                " which a deleted arm could not be. A ONE-THREAD-PER-WINDOW ARM WAS ALSO\n"
                " written and priced and IS gone: it lost 14 readings out of 14 (1.23x\n"
                " slower at N = 1, 1.76x at N = 2) because 200 windows is 200 THREADS --\n"
                " two blocks on a 48-SM part, leaving the machine idle, where one block\n"
                " per window is 12,800 threads.\n");
    for (int depth = 1; depth <= 2; ++depth) {
        const DevicePlaneBlockConstView a = (depth == 1) ? cx1 : cx2;
        const DevicePlaneBlockConstView b = (depth == 1) ? cy1 : cy2;
        const auto plain = [&] {
            bincv::cuda::impl::covarianceAlignedRunEnabled() = false;
            batch(a, b, dWins.data());
        };
        const auto aligned = [&] {
            bincv::cuda::impl::covarianceAlignedRunEnabled() = true;
            batch(a, b, dWins.data());
        };
        const PairedTiming p = timeKernelPaired(plain, aligned, kIters, kIters, kRounds, gStream);
        char na[64], nb[64];
        std::snprintf(na, sizeof na, "covariance N=%d, 31x31 per-word (ref)", depth);
        std::snprintf(nb, sizeof nb, "covariance N=%d, 31x31 funnel-aligned (opt)", depth);
        printPaired(na, nb, p, "kernel");
        bincv::cuda::impl::covarianceAlignedRunEnabled() = false;  // the shipped default
    }
    std::printf("\n PREDICTED BEFORE MEASURING, AND THE ARITHMETIC IS THE RULE: at 31x31\n"
                " with a %u-thread block the per-word arm has 31 rows x 2 words = 62\n"
                " units and the aligned arm 31, and BOTH are under %u -- so every thread\n"
                " makes exactly ONE pass either way and the reduction that follows is\n"
                " identical. The funnel therefore CANNOT show here, and ~1.00x above is\n"
                " the prediction confirmed rather than the arm failing. The work\n"
                " reduction can only reach the clock where the per-word arm needs MORE\n"
                " PASSES than the aligned one, so that is the shape it must be priced on.\n",
                64u, 64u);

    // THE SHAPE WHERE THE ARITHMETIC SAYS IT MUST SHOW. 31 wide x 240 tall:
    // 480 per-word units against 240 aligned, i.e. 8 passes against 4 over a
    // 64-thread block. The FORMULA is the bar -- the ratio's ceiling is 2.00x
    // and the gate is agreement with it. If the ratio does not move materially
    // toward 2.00x here, the work reduction is not reaching the clock at any
    // shape and the arm does not earn its second implementation.
    {
        // Generated so that every window lies WHOLLY INSIDE the frame: a window
        // clipped by the bottom edge would be shorter than 240 rows and the
        // pass count the rule is written against would not be the one running.
        // 4000 windows, not the tracker's 200: at 200 the whole batch is ~50 KB
        // and sits ON the launch floor, where halving the word visits cannot
        // show whatever it does to the work. This is Case A3's discipline
        // applied to the covariance -- raise the size until the comparison is
        // DECIDABLE rather than record a launch-bound ratio as a kernel result.
        // 4000 x 240 rows x 2 words x 4 loads is ~30 MB, comfortably clear.
        constexpr size_t kTallCount = 4000;
        std::vector<Rect> tallFixed;
        tallFixed.reserve(kTallCount);
        for (size_t i = 0; i < kTallCount; ++i)
            tallFixed.emplace_back(static_cast<int>((i * 37u) % (kW - 31u)),
                                   static_cast<int>((i * 53u) % (kH - 240u)), 31, 240);
        bincv::cuda::DeviceArray<Rect> dTall(tallFixed.size());
        cudaMemcpy(dTall.data(), tallFixed.data(), tallFixed.size() * sizeof(Rect),
                   cudaMemcpyHostToDevice);
        bincv::cuda::DeviceArray<DeviceGradientCovariance> dTallOut(tallFixed.size());
        const auto tallBatch = [&](bool on) {
            bincv::cuda::impl::covarianceAlignedRunEnabled() = on;
            bincv::cuda::gradientCovarianceBatchAsync(cx1, cy1, dTall.data(),
                                                      tallFixed.size(), dTallOut.data(),
                                                      gStream);
        };
        const PairedTiming p = timeKernelPaired([&] { tallBatch(false); },
                                                [&] { tallBatch(true); }, kIters, kIters,
                                                kRounds, gStream);
        std::printf("\n 4000 windows of 31 wide x 240 tall -- 8 passes against 4, so the\n"
                    " formula's ceiling is 2.00x and agreement with it is the gate:\n");
        printPaired("covariance N=1, 31x240 per-word (ref)",
                    "covariance N=1, 31x240 funnel-aligned (opt)", p, "kernel");
        bincv::cuda::impl::covarianceAlignedRunEnabled() = false;  // the shipped default
    }

    // ==================================================================
    // 5. CASE D controls for the covariance
    // ==================================================================
    std::printf("\n=====================================================================\n"
                " 5. CASE D controls -- two cases these switches' own gates exclude\n"
                "=====================================================================\n"
                " (a) A 40-PIXEL-WIDE window. One funnel shift reaches 32 bits, so the\n"
                "     aligned arm's own gate excludes any wider run and both settings\n"
                "     take the same per-word loop. MUST read ~1.00x.\n");
    {
        const auto plain = [&] {
            bincv::cuda::impl::covarianceAlignedRunEnabled() = false;
            batch(cx1, cy1, dWide.data());
        };
        const auto aligned = [&] {
            bincv::cuda::impl::covarianceAlignedRunEnabled() = true;
            batch(cx1, cy1, dWide.data());
        };
        const PairedTiming p = timeKernelPaired(plain, aligned, kIters, kIters, kRounds, gStream);
        printPaired("covariance 40px window, aligned OFF",
                    "covariance 40px window, aligned ON", p, "kernel", /*expect1x=*/true);
        bincv::cuda::impl::covarianceAlignedRunEnabled() = false;  // the shipped default
    }
    std::printf("\n (b) The SINGLE-REGION form with the ALIGNED-RUN switch flipped. That\n"
                "     switch selects a BATCH traversal and cannot reach a grid-stride\n"
                "     entry point whose region may be a whole frame. MUST read ~1.00x.\n");
    {
        DeviceGradientCovariance* one = nullptr;
        cudaMalloc(&one, sizeof(DeviceGradientCovariance));
        const Rect whole{0, 0, static_cast<int>(kW), static_cast<int>(kH)};
        const auto on = [&] {
            bincv::cuda::impl::covarianceAlignedRunEnabled() = true;
            bincv::cuda::gradientCovarianceAsync(cx1, cy1, whole, one, gStream);
        };
        const auto off = [&] {
            bincv::cuda::impl::covarianceAlignedRunEnabled() = false;
            bincv::cuda::gradientCovarianceAsync(cx1, cy1, whole, one, gStream);
        };
        const PairedTiming p = timeKernelPaired(on, off, kIters / 4, kIters / 4, kRounds, gStream);
        printPaired("single-region cov, aligned switch ON",
                    "single-region cov, aligned switch OFF", p, "kernel", /*expect1x=*/true);
        bincv::cuda::impl::covarianceAlignedRunEnabled() = false;  // the shipped default
        // The single-region form over a whole frame is also the arm with real
        // work in it, so its distance from the floor says whether anything in
        // section 4 was a kernel measurement at all.
        printArmVsFloor("single-region covariance, WHOLE FRAME", p.a, floor, "kernel");
        cudaFree(one);
    }

    // ==================================================================
    // 6. THE HOST ARM -- CONTEXT ONLY. Ruling R2, and a different clock.
    // ==================================================================
    std::printf("\n=====================================================================\n"
                " 6. THE HOST ARM -- CONTEXT, NOT A SHIP GATE\n"
                "=====================================================================\n"
                " Neither cv::cuda nor cv:: computes a 2x2 gradient covariance at any\n"
                " API level, so ruling R2 applies and this operation's SPEED VERDICT IS\n"
                " OUTSTANDING. The row below is the binCV HOST arm on this machine, on a\n"
                " DIFFERENT CLOCK (host steady_clock around a synchronous call, against\n"
                " CUDA events above), and this x86 host under WSL2 is recorded at 30-130%%\n"
                " spread for CPU arms. It is printed so the reader knows what the device\n"
                " is being compared to, and it decides nothing.\n");
    {
        SignedQuantMat<1, uint32_t> hx(static_cast<int>(kW), static_cast<int>(kH));
        SignedQuantMat<1, uint32_t> hy(static_cast<int>(kW), static_cast<int>(kH));
        bincv::derivativeX<1, uint32_t>(src1, hx);
        bincv::derivativeY<1, uint32_t>(src1, hy);
        int64_t sink = 0;
        std::vector<double> samples;
        for (int r = 0; r < 9; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            for (const Rect& w : wins) {
                const GradientCovariance g =
                    bincv::gradientCovariance<uint32_t>(hx.constMagnitude(0),
                                                        hy.constMagnitude(0), hx.constSign(),
                                                        hy.constSign(), w);
                sink += g.sumXX + g.sumYY + g.sumXY;
            }
            samples.push_back(std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count());
        }
        const Timing h = summarize(std::move(samples));
        printArm("HOST gradientCovariance, 200 windows N=1", h, "host steady_clock");
        std::printf("   (sink %lld -- kept so the loop is not elided)\n",
                    static_cast<long long>(sink));
    }

    // ==================================================================
    // 7. MEMORY -- binCV to binCV on meter 1, and the no-allocation reading
    // ==================================================================
    std::printf("\n=====================================================================\n"
                " 7. MEMORY\n"
                "=====================================================================\n");
    printMemoryHeader("the derivative-covariance family at 752x480");
    {
        const size_t planeBytes = bincv::cuda::rowWords(kW) * 4 * kH;
        printAllocSum("source binary level, N=1", planeBytes);
        printAllocSum("dx signed block (2 planes)", 2 * planeBytes);
        printAllocSum("dy signed block (2 planes)", 2 * planeBytes);
        printAllocSum("window array, 200 x 16 B", kWindows * sizeof(Rect));
        printAllocSum("result array, 200 x 24 B",
                      kWindows * sizeof(DeviceGradientCovariance));
        printAllocSum("FAMILY TOTAL, N=1", 5 * planeBytes + kWindows * (sizeof(Rect) +
                                           sizeof(DeviceGradientCovariance)));
        printAllocSum("FAMILY TOTAL, N=2 (src+2x3 planes)",
                      7 * planeBytes + kWindows * (sizeof(Rect) +
                                       sizeof(DeviceGradientCovariance)));
        std::printf("   [meter 1 is binCV-to-binCV ONLY. It is never divided into the\n"
                    "    cudaMemGetInfo readings in section 8.]\n");
    }
    {
        // C2: NO ALLOCATION INSIDE ANY KERNEL, as a reading rather than a claim.
        // The driver reserves in multi-MB units, so any allocation the batch made
        // would move this by at least one unit. Zero is the pass.
        const size_t step = measureDriverMeterStep();
        DeviceMemMeter m;
        for (int i = 0; i < 200; ++i) batch(cx2, cy2, dWins.data());
        cudaStreamSynchronize(gStream);
        const size_t delta = m.deltaBytes();
        std::printf("\n   [meter 2: cudaMemGetInfo] across 200 batch launches: %zu B\n"
                    "                             (driver step measured here: %.2f MB)\n"
                    "   C2 SCRATCH GATE: %s -- the covariance allocates nothing, forms no\n"
                    "   selector plane, and puts 24 B per window on the bus.\n",
                    delta, static_cast<double>(step) / (1024.0 * 1024.0),
                    delta == 0 ? "PASS (0 B)" : "FAIL");
    }

    // ==================================================================
    // 8. THE ROLE BAR -- cv::cuda::createDerivFilter, on THIS stream
    // ==================================================================
    std::printf("\n=====================================================================\n"
                " 8. CASE A ROLE BAR -- cv::cuda::createDerivFilter (cudafilters)\n"
                "=====================================================================\n");
#if BINCV_CUDA_DERIVCOV_OPENCV
    {
        cv::cuda::setDevice(0);
        { cv::cuda::GpuMat warm(16, 16, CV_8UC1); warm.setTo(cv::Scalar(0)); }
        cudaDeviceSynchronize();
        cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(gStream);

        cv::Mat hostFrame(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                          const_cast<uint8_t*>(frame.data()));
        cv::cuda::GpuMat gSrc;
        gSrc.upload(hostFrame);
        cv::cuda::GpuMat gDx(static_cast<int>(kH), static_cast<int>(kW), CV_16SC1);
        cv::cuda::GpuMat gDy(static_cast<int>(kH), static_cast<int>(kW), CV_16SC1);
        // ksize = 1 with dx = 1 gives getDerivKernels' 3-tap [-1, 0, 1] against a
        // size-1 column kernel -- the same tap and the same borders binCV runs.
        // The outputs differ in RANGE and STORAGE, not in sign structure:
        // {-255..255} in CV_16S against binCV's {-1, 0, +1} in two bit planes.
        // That is stated at the number and never folded into it.
        auto fx = cv::cuda::createDerivFilter(CV_8UC1, CV_16SC1, 1, 0, 1, false, 1.0,
                                              cv::BORDER_REFLECT101, cv::BORDER_REFLECT101);
        auto fy = cv::cuda::createDerivFilter(CV_8UC1, CV_16SC1, 0, 1, 1, false, 1.0,
                                              cv::BORDER_REFLECT101, cv::BORDER_REFLECT101);

        const auto cvArm = [&] {
            fx->apply(gSrc, gDx, cvStream);
            fy->apply(gSrc, gDy, cvStream);
        };
        const auto binArm = [&] {
            bincv::cuda::derivativeXY(s1, x1, y1, bincv::BORDER_REFLECT_101, false, gStream);
        };
        const PairedTiming p = timeKernelPaired(cvArm, binArm, kIters / 4, kIters / 4, kRounds, gStream);
        std::printf(" Both arms on ONE EXPLICIT STREAM. Arm A is OpenCV, arm B is binCV,\n"
                    " so B/A under 1.00x is binCV ahead. ROLE-ONLY: same taps, same\n"
                    " borders, same sign structure; different range and different storage\n"
                    " (CV_16S per pixel against 2 bits per pixel).\n");
        printPaired("cv::cuda::createDerivFilter, both axes",
                    "binCV derivativeXY N=1, both axes", p, "kernel");
        printArmVsFloor("  ...binCV's arm against the launch floor", p.b, floor, "kernel");
        std::printf("   READ THOSE TWO LINES TOGETHER. When binCV's arm is AT THE LAUNCH\n"
                    "   FLOOR the comparison is still decidable -- OpenCV's arm is far\n"
                    "   above it, so the ranges can separate -- but the honest statement\n"
                    "   is NOT 'binCV's kernel is 4x faster'. It is: binCV's derivative\n"
                    "   costs a LAUNCH at this size, and OpenCV's costs a launch plus its\n"
                    "   filter work. The binCV side is a floor reading, so the ratio is a\n"
                    "   lower bound on the gap and says nothing about binCV's kernel.\n");

        // A1: MEMORY, cudaMemGetInfo ON BOTH SIDES, absolute bytes, replicated so
        // the delta clears the driver's unit many times over.
        const size_t step = measureDriverMeterStep();
        std::printf("\n MEMORY, CASE A1. One meter across the boundary: cudaMemGetInfo,\n"
                    " taken identically on both sides over %d working sets (the driver\n"
                    " reserves in %.2f MB units, larger than one working set). The GATE\n"
                    " is arithmetic agreement with each side's predicted geometry; the\n"
                    " ratio is a REPORTED number and not a bar.\n",
                    kReplicas, static_cast<double>(step) / (1024.0 * 1024.0));
        size_t binDelta = 0, cvDelta = 0;
        {
            DeviceMemMeter m;
            std::vector<DeviceBinMat> keep;
            keep.reserve(static_cast<size_t>(kReplicas) * 3);
            for (int i = 0; i < kReplicas; ++i) {
                keep.emplace_back(static_cast<int>(kW), static_cast<int>(kH));
                keep.emplace_back(static_cast<int>(kW), static_cast<int>(2 * kH));
                keep.emplace_back(static_cast<int>(kW), static_cast<int>(2 * kH));
            }
            binDelta = m.deltaBytes();
        }
        {
            DeviceMemMeter m;
            std::vector<cv::cuda::GpuMat> keep;
            std::vector<cv::Ptr<cv::cuda::Filter>> filters;
            keep.reserve(static_cast<size_t>(kReplicas) * 3);
            for (int i = 0; i < kReplicas; ++i) {
                keep.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
                keep.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_16SC1);
                keep.emplace_back(static_cast<int>(kH), static_cast<int>(kW), CV_16SC1);
                // THE FILTER OBJECTS ARE PART OF WHAT A CALLER PAYS. A separable
                // filter holds its own intermediate buffer, and it is allocated
                // lazily -- so each is APPLIED once inside the metered scope.
                filters.push_back(cv::cuda::createDerivFilter(
                    CV_8UC1, CV_16SC1, 1, 0, 1, false, 1.0, cv::BORDER_REFLECT101,
                    cv::BORDER_REFLECT101));
                filters.push_back(cv::cuda::createDerivFilter(
                    CV_8UC1, CV_16SC1, 0, 1, 1, false, 1.0, cv::BORDER_REFLECT101,
                    cv::BORDER_REFLECT101));
                filters[filters.size() - 2]->apply(keep[keep.size() - 3],
                                                   keep[keep.size() - 2]);
                filters[filters.size() - 1]->apply(keep[keep.size() - 3],
                                                   keep[keep.size() - 1]);
            }
            cudaDeviceSynchronize();
            cvDelta = m.deltaBytes();
        }
        // THE METER'S OWN GRANULARITY IS PART OF THE READING. A delta can
        // undercount by up to one whole unit -- the allocator serves small
        // requests out of slack the driver already reserved -- so the honest
        // figure per working set is an INTERVAL of width step/kReplicas, not a
        // point. Quoting the point alone is how a 131,072 B reading of a
        // 230,400 B footprint gets published.
        const double binLo = static_cast<double>(binDelta) / kReplicas;
        const double binHi = static_cast<double>(binDelta + step) / kReplicas;
        const double cvLo = static_cast<double>(cvDelta) / kReplicas;
        const double cvHi = static_cast<double>(cvDelta + step) / kReplicas;
        const size_t predicted = 5 * bincv::cuda::rowWords(kW) * 4 * kH;
        std::printf("   [meter 2: cudaMemGetInfo] binCV  %9.0f .. %9.0f B/working set\n"
                    "                             PREDICTED %zu B -- %s\n",
                    binLo, binHi, predicted,
                    (static_cast<double>(predicted) >= binLo &&
                     static_cast<double>(predicted) <= binHi)
                        ? "AGREES, A1 arithmetic gate PASSES"
                        : "OUTSIDE the meter's interval -- the arithmetic is wrong and "
                          "the design is re-derived before anything ships");
        std::printf("   [meter 2: cudaMemGetInfo] OpenCV %9.0f .. %9.0f B/working set\n"
                    "                             (an UPPER BOUND: GpuMat may pool, and\n"
                    "                              anything a filter holds rather than\n"
                    "                              frees lands inside the delta)\n",
                    cvLo, cvHi);
        std::printf("   ratio, ONE meter both sides: %.1fx .. %.1fx smaller\n",
                    binHi > 0.0 ? cvLo / binHi : 0.0, binLo > 0.0 ? cvHi / binLo : 0.0);
        std::printf("   Read the binCV number against the prediction, NOT against the\n"
                    "   ratio: a miss there means the arithmetic is wrong and the design\n"
                    "   is re-derived before anything ships. The OpenCV number is an\n"
                    "   UPPER BOUND -- anything a filter holds rather than frees lands\n"
                    "   inside the delta -- and it INCLUDES the separable filters'\n"
                    "   CV_32F intermediates, which an earlier memory plan omitted.\n");
    }
#else
    std::printf(" NOT COMPILED IN. This target is always built, so the role bar lives\n"
                " behind BINCV_CUDA_DERIVCOV_OPENCV; point BINCV_CUDA_OPENCV_DIR at an\n"
                " OpenCV with cudafilters to take it.\n"
                " CASE A VERDICT: **BLOCKED** -- role bar UNMEASURED. No substitute bar\n"
                " is invented and no CPU number is quoted as a GPU one.\n");
#endif

    // ==================================================================
    // 9. THE VERDICTS
    // ==================================================================
    std::printf("\n=====================================================================\n"
                " 9. VERDICTS, against the rule printed at the top\n"
                "=====================================================================\n"
                " CASE A  derivative: read A3 (the floor) FIRST. If the binCV arm sits on\n"
                "         the floor, A2 is NOT DECIDABLE at 752x480 and is recorded so.\n"
                "         A1 is the two absolute readings in section 8 against their\n"
                "         predictions. WHAT THIS DOES NOT DECIDE: adoption. The number\n"
                "         that decides that is the derivative's share of a RESIDENT GPU\n"
                "         FRONTEND, which does not exist -- on the host the derivative is\n"
                "         3.0-3.3%% of the whole frontend, so even an infinite speedup on\n"
                "         it is worth ~1.03x there. The share verdict is OUTSTANDING, and\n"
                "         whether the op ships on A1+A2 alone is the OWNER'S call under\n"
                "         the 2026-09-15 both-axes rule. This file does not answer it.\n"
                " CASE B  fused arm: section 2. Ships only if it clears both printed\n"
                "         ranges; otherwise the arm and its switch are DELETED.\n"
                " CASE B' the COVARIANCE's two candidate optimized arms both FAILED this\n"
                "         same rule and neither ships enabled. One-thread-per-window lost\n"
                "         14/14 and is deleted; the funnel-aligned run lost 1.09-1.15x at\n"
                "         the size where the comparison is decidable and DEFAULTS OFF --\n"
                "         kept reachable only so a serial pass can re-take an indicative\n"
                "         number. It saves popcounts and not loads, and at N = 1 this\n"
                "         kernel is traffic-shaped, so the half it saves was never the\n"
                "         cost. The shipped covariance is ONE straightforward kernel.\n"
                " CASE C  covariance: C1 is green in tests/test_cuda_derivcov.cu (986\n"
                "         checks). C2 is the 0 B reading in section 7. **C3 IS A\n"
                "         STOP-AND-ASK: no cv::cuda or cv:: counterpart exists at any API\n"
                "         level, so the speed verdict is OUTSTANDING under ruling R2. The\n"
                "         host row in section 6 is context on a different clock and is\n"
                "         not a bar.** cornerHarris / createMinEigenValCorner compute a\n"
                "         response THROUGH a covariance and are the honest comparison for\n"
                "         a COMPOSED corner operation, which this family does not provide.\n"
                " CASE D  the three ~1.00x controls in sections 3 and 5. If any of them\n"
                "         is not ~1.00x, nothing else here is trustworthy.\n"
                "\n **INDICATIVE. The GPU may be shared during this run.** Aggregate at\n"
                " least seven process runs before quoting anything.\n");

    cudaStreamDestroy(gStream);
    return 0;
}
