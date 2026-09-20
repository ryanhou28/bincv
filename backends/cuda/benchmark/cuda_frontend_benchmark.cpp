// The frontend's entry point, priced: FAST, the minimum-eigenvalue response,
// good-features selection and sub-pixel refinement.
//
// ONE EXECUTABLE, ALWAYS BUILT, with the cv::cuda role arms behind
// BINCV_CUDA_FRONTEND_OPENCV inside it -- the shape cuda_window_benchmark and
// cuda_sensor_benchmark already use, and for the reason their CMake blocks give:
// scripts/verify_cuda.sh derives its benchmark list from the TEXT of
// benchmark/CMakeLists.txt and hand-excludes exactly one name, so a target that
// exists only when an OpenCV is pointed at is one the gate tries to build on
// every machine that has not built one. Without that OpenCV this binary still
// runs and still says something true: it reports the role bars as UNMEASURED and
// the verdicts that depend on them as BLOCKED.
//
// THE MEASUREMENT PROTOCOL, and every clause of it is a correction to something
// that went wrong before:
//
//   * BOTH SIDES ON ONE EXPLICIT STREAM. OpenCV synchronizes the whole device on
//     the default stream -- a `if (stream == 0) cudaSafeCall(cudaDeviceSynchronize())`
//     guard sits in cudev's grid transform and in every cudafilters filter. A
//     default-stream comparison measures against an arm nobody would use.
//   * INTERLEAVED, ALTERNATING ROUNDS. timeKernelPaired brackets both arms inside
//     every round and swaps their order round to round, so the quoted ratio is a
//     distribution of per-round ratios rather than a ratio of two medians taken
//     minutes apart.
//   * RANGES, NOT JUST MEDIANS. Every pair prints whether the two arms' sample
//     ranges are DISJOINT, as a fact beside the verdict. It is not the verdict:
//     that is measure_util.hpp's difference-against-spread rule on the per-round
//     ratio, which a single round slow in BOTH arms cannot veto.
//   * ONE METER PER COMPARISON, NAMED AT THE NUMBER. Allocation sums compare
//     binCV to binCV; a cudaMemGetInfo delta is the only meter readable on both
//     sides of the library boundary and is the only one used there. They are
//     never divided by each other.
//   * ONE RUN IS NOT A NUMBER. Every figure here is a median of interleaved
//     batches, and the header says plainly that a single process run on a shared
//     GPU is INDICATIVE.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/corner.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/fast.hpp"
#include "bincv/cuda/subpix.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/corner.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/fast.hpp"
#include "bincv/ops/subpix.hpp"
#include "bincv/quantMat.hpp"
#include "cuda_bench_util.hpp"

#if BINCV_CUDA_FRONTEND_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

namespace bc = bincv::cuda;
using namespace cudabench;
using Word = uint32_t;

namespace {

constexpr size_t kWidth = 752;
constexpr size_t kHeight = 480;
constexpr int kRounds = 11;

cudaStream_t gStream = nullptr;

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// A binarized frame with blob structure plus noise -- not pure noise, which
/// makes nearly every pixel a FAST corner and prices a case no caller has.
bincv::BinMat<Word> referenceFrame(uint64_t seed = 0x51EEDu) {
    bincv::BinMat<Word> m(static_cast<int>(kWidth), static_cast<int>(kHeight));
    for (size_t y = 0; y < kHeight; ++y) {
        Word* row = m.view().row(y);
        for (size_t x = 0; x < kWidth; ++x) {
            const bool blob = ((x / 17) + (y / 13)) % 2 == 0;
            const bool noise = (splitmix(seed) & 31u) == 0u;
            if (blob != noise) row[x / 32] |= (Word{1} << (x % 32));
        }
    }
    return m;
}

bincv::BinMat<Word> allOnesFrame() {
    bincv::BinMat<Word> m(static_cast<int>(kWidth), static_cast<int>(kHeight));
    for (size_t y = 0; y < kHeight; ++y) {
        Word* row = m.view().row(y);
        for (size_t x = 0; x < kWidth; ++x) row[x / 32] |= (Word{1} << (x % 32));
    }
    return m;
}

/// A wall clock with timeKernelPaired's protocol, for the one comparison whose
/// baseline is NOT a kernel: cv::cuda's good-features detector downloads its
/// candidate list and runs the minimum-distance spacing filter on the HOST, so a
/// CUDA-event clock on that side silently excludes half the operation.
#if BINCV_CUDA_FRONTEND_OPENCV
PairedTiming timeWallPaired(const std::function<void()>& bodyA,
                            const std::function<void()>& bodyB, int itersA, int itersB,
                            int repeats) {
    const auto batch = [&](const std::function<void()>& body, int iters) {
        cudaStreamSynchronize(gStream);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) body();
        cudaStreamSynchronize(gStream);
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    };
    for (int i = 0; i < itersA; ++i) bodyA();
    for (int i = 0; i < itersB; ++i) bodyB();
    cudaStreamSynchronize(gStream);

    std::vector<double> sa, sb;
    for (int r = 0; r < repeats; ++r) {
        double ta = 0.0, tb = 0.0;
        if (r % 2 == 0) {
            ta = batch(bodyA, itersA);
            tb = batch(bodyB, itersB);
        } else {
            tb = batch(bodyB, itersB);
            ta = batch(bodyA, itersA);
        }
        sa.push_back(ta);
        sb.push_back(tb);
    }
    // THE SUMMARY IS summarizePaired'S, not this function's. Assembled here it
    // left roundsFavouringA/B at zero, so the row printed a 0-0 sign split and
    // a sign-test p of 1 over rounds that were in fact unanimous -- and it took
    // the ratio's median from summarize(), whose even-count midpoint is the
    // arithmetic one that makes a ratio depend on which arm is the denominator.
    return summarizePaired(sa, sb);
}
#endif  // BINCV_CUDA_FRONTEND_OPENCV

void rule() {
    std::printf(
"============================================================================\n"
" THE DECISION RULE -- WRITTEN BEFORE ANY OF THESE NUMBERS EXISTED\n"
"============================================================================\n"
"\n"
" Three operations, three rules, because they have three different bars and no\n"
" project-wide number exists. Three magnitudes the family's own design proposed\n"
" were DELETED before measuring, and the reason is recorded beside each.\n"
"\n"
" OP1  detectFastAsync.  Role bar: cv::cuda::FastFeatureDetector::create(\n"
"      TYPE_9_16, threshold=128, nonmaxSuppression=false) on the SAME picture as\n"
"      CV_8U {0,255} -- the configuration ops/fast.hpp proves yields binCV's\n"
"      exact corner set. One side holds it at 1 bit/px, the other at 8.\n"
"   GATE   the device corner set must equal the HOST's byte for byte (the suite),\n"
"          and the OpenCV set must agree as a SET. A mismatch stops the\n"
"          comparison; it is not traded against speed.\n"
"   SPEED  required >= 1.00x kernel-resident, both sides on one explicit stream.\n"
"          Derived, not imported: binCV decides 32 pixels per lane-instruction\n"
"          and OpenCV decides one, so below parity the advantage was thrown away\n"
"          by the implementation rather than absent from the argument.\n"
"          *** THE DESIGN'S '>= 2.00x target' IS DELETED. *** It was load-bearing\n"
"          (it decided ship-unqualified against ship-with-a-shortfall-filed) and\n"
"          its only justification was circular. The project's own measurement of\n"
"          this identical algebra at 256-bit width -- eight times a CUDA lane --\n"
"          collected 1.50x on x86 and 2.37x on aarch64, so no number between 1.0\n"
"          and 32 is derivable from the instruction ratio. WHAT RATIO SHIPS THE\n"
"          FAST CLAIM UNQUALIFIED IS A JUDGEMENT NOBODY HAS MADE: it is an\n"
"          explicit STOP AND ASK, not a number filled in here.\n"
"   MEMORY *** THE DESIGN'S '>= 3.0x smaller' IS DELETED. *** Nothing derived 3.0\n"
"          and no identified failure mode lands there. In its place, two things\n"
"          that are actually checkable: (a) the INPUT-IMAGE component must measure\n"
"          8x smaller, which is true by construction and whose failure means the\n"
"          upload path is materialising bytes it should not; (b) the whole\n"
"          working set is REPORTED at each capacity with NO pass/fail number.\n"
"          The sentence that used to stand here -- that the per-corner record\n"
"          ratio is fixed by the host FastCorner struct at 16 B -- was WRONG and\n"
"          was costing 4 bytes a corner: the device record carries the bit-plane\n"
"          score, whose whole attainable range is an arc length in [1, 16], so it\n"
"          is 12 B and the host type widens it back on the way out. What is left\n"
"          fixed is the CAPACITY the caller chooses, which is the caller's.\n"
"          A ship gate on (b) is the owner's to set.\n"
"\n"
" OP2  goodFeaturesToTrackAsync.  Role bar: cv::cuda::createGoodFeaturesToTrack\n"
"      Detector(CV_8UC1, 200, 0.01, 33.333, 3, useHarris=false) over\n"
"      cv::cuda::createMinEigenValCorner. Role only and Tier 2 on BOTH sides:\n"
"      binarized [-1,0,1] taps against a Sobel over bytes.\n"
"   GATE   byte-identical Corner arrays and an identical result triple against\n"
"          the host's goodFeaturesToTrackStreaming (the suite).\n"
"   SPEED  *** NOT WRITABLE. STOP AND ASK. *** The design wrote '>= 1.00x' and\n"
"          then flagged the same number as unruled in its own open questions,\n"
"          which is exactly the laundering CLAUDE.md names -- and its stated\n"
"          derivation imports a HOST CPU-vs-CPU ratio (0.92x x86, 0.69x aarch64\n"
"          against cv::goodFeaturesToTrack) to set a GPU-vs-GPU bar, which it\n"
"          cannot do. What the owner is asked: this is the frontend's entry\n"
"          point; the fused arm's claim is the ABSENCE of every frame-sized\n"
"          intermediate; what speed result against cudaimgproc's detector ships\n"
"          it, and what result sends it back to be optimized? Until that ruling\n"
"          exists the ratio below is REPORTED and the verdict is PENDING.\n"
"   PROTOCOL  settled before measuring, because without it the ratio is\n"
"          unarguable in either direction. VERIFIED in the module source:\n"
"          cv::cuda::GoodFeaturesToTrackDetector::detect DOWNLOADS its sorted\n"
"          candidate list and runs the minDistance spacing filter ON THE HOST\n"
"          whenever minDistance >= 1 (cudaimgproc/src/gftt.cpp). The named\n"
"          configuration's minDistance is 33.333, so that is the path measured.\n"
"          'Kernel-resident time for the whole operation' is therefore not a\n"
"          quantity OpenCV's side HAS. So: WALL CLOCK around the complete detect\n"
"          call on both sides, data resident at entry and corners resident at\n"
"          exit, with OpenCV's internal round trip NAMED at the number. binCV's\n"
"          selection stays device-resident end to end, and that residency is a\n"
"          claim in its own right rather than a detail of the clock.\n"
"   MEMORY *** THE DESIGN'S '>= 5.0x smaller' IS DELETED. *** It fails its own\n"
"          stated purpose: computed from the design's own byte tables, the\n"
"          frame-float-map arm -- the arm the bar exists to reject -- lands at\n"
"          6.61x and PASSES. In its place the binary check the design already\n"
"          uses correctly for cornerSubPix: THE FUSED ARM ALLOCATES ZERO\n"
"          FRAME-SIZED DEVICE BYTES, at every frame size, checkable rather than a\n"
"          ratio -- with both arms' measured working sets reported beside it.\n"
"   ARM A/B  fused (0 frame-sized bytes) against the frame-map reference arm is\n"
"          settled by the project's OWN stated tie-break and not by an invented\n"
"          percentage: performance and footprint are co-equal, and when they\n"
"          conflict with no explicit choice made, MEMORY WINS. So if the fused\n"
"          arm is not slower than the reference arm, the fused arm ships. If the\n"
"          fused arm is SLOWER and the reference arm would clear a bar the owner\n"
"          has not set, the two goals genuinely conflict and that is a second\n"
"          stop-and-ask, not a number to fill in.\n"
"\n"
" OP3  cornerSubPixAsync.  *** THERE IS NO cv::cuda COUNTERPART. *** Not in\n"
"      cudaimgproc, cudafeatures2d, cudaarithm, cudawarping, cudafilters,\n"
"      cudaoptflow or cudastereo. Ruling R2 applies: it ships on correctness,\n"
"      memory and the host comparison, speed verdict recorded OUTSTANDING, and\n"
"      NO GPU-vs-GPU number is quoted or implied.\n"
"   GATE   byte-identical Point2f arrays (exact float compare, no tolerance) and\n"
"          identical {refined, singular, clamped, diverged} (the suite).\n"
"   DECIDES  not a standalone speed ratio -- 200 corners x 121-pixel windows is\n"
"          microseconds and a standalone ratio is dominated by the launch floor.\n"
"          The device arm ships if and only if it is CHEAPER THAN THE ROUND TRIP\n"
"          IT REPLACES:\n"
"            kernel(cornerSubPixAsync) + up(200x8 B) + down(200x8 B)\n"
"              <  down(the derivative data) + host cornerSubPix + up(200x8 B)\n"
"          Required: strictly less, both sides measured on this machine in this\n"
"          binary. AND the baseline is priced BOTH WAYS, because 'pick the right\n"
"          baseline' applies to the thing being replaced: the whole four-plane\n"
"          download (180 KB) AND the narrower alternative of downloading only the\n"
"          11-row bands around the corners. If the narrow one wins, it is the\n"
"          denominator.\n"
"   MEMORY required to allocate NOTHING frame-sized at any winHalf up to\n"
"          kMaxWinHalf. Binary, checkable, not a ratio.\n"
"\n"
" ALL THREE -- the vector-arm rule in its device spelling. Every optimized arm is\n"
" reachable at runtime, this benchmark prints the fast/reference ratio for each,\n"
" and the cases the fast path's OWN GATE excludes must read ~1.00x:\n"
"     arcLength 10  (no compile-time arc schedule -> fastTiledApplies is false)\n"
"     blockSize 7   (no bit-sliced box sums  -> cornerSlicedApplies is false)\n"
"     an all-ones frame for subPix (the set-bit skip can eliminate nothing)\n"
" If one of those does not read ~1.00x, the fast path is not running where it is\n"
" believed to be running, which is the exact failure the rule exists to catch.\n"
"\n"
" WHAT THESE MEASUREMENTS COVER. Each is ONE OPERATION IN A LOOP. Detection runs\n"
" at a duty cycle in a real frontend, so an operation ratio times that duty cycle\n"
" is what a frontend sees, and that multiplication belongs to a resident-frontend\n"
" example that does not exist yet. No end-to-end claim may be quoted from here.\n"
"\n");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return 77;
    }
    cudaStreamCreate(&gStream);

    std::printf("============================================================\n"
                "  binCV CUDA -- the frontend's entry point\n"
                "============================================================\n");
    printDevice();
    std::printf("\n *** THESE TIMINGS ARE INDICATIVE. *** They were taken on a SHARED GPU in\n"
                " a single process run. Medians of %d interleaved rounds with alternating\n"
                " arm order, and every pair reports whether its two sample ranges are\n"
                " disjoint -- but a number that ships needs a serial pass on an idle\n"
                " device, and nothing here is that pass.\n\n",
                kRounds);
    rule();

    const Timing floor = measureLaunchFloor(dim3(1), dim3(32), 100, 25, 250.0, gStream);
    printLaunchFloor(floor);
    std::printf("\n");

    // -----------------------------------------------------------------------
    // Inputs, uploaded once: every arm below runs on device-resident data.
    // -----------------------------------------------------------------------
    const bincv::BinMat<Word> frame = referenceFrame();
    const bincv::BinMat<Word> ones = allOnesFrame();
    bincv::TernaryMat<Word> hdx(static_cast<int>(kWidth), static_cast<int>(kHeight));
    bincv::TernaryMat<Word> hdy(static_cast<int>(kWidth), static_cast<int>(kHeight));
    bincv::derivativeX<1, Word>(frame, hdx);
    bincv::derivativeY<1, Word>(frame, hdy);

    bc::DeviceBinMat dframe(static_cast<int>(kWidth), static_cast<int>(kHeight));
    bc::DeviceBinMat dones(static_cast<int>(kWidth), static_cast<int>(kHeight));
    bc::DeviceBinMat dmagX(static_cast<int>(kWidth), static_cast<int>(kHeight));
    bc::DeviceBinMat dmagY(static_cast<int>(kWidth), static_cast<int>(kHeight));
    bc::DeviceBinMat dsignX(static_cast<int>(kWidth), static_cast<int>(kHeight));
    bc::DeviceBinMat dsignY(static_cast<int>(kWidth), static_cast<int>(kHeight));
    bc::upload(frame.constView(), dframe.view(), gStream);
    bc::upload(ones.constView(), dones.view(), gStream);
    bc::upload(hdx.constMagnitude(0), dmagX.view(), gStream);
    bc::upload(hdy.constMagnitude(0), dmagY.view(), gStream);
    bc::upload(hdx.constSign(), dsignX.view(), gStream);
    bc::upload(hdy.constSign(), dsignY.view(), gStream);
    cudaStreamSynchronize(gStream);

    // -----------------------------------------------------------------------
    // OP1 -- FAST
    // -----------------------------------------------------------------------
    std::printf("----------------------------------------------------------------\n"
                " OP1  detectFastAsync -- %zux%zu bit-plane, arcLength 9\n"
                "----------------------------------------------------------------\n",
                kWidth, kHeight);

    // Sized so NOTHING TRUNCATES on this frame. A truncated run cannot be compared
// against OpenCV's corner set at all -- the atomic decided which corners were
// stored -- so a capacity below the true count turns the role comparison's own
// gate into a coin toss.
const uint32_t kFastCapacity = 16384;
    bc::DeviceArray<bc::DeviceFastCorner> dfast(kFastCapacity);
    bc::DeviceAppendCounter fastCounter;
    const bc::DeviceFastCornerBuffer fastBuf(dfast.data(), fastCounter.devicePtr(),
                                             kFastCapacity);
    // REFERENCE, because this binary times BOTH arms against one buffer: the
    // ordered arm's own number is one uint32 per block and the sort arm would be
    // refused against it. A caller that runs only the shipped arm asks for
    // FastArm::Ordered and pays 380 bytes here instead of 256 KB -- which is what
    // the footprint table below prints, separately, for both.
    const size_t fastScratch =
        bc::fastScratchBytes(kWidth, kHeight, kFastCapacity, bc::FastArm::Reference);
    bc::DeviceArray<uint8_t> dfastScratch(fastScratch);

    // Carried out of the capacity pair so the role-bar verdict below can say WHERE
    // the shortfall is rather than only that there is one. Unused without the
    // OpenCV arms, which is the one build where there is no verdict to qualify.
    [[maybe_unused]] double sortShare = 0.0;
    [[maybe_unused]] double sortOnlyMs = 0.0;
    const auto runFast = [&](int arcLength) {
        fastCounter.reset(gStream);
        bc::detectFastAsync(dframe.constView(), fastBuf, dfastScratch.data(), fastScratch,
                            arcLength, gStream);
    };

    // How many corners this frame actually has, so the numbers below are read
    // against a known density rather than an assumed one.
    runFast(9);
    cudaStreamSynchronize(gStream);
    bc::DeviceAppendResult fastRes;
    bc::readAppendResult(fastBuf, fastRes, gStream);
    std::printf(" corners on this frame: %u  (%.2f%% of pixels, capacity %u%s)\n\n",
                fastRes.found(),
                100.0 * static_cast<double>(fastRes.found()) /
                    static_cast<double>(kWidth * kHeight),
                kFastCapacity, fastRes.truncated() ? ", TRUNCATED" : "");

    {
        // THE ORDERING ARM, and it is this section's headline. Both arms return
        // the host's raster order on a complete run; one gets there with a
        // single-block bitonic network over the stored corners, the other by
        // prefix-summing per-word popcounts so every corner is written at its
        // raster rank. No corner is ever compared with another on the fast arm.
        bincv::cuda::impl::fastMaskScoreEnabled() = true;
        bincv::cuda::impl::fastTiledEnabled() = true;
        const PairedTiming ordered = timeKernelPaired(
            [&] { bincv::cuda::impl::fastOrderedEnabled() = true; runFast(9); },
            [&] { bincv::cuda::impl::fastOrderedEnabled() = false; runFast(9); }, 10, 10,
            kRounds, gStream);
        printPaired("detectFast, ORDERED arm (prefix sum, no sort)",
                    "detectFast, append-and-SORT arm", ordered, "kernel");
        std::printf("   off-switch ratio sort/ordered: %.2fx  at %u corners\n\n",
                    ordered.ratioMedian, fastRes.found());

        // THE GATE-EXCLUDED CONTROL for the ordering arm: a capacity too small
        // to hold one uint32 per block in the caller's scratch. Both switch
        // positions then run the append-and-sort arm and this MUST read ~1.00x.
        bincv::cuda::impl::fastOrderedEnabled() = true;
        bc::DeviceArray<bc::DeviceFastCorner> tiny(8);
        bc::DeviceAppendCounter tinyCounter;
        const bc::DeviceFastCornerBuffer tinyBuf(tiny.data(), tinyCounter.devicePtr(), 8u);
        const size_t tinyBytes =
            bc::fastScratchBytes(kWidth, kHeight, 8, bc::FastArm::Reference);
        bc::DeviceArray<uint8_t> tinyScratch(tinyBytes + 16);
        const auto runTiny = [&] {
            tinyCounter.reset(gStream);
            bc::detectFastAsync(dframe.constView(), tinyBuf, tinyScratch.data(), tinyBytes, 9,
                                gStream);
        };
        const PairedTiming orderControl = timeKernelPaired(
            [&] { bincv::cuda::impl::fastOrderedEnabled() = true; runTiny(); },
            [&] { bincv::cuda::impl::fastOrderedEnabled() = false; runTiny(); }, 10, 10,
            kRounds, gStream);
        printPaired("CONTROL capacity 8, switch ON  (gate excludes it)",
                    "CONTROL capacity 8, switch OFF", orderControl, "kernel",
                    /*expect1x=*/true);
        std::printf("   fastOrderedApplies(%zu, %zu, 8) = %s -- the gate's own answer.\n\n",
                    kWidth, kHeight,
                    bincv::cuda::impl::fastOrderedApplies(kWidth, kHeight, 8) ? "true"
                                                                             : "FALSE");
        bincv::cuda::impl::fastOrderedEnabled() = true;

        const PairedTiming tiled = timeKernelPaired(
            [&] { bincv::cuda::impl::fastTiledEnabled() = true; runFast(9); },
            [&] { bincv::cuda::impl::fastTiledEnabled() = false; runFast(9); }, 10, 10,
            kRounds, gStream);
        printPaired("detectFast, TILED arm (arc 9)", "detectFast, reference arm (arc 9)",
                    tiled, "kernel");
        std::printf("   off-switch ratio reference/tiled: %.2fx\n\n",
                    tiled.ratioMedian);

        bincv::cuda::impl::fastTiledEnabled() = true;
        const PairedTiming score = timeKernelPaired(
            [&] { bincv::cuda::impl::fastMaskScoreEnabled() = true; runFast(9); },
            [&] { bincv::cuda::impl::fastMaskScoreEnabled() = false; runFast(9); }, 10, 10,
            kRounds, gStream);
        printPaired("detectFast, score off the ARC MASKS", "detectFast, score by RING PEEL",
                    score, "kernel");
        std::printf("\n");

        // THE GATE-EXCLUDED CONTROL. fastTiledApplies(10) is false, so both switch
        // positions run the reference arm and this MUST read ~1.00x.
        bincv::cuda::impl::fastMaskScoreEnabled() = true;
        const PairedTiming control = timeKernelPaired(
            [&] { bincv::cuda::impl::fastTiledEnabled() = true; runFast(10); },
            [&] { bincv::cuda::impl::fastTiledEnabled() = false; runFast(10); }, 10, 10,
            kRounds, gStream);
        printPaired("CONTROL arc 10, switch ON  (gate excludes it)",
                    "CONTROL arc 10, switch OFF", control, "kernel", /*expect1x=*/true);
        std::printf("   fastTiledApplies(10) = %s -- the gate's own answer, not a\n"
                    "   restatement of it.\n\n",
                    bincv::cuda::impl::fastTiledApplies(10) ? "true" : "FALSE");
        bincv::cuda::impl::fastTiledEnabled() = true;
    }

    // WHAT THE ORDERING COSTS NOW, isolated the same way it was when it was the
    // whole operation: two capacities over IDENTICAL detection work, so the only
    // difference is how many corners are ordered. With the append-and-sort arm
    // this pair read 0.03 ms against 2.24 ms, because a single-block bitonic
    // network is O(n log^2 n) on one SM with 47 idle and its cost tracks
    // nextPow2(stored). With the prefix-sum arm the ordering is O(n) and the two
    // capacities do the same work, so the pair should now read near 1.00x -- and
    // if it does not, the ordering still scales with the corner count.
    {
        bc::DeviceArray<bc::DeviceFastCorner> small(512);
        bc::DeviceAppendCounter smallCounter;
        const bc::DeviceFastCornerBuffer smallBuf(small.data(), smallCounter.devicePtr(), 512u);
        const size_t smallBytes =
            bc::fastScratchBytes(kWidth, kHeight, 512, bc::FastArm::Reference);
        bc::DeviceArray<uint8_t> smallScratch(smallBytes);
        const PairedTiming bySort = timeKernelPaired(
            [&] {
                smallCounter.reset(gStream);
                bc::detectFastAsync(dframe.constView(), smallBuf, smallScratch.data(),
                                    smallBytes, 9, gStream);
            },
            [&] { runFast(9); }, 10, 10, kRounds, gStream);
        printPaired("detectFast, capacity 512  (orders 512 corners)",
                    "detectFast, full capacity (orders every stored corner)", bySort, "kernel");
        std::printf("   IDENTICAL DETECTION WORK on both sides -- every pixel of the frame\n"
                    "   is tested either way, and only the number of STORED corners differs.\n"
                    "   The gap is therefore the ordering. A ratio near 1.00x is the claim:\n"
                    "   the ordering no longer scales with the corner count, because it is a\n"
                    "   prefix sum over WORDS and not a comparison network over corners.\n\n");
        sortShare = bySort.ratioMedian;
        sortOnlyMs = bySort.a.medianMs;
    }

    printMemoryHeader("detectFastAsync, binCV against binCV");
    {
        const size_t plane = bc::rowWords(kWidth) * kHeight * sizeof(uint32_t);
        // BOTH ARMS' SCRATCH, side by side, because they are now different numbers
        // and the gap is the whole of what a caller of the shipped arm stopped
        // paying. The ordered column is what production allocates; the reference
        // column is what a binary that flips the switch has to.
        std::printf("   record: %zu bytes (int x, int y, int score)\n",
                    sizeof(bc::DeviceFastCorner));
        for (uint32_t cap : {200u, 2048u, 8192u, 32768u}) {
            const size_t ordered =
                bc::fastScratchBytes(kWidth, kHeight, cap, bc::FastArm::Ordered);
            const size_t reference =
                bc::fastScratchBytes(kWidth, kHeight, cap, bc::FastArm::Reference);
            const size_t total = plane + static_cast<size_t>(cap) * sizeof(bc::DeviceFastCorner) +
                                 ordered + sizeof(uint32_t);
            std::printf("   capacity %5u: plane %.1f KB + corners %.1f KB + scratch %.2f KB"
                        " = %.1f KB   (reference arm would want %.1f KB of scratch)\n",
                        cap, static_cast<double>(plane) / 1024.0,
                        static_cast<double>(cap * sizeof(bc::DeviceFastCorner)) / 1024.0,
                        static_cast<double>(ordered) / 1024.0,
                        static_cast<double>(total) / 1024.0,
                        static_cast<double>(reference) / 1024.0);
        }
        printAllocSum("input bit plane (1 bit/px)", plane);
        std::printf("   The same picture as CV_8U is %.1f KB unpitched -- an 8.00x ratio on\n"
                    "   the INPUT-IMAGE COMPONENT, which is the construction fact the rule\n"
                    "   requires. The whole-working-set ratio is reported against OpenCV's\n"
                    "   measured delta below and carries NO pass/fail number.\n\n",
                    static_cast<double>(kWidth * kHeight) / 1024.0);
    }

    // -----------------------------------------------------------------------
    // OP2 -- the response map and the whole operation
    // -----------------------------------------------------------------------
    std::printf("----------------------------------------------------------------\n"
                " OP2  the corner response, and goodFeaturesToTrack end to end\n"
                "----------------------------------------------------------------\n");

    bc::DeviceImage<float> dmap(static_cast<int>(kWidth), static_cast<int>(kHeight));
    {
        const PairedTiming sliced = timeKernelPaired(
            [&] {
                bincv::cuda::impl::cornerSlicedEnabled() = true;
                bc::cornerMinEigenValAsync(dmagX.constView(), dmagY.constView(),
                                           dsignX.constView(), dsignY.constView(), 3,
                                           dmap.view(), gStream);
            },
            [&] {
                bincv::cuda::impl::cornerSlicedEnabled() = false;
                bc::cornerMinEigenValAsync(dmagX.constView(), dmagY.constView(),
                                           dsignX.constView(), dsignY.constView(), 3,
                                           dmap.view(), gStream);
            },
            5, 5, kRounds, gStream);
        printPaired("cornerMinEigenVal bs 3, BIT-SLICED arm",
                    "cornerMinEigenVal bs 3, per-pixel window arm", sliced, "kernel");
        std::printf("   off-switch ratio window/sliced: %.2fx\n\n", sliced.ratioMedian);

        // THE GATE-EXCLUDED CONTROL: cornerSlicedApplies(7) is false, so both
        // switch positions run the window arm and this MUST read ~1.00x.
        const PairedTiming control = timeKernelPaired(
            [&] {
                bincv::cuda::impl::cornerSlicedEnabled() = true;
                bc::cornerMinEigenValAsync(dmagX.constView(), dmagY.constView(),
                                           dsignX.constView(), dsignY.constView(), 7,
                                           dmap.view(), gStream);
            },
            [&] {
                bincv::cuda::impl::cornerSlicedEnabled() = false;
                bc::cornerMinEigenValAsync(dmagX.constView(), dmagY.constView(),
                                           dsignX.constView(), dsignY.constView(), 7,
                                           dmap.view(), gStream);
            },
            2, 2, kRounds, gStream);
        printPaired("CONTROL bs 7, switch ON  (gate excludes it)", "CONTROL bs 7, switch OFF",
                    control, "kernel", /*expect1x=*/true);
        std::printf("   cornerSlicedApplies(7) = %s\n\n",
                    bincv::cuda::impl::cornerSlicedApplies(7) ? "true" : "FALSE");
        bincv::cuda::impl::cornerSlicedEnabled() = true;
    }

    const uint32_t kCandidateCapacity = 65536;
    const uint32_t kCornerCapacity = 4096;
    bc::DeviceArray<bc::DeviceCorner> dcand(kCandidateCapacity);
    bc::DeviceAppendCounter candCounter;
    bc::DeviceArray<uint32_t> dmaxBits(1);
    const size_t gfScratch = bc::goodFeaturesScratchBytes(kCandidateCapacity);
    bc::DeviceArray<uint8_t> dgfScratch(gfScratch);
    bc::DeviceArray<bc::DeviceCorner> dcorners(kCornerCapacity);
    bc::DeviceArray<bc::DeviceCornerResult> dresult(1);

    bincv::GoodFeaturesParams params;  // the frontend's own four values
    bc::DeviceGoodFeaturesWorkspace work;
    work.candidates =
        bc::DeviceCornerBuffer(dcand.data(), candCounter.devicePtr(), kCandidateCapacity);
    work.maxBits = dmaxBits.data();
    work.scratch = dgfScratch.data();
    work.scratchBytes = gfScratch;
    work.frameMap = dmap.view();

    const auto runCorners = [&]() {
        candCounter.reset(gStream);
        cudaMemsetAsync(dmaxBits.data(), 0, sizeof(uint32_t), gStream);
        bc::goodFeaturesToTrackAsync(dmagX.constView(), dmagY.constView(), dsignX.constView(),
                                     dsignY.constView(), params, work, dcorners.data(),
                                     kCornerCapacity, dresult.data(), gStream);
    };

    runCorners();
    cudaStreamSynchronize(gStream);
    bc::DeviceCornerResult cr{};
    cudaMemcpy(&cr, dresult.data(), sizeof(cr), cudaMemcpyDeviceToHost);
    uint32_t candFound = 0;
    cudaMemcpy(&candFound, candCounter.devicePtr(), sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::printf(" this frame: %u raw 3x3 maxima appended, %u ranked, %u corners kept\n"
                " (candidate capacity %u -- the number a caller must size, and it is NOT\n"
                "  maxCorners; the design's guess at this was never counted before now)\n\n",
                candFound, cr.candidatesRanked, cr.count, kCandidateCapacity);

    {
        const PairedTiming arms = timeKernelPaired(
            [&] { bincv::cuda::impl::cornerFusedEnabled() = true; runCorners(); },
            [&] { bincv::cuda::impl::cornerFusedEnabled() = false; runCorners(); }, 5, 5,
            kRounds, gStream);
        printPaired("goodFeaturesToTrack, FUSED arm (0 frame-sized bytes)",
                    "goodFeaturesToTrack, frame-map reference arm", arms, "kernel");
        std::printf("   off-switch ratio reference/fused: %.2fx\n"
                    "   ARM A/B VERDICT under the project's own tie-break (memory wins a\n"
                    "   conflict nobody has ruled on): the fused arm materialises ZERO\n"
                    "   frame-sized device bytes; the reference arm materialises %.2f MB.\n"
                    "   %s\n\n",
                    arms.ratioMedian,
                    static_cast<double>(kWidth * kHeight * sizeof(float)) / (1024.0 * 1024.0),
                    arms.ratioMedian >= 1.0
                        ? "The fused arm is not slower, so the fused arm ships."
                        : "The fused arm is SLOWER -- the two goals conflict and that is a "
                          "STOP AND ASK.");
        bincv::cuda::impl::cornerFusedEnabled() = true;
    }

    // THE TWO SELECTION ARMS. The tail the reference positions run -- one block,
    // a bitonic network over 16-byte corner records, then a greedy filter that
    // rescans every surviving candidate once per acceptance -- measured 5.6 ms
    // of sort and 23.1 ms of spacing on a real frame with 25,115 candidates.
    // Both are switchable and both are held to the same corner array by the
    // suite.
    {
        const PairedTiming sortArm = timeKernelPaired(
            [&] { bincv::cuda::impl::cornerSortParallelEnabled() = true; runCorners(); },
            [&] { bincv::cuda::impl::cornerSortParallelEnabled() = false; runCorners(); }, 5, 5,
            kRounds, gStream);
        printPaired("goodFeaturesToTrack, DEVICE-WIDE sort ladder",
                    "goodFeaturesToTrack, one-block sort", sortArm, "kernel");
        std::printf("   off-switch ratio one-block/device-wide: %.2fx\n\n",
                    sortArm.ratioMedian);
        bincv::cuda::impl::cornerSortParallelEnabled() = true;

        const PairedTiming spaceArm = timeKernelPaired(
            [&] { bincv::cuda::impl::cornerSpacingChunkedEnabled() = true; runCorners(); },
            [&] { bincv::cuda::impl::cornerSpacingChunkedEnabled() = false; runCorners(); }, 5,
            5, kRounds, gStream);
        printPaired("goodFeaturesToTrack, CHUNKED spacing (test)",
                    "goodFeaturesToTrack, kill-forward spacing", spaceArm, "kernel");
        std::printf("   off-switch ratio kill/chunked: %.2fx\n"
                    "   The chunked arm is the HOST's own loop -- for each candidate in rank\n"
                    "   order, accept it when no accepted corner is within minDistance --\n"
                    "   with the test spread over a block. The reference arm reaches the same\n"
                    "   answer by marking losers, which is one pass over every survivor per\n"
                    "   acceptance. It also keeps the host's DOUBLE distance comparison,\n"
                    "   which is what makes the suite's agreement the proof that the chunked\n"
                    "   arm's integer test is the same predicate.\n\n",
                    spaceArm.ratioMedian);
        bincv::cuda::impl::cornerSpacingChunkedEnabled() = true;

        // GATE-EXCLUDED CONTROLS. minDistance < 1 takes gftt.cpp's own `else`
        // branch, so there is no spacing pass for either switch position to
        // differ over; both MUST read ~1.00x.
        bincv::GoodFeaturesParams noSpace = params;
        noSpace.minDistance = 0.0;
        const auto runNoSpace = [&] {
            candCounter.reset(gStream);
            cudaMemsetAsync(dmaxBits.data(), 0, sizeof(uint32_t), gStream);
            bc::goodFeaturesToTrackAsync(dmagX.constView(), dmagY.constView(),
                                         dsignX.constView(), dsignY.constView(), noSpace, work,
                                         dcorners.data(), kCornerCapacity, dresult.data(),
                                         gStream);
        };
        const PairedTiming spaceControl = timeKernelPaired(
            [&] { bincv::cuda::impl::cornerSpacingChunkedEnabled() = true; runNoSpace(); },
            [&] { bincv::cuda::impl::cornerSpacingChunkedEnabled() = false; runNoSpace(); }, 5,
            5, kRounds, gStream);
        printPaired("CONTROL minDistance 0, chunked ON  (gate excludes it)",
                    "CONTROL minDistance 0, chunked OFF", spaceControl, "kernel",
                    /*expect1x=*/true);
        std::printf("   cornerSpacingChunkedApplies(minDistance 0) = %s\n\n",
                    bincv::cuda::impl::cornerSpacingChunkedApplies(noSpace) ? "true" : "FALSE");
        bincv::cuda::impl::cornerSpacingChunkedEnabled() = true;

        // And the sort ladder's own gate: at a candidate capacity inside one
        // shared-memory chunk the "device-wide" ladder IS one block sorting one
        // chunk, so both positions run the same network.
        const uint32_t kSmallPool = 1024;
        bc::DeviceArray<bc::DeviceCorner> smallCand(kSmallPool);
        bc::DeviceAppendCounter smallCounter;
        bc::DeviceArray<uint8_t> smallScratch(bc::goodFeaturesScratchBytes(kSmallPool));
        bc::DeviceGoodFeaturesWorkspace smallWork = work;
        smallWork.candidates =
            bc::DeviceCornerBuffer(smallCand.data(), smallCounter.devicePtr(), kSmallPool);
        smallWork.scratch = smallScratch.data();
        smallWork.scratchBytes = bc::goodFeaturesScratchBytes(kSmallPool);
        const auto runSmall = [&] {
            smallCounter.reset(gStream);
            cudaMemsetAsync(dmaxBits.data(), 0, sizeof(uint32_t), gStream);
            bc::goodFeaturesToTrackAsync(dmagX.constView(), dmagY.constView(),
                                         dsignX.constView(), dsignY.constView(), params,
                                         smallWork, dcorners.data(), kCornerCapacity,
                                         dresult.data(), gStream);
        };
        const PairedTiming sortControl = timeKernelPaired(
            [&] { bincv::cuda::impl::cornerSortParallelEnabled() = true; runSmall(); },
            [&] { bincv::cuda::impl::cornerSortParallelEnabled() = false; runSmall(); }, 5, 5,
            kRounds, gStream);
        printPaired("CONTROL pool 1024, ladder ON  (gate excludes it)",
                    "CONTROL pool 1024, ladder OFF", sortControl, "kernel", /*expect1x=*/true);
        std::printf("   cornerSortParallelApplies(1024) = %s  (this pool OVERFLOWS on this\n"
                    "   frame, which is a refusal and not a wrong answer -- the control is\n"
                    "   timing the launch ladder, which is what it is for)\n\n",
                    bincv::cuda::impl::cornerSortParallelApplies(kSmallPool) ? "true" : "FALSE");
        bincv::cuda::impl::cornerSortParallelEnabled() = true;
    }

    // WHERE THE TIME ACTUALLY IS, isolated -- because the family's design
    // inherited this on trust and the review was right to say so. The selection
    // tail is ONE BLOCK: a bitonic sort over the candidates, then the greedy
    // spacing filter. Turning the spacing off (minDistance < 1 takes gftt.cpp's
    // own `else` branch) leaves the sort alone, so the pair separates them.
    {
        bincv::GoodFeaturesParams noSpacing = params;
        noSpacing.minDistance = 0.0;
        const PairedTiming split = timeKernelPaired(
            [&] {
                candCounter.reset(gStream);
                cudaMemsetAsync(dmaxBits.data(), 0, sizeof(uint32_t), gStream);
                bc::goodFeaturesToTrackAsync(dmagX.constView(), dmagY.constView(),
                                             dsignX.constView(), dsignY.constView(), noSpacing,
                                             work, dcorners.data(), kCornerCapacity,
                                             dresult.data(), gStream);
            },
            [&] { runCorners(); }, 5, 5, kRounds, gStream);
        printPaired("whole op, spacing OFF (sort only in the tail)",
                    "whole op, spacing ON  (sort + greedy filter)", split, "kernel");
        std::printf("   The tail is still where this operation's time is -- %u raw maxima\n"
                    "   on this frame, %.3f ms with the spacing filter on -- but the split is\n"
                    "   no longer what it was. The sort is now a device-wide ladder and the\n"
                    "   spacing filter tests rather than kills, so what remains is the\n"
                    "   spacing pass itself, and it is ONE BLOCK because the acceptances are\n"
                    "   sequential. THE OPEN ITEM, named rather than left to be\n"
                    "   rediscovered: that pass costs `ranked x kept` distance tests, and a\n"
                    "   spatial index over the accepted points would make it `ranked x O(1)`\n"
                    "   -- accepted corners are pairwise at least minDistance apart, so a\n"
                    "   cell of that side holds at most four of them. Not done here.\n\n",
                    candFound, split.b.medianMs);
    }

    printMemoryHeader("goodFeaturesToTrackAsync, binCV against binCV");
    {
        const size_t plane = bc::rowWords(kWidth) * kHeight * sizeof(uint32_t);
        const size_t fused = 4 * plane +
                             static_cast<size_t>(kCandidateCapacity) * sizeof(bc::DeviceCorner) +
                             gfScratch +
                             static_cast<size_t>(kCornerCapacity) * sizeof(bc::DeviceCorner) +
                             sizeof(uint32_t) + sizeof(bc::DeviceCornerResult);
        const size_t map = kWidth * kHeight * sizeof(float);
        printAllocSum("four input bit planes", 4 * plane);
        printAllocSum("candidate buffer + sort scratch",
                      static_cast<size_t>(kCandidateCapacity) * sizeof(bc::DeviceCorner) +
                          gfScratch);
        printAllocSum("FUSED arm, whole working set", fused);
        printAllocSum("reference arm adds a frame float map", map);
        printAllocSum("reference arm, whole working set", fused + map);
        std::printf("   THE BINARY CHECK THE RULE ASKS FOR: frame-sized device bytes in the\n"
                    "   fused arm = 0. Not a ratio, and it does not become one.\n\n");
    }

    // -----------------------------------------------------------------------
    // OP3 -- cornerSubPix
    // -----------------------------------------------------------------------
    std::printf("----------------------------------------------------------------\n"
                " OP3  cornerSubPixAsync -- NO cv::cuda COUNTERPART (ruling R2)\n"
                "----------------------------------------------------------------\n");

    const uint32_t kSubPixCorners = 200;
    std::vector<bincv::Point2f> seeds(kSubPixCorners);
    {
        uint64_t s = 0x5BADu;
        for (uint32_t i = 0; i < kSubPixCorners; ++i) {
            seeds[i].x = static_cast<float>(20 + splitmix(s) % (kWidth - 40));
            seeds[i].y = static_cast<float>(20 + splitmix(s) % (kHeight - 40));
        }
    }
    const bincv::SubPixParams sp;  // winHalf 5, the shipped operating point
    bc::DeviceSubPixMask mask(sp);
    bc::DeviceArray<float> dseeds(kSubPixCorners * 2);
    bc::DeviceArray<bc::DeviceSubPixResult> dsub(1);
    bc::DeviceBinMat donesMag(static_cast<int>(kWidth), static_cast<int>(kHeight));
    bc::upload(ones.constView(), donesMag.view(), gStream);
    cudaStreamSynchronize(gStream);

    const auto runSubPix = [&](bool dense) {
        bincv::cuda::impl::subPixSkipEnabled() = !dense;
        cudaMemcpyAsync(dseeds.data(), seeds.data(), seeds.size() * sizeof(bincv::Point2f),
                        cudaMemcpyHostToDevice, gStream);
        cudaMemsetAsync(dsub.data(), 0, sizeof(bc::DeviceSubPixResult), gStream);
        bc::cornerSubPixAsync(dmagX.constView(), dmagY.constView(), dsignX.constView(),
                              dsignY.constView(), dseeds.data(), kSubPixCorners, sp,
                              mask.devicePtr(), dsub.data(), gStream);
    };

    {
        // THE SPREAD ARM, and it is this section's headline. One corner per thread
        // is a bit-exactness requirement, so the only lever is how many SMs the
        // corners reach: a 256-thread block puts 200 corners on ONE SM, and this
        // part issues FP64 at 1/64 of FP32 -- two pipes for the whole frame.
        const PairedTiming spread = timeKernelPaired(
            [&] { bincv::cuda::impl::subPixSpreadEnabled() = true; runSubPix(false); },
            [&] { bincv::cuda::impl::subPixSpreadEnabled() = false; runSubPix(false); }, 20, 20,
            kRounds, gStream);
        printPaired("cornerSubPix, SPREAD launch (one warp per block)",
                    "cornerSubPix, packed 256-thread blocks", spread, "kernel");
        std::printf("   off-switch ratio packed/spread: %.2fx at %u corners, which is\n"
                    "   %u blocks against 1.\n\n",
                    spread.ratioMedian, kSubPixCorners, (kSubPixCorners + 31u) / 32u);
        bincv::cuda::impl::subPixSpreadEnabled() = true;

        // THE SPREAD ARM'S GATE-EXCLUDED CONTROL: at 32 corners or fewer both
        // geometries are a single block on a single SM, so this MUST read ~1.00x.
        const auto runSubPixFew = [&] {
            cudaMemcpyAsync(dseeds.data(), seeds.data(), 32 * sizeof(bincv::Point2f),
                            cudaMemcpyHostToDevice, gStream);
            cudaMemsetAsync(dsub.data(), 0, sizeof(bc::DeviceSubPixResult), gStream);
            bc::cornerSubPixAsync(dmagX.constView(), dmagY.constView(), dsignX.constView(),
                                  dsignY.constView(), dseeds.data(), 32u, sp, mask.devicePtr(),
                                  dsub.data(), gStream);
        };
        const PairedTiming spreadControl = timeKernelPaired(
            [&] { bincv::cuda::impl::subPixSpreadEnabled() = true; runSubPixFew(); },
            [&] { bincv::cuda::impl::subPixSpreadEnabled() = false; runSubPixFew(); }, 20, 20,
            kRounds, gStream);
        printPaired("CONTROL 32 corners, spread ON  (gate excludes it)",
                    "CONTROL 32 corners, spread OFF", spreadControl, "kernel",
                    /*expect1x=*/true);
        std::printf("   subPixSpreadApplies(32) = %s\n\n",
                    bincv::cuda::impl::subPixSpreadApplies(32u) ? "true" : "FALSE");
        bincv::cuda::impl::subPixSpreadEnabled() = true;

        const PairedTiming skip = timeKernelPaired([&] { runSubPix(false); },
                                                   [&] { runSubPix(true); }, 20, 20, kRounds,
                                                   gStream);
        printPaired("cornerSubPix, SET-BIT skip arm", "cornerSubPix, dense window arm", skip,
                    "kernel");
        std::printf("   off-switch ratio dense/skip: %.2fx\n\n", skip.ratioMedian);

        // THE GATE-EXCLUDED CONTROL: on an all-ones magnitude plane the skip can
        // eliminate nothing, so the two arms do the same work and this MUST read
        // ~1.00x.
        const auto runSubPixOnes = [&](bool dense) {
            bincv::cuda::impl::subPixSkipEnabled() = !dense;
            cudaMemcpyAsync(dseeds.data(), seeds.data(), seeds.size() * sizeof(bincv::Point2f),
                            cudaMemcpyHostToDevice, gStream);
            cudaMemsetAsync(dsub.data(), 0, sizeof(bc::DeviceSubPixResult), gStream);
            bc::cornerSubPixAsync(donesMag.constView(), donesMag.constView(),
                                  dsignX.constView(), dsignY.constView(), dseeds.data(),
                                  kSubPixCorners, sp, mask.devicePtr(), dsub.data(), gStream);
        };
        const PairedTiming control =
            timeKernelPaired([&] { runSubPixOnes(false); }, [&] { runSubPixOnes(true); }, 20,
                             20, kRounds, gStream);
        printPaired("CONTROL all-ones window, skip ON", "CONTROL all-ones window, skip OFF",
                    control, "kernel", /*expect1x=*/true);
        std::printf("   Every magnitude bit set, so the skip removes nothing and the two\n"
                    "   arms are the same work. A ratio far from 1.00x here would mean the\n"
                    "   switch is selecting something other than the skip.\n\n");
        bincv::cuda::impl::subPixSkipEnabled() = true;
    }

    // THE RULE'S OWN COMPARISON: the device arm against the round trip it
    // replaces, both sides measured here, and the baseline priced BOTH ways.
    {
        const size_t planeBytes = bc::rowWords(kWidth) * kHeight * sizeof(uint32_t);
        std::vector<uint32_t> host(4 * bc::rowWords(kWidth) * kHeight);
        const Timing deviceArm = timeKernel([&] { runSubPix(false); }, 20, kRounds, gStream);

        const Timing wholeDownload = timeKernel(
            [&] {
                cudaMemcpyAsync(host.data(), dmagX.constView().ptr, planeBytes,
                                cudaMemcpyDeviceToHost, gStream);
                cudaMemcpyAsync(host.data() + bc::rowWords(kWidth) * kHeight,
                                dmagY.constView().ptr, planeBytes, cudaMemcpyDeviceToHost,
                                gStream);
                cudaMemcpyAsync(host.data() + 2 * bc::rowWords(kWidth) * kHeight,
                                dsignX.constView().ptr, planeBytes, cudaMemcpyDeviceToHost,
                                gStream);
                cudaMemcpyAsync(host.data() + 3 * bc::rowWords(kWidth) * kHeight,
                                dsignY.constView().ptr, planeBytes, cudaMemcpyDeviceToHost,
                                gStream);
            },
            5, kRounds, gStream);

        // The narrower alternative, priced rather than assumed away: only the
        // 11-row bands around the corners. One pitched copy per corner is the
        // shape a caller would actually write.
        const Timing bandDownload = timeKernel(
            [&] {
                const size_t rowBytes = bc::rowWords(kWidth) * sizeof(uint32_t);
                for (uint32_t i = 0; i < kSubPixCorners; ++i) {
                    const size_t y0 = static_cast<size_t>(seeds[i].y) > 5
                                          ? static_cast<size_t>(seeds[i].y) - 5
                                          : 0;
                    cudaMemcpyAsync(host.data(),
                                    dmagX.constView().ptr + y0 * dmagX.constView().stride,
                                    11 * rowBytes, cudaMemcpyDeviceToHost, gStream);
                }
            },
            1, kRounds, gStream);

        // Host refinement of the same corners, on the same machine.
        std::vector<bincv::Point2f> hostSeeds = seeds;
        const auto t0 = std::chrono::steady_clock::now();
        int hostIters = 0;
        for (; hostIters < 20; ++hostIters) {
            hostSeeds = seeds;
            bincv::cornerSubPix<1, Word>(hdx, hdy, hostSeeds.data(), hostSeeds.size(), sp);
        }
        const double hostMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count() /
            hostIters;

        const double left = deviceArm.medianMs;  // the upload/download of 200x8 B is inside it
        const double rightWhole = wholeDownload.medianMs + hostMs;
        const double rightBands = bandDownload.medianMs + hostMs;
        std::printf(" THE ROUND-TRIP RULE, both sides on this machine in this binary:\n");
        printArmVsFloor("LEFT  device arm (kernel + 1.6 KB each way)", deviceArm, floor,
                        "kernel");
        std::printf(" %-44s %9.3f ms  [kernel]\n"
                    " %-44s %9.3f ms  [kernel, 200 pitched copies]\n"
                    " %-44s %9.3f ms  [wall, host]\n",
                    "RIGHT four-plane download (180 KB)", wholeDownload.medianMs,
                    "RIGHT 11-row bands around 200 corners", bandDownload.medianMs,
                    "RIGHT host cornerSubPix, same 200 corners", hostMs);
        std::printf("   LEFT %.4f ms   vs   RIGHT(whole planes) %.4f ms   -> %s\n"
                    "   LEFT %.4f ms   vs   RIGHT(bands)        %.4f ms   -> %s\n",
                    left, rightWhole, left < rightWhole ? "RULE MET" : "RULE MISSED", left,
                    rightBands, left < rightBands ? "RULE MET" : "RULE MISSED");
        std::printf("   The tighter of the two baselines is the denominator, per 'pick the\n"
                    "   right baseline'. %s\n",
                    rightBands < rightWhole
                        ? "The band download is the tighter one and is what decides."
                        : "The whole-plane download is the tighter one, which is itself a "
                          "finding: 200 pitched copies cost more than one big one.");

        // THE SAME INEQUALITY, MEASURED AS ONE THING INSTEAD OF THREE.
        //
        // The three medians above were taken in three separate timed passes and
        // then added, so the comparison carries all three passes' drift and
        // offers nothing to decide with: "0.0387 < 0.0414" is a statement about
        // two numbers, not about two alternatives. It has been reported as met
        // in six runs out of seven for exactly that reason -- the seventh is
        // not a different machine, it is the same three passes landing
        // differently.
        //
        // So the inequality is also run as ONE PAIRED COMPARISON. Each round
        // times the device arm and the whole round trip back to back, in
        // alternating order, and yields one observation of the ratio with the
        // drift divided out. It is on the WALL CLOCK on both sides because the
        // round trip is mostly not device work -- a CUDA-event bracket around
        // it would time the copies and silently drop the host refinement,
        // which is the largest term in it.
        {
            // BOTH ARMS ARE THE INEQUALITY AS WRITTEN, down to the 1.6 KB
            // transfers. The rule is
            //   kernel + up(200x8 B) + down(200x8 B)
            //     < down(the derivative data) + host cornerSubPix + up(200x8 B)
            // and the two small transfers sit on OPPOSITE sides, so leaving
            // both out very nearly cancels and leaving out one does not. They
            // are ~1.6 KB each and cost microseconds; they are here because a
            // rule measured is not a rule approximated.
            std::vector<bincv::Point2f> refined(kSubPixCorners);
            const auto leftArm = [&] {
                runSubPix(false);
                cudaMemcpyAsync(refined.data(), dseeds.data(),
                                kSubPixCorners * sizeof(bincv::Point2f),
                                cudaMemcpyDeviceToHost, gStream);
                cudaStreamSynchronize(gStream);
            };
            // THE BASELINE IS THE TIGHTER OF THE TWO, which on this machine is
            // the WHOLE-PLANE download and not the bands: 200 pitched copies of
            // 11 rows cost about fifty times one copy of four planes, so
            // pairing against the bands would be measuring against a spelling
            // nobody would keep. Which one is tighter is READ OFF the two
            // single-arm timings above rather than assumed, because the answer
            // is a property of this machine's copy engine.
            const bool bandsAreTighter = bandDownload.medianMs < wholeDownload.medianMs;
            const auto rightArm = [&] {
                const size_t rowBytes = bc::rowWords(kWidth) * sizeof(uint32_t);
                if (bandsAreTighter) {
                    for (uint32_t i = 0; i < kSubPixCorners; ++i) {
                        const size_t y0 = static_cast<size_t>(seeds[i].y) > 5
                                              ? static_cast<size_t>(seeds[i].y) - 5
                                              : 0;
                        cudaMemcpyAsync(
                            host.data(),
                            dmagX.constView().ptr + y0 * dmagX.constView().stride,
                            11 * rowBytes, cudaMemcpyDeviceToHost, gStream);
                    }
                } else {
                    const size_t planeWords = bc::rowWords(kWidth) * kHeight;
                    cudaMemcpyAsync(host.data(), dmagX.constView().ptr, planeBytes,
                                    cudaMemcpyDeviceToHost, gStream);
                    cudaMemcpyAsync(host.data() + planeWords, dmagY.constView().ptr,
                                    planeBytes, cudaMemcpyDeviceToHost, gStream);
                    cudaMemcpyAsync(host.data() + 2 * planeWords, dsignX.constView().ptr,
                                    planeBytes, cudaMemcpyDeviceToHost, gStream);
                    cudaMemcpyAsync(host.data() + 3 * planeWords, dsignY.constView().ptr,
                                    planeBytes, cudaMemcpyDeviceToHost, gStream);
                }
                cudaStreamSynchronize(gStream);
                std::vector<bincv::Point2f> pts = seeds;
                bincv::cornerSubPix<1, Word>(hdx, hdy, pts.data(), pts.size(), sp);
                cudaMemcpyAsync(dseeds.data(), pts.data(),
                                kSubPixCorners * sizeof(bincv::Point2f),
                                cudaMemcpyHostToDevice, gStream);
                cudaStreamSynchronize(gStream);
            };
            const PairedTiming rt = timeHostPaired(leftArm, rightArm, 20, 10, kRounds);
            std::printf("\n   THE INEQUALITY AS ONE PAIRED MEASUREMENT (wall clock both\n"
                        "   sides, against the %s -- the tighter baseline):\n",
                        bandsAreTighter ? "band download" : "whole-plane download");
            printPaired("LEFT  device arm, enqueue + sync",
                        bandsAreTighter ? "RIGHT band download + host cornerSubPix"
                                        : "RIGHT whole-plane download + host cornerSubPix",
                        rt, "WALL");
            std::printf("   Arm A is the device arm, so a ratio ABOVE 1.00x is the round\n"
                        "   trip costing more, which is the rule being MET. The verdict\n"
                        "   line above says whether this run can tell them apart at all.\n");
        }
        std::printf("   THE LEFT SIDE IS AT ITS CEILING AT THIS CORNER COUNT, and the number\n"
                    "   comes from the profiler rather than from an A/B. One corner per\n"
                    "   thread is the bit-exactness requirement, so 200 corners are 7 warps\n"
                    "   of work; spread one warp per block they occupy 7 SMs at ONE warp\n"
                    "   each, and a single warp cannot keep an SM's two FP64 pipes busy --\n"
                    "   measured, sm__inst_executed_pipe_fp64 reads 45.7%% of peak on the\n"
                    "   active SMs. So the spread buys 7 x 0.457 = 3.20x where a naive\n"
                    "   reading of 'seven times the SMs' predicts 7x, and the arm ratio\n"
                    "   above reads 3.18x. The remaining 2.2x needs a SECOND warp per SM,\n"
                    "   which needs about 450 corners; at 200 there is not enough work to\n"
                    "   fill the part and no launch geometry changes that.\n\n");
        printMemoryHeader("cornerSubPixAsync");
        printAllocSum("Gaussian mask, winHalf 5", mask.size() * sizeof(double));
        printAllocSum("corner positions, 200", kSubPixCorners * 2 * sizeof(float));
        printAllocSum("result counters", sizeof(bc::DeviceSubPixResult));
        std::printf("   Frame-sized device bytes: 0, at every winHalf up to kMaxWinHalf.\n"
                    "   That is the rule's requirement and it is binary. The four derivative\n"
                    "   planes are already resident for the corner op and are NOT charged\n"
                    "   twice here.\n\n");
        bincv::cuda::impl::subPixSkipEnabled() = true;
    }

    // -----------------------------------------------------------------------
    // The role bars
    // -----------------------------------------------------------------------
    std::printf("----------------------------------------------------------------\n"
                " THE ROLE BARS\n"
                "----------------------------------------------------------------\n");

#if BINCV_CUDA_FRONTEND_OPENCV
    {
        cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(gStream);
        cv::Mat hostBytes(static_cast<int>(kHeight), static_cast<int>(kWidth), CV_8UC1);
        for (size_t y = 0; y < kHeight; ++y) {
            for (size_t x = 0; x < kWidth; ++x) {
                const bool bit =
                    (frame.constView().row(y)[x / 32] >> (x % 32)) & 1u;
                hostBytes.at<uint8_t>(static_cast<int>(y), static_cast<int>(x)) =
                    bit ? 255 : 0;
            }
        }
        cv::cuda::GpuMat gsrc(hostBytes);

        // ---- OP1: cv::cuda::FastFeatureDetector -------------------------------
        cv::Ptr<cv::cuda::FastFeatureDetector> cvFast = cv::cuda::FastFeatureDetector::create(
            128, false, cv::FastFeatureDetector::TYPE_9_16,
            static_cast<int>(kFastCapacity));
        cv::cuda::GpuMat cvKeypoints;
        const PairedTiming fastRole = timeKernelPaired(
            [&] { runFast(9); }, [&] { cvFast->detectAsync(gsrc, cvKeypoints, cv::noArray(), cvStream); },
            10, 10, kRounds, gStream);
        printPaired("binCV detectFastAsync (1 bit/px)",
                    "cv::cuda::FastFeatureDetector (CV_8U, nms off)", fastRole, "kernel");
        std::printf("   ratio binCV/OpenCV (>1 means binCV is FASTER): %.2fx\n"
                    "   RULE: required >= 1.00x -> %s.  Whether a ratio above parity ships\n"
                    "   the FAST claim UNQUALIFIED is the owner's to rule; this binary does\n"
                    "   not invent that number.\n",
                    fastRole.ratioMedian,
                    fastRole.ratioMedian >= 1.0 ? "MET" : "MISSED");
        std::printf("   WHERE THE COST USED TO BE. With the append-and-sort arm this\n"
                    "   operation measured 11.7x OpenCV's time and the whole of it was one\n"
                    "   single-block raster sort: 2.236 ms of a 2.236 ms call, against\n"
                    "   0.033 ms for the detector with the sort capped out. The capacity\n"
                    "   pair above is that isolation re-run -- %.2fx now, %.3f ms capped\n"
                    "   against %.3f ms for OpenCV -- and the raster order is now produced\n"
                    "   by a prefix sum over per-word popcounts, so nothing in this\n"
                    "   operation compares two corners. The open question that shortfall\n"
                    "   raised -- must a device detector return RASTER ORDER at all, or is\n"
                    "   the corner SET the contract -- is therefore WITHDRAWN rather than\n"
                    "   answered: the order is returned and it no longer costs anything.\n",
                    sortShare, sortOnlyMs, fastRole.b.medianMs);
        // Whether this is a result is printPaired's verdict above -- the
        // project's difference-against-spread rule. Overlapping ranges are a
        // fact worth flagging next to a headline number, not a second veto.
        if (!fastRole.separated()) {
            std::printf("   (the two arms' sample RANGES overlap, which is a fact and not"
                        " the verdict:\n"
                        "   the verdict above is measure_util.hpp's rule on the per-round"
                        " ratio.)\n");
        }
        std::printf("\n");

        // Corner-set agreement is a GATE, not a trade -- checked as a SET,
        // because OpenCV's emission order is its own.
        {
            runFast(9);
            cudaStreamSynchronize(gStream);
            bc::DeviceAppendResult r;
            bc::readAppendResult(fastBuf, r, gStream);
            std::vector<bc::DeviceFastCorner> mine(r.acceptTruncated());
            if (!mine.empty()) {
                bc::downloadAppended(fastBuf, r, mine.data(), gStream);
                cudaStreamSynchronize(gStream);
            }
            cvFast->detectAsync(gsrc, cvKeypoints, cv::noArray(), cvStream);
            cvStream.waitForCompletion();
            cv::Mat kp;
            cvKeypoints.download(kp);
            // cudafeatures2d packs the location as a short2 in LOCATION_ROW of a
            // CV_32FC1 matrix -- decoded here rather than counted, because the
            // gate is that the two SETS agree, not that two numbers do.
            std::vector<uint64_t> theirs;
            const short* loc = kp.ptr<short>(cv::cuda::FastFeatureDetector::LOCATION_ROW);
            for (int i = 0; i < kp.cols; ++i) {
                theirs.push_back((static_cast<uint64_t>(static_cast<uint16_t>(loc[2 * i + 1]))
                                  << 32) |
                                 static_cast<uint32_t>(static_cast<uint16_t>(loc[2 * i])));
            }
            std::vector<uint64_t> ours;
            for (const bc::DeviceFastCorner& c : mine) {
                ours.push_back((static_cast<uint64_t>(static_cast<uint32_t>(c.y)) << 32) |
                               static_cast<uint32_t>(c.x));
            }
            std::sort(theirs.begin(), theirs.end());
            std::sort(ours.begin(), ours.end());
            const bool setsAgree = theirs == ours;
            std::printf("   CORNER-SET AGREEMENT GATE: binCV %zu, OpenCV %d keypoints -- %s\n"
                        "   ops/fast.hpp proves the two accept the SAME corners on binary\n"
                        "   content at any threshold in [1,254]; positions compared as a set,\n"
                        "   because OpenCV's emission order is its own. A mismatch STOPS the\n"
                        "   comparison -- it is a gate, not something traded against speed.\n"
                        "   %s\n\n",
                        ours.size(), kp.cols,
                        setsAgree ? "SETS AGREE" : "*** SETS DIFFER ***",
                        setsAgree ? ""
                                  : "   The speed and memory ratios above are therefore NOT a"
                                    " like-for-like\n   comparison and must not be quoted.");
        }

        // ---- OP1 memory, meter 2, both sides ---------------------------------
        {
            const size_t step = measureDriverMeterStep();
            const int replicas = 16;
            size_t mineDelta = 0, theirsDelta = 0;
            {
                DeviceMemMeter meter;
                std::vector<bc::DeviceBinMat> planes;
                std::vector<bc::DeviceArray<bc::DeviceFastCorner>> outs;
                std::vector<bc::DeviceArray<uint8_t>> scratches;
                for (int i = 0; i < replicas; ++i) {
                    planes.emplace_back(static_cast<int>(kWidth), static_cast<int>(kHeight));
                    outs.emplace_back(kFastCapacity);
                    // The SHIPPED arm's number: this probe is what a production
                    // caller allocates, and production never flips the switch.
                    scratches.emplace_back(bc::fastScratchBytes(kWidth, kHeight,
                                                                kFastCapacity,
                                                                bc::FastArm::Ordered));
                }
                mineDelta = meter.deltaBytes() / replicas;
            }
            {
                DeviceMemMeter meter;
                std::vector<cv::cuda::GpuMat> srcs;
                std::vector<cv::cuda::GpuMat> kps;
                std::vector<cv::Ptr<cv::cuda::FastFeatureDetector>> dets;
                for (int i = 0; i < replicas; ++i) {
                    srcs.emplace_back(hostBytes);
                    dets.push_back(cv::cuda::FastFeatureDetector::create(
                        128, false, cv::FastFeatureDetector::TYPE_9_16,
                        static_cast<int>(kFastCapacity)));
                    kps.emplace_back();
                    dets.back()->detectAsync(srcs.back(), kps.back(), cv::noArray(), cvStream);
                }
                cvStream.waitForCompletion();
                theirsDelta = meter.deltaBytes() / replicas;
            }
            printMemoryHeader("detectFast against cv::cuda::FastFeatureDetector");
            printDriverDelta("binCV, per working set", mineDelta, step);
            printDriverDelta("OpenCV, per working set", theirsDelta, step);
            std::printf("   ratio OpenCV/binCV at capacity %u: %.2fx.  REPORTED, NOT GATED:\n"
                        "   binCV's side is %zu B a corner of output plus %zu B of scratch for\n"
                        "   the whole frame; OpenCV's is a short2 column at capacity plus a\n"
                        "   two-row float output at the count it found.\n"
                        "   OpenCV's reading is an UPPER BOUND: GpuMat can pool, and anything\n"
                        "   held rather than freed lands inside the delta.\n\n",
                        kFastCapacity,
                        mineDelta > 0 ? static_cast<double>(theirsDelta) /
                                            static_cast<double>(mineDelta)
                                      : 0.0,
                        sizeof(bc::DeviceFastCorner),
                        bc::fastScratchBytes(kWidth, kHeight, kFastCapacity,
                                             bc::FastArm::Ordered));
        }

        // ---- OP2: cv::cuda::createGoodFeaturesToTrackDetector ----------------
        {
            cv::Ptr<cv::cuda::CornersDetector> cvGftt =
                cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, params.maxCorners,
                                                            params.qualityLevel,
                                                            params.minDistance,
                                                            params.blockSize, false);
            cv::cuda::GpuMat cvCorners;
            const PairedTiming gfttRole = timeWallPaired(
                [&] { runCorners(); },
                [&] { cvGftt->detect(gsrc, cvCorners, cv::noArray(), cvStream); }, 5, 5,
                kRounds);
            printPaired("binCV goodFeaturesToTrackAsync (device-resident)",
                        "cv::cuda gftt over createMinEigenValCorner", gfttRole, "WALL");
            std::printf("   WALL CLOCK ON BOTH SIDES, and the reason is on OpenCV's side of\n"
                        "   the comparison rather than ours: at minDistance %.3f its detect()\n"
                        "   DOWNLOADS the sorted candidate list and runs the spacing filter on\n"
                        "   the HOST, then uploads the survivors. A CUDA-event clock would\n"
                        "   silently exclude that pass and flatter OpenCV. binCV's selection\n"
                        "   never leaves the device -- that residency has no OpenCV\n"
                        "   counterpart at all, and it is a claim in its own right.\n"
                        "   ratio binCV/OpenCV: %.2fx.  VERDICT: PENDING AN OWNER RULING --\n"
                        "   the speed bar for this op was NOT WRITABLE (see the rule above).\n",
                        params.minDistance, gfttRole.ratioMedian);
            if (!gfttRole.separated()) {
                std::printf("   (the two arms' sample RANGES overlap -- a fact, not the"
                            " verdict; see the\n   rule printed above.)\n");
            }
            std::printf("\n");

            const size_t step = measureDriverMeterStep();
            const int replicas = 8;
            size_t mineDelta = 0, theirsDelta = 0;
            {
                DeviceMemMeter meter;
                std::vector<bc::DeviceBinMat> planes;
                std::vector<bc::DeviceArray<bc::DeviceCorner>> cands;
                std::vector<bc::DeviceArray<uint8_t>> scratches;
                for (int i = 0; i < replicas; ++i) {
                    for (int p = 0; p < 4; ++p)
                        planes.emplace_back(static_cast<int>(kWidth), static_cast<int>(kHeight));
                    cands.emplace_back(kCandidateCapacity);
                    scratches.emplace_back(gfScratch);
                }
                mineDelta = meter.deltaBytes() / replicas;
            }
            {
                DeviceMemMeter meter;
                std::vector<cv::cuda::GpuMat> srcs, outs;
                std::vector<cv::Ptr<cv::cuda::CornersDetector>> dets;
                for (int i = 0; i < replicas; ++i) {
                    srcs.emplace_back(hostBytes);
                    dets.push_back(cv::cuda::createGoodFeaturesToTrackDetector(
                        CV_8UC1, params.maxCorners, params.qualityLevel, params.minDistance,
                        params.blockSize, false));
                    outs.emplace_back();
                    dets.back()->detect(srcs.back(), outs.back(), cv::noArray(), cvStream);
                }
                cvStream.waitForCompletion();
                theirsDelta = meter.deltaBytes() / replicas;
            }
            printMemoryHeader("goodFeaturesToTrack against cv::cuda's detector");
            printDriverDelta("binCV FUSED, per working set", mineDelta, step);
            printDriverDelta("OpenCV, per working set", theirsDelta, step);
            std::printf("   ratio OpenCV/binCV: %.2fx.  The rule's own requirement is the\n"
                        "   BINARY one above -- zero frame-sized device bytes in the fused\n"
                        "   arm -- and this ratio is reported beside it, not in place of it.\n"
                        "   OpenCV's reading is an upper bound (GpuMat can pool).\n\n",
                        mineDelta > 0 ? static_cast<double>(theirsDelta) /
                                            static_cast<double>(mineDelta)
                                      : 0.0);
        }
    }
#else
    std::printf(" OP1 detectFastAsync            role bar UNMEASURED -> verdict BLOCKED\n"
                " OP2 goodFeaturesToTrackAsync   role bar UNMEASURED -> verdict BLOCKED\n"
                "\n"
                " This binary was built without an OpenCV carrying cudafeatures2d and\n"
                " cudaimgproc, so the two role bars could not be run. Under the owner's\n"
                " both-axes ship rule a missing role comparison is NOT a memory argument\n"
                " that carries the op anyway: the verdict is BLOCKED, and no CPU number is\n"
                " quoted in a GPU bar's place. Point -DBINCV_CUDA_OPENCV_DIR at such a\n"
                " build to run them.\n\n");
#endif
    std::printf(" OP3 cornerSubPixAsync          NO cv::cuda COUNTERPART EXISTS\n"
                "                                -> speed verdict OUTSTANDING (ruling R2).\n"
                "     It ships on correctness, memory and the ROUND-TRIP rule above. The\n"
                "     host library's 13.70x against cv::cornerSubPix is a HOST result and\n"
                "     is not restated here as a device claim.\n\n");

    cudaStreamDestroy(gStream);
    return 0;
}
