// The sensor stage priced at birth: threshold, binarize, edgeThreshold.
//
// WHAT EACH NUMBER HERE IS, stated before any of them is printed, because for
// ops this cheap the difference between "the kernel" and "the launcher" is the
// whole result:
//
//   * Every device time is KERNEL-RESIDENT (CUDA events around enqueued work) and
//     is printed next to the MEASURED launch floor and the geometry's bandwidth
//     floor. An arm at the launch floor is reporting launch overhead; the printer
//     says so in words rather than leaving it to the reader.
//   * Every ratio between two device arms comes from timeKernelPaired, which
//     brackets both arms inside every round and alternates their order, and which
//     reports whether the two sample RANGES are disjoint. A ratio whose ranges
//     overlap is not a result, however far from 1.00x its median sits. That is
//     also this family's stated rule for the vector arm: it must beat the arm it
//     replaces by more than both arms' printed spreads.
//   * Every host time is a CPU arm on this machine and is NEVER quoted as a GPU
//     bar. It is here because binarize has no OpenCV counterpart at any API level
//     and the host arm is one of the three references that stand in for one.
//   * Memory is reported beside speed, one meter per comparison, named at the
//     number. Meter 1 (allocation sum) for binCV against binCV; meter 2
//     (cudaMemGetInfo delta) only where the comparison crosses to OpenCV.
//
// THE TWO SIZES. 752x480 is the reference frame the resident pipeline runs and
// the size every other table in the backend's report uses; it carries the bars.
// 3840x2160 is reported so the kernel is visible at all, and carries NO bar --
// the premise that it makes launch overhead negligible is itself unmeasured
// (4K threshold traffic / this part's peak bandwidth is ~0.015 ms against a
// launch floor this benchmark measures at roughly a third of that).
//
// THE VECTOR ARM'S GATE-EXCLUDED CONTROL. A runtime switch is half the rule; the
// other half is a shape the fast arm's own gate REJECTS, which must then read
// ~1.00x. Two are timed here -- a uint16 source and the Forward spatial mode --
// and `edgeVectorApplies` is asked rather than restated, so the control cannot
// drift away from the gate it is controlling for.

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/edge.hpp"
#include "bincv/cuda/pack.hpp"
#include "bincv/cuda/threshold.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/edge.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/ops/threshold.hpp"
#include "bincv/quantMat.hpp"
#include "cuda_bench_util.hpp"
#include "measure_util.hpp"

#ifdef BINCV_CUDA_SENSOR_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {

struct Geometry {
    size_t w, h;
    const char* label;
    bool carriesBars;
};

const Geometry kGeometries[] = {
    {752, 480, "752x480 (reference frame -- the bars are here)", true},
    {3840, 2160, "3840x2160 (kernel visibility only -- NO bar)", false},
};

template <typename T>
std::vector<T> randomFrame(size_t w, size_t h, uint64_t seed) {
    std::vector<T> img(w * h);
    for (auto& v : img) v = static_cast<T>(measure::nextRandom(seed) >> 40);
    return img;
}

double cpuMedianMs(const char* name, std::function<void(int)> body, double* spreadPct) {
    std::vector<measure::Bench> arm = {{name, std::move(body)}};
    const auto t = measure::measureInterleaved(arm, 5, 25.0);
    if (spreadPct != nullptr) *spreadPct = t[0].spreadPct();
    return t[0].medianNs / 1e6;
}

/// @brief Peak DRAM bandwidth in bytes/ms, read off the device.
double peakBytesPerMs() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);
    const double gbPerSec =
        2.0 * p.memoryClockRate * (p.memoryBusWidth / 8.0) / 1.0e6;  // GB/s
    return gbPerSec * 1.0e9 / 1000.0;
}

/// @brief One arm, with BOTH floors it stands on named on the following line.
void printArmWithFloors(const char* name, const cudabench::Timing& t,
                        const cudabench::Timing& floor, size_t trafficBytes,
                        double bytesPerMs) {
    cudabench::printArmVsFloor(name, t, floor, "kernel");
    const double bwFloor = static_cast<double>(trafficBytes) / bytesPerMs;
    std::printf("   %-42s %9.4f ms   (%zu B of unavoidable traffic at peak)\n",
                "bandwidth floor for this op's traffic", bwFloor, trafficBytes);
}

/// @brief Meter 1, the allocation sum, with the formula that produced it. The
/// bars in this family are FORMULAS, not round numbers, so the formula is
/// printed with the byte count and the reader can check the arithmetic.
void printFormula(const char* what, const char* formula, size_t bytes) {
    std::printf("   [meter 1: allocation sum ] %-34s %9.1f KB   %s\n", what,
                static_cast<double>(bytes) / 1024.0, formula);
}

/// @brief A gate-excluded control, with the one thing that makes its verdict
/// readable printed next to it.
/// @note The control asks: does a shape the fast arm's own gate REJECTS time the
/// same with the switch on and off? "The same" is only a decidable question
/// while the arms are above the launch floor. When they are not, the
/// per-round ratio scatters over a factor of ten on this host and the
/// printer's +/-5% band is noise either way -- so the floor share is printed
/// with the control and the reader is told which case they are looking at.
/// The control is READ at the geometry where the arms are visible; it is
/// printed at both so a regression at either cannot hide.
void printControl(const char* nameA, const char* nameB, const cudabench::PairedTiming& p,
                  const cudabench::Timing& floor) {
    cudabench::printPaired(nameA, nameB, p, "kernel", /*expect1x=*/true);
    const double share =
        p.a.medianMs > 0.0 ? floor.medianMs / p.a.medianMs : 0.0;
    if (share >= 0.25) {
        std::printf("   READING: both arms are within %.0f%% of the launch floor, so this\n"
                    "   control's verdict is not decidable at this geometry -- read it in\n"
                    "   the section where the arms are visible above the floor.\n",
                    share * 100.0);
    } else {
        std::printf("   READING: the arms are %.1fx the launch floor, so this control's\n"
                    "   verdict IS decidable here.\n",
                    share > 0.0 ? 1.0 / share : 0.0);
    }
}

size_t bitBytes(size_t w, size_t h) { return h * bincv::cuda::rowWords(w) * 4; }

} // namespace

// ---------------------------------------------------------------------------

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    std::printf("=== CUDA sensor stage: threshold, binarize, edgeThreshold ===\n");
    cudabench::printDevice();
    std::printf(" Device times are KERNEL-RESIDENT. Host times are this machine's CPU arm\n"
                " and are never a GPU bar. Memory is beside speed, one meter per number.\n\n");

    const auto floor = cudabench::measureLaunchFloor();
    cudabench::printLaunchFloor(floor);
    const double bytesPerMs = peakBytesPerMs();
    std::printf("\n");

    for (const auto& geo : kGeometries) {
        const size_t w = geo.w, h = geo.h;
        std::printf("===========================================================\n");
        std::printf(" %s\n", geo.label);
        if (!geo.carriesBars) {
            std::printf(" NO BAR is attached to any speed number in this section. It exists\n"
                        " only because at the reference size these ops sit on the launch\n"
                        " floor, and an arm that cannot be seen cannot be compared. Read the\n"
                        " floors printed under each arm before reading the arm.\n");
        }
        std::printf("===========================================================\n");

        // A tiny arm needs a longer batch: back-to-back enqueues pipeline, so a
        // 200-enqueue batch at the reference size is what makes its sample a
        // measurement of anything at all rather than of one launch's jitter.
        const int iters = (w <= 1024) ? 200 : 20;
        const auto frame8 = randomFrame<uint8_t>(w, h, 0x5E507 + w);
        const auto frame16 = randomFrame<uint16_t>(w, h, 0x16B17 + w);

        bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::DeviceImage<uint16_t> dImg16(static_cast<int>(w),
                                                  static_cast<int>(h));
        bincv::cuda::uploadImage<uint8_t>(frame8.data(), w, h, w, dImg.view());
        bincv::cuda::uploadImage<uint16_t>(frame16.data(), w, h, w, dImg16.view());
        bincv::cuda::DeviceBinMat dBits(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::DeviceBinMat dPlanes(static_cast<int>(w), static_cast<int>(2 * h));
        bincv::cuda::packQuant(dImg.constView(), dPlanes.view(), 2);
        const bincv::cuda::DeviceBinMatConstView blockView = dPlanes.constView();
        const bincv::cuda::DevicePlaneBlockConstView planes2{blockView.ptr, w, h,
                                                             blockView.stride, 2};
        cudaDeviceSynchronize();

        // ---------------- memory, stated as the formulas the bars are ----------
        std::printf("\n");
        cudabench::printMemoryHeader("the sensor stage's own working sets");
        printFormula("wide frame, uint8", "height * width * sizeof(SrcT)", h * w);
        printFormula("bit frame", "height * rowWords(width) * 4", bitBytes(w, h));
        printFormula("plane block, n = 2", "n * height * rowWords(width) * 4",
                     2 * bitBytes(w, h));
        printFormula("threshold working set", "wide + bits", h * w + bitBytes(w, h));
        printFormula("binarize working set, n = 2", "2*planes + bits",
                     2 * bitBytes(w, h) + bitBytes(w, h));
        printFormula("edgeThreshold working set", "wide + bits", h * w + bitBytes(w, h));
        std::printf("   packed-vs-byte output ratio  %.4fx   "
                    "(height*width*1) / (height*rowWords(width)*4)\n",
                    static_cast<double>(h * w) / static_cast<double>(bitBytes(w, h)));
        std::printf("   Exactly 8.0000x only where width %% 32 == 0; %zu %s.\n",
                    w, (w % 32 == 0) ? "is" : "is not");
        std::printf("   Device scratch allocated by any op here: 0 B. "
                    "Shared memory per block: 0 B.\n");

        // ---------------- threshold ----------------
        std::printf("\n -- threshold (TIER 1): one cutoff and a dispatch, no new kernel\n");
        {
            const auto t = cudabench::timeKernel(
                [&] { bincv::cuda::threshold(dImg.constView(), dBits.view(), 127.0); },
                iters);
            printArmWithFloors("cuda::threshold, device", t, floor,
                               h * w + bitBytes(w, h), bytesPerMs);
            bincv::BinMat<uint32_t> hostBits(static_cast<int>(w), static_cast<int>(h));
            double spread = 0.0;
            const double hostMs = cpuMedianMs(
                "host",
                [&](int) {
                    bincv::packBits<bincv::PackRule::GreaterEqual, uint8_t, uint32_t>(
                        frame8.data(), w, h, w, hostBits.view(), uint8_t{128});
                },
                &spread);
            std::printf("   %-42s %9.4f ms   spread %3.0f%%  [CPU arm, NOT a GPU bar]\n",
                        "host bincv::threshold body (same machine)", hostMs, spread);
        }

        // ---------------- binarize ----------------
        std::printf("\n -- binarize (TIER 3, no OpenCV counterpart at any API level)\n");
        {
            const auto t = cudabench::timeKernel(
                [&] { bincv::cuda::binarize(planes2, dBits.view(), 1u); }, iters);
            printArmWithFloors("cuda::binarize, n = 2, device", t,
                               floor, 2 * bitBytes(w, h) + bitBytes(w, h), bytesPerMs);

            // The cost-of-shape reference, interleaved against it: same output,
            // 2.94x less traffic in. Stated in advance as NOT a speed bar -- at
            // the reference size binarize launches rowWords*height threads, which
            // is an occupancy floor set by one-thread-per-output-word, not by the
            // kernel.
            const auto p = cudabench::timeKernelPaired(
                [&] { bincv::cuda::packBits(dImg.constView(), dBits.view(),
                                            bincv::PackRule::GreaterEqual, uint8_t{128}); },
                [&] { bincv::cuda::binarize(planes2, dBits.view(), 1u); }, iters, iters);
            cudabench::printPaired("A: cuda::packBits (byte in, cost of shape)",
                                   "B: cuda::binarize n = 2 (bits in)", p, "kernel");
            std::printf("   traffic A %zu B, traffic B %zu B = %.3fx less for B.\n",
                        h * w + bitBytes(w, h), 3 * bitBytes(w, h),
                        static_cast<double>(h * w + bitBytes(w, h)) /
                            static_cast<double>(3 * bitBytes(w, h)));
            std::printf("   binarize launches %zu threads here (%zu warps): the shape's\n"
                        "   occupancy floor, named BEFORE the number, not after it.\n",
                        bincv::cuda::rowWords(w) * h, bincv::cuda::rowWords(w) * h / 32);

            bincv::QuantMat<2, uint32_t> hostQ(static_cast<int>(w), static_cast<int>(h));
            bincv::BinMatView<uint32_t> hostPlanes[2];
            for (size_t p2 = 0; p2 < 2; ++p2) hostPlanes[p2] = hostQ.plane(p2);
            bincv::packQuant<bincv::QuantRule::Scale, 2, uint8_t, uint32_t>(
                frame8.data(), w, h, w, hostPlanes);
            bincv::BinMat<uint32_t> hostOut(static_cast<int>(w), static_cast<int>(h));
            double spread = 0.0;
            const double hostMs = cpuMedianMs(
                "host", [&](int) { bincv::binarize<2, uint32_t>(hostQ, hostOut.view(), 1u); },
                &spread);
            std::printf("   %-42s %9.4f ms   spread %3.0f%%  [CPU arm, NOT a GPU bar]\n",
                        "host bincv::binarize n = 2 (same machine)", hostMs, spread);
            std::printf("   SPEED VERDICT: OUTSTANDING. There is no GPU baseline for an\n"
                        "   N-bit input because OpenCV has no N-bit image type; this op is\n"
                        "   priced against the resident device sensor pipeline when one\n"
                        "   exists, and no substitute bar is invented for it here.\n");
        }

        // ---------------- edgeThreshold, both arms ----------------
        std::printf("\n -- edgeThreshold (TIER 3): the byte-lane arm against its own reference\n");
        {
            auto run = [&](bool vectorOn) {
                return [&, vectorOn] {
                    bincv::cuda::impl::edgeVectorEnabled() = vectorOn;
                    bincv::cuda::edgeThreshold(dImg.constView(), dBits.view(),
                                               uint8_t{17});
                };
            };
            const bool gateAccepts = bincv::cuda::impl::edgeVectorApplies(
                w, dImg.getStride(), nullptr, 1, bincv::EdgeSpatial::Wide, 17);
            std::printf("   edgeVectorApplies(width=%zu, stride=%zu, uint8, Wide, tp=17)"
                        " = %s\n",
                        w, dImg.getStride(), gateAccepts ? "TRUE" : "FALSE");
            const auto p = cudabench::timeKernelPaired(run(false), run(true), iters, iters);
            cudabench::printPaired("A: reference arm (1 pixel/lane, ballot)",
                                   "B: byte-lane arm (4 pixels/lane)", p, "kernel");
            printArmWithFloors("edgeThreshold, arm in use", p.b, floor,
                               h * w + bitBytes(w, h), bytesPerMs);
            bincv::cuda::impl::edgeVectorEnabled() = true;

            // GATE-EXCLUDED CONTROL 1: a uint16 source. The byte-lane instructions
            // are byte-lane, so the gate rejects it and BOTH arms are the reference
            // arm. This must read ~1.00x; anything else means the switch is not
            // switching and every ratio above is one arm timed twice.
            bincv::cuda::DeviceBinMat dBits16(static_cast<int>(w), static_cast<int>(h));
            auto run16 = [&](bool vectorOn) {
                return [&, vectorOn] {
                    bincv::cuda::impl::edgeVectorEnabled() = vectorOn;
                    bincv::cuda::edgeThreshold(dImg16.constView(), dBits16.view(),
                                               uint16_t{17});
                };
            };
            std::printf("   GATE-EXCLUDED CONTROL 1 -- uint16 source, "
                        "edgeVectorApplies = %s\n",
                        bincv::cuda::impl::edgeVectorApplies(
                            w, dImg16.getStride(), nullptr, 2, bincv::EdgeSpatial::Wide,
                            17)
                            ? "TRUE (WRONG)"
                            : "FALSE");
            const auto c1 =
                cudabench::timeKernelPaired(run16(false), run16(true), iters, iters, 25);
            printControl("A: uint16, switch off", "B: uint16, switch on", c1, floor);
            bincv::cuda::impl::edgeVectorEnabled() = true;

            // GATE-EXCLUDED CONTROL 2: EdgeSpatial::Forward, which the gate also
            // rejects -- exactly as the host's own AVX2 arm covers only Wide.
            auto runFwd = [&](bool vectorOn) {
                return [&, vectorOn] {
                    bincv::cuda::impl::edgeVectorEnabled() = vectorOn;
                    bincv::cuda::edgeThreshold(dImg.constView(), dBits.view(), uint8_t{17},
                                               bincv::EdgeCombine::Or,
                                               bincv::EdgeRelation::Ge,
                                               bincv::EdgeSpatial::Forward);
                };
            };
            std::printf("   GATE-EXCLUDED CONTROL 2 -- EdgeSpatial::Forward, "
                        "edgeVectorApplies = %s\n",
                        bincv::cuda::impl::edgeVectorApplies(
                            w, dImg.getStride(), nullptr, 1, bincv::EdgeSpatial::Forward,
                            17)
                            ? "TRUE (WRONG)"
                            : "FALSE");
            const auto c2 =
                cudabench::timeKernelPaired(runFwd(false), runFwd(true), iters, iters, 25);
            printControl("A: Forward, switch off", "B: Forward, switch on", c2, floor);
            bincv::cuda::impl::edgeVectorEnabled() = true;

            bincv::BinMat<uint32_t> hostEdgeOut(static_cast<int>(w), static_cast<int>(h));
            double spread = 0.0;
            const double hostMs = cpuMedianMs(
                "host",
                [&](int) {
                    bincv::edgeThreshold<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                                         bincv::EdgeSpatial::Wide, uint8_t, uint32_t>(
                        frame8.data(), w, h, w, hostEdgeOut.view(), uint8_t{17});
                },
                &spread);
            std::printf("   %-42s %9.4f ms   spread %3.0f%%  [CPU arm, NOT a GPU bar]\n",
                        "host bincv::edgeThreshold (same machine)", hostMs, spread);
        }

        // ---------------- the composed shape a caller actually issues ----------
        std::printf("\n -- composed: a resident wide frame to bits, the shape the sensor\n"
                    "    stage runs. One launch, no wide intermediate.\n");
        {
            const auto t = cudabench::timeKernel(
                [&] {
                    bincv::cuda::edgeThreshold(dImg.constView(), dBits.view(),
                                               uint8_t{17});
                },
                iters);
            printArmWithFloors("resident frame -> edge bits (1 launch)", t, floor,
                               h * w + bitBytes(w, h), bytesPerMs);
        }
        std::printf("\n");
    }

    // -----------------------------------------------------------------------
    // The role comparison. Built only where an OpenCV with cudaarithm,
    // cudafilters and cudaimgproc exists -- and where it does not, this family's
    // ship rule says threshold and edgeThreshold land BLOCKED rather than merged
    // on the memory argument alone, so the absence is printed as a verdict.
    // -----------------------------------------------------------------------
#ifdef BINCV_CUDA_SENSOR_OPENCV
    {
        const size_t w = kGeometries[0].w, h = kGeometries[0].h;
        std::printf("===========================================================\n");
        std::printf(" ROLE COMPARISON vs cv::cuda, %zux%zu (the bars are here)\n", w, h);
        std::printf(" OpenCV %s\n", CV_VERSION);
        std::printf("===========================================================\n");

        const auto frame8 = randomFrame<uint8_t>(w, h, 0xE06E5u);
        cv::Mat hostSrc(static_cast<int>(h), static_cast<int>(w), CV_8UC1,
                        const_cast<uint8_t*>(frame8.data()));
        bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::uploadImage<uint8_t>(frame8.data(), w, h, w, dImg.view());
        bincv::cuda::DeviceBinMat dBits(static_cast<int>(w), static_cast<int>(h));

        const size_t step = cudabench::measureDriverMeterStep();

        // ---- threshold: THIS COMPARISON LIVES IN cuda_role_benchmark ---------
        //
        // A cv::cuda::threshold role pair used to run here, and it was measured
        // with no stream argument on either arm -- that is, on the default
        // stream, where OpenCV's own `if (stream == 0) cudaDeviceSynchronize()`
        // guard charges the comparison a whole-device synchronize that no
        // caller would pay. It printed "binCV is 7.35x FASTER". On one explicit
        // stream the same comparison reads 1.000x / 0.908x / 0.744x across
        // 752x480, 1920x1080 and 3840x2160.
        //
        // It is DELETED rather than repaired, because cuda_role_benchmark
        // already owns this pair, on one explicit stream, at three geometries,
        // over seven process runs. Repairing it here would have left two
        // spellings of one role comparison to drift apart, and the surviving
        // copy should be the one with the protocol in it.
        std::printf("\n -- threshold: see cuda_role_benchmark (one explicit stream,\n"
                    "    three geometries). Not measured here: a default-stream pair\n"
                    "    charges OpenCV a device-wide synchronize no caller pays.\n");


        // ---- edgeThreshold: 1 launch against the composed spelling -----------
        std::printf("\n -- edgeThreshold: the composed cv::cuda spelling, LAUNCHES COUNTED\n");
        {
            // createDerivFilter, NOT createLinearFilter: the latter asserts that
            // the destination depth equals the source depth, so 8U -> 16S throws
            // at construction. At ksize == 1 getDerivKernels returns exactly
            // [-1, 0, 1], which is the SAME computation binCV performs rather
            // than an approximation of it, and the filter defaults to
            // BORDER_REFLECT_101, which is binCV's own border rule.
            cv::Ptr<cv::cuda::Filter> fx = cv::cuda::createDerivFilter(
                CV_8UC1, CV_16SC1, 1, 0, 1, false, 1.0, cv::BORDER_REFLECT_101,
                cv::BORDER_REFLECT_101);
            cv::Ptr<cv::cuda::Filter> fy = cv::cuda::createDerivFilter(
                CV_8UC1, CV_16SC1, 0, 1, 1, false, 1.0, cv::BORDER_REFLECT_101,
                cv::BORDER_REFLECT_101);
            cv::cuda::GpuMat gSrc, gdx, gdy;
            gSrc.upload(hostSrc);
            gdx.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
            gdy.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);

            // The chain, in the version most favourable to OpenCV: abs and
            // threshold in place, so nothing extra is allocated for them.
            const auto chain = [&] {
                fx->apply(gSrc, gdx);
                fy->apply(gSrc, gdy);
                cv::cuda::abs(gdx, gdx);
                cv::cuda::abs(gdy, gdy);
                cv::cuda::threshold(gdx, gdx, 16.0, 1.0, cv::THRESH_BINARY);
                cv::cuda::threshold(gdy, gdy, 16.0, 1.0, cv::THRESH_BINARY);
                cv::cuda::bitwise_or(gdx, gdy, gdx);
            };
            chain();
            cudaDeviceSynchronize();

            const auto p = cudabench::timeKernelPaired(
                chain,
                [&] {
                    bincv::cuda::edgeThreshold(dImg.constView(), dBits.view(),
                                               uint8_t{17});
                },
                10, 20);
            cudabench::printPaired("A: cv::cuda deriv+abs+threshold+or -> CV_16S",
                                   "B: cuda::edgeThreshold -> 1 bit/pixel", p, "kernel");
            const double speedup = p.ratioMedian > 0.0 ? 1.0 / p.ratioMedian : 0.0;
            // THE 3x IS THE AUTHOR'S OWN BAR AND IT IS UNCHANGED. What changed
            // is only how "is this difference real" is answered: the range
            // test has been replaced by measure_util.hpp's own rule, which
            // printPaired prints in full above. The magnitude still has to be
            // cleared as well as the noise.
            const bool real = p.differenceClearsNoise(cudabench::runToRunScatterFactor());
            std::printf("   binCV is %.2fx %s.   BAR: >= 3x.  %s\n", speedup,
                        p.ratioMedian < 1.0 ? "FASTER" : "SLOWER",
                        (speedup >= 3.0 && real) ? "MET"
                                                 : (speedup >= 3.0 ? "NOT MET: the"
                                                                     " magnitude is there"
                                                                     " but the difference"
                                                                     " is a null result"
                                                                   : "NOT MET"));
            std::printf("   LAUNCH COUNT, and it is derived from the API rather than\n"
                        "   profiled (no profiler runs on this machine): createDerivFilter\n"
                        "   is SEPARABLE, so each apply is a row pass and a column pass.\n"
                        "   2 filters x 2 passes + 2 abs + 2 threshold + 1 or = 9 launches\n"
                        "   against binCV's 1. OpenCV's output is a CV_16S map, not bits;\n"
                        "   no cv::cuda operation emits one bit per pixel, which is why\n"
                        "   this is a composed spelling and not a counterpart.\n");

            printFormula("binCV working set", "wide + bits", h * w + bitBytes(w, h));
            printFormula("OpenCV working set",
                         "src 8U + dx 16S + dy 16S + separable buf 16S + mask 8U",
                         h * w + 3 * (2 * h * w) + h * w);
            const double memRatio =
                static_cast<double>(h * w + 3 * (2 * h * w) + h * w) /
                static_cast<double>(h * w + bitBytes(w, h));
            std::printf("   working-set ratio %.3fx   BAR: >= 5x.  %s\n", memRatio,
                        memRatio >= 5.0 ? "MET" : "NOT MET");

            // Meter 2, the only meter readable on both sides, measured the same
            // way on each. Its granularity is printed with it.
            {
                cudabench::DeviceMemMeter m;
                cv::cuda::GpuMat a, b, c;
                a.create(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
                b.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
                c.create(static_cast<int>(h), static_cast<int>(w), CV_16SC1);
                cudabench::printDriverDelta("OpenCV GpuMat trio (upper reading)",
                                            m.deltaBytes(), step);
                cudabench::printPitch("GpuMat CV_8UC1", a.step, static_cast<size_t>(a.rows),
                                      w);
                cudabench::printPitch("GpuMat CV_16SC1", b.step,
                                      static_cast<size_t>(b.rows), 2 * w);
            }
            {
                cudabench::DeviceMemMeter m;
                bincv::cuda::DeviceImage<uint8_t> a(static_cast<int>(w),
                                                    static_cast<int>(h));
                bincv::cuda::DeviceBinMat b(static_cast<int>(w), static_cast<int>(h));
                cudabench::printDriverDelta("binCV wide + bits", m.deltaBytes(), step);
                cudabench::printPitch("DeviceBinMat", b.getAlignedWidth() * 4,
                                      b.getHeight(), bincv::cuda::rowWords(w) * 4);
                (void)a;
            }
        }

        // ---- the one-call role reference, reported with NO bar ---------------
        std::printf("\n -- cv::cuda::CannyEdgeDetector (cudaimgproc): reported, NO BAR.\n"
                    "    A DIFFERENT algorithm (Sobel + non-maximum suppression +\n"
                    "    hysteresis) producing a DIFFERENT map at one byte per pixel. It\n"
                    "    is here only so a reader knows what an existing GPU library costs\n"
                    "    to turn a frame into an edge mask. No correctness or magnitude\n"
                    "    claim attaches to it.\n");
        {
            cv::Ptr<cv::cuda::CannyEdgeDetector> canny =
                cv::cuda::createCannyEdgeDetector(50.0, 100.0, 3, false);
            cv::cuda::GpuMat gSrc, gEdge;
            gSrc.upload(hostSrc);
            canny->detect(gSrc, gEdge);
            cudaDeviceSynchronize();
            const auto t =
                cudabench::timeKernel([&] { canny->detect(gSrc, gEdge); }, 10, 9);
            cudabench::printArmVsFloor("cv::cuda::Canny -> CV_8U map", t, floor, "kernel");
        }
        std::printf("\n");
    }
#else
    std::printf("===========================================================\n");
    std::printf(" ROLE COMPARISON: NOT BUILT.\n");
    std::printf(" cuda_sensor_benchmark was configured without an OpenCV carrying\n"
                " cudaarithm + cudafilters + cudaimgproc, so cv::cuda::threshold and\n"
                " the composed cv::cuda edge chain were not timed. Under this\n"
                " project's both-axes ship rule that means `threshold` and\n"
                " `edgeThreshold` have no role comparison and land BLOCKED -- correct,\n"
                " priced against the host arm and the launch floor, and recorded as\n"
                " blocked, not merged on the memory argument alone.\n");
    std::printf("===========================================================\n");
#endif
    return 0;
}
