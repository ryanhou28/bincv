// The pyramid, the resident ladder and the shift, priced at birth.
//
// READ THE DECISION RULE OFF THIS FILE'S SECTION HEADERS -- it was written
// before any of these numbers existed, and the sections are in its order.
//
//   (1) FOOTPRINT is an EQUALITY, not a threshold. The ladder's bytes are a
//       closed formula that DevicePyramid::sizeInBytes() computes; the pass is
//       that the allocation equals it exactly, and the ratio against a CV_8U
//       ladder is then whatever the two formulas say. cudaMemGetInfo is NOT the
//       meter for that comparison -- its step here is 2 MB against a 384 KB
//       difference, and a meter that cannot resolve the difference cannot fail.
//
//   (2) TIME at the reference frame size is LAUNCH-BOUND and says so. The whole
//       ladder is 93.5 KB, far inside this device's 4 MB L2, so neither binCV
//       nor a byte-per-pixel arm pays the DRAM traffic the format ratio
//       describes. At 752x480 the traffic ratio is a FOOTPRINT claim. The 4K
//       rows below are where it is allowed to become a speed claim.
//
//   (3) THE DECIDING NUMBER IS NOT A MICROBENCHMARK. It is the build's cost
//       inside a named resident pipeline, and the tracker that will own that
//       pipeline does not exist yet. The stand-in was named in advance and is
//       built from shipped ops only: uploadImage -> packBits -> buildPyramidBox,
//       with the build's SHARE of the sequence printed.
//
//   (6) shift's byte-side alternative IS NOT A KERNEL. An integer-pixel
//       translation of a byte image is a pitched 2-D DMA, which spends zero ALU
//       instructions per pixel. binCV's one funnel-shift is being compared
//       against none, not against thirty-two, and the DMA arm here is if
//       anything optimistic -- it does not even pay for the rim. The claim is
//       the 8x footprint; the time is reported as what it is.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/pack.hpp"
#include "bincv/cuda/pyramid.hpp"
#include "bincv/cuda/shift.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/ops/pyramid.hpp"
#include "bincv/ops/shift.hpp"
#include "bincv/quantMat.hpp"
#include "cuda_bench_util.hpp"
#include "measure_util.hpp"

#if defined(BINCV_CUDA_BENCH_OPENCV)
#  include <opencv2/core.hpp>
#  include <opencv2/core/cuda.hpp>
#  include <opencv2/cudawarping.hpp>
#endif

namespace {

constexpr size_t kW = 752, kH = 480;
constexpr size_t kW4K = 3840, kH4K = 2160;

std::vector<uint8_t> randomFrame(size_t w, size_t h, uint64_t seed) {
    std::vector<uint8_t> img(w * h);
    for (auto& v : img) v = static_cast<uint8_t>(measure::nextRandom(seed) >> 40);
    return img;
}

double cpuMedianMs(const char* name, std::function<void(int)> body) {
    std::vector<measure::Bench> arm = {{name, std::move(body)}};
    const auto t = measure::measureInterleaved(arm, 5, 25.0);
    return t[0].medianNs / 1e6;
}

template <size_t N>
bincv::BinMatConstView<uint32_t> stackOf(const bincv::QuantMat<N, uint32_t>& m) {
    return bincv::BinMatConstView<uint32_t>(m.data(), m.getWidth(), N * m.getHeight(),
                                            m.getAlignedWidth());
}

template <size_t N>
void fillQuant(bincv::QuantMat<N, uint32_t>& m, uint64_t seed) {
    const auto frame = randomFrame(m.getWidth(), m.getHeight(), seed);
    bincv::BinMatView<uint32_t> planes[N];
    for (size_t p = 0; p < N; ++p) planes[p] = m.plane(p);
    bincv::packQuant<bincv::QuantRule::Scale, N, uint8_t, uint32_t>(
        frame.data(), m.getWidth(), m.getHeight(), m.getWidth(), planes);
}

// ---------------------------------------------------------------------------
// One pyrDownBox step, both arms, interleaved.
// ---------------------------------------------------------------------------
struct StepArms {
    bincv::cuda::DeviceBinMat src;
    bincv::cuda::DeviceBinMat dst;
    size_t nIn;
    size_t nOut;

    StepArms(size_t w, size_t h, size_t nIn_, size_t nOut_)
        : src(static_cast<int>(w), static_cast<int>(nIn_ * h)),
          dst(static_cast<int>(bincv::pyrDownWidth(w)),
              static_cast<int>(nOut_ * bincv::pyrDownHeight(h))),
          nIn(nIn_),
          nOut(nOut_) {}

    void run() {
        bincv::cuda::pyrDownBox(bincv::cuda::planeBlock(src.constView(), nIn),
                                bincv::cuda::planeBlock(dst.view(), nOut));
    }
};

void priceOneStep(const char* label, size_t w, size_t h, size_t nIn, size_t nOut,
                  const cudabench::Timing& floor) {
    StepArms arms(w, h, nIn, nOut);
    const size_t threadsFast = bincv::cuda::rowWords(bincv::pyrDownWidth(w)) *
                               bincv::pyrDownHeight(h);

    const auto paired = cudabench::timeKernelPaired(
        [&] {
            bincv::cuda::impl::pyrBitSlicedEnabled() = false;
            arms.run();
        },
        [&] {
            bincv::cuda::impl::pyrBitSlicedEnabled() = true;
            arms.run();
        },
        50, 50, 15);
    std::printf("\n %s   (%zu->%zu bits, fast arm covered: %s)\n", label, nIn, nOut,
                bincv::cuda::pyrFastArmCovers(nIn, nOut) ? "YES" : "NO");
    std::printf("   arm B is one thread per destination word: %zu threads here"
                " (arm A is 32x that).\n",
                threadsFast);
    cudabench::printPaired("  A  ballot, warp per dst word (reference)",
                           "  B  bit-sliced, one thread per dst word", paired, "kernel",
                           !bincv::cuda::pyrFastArmCovers(nIn, nOut));
    cudabench::printArmVsFloor("      arm A against the launch floor", paired.a, floor,
                               "kernel");
    bincv::cuda::impl::pyrBitSlicedEnabled() = true;
}

} // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }

    std::printf("=== CUDA pyramid, resident ladder and shift ===\n");
    cudabench::printDevice();
    std::printf(" THE GPU MAY BE SHARED. A number taken on a contended device is\n"
                " indicative only; the committed figures come from a serial pass.\n\n");

    const auto floor = cudabench::measureLaunchFloor();
    cudabench::printLaunchFloor(floor);

    // -----------------------------------------------------------------------
    // (1) FOOTPRINT -- an equality against a closed formula
    // -----------------------------------------------------------------------
    using Ladder = bincv::cuda::DevicePyramid<1, 3, 4, 5>;
    std::printf("\n--- (1) FOOTPRINT: the resident ladder, {1,3,4,5} at %zux%zu ---\n", kW,
                kH);
    cudabench::printMemoryHeader("the resident pyramid");
    {
        // The formula, level by level, so the number is derived rather than
        // reported. stride = ceil(width/32) words at this backend's tight
        // default; rounding these rows to 128 B would cost 2.65x.
        size_t w = kW, h = kH, formula = 0;
        const size_t bits[4] = {1, 3, 4, 5};
        size_t byteLadder = 0;
        for (size_t i = 0; i < 4; ++i) {
            const size_t words = bits[i] * h * bincv::cuda::rowWords(w);
            std::printf("     L%zu %4zux%-4zu %zu plane(s) stride %3zu -> %8zu B\n", i, w, h,
                        bits[i], bincv::cuda::rowWords(w), words * 4);
            formula += words * 4;
            byteLadder += w * h;
            w = bincv::pyrDownWidth(w);
            h = bincv::pyrDownHeight(h);
        }
        Ladder ladder(static_cast<int>(kW), static_cast<int>(kH));
        cudabench::printAllocSum("binCV ladder, ONE allocation", ladder.sizeInBytes());
        cudabench::printAllocSum("the closed formula above", formula);
        std::printf("   EQUALITY (the decision rule's pass): %s\n",
                    ladder.sizeInBytes() == formula
                        ? "allocation == formula, to the byte"
                        : "MISMATCH -- an allocation bug, not a result");
        cudabench::printAllocSum("the tracker's resident PAIR", 2 * ladder.sizeInBytes());
        cudabench::printAllocSum("same-shaped CV_8U ladder, tight", byteLadder);
        std::printf("   ratio, both computed and neither measured: %.3fx smaller\n",
                    static_cast<double>(byteLadder) /
                        static_cast<double>(ladder.sizeInBytes()));
        std::printf("   At GpuMat's 512-byte row pitch the byte ladder is 583,680 B,\n"
                    "   i.e. 6.096x. Both figures are arithmetic on the two formats.\n");

        // METER 2 as a CROSS-CHECK ONLY, and at a size where it can resolve.
        const size_t step = cudabench::measureDriverMeterStep();
        cudabench::DeviceMemMeter meter;
        {
            std::vector<Ladder> many;
            many.reserve(64);
            for (int i = 0; i < 64; ++i)
                many.emplace_back(static_cast<int>(kW), static_cast<int>(kH));
            cudabench::printDriverDelta("64 ladders (cross-check only)", meter.deltaBytes(),
                                        step);
            std::printf("   [why 64] one ladder is %.1f KB and this meter's step is\n"
                        "   %.2f MB, so a single ladder cannot move it. Meter 1 above is\n"
                        "   what measures this working set; this line only confirms that\n"
                        "   64 of them reserve about 64 times as much.\n",
                        static_cast<double>(many[0].sizeInBytes()) / 1024.0,
                        static_cast<double>(step) / (1024.0 * 1024.0));
        }
    }

    // -----------------------------------------------------------------------
    // (4) ARM A vs ARM B -- per level, because the regime changes down the ladder
    // -----------------------------------------------------------------------
    std::printf("\n--- (4) pyrDownBox: reference arm vs bit-sliced arm, PER LEVEL ---\n");
    std::printf(" The prediction, written before measuring: arm B favoured at L0->L1,\n"
                " contested at L1->L2, predicted to LOSE at L2->L3, where it launches\n"
                " fewer than six warps on a %d-SM part. ptxas reports ZERO local-memory\n"
                " spill on every shipped instantiation (22-52 registers, 8->8 the widest).\n",
                [] {
                    int dev = 0;
                    cudaGetDevice(&dev);
                    cudaDeviceProp prop{};
                    cudaGetDeviceProperties(&prop, dev);
                    return prop.multiProcessorCount;
                }());
    priceOneStep("L0->L1  752x480", kW, kH, 1, 3, floor);
    priceOneStep("L1->L2  376x240", kW / 2, kH / 2, 3, 4, floor);
    priceOneStep("L2->L3  188x120", kW / 4, kH / 4, 4, 5, floor);
    priceOneStep("4K base 3840x2160", kW4K, kH4K, 1, 3, floor);
    std::printf("\n The 4K row is where the traffic ratio is allowed to be a SPEED claim:\n"
                " a 3840x2160 1-bit level is 1,036,800 B and the CV_8U one is 8,294,400 B,\n"
                " and only the second exceeds this device's L2.\n");

    // The mandated gate-excluded case.
    std::printf("\n--- THE GATE-EXCLUDED CASE (required to read ~1.00x) ---\n");
    std::printf(" 2->7 bits is outside the fast arm's instantiation set, so BOTH switch\n"
                " positions run the reference arm. Anything other than ~1.00x here means\n"
                " the fast arm is not running where the rows above say it is.\n"
                " RUN AT 4K ON PURPOSE: at 752x480 both arms sit ON the launch floor with\n"
                " a 100%%-plus spread, and a ratio between two arms that are both launch\n"
                " overhead cannot show 1.00x however identical the code is. At 3840x2160\n"
                " the reference arm is several times the floor, so the control can\n"
                " actually demonstrate what it claims.\n");
    priceOneStep("4K base 3840x2160 GATE-EXCLUDED", kW4K, kH4K, 2, 7, floor);

    // -----------------------------------------------------------------------
    // (2) THE WHOLE LADDER against the host arm, and against cv::cuda
    // -----------------------------------------------------------------------
    std::printf("\n--- (2) buildPyramidBox: the whole ladder, one frame ---\n");
    {
        bincv::QuantMat<1, uint32_t> base(static_cast<int>(kW), static_cast<int>(kH));
        fillQuant<1>(base, 0xBEEFu);

        Ladder dev(static_cast<int>(kW), static_cast<int>(kH));
        bincv::cuda::upload(stackOf<1>(base), dev.levelAt(0).block());

        bincv::Pyramid<uint32_t, 1, 3, 4, 5> host(static_cast<int>(kW),
                                                  static_cast<int>(kH));
        fillQuant<1>(host.level<0>(), 0xBEEFu);
        const double hostMs = cpuMedianMs("host build", [&](int) {
            host.build<bincv::PyrDownFilter::Box2x2, bincv::PyrDownBorder::Replicate>();
            measure::g_sink += host.level<3>().data()[0];
        });
        std::printf(" %-44s %9.3f ms  [host CPU]\n", "binCV host ladder, same filter",
                    hostMs);

#if defined(BINCV_CUDA_BENCH_OPENCV)
        {
            cv::Mat hostFrame(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1);
            const auto frame = randomFrame(kW, kH, 0xBEEFu);
            std::memcpy(hostFrame.data, frame.data(), frame.size());
            cv::cuda::GpuMat l0;
            l0.upload(hostFrame);
            cv::cuda::GpuMat l1, l2, l3;
            // Sized once, outside the timed region, so the comparison is build
            // against build rather than build against allocate.
            cv::cuda::resize(l0, l1, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
            cv::cuda::resize(l1, l2, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
            cv::cuda::resize(l2, l3, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
            cudaDeviceSynchronize();

            // INTERLEAVED, not two separate timers: these arms differ by ~6x on
            // a host whose launch overhead wanders by 100%, and a ratio of two
            // separately-measured medians carries the drift between them.
            const auto paired = cudabench::timeKernelPaired(
                [&] { bincv::cuda::buildPyramidBox(dev); },
                [&] {
                    cv::cuda::resize(l0, l1, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
                    cv::cuda::resize(l1, l2, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
                    cv::cuda::resize(l2, l3, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
                },
                50, 50, 15);
            cudabench::printPaired("binCV buildPyramidBox (3 launches)",
                                   "cv::cuda::resize INTER_AREA x3 (SAME filter)", paired,
                                   "kernel");
            cudabench::printArmVsFloor("   binCV against the launch floor", paired.a, floor,
                                       "kernel");

            const auto pyr = cudabench::timeKernel(
                [&] {
                    cv::cuda::pyrDown(l0, l1);
                    cv::cuda::pyrDown(l1, l2);
                    cv::cuda::pyrDown(l2, l3);
                },
                50, 15);
            cudabench::printArmVsFloor("cv::cuda::pyrDown x3 (different filter)", pyr, floor,
                                       "kernel");
            std::printf("   Only 752x480 is like-for-like: cv::cuda::resize computes\n"
                        "   dsize = saturate_cast<int>(753*0.5) = 376 at an ODD width where\n"
                        "   binCV's pyrDownWidth(753) = 377, so an odd-width row would not\n"
                        "   be the same operation on both sides.\n");

            // METER 3 on the byte side: the pitch READ BACK from the GpuMats,
            // so the "512-byte pitch" figure is measured rather than assumed.
            const cv::cuda::GpuMat* levels[4] = {&l0, &l1, &l2, &l3};
            size_t pitched = 0;
            for (const cv::cuda::GpuMat* m : levels) {
                cudabench::printPitch("GpuMat level", m->step,
                                      static_cast<size_t>(m->rows),
                                      static_cast<size_t>(m->cols));
                pitched += m->step * static_cast<size_t>(m->rows);
            }
            cudabench::printAllocSum("CV_8U ladder at its READ-BACK pitch", pitched);
            std::printf("   binCV is %.3fx smaller than that, measured pitch against\n"
                        "   computed formula -- the only cross-library figure here.\n",
                        static_cast<double>(pitched) /
                            static_cast<double>(dev.sizeInBytes()));
        }
#else
        {
            const auto gpu = cudabench::timeKernel(
                [&] { bincv::cuda::buildPyramidBox(dev); }, 50, 15);
            cudabench::printArmVsFloor("binCV buildPyramidBox (3 launches)", gpu, floor,
                                       "kernel");
        }
        std::printf("   cv::cuda::resize / cv::cuda::pyrDown arms: NOT BUILT.\n"
                    "   Configure with -DBINCV_CUDA_OPENCV_DIR pointing at an OpenCV\n"
                    "   built with cudawarping; no packaged OpenCV ships the CUDA modules.\n"
                    "   Until then the GPU role comparison for this family is OUTSTANDING\n"
                    "   and no substitute bar is quoted in its place.\n");
#endif
        std::printf("   DISPOSITION (written before measuring): binCV better with DISJOINT\n"
                    "   ranges = a win; ranges OVERLAP = a TIE, which PASSES and reads\n"
                    "   \"a wash on time, 5.006x on memory\"; binCV worse with disjoint\n"
                    "   ranges = does not merge, and goes to the owner.\n");
    }

    // -----------------------------------------------------------------------
    // (3) THE DECIDING NUMBER'S STAND-IN -- a named resident pipeline
    // -----------------------------------------------------------------------
    std::printf("\n--- (3) THE RESIDENT PIPELINE STAND-IN: uploadImage -> packBits ->"
                " buildPyramidBox ---\n");
    std::printf(" Issue #59's device tracker does not exist. This stand-in was named in\n"
                " the decision rule BEFORE measuring and uses only shipped ops. The number\n"
                " that decides this family is the build's SHARE of a real frame, not the\n"
                " microbenchmark ratios above.\n");
    {
        const auto frame = randomFrame(kW, kH, 0x1234u);
        bincv::cuda::DeviceImage<uint8_t> wide(static_cast<int>(kW), static_cast<int>(kH));
        Ladder dev(static_cast<int>(kW), static_cast<int>(kH));

        const auto whole = cudabench::timeKernel(
            [&] {
                bincv::cuda::uploadImage(frame.data(), kW, kH, kW, wide.view());
                bincv::cuda::packBits(wide.constView(), dev.levelAt(0).block(),
                                      bincv::PackRule::GreaterEqual, uint8_t{128});
                bincv::cuda::buildPyramidBox(dev);
            },
            20, 15);
        const auto noBuild = cudabench::timeKernel(
            [&] {
                bincv::cuda::uploadImage(frame.data(), kW, kH, kW, wide.view());
                bincv::cuda::packBits(wide.constView(), dev.levelAt(0).block(),
                                      bincv::PackRule::GreaterEqual, uint8_t{128});
            },
            20, 15);
        cudabench::printArm("per frame: upload + pack + build", whole, "kernel");
        cudabench::printArm("per frame: upload + pack only", noBuild, "kernel");
        const double share =
            whole.medianMs > 0.0 ? (whole.medianMs - noBuild.medianMs) / whole.medianMs : 0.0;
        std::printf("   the ladder build is %.0f%% of this frame's device time, and the\n"
                    "   two arms' spreads are %.0f%% and %.0f%%. Read the share against\n"
                    "   those spreads before reading anything into it.\n",
                    share * 100.0, whole.spreadPct(), noBuild.spreadPct());
        cudabench::printAllocSum("resident device bytes: ladder", dev.sizeInBytes());
        cudabench::printAllocSum("resident device bytes: wide staging frame", kW * kH);
    }

    // -----------------------------------------------------------------------
    // (6) shift
    // -----------------------------------------------------------------------
    std::printf("\n--- (6) shift, %zux%zu ---\n", kW, kH);
    {
        bincv::BinMat<uint32_t> hostSrc(static_cast<int>(kW), static_cast<int>(kH));
        fillQuant<1>(hostSrc, 0x77u);
        bincv::BinMat<uint32_t> hostDst(static_cast<int>(kW), static_cast<int>(kH));
        bincv::cuda::DeviceBinMat dSrc(static_cast<int>(kW), static_cast<int>(kH));
        bincv::cuda::DeviceBinMat dDst(static_cast<int>(kW), static_cast<int>(kH));
        bincv::cuda::upload<uint32_t>(hostSrc.plane(0), dSrc.view());

        const auto paired = cudabench::timeKernelPaired(
            [&] {
                bincv::cuda::impl::shiftFunnelEnabled() = false;
                bincv::cuda::shift(dSrc.constView(), dDst.view(), 5, 3);
            },
            [&] {
                bincv::cuda::impl::shiftFunnelEnabled() = true;
                bincv::cuda::shift(dSrc.constView(), dDst.view(), 5, 3);
            },
            100, 100, 15);
        cudabench::printPaired("  A  two-shift-or (reference, with the UB branch)",
                               "  B  __funnelshift (default)", paired, "kernel");
        bincv::cuda::impl::shiftFunnelEnabled() = true;
        cudabench::printArmVsFloor("      the funnel arm against the launch floor", paired.b,
                                   floor, "kernel");

        // THE RIM. Not a footnote: at 752 px a row is 24 words, so |dx| = 100
        // puts 5 of them (21%) on the divergent per-pixel path.
        const auto constant = cudabench::timeKernel(
            [&] { bincv::cuda::shift(dSrc.constView(), dDst.view(), 100, 0); }, 100, 15);
        const auto reflect = cudabench::timeKernel(
            [&] {
                bincv::cuda::shift(dSrc.constView(), dDst.view(), 100, 0,
                                   bincv::BORDER_REFLECT_101);
            },
            100, 15);
        cudabench::printArm("  dx=100, BORDER_CONSTANT (no rim)", constant, "kernel");
        cudabench::printArm("  dx=100, BORDER_REFLECT_101 (5 of 24 words)", reflect,
                            "kernel");

        // THE BYTE-SIDE ALTERNATIVE IS A DMA, NOT A KERNEL.
        bincv::cuda::DeviceImage<uint8_t> byteSrc(static_cast<int>(kW),
                                                  static_cast<int>(kH));
        bincv::cuda::DeviceImage<uint8_t> byteDst(static_cast<int>(kW),
                                                  static_cast<int>(kH));
        const auto dma = cudabench::timeKernel(
            [&] {
                cudaMemcpy2DAsync(byteDst.view().ptr, kW, byteSrc.constView().ptr + 5, kW,
                                  kW - 5, kH, cudaMemcpyDeviceToDevice);
            },
            100, 15);
        std::printf(" %-44s %9.3f ms  spread %4.0f%%  [kernel]\n",
                    "  byte side: cudaMemcpy2D ROI copy (DMA)", dma.medianMs,
                    dma.spreadPct());
        std::printf("   THIS IS THE HONEST BAR AND IT IS AN OPTIMISTIC ONE: the DMA does\n"
                    "   no rim at all (a caller adds copyMakeBorder), spends ZERO ALU\n"
                    "   instructions per pixel, and both 46 KB and 361 KB sit inside this\n"
                    "   device's L2. So binCV's one funnel-shift instruction is compared\n"
                    "   against NONE, not against thirty-two, and the design's \"structural\n"
                    "   twice over\" claim does not survive that. What survives is memory.\n");

        const double hostMs = cpuMedianMs("host shift", [&](int) {
            bincv::shift<uint32_t>(hostSrc.plane(0), hostDst.plane(0), 5, 3);
            measure::g_sink += hostDst.data()[0];
        });
        std::printf(" %-44s %9.3f ms  [host CPU]\n", "  binCV host shift, same offsets",
                    hostMs);

        cudabench::printMemoryHeader("shift's two sides");
        const size_t bits = kH * bincv::cuda::rowWords(kW) * 4;
        cudabench::printAllocSum("binCV: height*rowWords(width)*4", bits);
        cudabench::printAllocSum("byte side: height*width*1", kW * kH);
        std::printf("   ratio %.4fx -- the FORMULA, not \"about 8x\". It is exactly\n"
                    "   8.0000x wherever width is a multiple of 32 (3840: 8,294,400 B\n"
                    "   against 1,036,800 B) and %.4fx at 752, where the tail word\n"
                    "   carries 16 padding bits.\n",
                    static_cast<double>(kW * kH) / static_cast<double>(bits),
                    static_cast<double>(kW * kH) / static_cast<double>(bits));
        std::printf("   shift has NO OpenCV counterpart at any API level, on either side\n"
                    "   of the bus. Its GPU speed verdict against a resident pipeline is\n"
                    "   recorded as OUTSTANDING; the DMA row above is the byte-side\n"
                    "   ALTERNATIVE and is labelled as such, not as an OpenCV bar.\n");
    }

    std::printf("\n");
    return 0;
}
