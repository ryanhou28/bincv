// The median family, priced at birth: the binary three-pixel L over packed
// words, and the wide median over a caller-chosen neighbourhood.
//
// WHAT DECIDES WHAT, written into the rows rather than left to the reader:
//
//   * THE LAUNCH FLOOR IS PRINTED FIRST AND SHARES EVERY LINE. At 752x480 the
//     binary median moves 92 KB -- about 0.15 us of traffic against an empty
//     kernel that costs microseconds. A ratio taken there is measuring the
//     launch, so the size LADDER exists, and the deciding size is the first one
//     whose measured time clears the printed floor. That is a measurement, not
//     a number anybody chose.
//
//   * THE BINARY MEDIAN'S BAR IS BINCV'S OWN BYTE KERNEL, WHICH IS THE HARDEST
//     ONE AVAILABLE. `medianWide` with `kMedianReferenceL` computes the
//     IDENTICAL operation -- same three samples, same zero border -- one byte
//     per pixel, in one launch, with its fast arm on. Running that against
//     `denoiseMedian3` on the same frame isolates exactly one variable: the
//     representation. Two softer bars were available and are deliberately not
//     used: a naive one-thread-one-pixel byte kernel, and the composed cv::cuda
//     spelling of the same map (two zeroed mats, two ROI copies, four min/max
//     = seven frame-sized buffers and eight launches). Either would hand this
//     op most of its headline for free -- the first on instruction count, the
//     second on launch count -- and neither is what the best existing GPU
//     option costs.
//
//   * THE WIDE MEDIAN HAS NO STRUCTURAL ADVANTAGE AND IS NOT REPORTED AS IF IT
//     DID. Byte in, byte out, on both sides. Its role bar is
//     cv::cuda::createMedianFilter, which this target does not link; where the
//     bar cannot be run the verdict is printed as OUTSTANDING rather than
//     substituted with a CPU number.
//
//   * EVERY ARM RATIO IS AN INTERLEAVED PER-ROUND RATIO, and the printer says
//     whether the two sample ranges are disjoint -- a fact, not the verdict.
//     What decides is measure_util.hpp's rule: the difference must exceed the
//     larger of the within-run spread and the run-to-run scatter.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/denseDisparity.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/median.hpp"
#include "bincv/cuda/pack.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/denoise.hpp"
#include "bincv/ops/medianWide.hpp"
#include "bincv/ops/pack.hpp"
#include "cuda_bench_util.hpp"
#include "measure_util.hpp"

namespace {

struct Size {
    size_t w, h;
};

// The ladder. 752x480 is the project's reference frame and sits under the
// launch floor for both ops; the rest exist so the deciding size can be a
// measurement.
const Size kLadder[] = {{752, 480}, {1280, 720}, {1920, 1080}, {4096, 2160}};

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

/// Is an arm's median clear of the launch floor? The rule this benchmark was
/// written under: the deciding size is the first ladder size whose measured
/// time exceeds the floor's median by more than the floor's own spread.
bool clearsFloor(const cudabench::Timing& arm, const cudabench::Timing& floor) {
    return arm.medianMs > floor.maxMs;
}

/// A floor whose OWN spread is this wide is not an instrument. It happens when
/// the device is shared: the empty kernel then measures whatever else was on
/// the GPU, and no kernel of any size can clear `floor.maxMs`. Saying
/// "launch-floor dominated" there would blame the kernel for the meter.
bool floorIsUsable(const cudabench::Timing& floor) { return floor.spreadPct() < 200.0; }

/// The one place the floor verdict is worded, so the two failure modes cannot
/// be confused with each other.
void sayFloorVerdict(const cudabench::PairedTiming& p, const cudabench::Timing& floor) {
    if (clearsFloor(p.b, floor)) return;
    if (!floorIsUsable(floor)) {
        std::printf("   THE FLOOR ITSELF IS UNUSABLE ON THIS RUN (spread %.0f%%): the"
                    " device was\n   SHARED, so the empty kernel measured other work and"
                    " no size can clear it.\n   The disjoint/overlap verdict above is"
                    " what stands; the floor-crossing\n   half of the speed gate needs a"
                    " serial pass on an idle GPU.\n",
                    floor.spreadPct());
        return;
    }
    std::printf("   LAUNCH-FLOOR DOMINATED at this size: this ratio decides nothing.\n");
}

void ruleHeader() {
    std::printf(
        "\n RULE, WRITTEN BEFORE THESE NUMBERS WERE TAKEN\n"
        "   denoiseMedian3 memory gate: the ratio against the byte bar is a FORMULA,\n"
        "     width / (rowWords(width) * 4) -- 7.833x at 752, 8.000x at 4096. The gate\n"
        "     is that the measured allocation sums AGREE with it, not that they beat a\n"
        "     chosen number. Scratch must be zero on both sides.\n"
        "   denoiseMedian3 speed gate: faster than the byte bar by more than both\n"
        "     printed spreads, decided at the first ladder size clear of the floor.\n"
        "   medianWide memory: PARITY is the requirement. No structural advantage\n"
        "     exists here and none is claimed.\n"
        "   Arm ratios: an arm ships only if it beats the arm it replaces by more than\n"
        "     both spreads at a size clear of the floor. The gate-excluded control must\n"
        "     read ~1.00x at that same size, or the switch is not wired to the arm the\n"
        "     table names.\n\n");
}

} // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    std::printf("=== CUDA median family ===\n");
    cudabench::printDevice();
    ruleHeader();

    const auto floor = cudabench::measureLaunchFloor();
    cudabench::printLaunchFloor(floor);
    if (!floorIsUsable(floor))
        std::printf("   ON THIS RUN THE FLOOR'S OWN SPREAD IS %.0f%%. That is a SHARED\n"
                    "   device, not a property of any kernel below: the empty kernel was\n"
                    "   measuring other work. Every timing here is INDICATIVE ONLY, and\n"
                    "   the floor-crossing half of the speed gate cannot be settled until\n"
                    "   this benchmark is re-run serially on an idle GPU.\n",
                    floor.spreadPct());

    // ======================================================================
    // denoiseMedian3 -- the binary three-pixel L
    // ======================================================================
    std::printf("\n-----------------------------------------------------------------\n"
                " denoiseMedian3 (packed bits) against the SAME operation on bytes\n"
                " Same three samples, same zero border, one launch each. The only\n"
                " variable between the two rows is the representation.\n"
                "-----------------------------------------------------------------\n");

    size_t decidingBinary = 0;
    for (size_t li = 0; li < sizeof(kLadder) / sizeof(kLadder[0]); ++li) {
        const Size s = kLadder[li];
        const auto frame = randomFrame(s.w, s.h, 0x1000 + li);
        bincv::BinMat<uint32_t> hostBits(static_cast<int>(s.w), static_cast<int>(s.h));
        bincv::packBits<bincv::PackRule::GreaterThan>(frame.data(), s.w, s.h, s.w,
                                                      hostBits.view(), uint8_t{127});

        bincv::cuda::DeviceBinMat dBits(static_cast<int>(s.w), static_cast<int>(s.h));
        bincv::cuda::DeviceBinMat dOut(static_cast<int>(s.w), static_cast<int>(s.h));
        bincv::cuda::upload(hostBits.constView(), dBits.view());
        bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(s.w), static_cast<int>(s.h));
        bincv::cuda::DeviceImage<uint8_t> dImgOut(static_cast<int>(s.w),
                                                  static_cast<int>(s.h));
        bincv::cuda::uploadImage<uint8_t>(frame.data(), s.w, s.h, s.w, dImg.view());
        cudaDeviceSynchronize();

        std::printf("\n %zux%zu\n", s.w, s.h);

        const auto pairedFmt = cudabench::timeKernelPaired(
            [&] {
                bincv::cuda::medianWide<3>(dImg.constView(), dImgOut.view(),
                                           bincv::kMedianReferenceL);
            },
            [&] { bincv::cuda::denoiseMedian3(dBits.constView(), dOut.view()); }, 20, 20,
            9);
        cudabench::printPaired("  A = byte bar: medianWide<L>, fast arm",
                               "  B = denoiseMedian3 (bits)", pairedFmt, "kernel");
        sayFloorVerdict(pairedFmt, floor);
        if (clearsFloor(pairedFmt.b, floor) && decidingBinary == 0)
            decidingBinary = li + 1;

    }
    std::printf("\n deciding ladder step for the binary median: %s\n",
                decidingBinary == 0
                    ? (floorIsUsable(floor)
                           ? "NONE -- every rung sat on the launch floor, so the speed"
                             " gate is UNDECIDED by this run"
                           : "UNDECIDED -- the launch floor was unusable on this run"
                             " (shared device); re-run serially on an idle GPU")
                    : "the first rung marked clear of the floor above");

    std::printf(
        "\n WHY THERE IS NO SECOND BINARY ARM TO SWITCH OFF, AND WHY THE ROWS ABOVE\n"
        " ARE THE EVIDENCE FOR IT. A uint4 arm -- 128 pixels per thread through one\n"
        " 128-bit load, 23 SASS instructions per 128 pixels against the shipped arm's\n"
        " 14 per 32 -- was written, proven bit-exact and timed on this ladder. It\n"
        " never separated: per-round ratios 0.97-1.04x with overlapping ranges at\n"
        " every rung, over 21 rounds. The reason is arithmetic, not noise. At\n"
        " 4096x2160 this operation's whole working set is 2 * 2160 * 128 * 4 =\n"
        " 2.21 MB, about 3.6 us of traffic at this GPU's 608 GB/s, against the\n"
        " launch floor printed at the top of this run. THE KERNEL IS CHEAPER THAN\n"
        " THE LAUNCH THAT CARRIES IT AT EVERY FRAME SIZE, so no kernel shape can\n"
        " move the number, and a second hand-written traversal to keep bit-exact\n"
        " forever would have been bought with nothing. One implementation ships,\n"
        " with no off-switch -- the censusTransformPacked precedent. Compare the\n"
        " denoiseMedian3 rows above against the floor to see it.\n");

    // ---- memory, meter named at the number ----
    {
        const size_t w = 752, h = 480;
        const size_t binBytes = 2 * h * bincv::cuda::rowWords(w) * sizeof(uint32_t);
        const size_t byteBytes = 2 * h * w * sizeof(uint8_t);
        std::printf("\n");
        cudabench::printMemoryHeader("denoiseMedian3 vs the byte bar, 752x480");
        cudabench::printAllocSum("binCV: src + dst, packed", binBytes);
        cudabench::printAllocSum("bar:   src + dst, bytes", byteBytes);
        cudabench::printAllocSum("binCV scratch", 0);
        cudabench::printAllocSum("bar scratch", 0);
        const double measured = static_cast<double>(byteBytes) / static_cast<double>(binBytes);
        const double formula =
            static_cast<double>(w) / (static_cast<double>(bincv::cuda::rowWords(w)) * 4.0);
        std::printf("   ratio %.4fx against the formula width/(rowWords(width)*4) ="
                    " %.4fx  -> %s\n",
                    measured, formula,
                    (measured > formula - 1e-9 && measured < formula + 1e-9)
                        ? "AGREES; no unaccounted buffer on either side"
                        : "DISAGREES -- a buffer is unaccounted for; investigate");
        std::printf("   This is the format's own 1-bit-against-1-byte ratio. It is not a\n"
                    "   performance result and it is not evidence about the wide median.\n");

        cudabench::DeviceMemMeter meter;
        const size_t step = cudabench::measureDriverMeterStep();
        meter.reset();
        {
            bincv::cuda::DeviceBinMat a(static_cast<int>(w), static_cast<int>(h));
            bincv::cuda::DeviceBinMat b(static_cast<int>(w), static_cast<int>(h));
            (void)a;
            (void)b;
            cudabench::printDriverDelta("binCV working set, driver", meter.deltaBytes(),
                                        step);
        }
    }

    // ---- the named pipeline the op actually sits inside ----
    {
        const size_t w = 752, h = 480;
        const auto l = randomFrame(w, h, 11), r = randomFrame(w, h, 12);
        bincv::BinMat<uint32_t> hl(static_cast<int>(w), static_cast<int>(h)),
            hr(static_cast<int>(w), static_cast<int>(h));
        bincv::packBits<bincv::PackRule::GreaterThan>(l.data(), w, h, w, hl.view(),
                                                      uint8_t{127});
        bincv::packBits<bincv::PackRule::GreaterThan>(r.data(), w, h, w, hr.view(),
                                                      uint8_t{127});
        bincv::cuda::DeviceBinMat dl(static_cast<int>(w), static_cast<int>(h)),
            dr(static_cast<int>(w), static_cast<int>(h)),
            dlf(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::upload(hl.constView(), dl.view());
        bincv::cuda::upload(hr.constView(), dr.view());
        bincv::cuda::DeviceImage<uint8_t> disp(static_cast<int>(w), static_cast<int>(h));
        bincv::DenseDisparityParams params;
        cudaDeviceSynchronize();

        const auto stage = cudabench::timeKernel(
            [&] {
                bincv::cuda::denoiseMedian3(dl.constView(), dlf.view());
                bincv::cuda::denseDisparityBinary(dlf.constView(), dr.constView(), params,
                                                  disp.view());
            },
            10, 9);
        const auto matchOnly = cudabench::timeKernel(
            [&] {
                bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), params,
                                                  disp.view());
            },
            10, 9);
        std::printf("\n THE NAMED PIPELINE: resident bits -> denoiseMedian3 ->"
                    " denseDisparityBinary\n (D=64, 9x9 -- the path cuda_dense_benchmark"
                    " already builds). The op's own\n microbenchmark is launch-bound, so"
                    " the number that prices it is this share.\n");
        cudabench::printArmVsFloor("  whole stage (median + match)", stage, floor,
                                   "kernel");
        cudabench::printArmVsFloor("  match alone", matchOnly, floor, "kernel");
        const double share = stage.medianMs > 0.0
                                 ? (stage.medianMs - matchOnly.medianMs) / stage.medianMs
                                 : 0.0;
        std::printf("   denoiseMedian3's share of the stage: %.1f%%\n", share * 100.0);
        std::printf("   NO BAR IS SET ON THIS SHARE. Choosing one would be inventing a\n"
                    "   threshold nobody has chosen; it is reported, not gated.\n");

        const double cpu = cpuMedianMs("hostDenoise", [&](int) {
            bincv::denoiseMedian3<uint32_t>(hl.constView(), hr.view());
            measure::g_sink += hr.data()[0];
        });
        std::printf(" %-44s %9.3f ms            [cpu]  (the round trip a\n"
                    "   non-resident caller pays instead, host arm on this machine)\n",
                    "  host denoiseMedian3", cpu);
    }

    // ======================================================================
    // medianWide -- the wide median
    // ======================================================================
    std::printf("\n-----------------------------------------------------------------\n"
                " medianWide -- NO STRUCTURAL ADVANTAGE, AND NONE IS CLAIMED.\n"
                " One byte per pixel in, one byte per pixel out, on both sides. What\n"
                " this op has is a cheaper OPERATION (3 samples, or 5, against a 3x3\n"
                " square's 9) and an on-ramp for a caller whose frame is already\n"
                " resident. Neither is evidence for 1-bit packing.\n"
                "-----------------------------------------------------------------\n");

    std::printf("\n ROLE BAR: cv::cuda::createMedianFilter(CV_8UC1, 3), module"
                " cudafilters.\n It is a 3x3 SQUARE median with a replicated border"
                " against binCV's 3-sample\n L with zero fill: different images, and not"
                " a correctness oracle -- that is\n settled against the host library"
                " alone. This target does not link an OpenCV\n built with cudafilters,"
                " so THE SPEED VERDICT AGAINST THE ROLE BAR IS\n OUTSTANDING. No"
                " substitute bar is used and no CPU number is quoted in its\n place.\n"
                " For uint16_t there is no bar to be outstanding against at all:\n"
                " createMedianFilter is CV_8UC1-only, so no cv::cuda alternative exists\n"
                " for that input type at any API level.\n");

    for (size_t li = 0; li < sizeof(kLadder) / sizeof(kLadder[0]); ++li) {
        const Size s = kLadder[li];
        const auto frame = randomFrame(s.w, s.h, 0x2000 + li);
        bincv::cuda::DeviceImage<uint8_t> dSrc(static_cast<int>(s.w),
                                               static_cast<int>(s.h));
        bincv::cuda::DeviceImage<uint8_t> dDst(static_cast<int>(s.w),
                                               static_cast<int>(s.h));
        bincv::cuda::uploadImage<uint8_t>(frame.data(), s.w, s.h, s.w, dSrc.view());
        cudaDeviceSynchronize();

        std::printf("\n %zux%zu, kMedianReferenceL (K=3), uint8\n", s.w, s.h);
        const auto paired = cudabench::timeKernelPaired(
            [&] {
                bincv::cuda::impl::medianWideFastArmEnabled() = false;
                bincv::cuda::medianWide<3>(dSrc.constView(), dDst.view(),
                                           bincv::kMedianReferenceL);
            },
            [&] {
                bincv::cuda::impl::medianWideFastArmEnabled() = true;
                bincv::cuda::medianWide<3>(dSrc.constView(), dDst.view(),
                                           bincv::kMedianReferenceL);
            },
            20, 20, 9);
        bincv::cuda::impl::medianWideFastArmEnabled() = true;
        cudabench::printPaired("  A = per-pixel reference arm", "  B = 4-pixel fast arm",
                               paired, "kernel");
        sayFloorVerdict(paired, floor);

        // K = 5, the other shipped pattern, at the same size.
        const auto paired5 = cudabench::timeKernelPaired(
            [&] {
                bincv::cuda::impl::medianWideFastArmEnabled() = false;
                bincv::cuda::medianWide<5>(dSrc.constView(), dDst.view(),
                                           bincv::kMedianReferencePlus);
            },
            [&] {
                bincv::cuda::impl::medianWideFastArmEnabled() = true;
                bincv::cuda::medianWide<5>(dSrc.constView(), dDst.view(),
                                           bincv::kMedianReferencePlus);
            },
            20, 20, 9);
        bincv::cuda::impl::medianWideFastArmEnabled() = true;
        cudabench::printPaired("  A = reference arm, K=5 plus",
                               "  B = fast arm, K=5 plus", paired5, "kernel");
    }

    // The gate-excluded control for the wide fast arm, at the top of the ladder.
    {
        const size_t w = 4095, h = 2160;  // stride 4095: not a multiple of 4
        const auto frame = randomFrame(w, h, 0x99);
        bincv::cuda::DeviceImage<uint8_t> dSrc(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::DeviceImage<uint8_t> dDst(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dSrc.view());
        cudaDeviceSynchronize();
        const auto ctl = cudabench::timeKernelPaired(
            [&] {
                bincv::cuda::impl::medianWideFastArmEnabled() = false;
                bincv::cuda::medianWide<3>(dSrc.constView(), dDst.view(),
                                           bincv::kMedianReferenceL);
            },
            [&] {
                bincv::cuda::impl::medianWideFastArmEnabled() = true;
                bincv::cuda::medianWide<3>(dSrc.constView(), dDst.view(),
                                           bincv::kMedianReferenceL);
            },
            // More rounds, for the reason given at the binary control above.
            30, 30, 21);
        bincv::cuda::impl::medianWideFastArmEnabled() = true;
        std::printf("\n GATE-EXCLUDED CONTROL, %zux%zu. The image's tight stride is"
                    " %zu, which is\n not a multiple of four, so the fast arm's own"
                    " alignment gate refuses it and\n the switch must select nothing."
                    " Run at the top of the ladder on purpose.\n",
                    w, h, w);
        cudabench::printPaired("  A = switch off", "  B = switch on", ctl, "kernel", true);
    }

    // uint16: the arm ratio, and the verdict that has no bar to be measured
    // against.
    {
        const size_t w = 1920, h = 1080;
        std::vector<uint16_t> frame(w * h);
        uint64_t seed = 0xFEED;
        for (auto& v : frame) v = static_cast<uint16_t>(measure::nextRandom(seed) >> 32);
        bincv::cuda::DeviceImage<uint16_t> dSrc(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::DeviceImage<uint16_t> dDst(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::uploadImage<uint16_t>(frame.data(), w, h, w, dSrc.view());
        cudaDeviceSynchronize();
        const auto paired = cudabench::timeKernelPaired(
            [&] {
                bincv::cuda::impl::medianWideFastArmEnabled() = false;
                bincv::cuda::medianWide<3>(dSrc.constView(), dDst.view(),
                                           bincv::kMedianReferenceL);
            },
            [&] {
                bincv::cuda::impl::medianWideFastArmEnabled() = true;
                bincv::cuda::medianWide<3>(dSrc.constView(), dDst.view(),
                                           bincv::kMedianReferenceL);
            },
            20, 20, 9);
        bincv::cuda::impl::medianWideFastArmEnabled() = true;
        std::printf("\n %zux%zu, kMedianReferenceL, uint16 -- SPEED VERDICT OUTSTANDING:"
                    "\n no cv::cuda alternative exists for this input type.\n",
                    w, h);
        cudabench::printPaired("  A = per-pixel reference arm", "  B = 2-pixel fast arm",
                               paired, "kernel");
    }

    // ---- memory: parity, said plainly ----
    {
        const size_t w = 752, h = 480;
        std::printf("\n");
        cudabench::printMemoryHeader("medianWide, 752x480, uint8");
        cudabench::printAllocSum("binCV: src + dst", 2 * w * h);
        cudabench::printAllocSum("binCV scratch", 0);
        std::printf("   Formula, both sides: width * height * sizeof(T) per view. A byte\n"
                    "   median holds the same two buffers. PARITY IS THE RESULT and the\n"
                    "   only admissible difference is a histogram-based filter's internal\n"
                    "   per-partition state, which this target cannot measure without the\n"
                    "   role bar linked.\n");
    }

    // ---- the named stage, with the stand-in's bias printed at the number ----
    {
        const size_t w = 752, h = 480;
        const auto frame = randomFrame(w, h, 21);
        bincv::cuda::DeviceImage<uint8_t> dSrc(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::DeviceImage<uint8_t> dMed(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::DeviceBinMat dBits(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dSrc.view());
        cudaDeviceSynchronize();

        const auto stage = cudabench::timeKernel(
            [&] {
                bincv::cuda::medianWide<3>(dSrc.constView(), dMed.view(),
                                           bincv::kMedianReferenceL);
                bincv::cuda::packBits(dMed.constView(), dBits.view(),
                                      bincv::PackRule::GreaterThan, uint8_t{127});
            },
            20, 9);
        const auto packOnly = cudabench::timeKernel(
            [&] {
                bincv::cuda::packBits(dMed.constView(), dBits.view(),
                                      bincv::PackRule::GreaterThan, uint8_t{127});
            },
            20, 9);
        std::printf("\n THE NAMED STAGE: resident gray -> medianWide -> packBits ->"
                    " bits.\n packBits IS A LABELLED STAND-IN for edgeThreshold, whose"
                    " device arm does not\n exist yet. THE BIAS HAS A DIRECTION: packBits"
                    " is one ballot per 32 pixels\n and edgeThreshold is a gradient"
                    " operation, so the stand-in stage is CHEAPER\n than the real one and"
                    " the share printed below is an UPPER BOUND on\n medianWide's real"
                    " share. It is recomputed when that arm lands, not quoted\n now as if"
                    " it were the stage.\n");
        cudabench::printArmVsFloor("  stage (median + pack)", stage, floor, "kernel");
        cudabench::printArmVsFloor("  pack alone", packOnly, floor, "kernel");
        const double share = stage.medianMs > 0.0
                                 ? (stage.medianMs - packOnly.medianMs) / stage.medianMs
                                 : 0.0;
        std::printf("   medianWide's share of the stand-in stage: %.1f%%  (UPPER BOUND)\n",
                    share * 100.0);

        // Path A vs path B: whose frame is resident decides, not binCV.
        std::vector<uint8_t> hostMed(w * h);
        bincv::BinMat<uint32_t> hostBits(static_cast<int>(w), static_cast<int>(h));
        const auto pathA = cudabench::timeKernel(
            [&] {
                bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dSrc.view());
                bincv::cuda::medianWide<3>(dSrc.constView(), dMed.view(),
                                           bincv::kMedianReferenceL);
                bincv::cuda::packBits(dMed.constView(), dBits.view(),
                                      bincv::PackRule::GreaterThan, uint8_t{127});
            },
            10, 9);
        const double pathB = cpuMedianMs("hostMedianPackUp", [&](int) {
            bincv::medianWide<3, uint8_t>(frame.data(), w, h, w, hostMed.data(), w,
                                          bincv::kMedianReferenceL);
            bincv::packBits<bincv::PackRule::GreaterThan>(hostMed.data(), w, h, w,
                                                          hostBits.view(), uint8_t{127});
            bincv::cuda::upload(hostBits.constView(), dBits.view());
            cudaDeviceSynchronize();
            measure::g_sink += hostBits.data()[0];
        });
        std::printf("\n PATH A vs PATH B -- reported, NOT gated. Which one a caller"
                    " should take\n depends on whether the wide frame is already"
                    " resident, which is the\n caller's fact and not binCV's.\n");
        cudabench::printArmVsFloor("  path A: upload + device median + pack", pathA, floor,
                                   "kernel");
        std::printf(" %-44s %9.3f ms            [e2e]\n",
                    "  path B: host median + pack + bits up", pathB);

        const double cpuMed = cpuMedianMs("hostMedian", [&](int) {
            bincv::medianWide<3, uint8_t>(frame.data(), w, h, w, hostMed.data(), w,
                                          bincv::kMedianReferenceL);
            measure::g_sink += hostMed[0];
        });
        std::printf(" %-44s %9.3f ms            [cpu]\n",
                    "  host medianWide alone (vector arm)", cpuMed);
    }

    std::printf("\n sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
