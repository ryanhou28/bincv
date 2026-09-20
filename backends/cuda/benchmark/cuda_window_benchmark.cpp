// The window family on the device: morphology's arms, its role comparison
// against cv::cuda, and its footprint.
//
// WHAT THIS BINARY IS FOR, stated before any number appears in it. binCV's
// morphology is a DEAD HEAT on a CPU -- erode 3x3 measures 1.04x on x86 and
// 1.00x on aarch64 against cv::erode, and the 5x5 ellipse LOSES at 0.32x --
// for one reason: an AVX2 register holds 32 bytes, so a vectorised BYTE kernel
// gets the same 32 pixels per instruction a uint32 word gets. A CUDA thread has
// no such register. Its widest byte-lane primitive is FOUR pixels, and
// __vminu4 / __vmaxu4 -- the two a byte morphology is built on -- are not even
// hardware on sm_86: six instructions each, measured from SASS on this machine.
// A packed word is still 32 pixels per instruction. So the prediction under
// test is that the CPU's dead heat INVERTS here. This binary is where that
// prediction is confirmed or refuted; it is asserted nowhere else.
//
// THE THREE MEASUREMENT RULES THIS FILE FOLLOWS, from CLAUDE.md:
//   * memory and speed are reported TOGETHER, one meter per comparison, named
//     at the number;
//   * every arm has a runtime off-switch AND a case the arm's own gate
//     excludes, which must read ~1.00x -- otherwise the switch is not selecting
//     what the table says it is;
//   * a microbenchmark ratio is not an end-to-end result. NO in-repo device
//     pipeline calls morphology today, so this file prints morphology's
//     ABSOLUTE per-frame cost and prints no "share" of any pipeline. Inserting
//     an OPEN into the dense-stereo path to manufacture a denominator would be
//     binCV taking an algorithm position it says it does not take.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/morphology.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/morphology.hpp"
#include "bincv/ops/pack.hpp"
#include "cuda_bench_util.hpp"

#if defined(BINCV_CUDA_HAVE_OPENCV)
#  include <opencv2/core.hpp>
#  include <opencv2/core/cuda.hpp>
#  include <opencv2/cudafilters.hpp>
#  include <opencv2/imgproc.hpp>
#endif

namespace {

using bincv::BinMat;
using bincv::StructuringElement;
using namespace cudabench;

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// A frame with real structure rather than salt-and-pepper: a morphology on
/// uniform noise erodes to nothing at 3x3 and measures a degenerate image.
std::vector<uint8_t> blobFrame(size_t w, size_t h, uint64_t seed) {
    std::vector<uint8_t> img(w * h, 0);
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) {
            const uint64_t n = splitmix(seed);
            const double fx = static_cast<double>(x) * 0.06;
            const double fy = static_cast<double>(y) * 0.07;
            const double v = 128.0 + 90.0 * (0.5 * std::sin(fx) + 0.5 * std::cos(fy)) +
                             20.0 * (static_cast<double>(n & 0xFFu) / 255.0 - 0.5);
            img[y * w + x] = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    return img;
}

BinMat<uint32_t> packedFrame(size_t w, size_t h, uint64_t seed) {
    const auto img = blobFrame(w, h, seed);
    BinMat<uint32_t> m(static_cast<int>(w), static_cast<int>(h));
    bincv::packBits<bincv::PackRule::GreaterThan>(img.data(), w, h, w, m.view(),
                                                  uint8_t{127});
    return m;
}

struct Frame {
    const char* name;
    size_t width;
    size_t height;
};

const Frame kFrames[2] = {{"752x480 (project reference)", 752, 480},
                          {"1920x1080", 1920, 1080}};

// ---------------------------------------------------------------------------
// Arm switching, in one place so a forgotten restore cannot leak into the next
// table.
// ---------------------------------------------------------------------------
struct ArmState {
    bool fast;
    bool wordBorder;
    bool fused;
};

ArmState saveArms() {
    return ArmState{bincv::cuda::impl::morphFastArmEnabled(),
                    bincv::cuda::impl::morphWordBorderEnabled(),
                    bincv::cuda::impl::morphAndNotFusedEnabled()};
}

void restoreArms(const ArmState& s) {
    bincv::cuda::impl::morphFastArmEnabled() = s.fast;
    bincv::cuda::impl::morphWordBorderEnabled() = s.wordBorder;
    bincv::cuda::impl::morphAndNotFusedEnabled() = s.fused;
}

// ---------------------------------------------------------------------------
// BAR 2 -- binCV against binCV. An arm becomes the default only if it beats the
// arm it replaces by more than the printed spread AND the two sample ranges are
// disjoint, at BOTH frame sizes.
// ---------------------------------------------------------------------------
void internalArms(const Frame& f, double floorMs) {
    const int w = static_cast<int>(f.width);
    const int h = static_cast<int>(f.height);
    const BinMat<uint32_t> host = packedFrame(f.width, f.height, 0xC0FFEEu);

    bincv::cuda::DeviceBinMat src(w, h);
    bincv::cuda::DeviceBinMat dst(w, h);
    bincv::cuda::DeviceBinMat scratch(w, h);
    cudaError_t up = bincv::cuda::upload(host.constView(), src.view());
    if (up != cudaSuccess) {
        std::printf("  upload failed: %s\n", cudaGetErrorString(up));
        return;
    }

    const auto rect3 = bincv::cuda::toDeviceElement(bincv::rect3x3());
    const auto ell5 = bincv::cuda::toDeviceElement(StructuringElement::ellipse(5, 5));
    const auto wide65 = bincv::cuda::toDeviceElement(StructuringElement::rect(65, 1));

    const ArmState saved = saveArms();

    std::printf("\n BAR 2 -- binCV against binCV, %s. [kernel-resident, interleaved]\n",
                f.name);
    // WHAT A RATIO HERE CAN AND CANNOT SAY, printed before the ratios so it
    // cannot be read as an excuse afterwards. An erode moves src + dst once.
    {
        bincv::cuda::DeviceBinMat probe(w, h);
        const double bytes =
            2.0 * static_cast<double>(f.height * probe.getAlignedWidth() * 4);
        const double dramMs = bytes / (608.0e9) * 1.0e3;  // 608 GB/s, printed above
        std::printf("   This op's compulsory traffic here is %.0f B (src + dst, meter 1),\n"
                    "   which at this part's %.0f GB/s peak is %.4f ms -- against a launch\n"
                    "   floor of %.4f ms. binCV's binary morphology is therefore LAUNCH-\n"
                    "   BOUND at this frame size, and a ratio between two arms that are\n"
                    "   both under the floor is reporting almost none of either. Where the\n"
                    "   ranges below OVERLAP, that is what is being said.\n",
                    bytes, 608.0, dramMs, floorMs);
    }

    // ---- the 3x3 constant-offset specialization against the general kernel --
    {
        const auto general = [&] {
            bincv::cuda::impl::morphFastArmEnabled() = false;
            bincv::cuda::erode(src.constView(), dst.view(), rect3, bincv::BORDER_CONSTANT);
        };
        const auto fast = [&] {
            bincv::cuda::impl::morphFastArmEnabled() = true;
            bincv::cuda::erode(src.constView(), dst.view(), rect3, bincv::BORDER_CONSTANT);
        };
        const PairedTiming p = timeKernelPaired(general, fast, 50, 50, 11);
        printPaired("erode rect3x3 CONSTANT, general element kernel",
                    "erode rect3x3 CONSTANT, 3x3 specialization", p, "kernel");
        std::printf("   what the switch removes: the runtime trip count over element\n"
                    "   cells and the data-dependent shift count per cell. The HOST\n"
                    "   measured that same substitution at 2.1x-3.7x.\n");
    }

    // ---- the gate-excluded control for that switch ------------------------
    {
        const auto off = [&] {
            bincv::cuda::impl::morphFastArmEnabled() = false;
            bincv::cuda::erode(src.constView(), dst.view(), ell5, bincv::BORDER_CONSTANT);
        };
        const auto on = [&] {
            bincv::cuda::impl::morphFastArmEnabled() = true;
            bincv::cuda::erode(src.constView(), dst.view(), ell5, bincv::BORDER_CONSTANT);
        };
        const PairedTiming p = timeKernelPaired(off, on, 30, 30, 11);
        printPaired("erode ellipse5x5, fast arm OFF  [GATE-EXCLUDED]",
                    "erode ellipse5x5, fast arm ON   [GATE-EXCLUDED]", p, "kernel", true);
        std::printf("   the fast arm's gate is (rows==3 && cols==3 && anchor==(1,1)),\n"
                    "   which a 5x5 ellipse fails -- so the switch selects nothing here\n"
                    "   and anything but ~1.00x would mean it is not doing what the\n"
                    "   table above says it is. Like the ERODE control further down, this\n"
                    "   pair is launch-bound and may not resolve 1.00x; the row below\n"
                    "   repeats the same check where it can.\n");
    }

    // ---- the same control, on a pair far above the launch floor ------------
    {
        const auto off = [&] {
            bincv::cuda::impl::morphFastArmEnabled() = false;
            bincv::cuda::erode(src.constView(), dst.view(), wide65, bincv::BORDER_CONSTANT);
        };
        const auto on = [&] {
            bincv::cuda::impl::morphFastArmEnabled() = true;
            bincv::cuda::erode(src.constView(), dst.view(), wide65, bincv::BORDER_CONSTANT);
        };
        const PairedTiming p = timeKernelPaired(off, on, 8, 8, 11);
        printPaired("erode rect65x1 CONSTANT, fast arm OFF [GATE-EXCLUDED]",
                    "erode rect65x1 CONSTANT, fast arm ON  [GATE-EXCLUDED]", p, "kernel",
                    true);
        std::printf("   A 65x1 element is not 3x3 either, so the switch again selects\n"
                    "   nothing -- and this pair is far above the launch floor, so here\n"
                    "   ~1.00x is a measurement. This is the row that says the fast arm's\n"
                    "   switch is attached to the gate it claims.\n");
    }

    // ---- the word-parallel border against the per-pixel banded fixup -------
    {
        const auto banded = [&] {
            bincv::cuda::impl::morphWordBorderEnabled() = false;
            bincv::cuda::erode(src.constView(), dst.view(), rect3,
                               bincv::BORDER_REFLECT_101);
        };
        const auto wordwise = [&] {
            bincv::cuda::impl::morphWordBorderEnabled() = true;
            bincv::cuda::erode(src.constView(), dst.view(), rect3,
                               bincv::BORDER_REFLECT_101);
        };
        const PairedTiming p = timeKernelPaired(banded, wordwise, 30, 50, 11);
        printPaired("erode rect3x3 REFLECT_101, per-pixel banded fixup",
                    "erode rect3x3 REFLECT_101, word-parallel border", p, "kernel");
        std::printf("   the banded arm is the HOST's shape: every destination column\n"
                    "   within reachX of an edge recomputed one pixel at a time. The\n"
                    "   word arm builds the virtual out-of-row word instead -- __brev\n"
                    "   of a funnel shift, two instructions for 32 pixels -- which is\n"
                    "   a primitive x86 does not have, and is why this family's four\n"
                    "   non-constant borders are not the excused cases here that they\n"
                    "   are on the host.\n");
    }

    // ---- the gate-excluded control for that switch ------------------------
    {
        const auto off = [&] {
            bincv::cuda::impl::morphWordBorderEnabled() = false;
            bincv::cuda::erode(src.constView(), dst.view(), wide65,
                               bincv::BORDER_REFLECT_101);
        };
        const auto on = [&] {
            bincv::cuda::impl::morphWordBorderEnabled() = true;
            bincv::cuda::erode(src.constView(), dst.view(), wide65,
                               bincv::BORDER_REFLECT_101);
        };
        const PairedTiming p = timeKernelPaired(off, on, 8, 8, 11);
        printPaired("erode rect65x1 REFLECT_101, word border OFF [GATE-EXCLUDED]",
                    "erode rect65x1 REFLECT_101, word border ON  [GATE-EXCLUDED]", p,
                    "kernel", true);
        std::printf("   the word-border gate is (border != CONSTANT && reachX < 32 &&\n"
                    "   width >= 64). A 65-wide centred element has reachX == 32, so\n"
                    "   both arms take the banded path and the ratio must read ~1.00x.\n");
    }

    // ---- the fused subtraction against the two-launch spelling -------------
    {
        const auto twoLaunch = [&] {
            bincv::cuda::impl::morphAndNotFusedEnabled() = false;
            bincv::cuda::morphologyEx(src.constView(), dst.view(), bincv::MORPH_GRADIENT,
                                      rect3, scratch.view(), bincv::BORDER_CONSTANT);
        };
        const auto fused = [&] {
            bincv::cuda::impl::morphAndNotFusedEnabled() = true;
            bincv::cuda::morphologyEx(src.constView(), dst.view(), bincv::MORPH_GRADIENT,
                                      rect3, scratch.view(), bincv::BORDER_CONSTANT);
        };
        const PairedTiming p = timeKernelPaired(twoLaunch, fused, 30, 30, 11);
        printPaired("morphologyEx GRADIENT, bitwiseNot + bitwiseAnd (4 launches)",
                    "morphologyEx GRADIENT, fused a & ~b        (3 launches)", p,
                    "kernel");
        std::printf("   ONE launch saved, once per GRADIENT call -- not three. It fires\n"
                    "   once per call for GRADIENT, TOPHAT and BLACKHAT alike.\n");
    }

    // ---- the gate-excluded control for that switch ------------------------
    {
        const auto off = [&] {
            bincv::cuda::impl::morphAndNotFusedEnabled() = false;
            bincv::cuda::morphologyEx(src.constView(), dst.view(), bincv::MORPH_ERODE,
                                      rect3, scratch.view(), bincv::BORDER_CONSTANT);
        };
        const auto on = [&] {
            bincv::cuda::impl::morphAndNotFusedEnabled() = true;
            bincv::cuda::morphologyEx(src.constView(), dst.view(), bincv::MORPH_ERODE,
                                      rect3, scratch.view(), bincv::BORDER_CONSTANT);
        };
        const PairedTiming p = timeKernelPaired(off, on, 50, 50, 11);
        printPaired("morphologyEx ERODE, fusion OFF [GATE-EXCLUDED]",
                    "morphologyEx ERODE, fusion ON  [GATE-EXCLUDED]", p, "kernel", true);
        std::printf("   MORPH_ERODE performs no subtraction, so the fusion switch has\n"
                    "   nothing to select and the ratio must read ~1.00x.\n"
                    "   THIS CONTROL HAS NO RESOLUTION AT THIS SIZE and the row above may\n"
                    "   say so: both arms are one launch-bound kernel, their per-round\n"
                    "   range spans 1.00x, and the median of a ratio between two identical\n"
                    "   launch-bound paths wanders with the host. A control that cannot\n"
                    "   resolve 1.00x cannot detect a mis-attached switch either, so the\n"
                    "   same check is repeated below on a pair that CAN.\n");
    }

    // ---- the same gate-excluded control, on a pair with resolution ---------
    {
        const auto off = [&] {
            bincv::cuda::impl::morphAndNotFusedEnabled() = false;
            bincv::cuda::morphologyEx(src.constView(), dst.view(), bincv::MORPH_OPEN,
                                      wide65, scratch.view(), bincv::BORDER_CONSTANT);
        };
        const auto on = [&] {
            bincv::cuda::impl::morphAndNotFusedEnabled() = true;
            bincv::cuda::morphologyEx(src.constView(), dst.view(), bincv::MORPH_OPEN,
                                      wide65, scratch.view(), bincv::BORDER_CONSTANT);
        };
        const PairedTiming p = timeKernelPaired(off, on, 6, 6, 11);
        printPaired("morphologyEx OPEN rect65x1, fusion OFF [GATE-EXCLUDED]",
                    "morphologyEx OPEN rect65x1, fusion ON  [GATE-EXCLUDED]", p, "kernel",
                    true);
        std::printf("   MORPH_OPEN performs no subtraction either, so the switch again\n"
                    "   selects nothing -- but a 65-wide element puts both arms far above\n"
                    "   the launch floor, so here ~1.00x is a measurement rather than a\n"
                    "   coin toss. This is the row that says the switch is attached.\n");
    }

    restoreArms(saved);
}

/// The absolute per-frame cost, which is the number a caller can put into
/// THEIR pipeline's arithmetic. Deliberately not divided by anything here.
void absoluteCost(const Frame& f, const Timing& floor) {
    const int w = static_cast<int>(f.width);
    const int h = static_cast<int>(f.height);
    const BinMat<uint32_t> host = packedFrame(f.width, f.height, 0x5EEDu);
    bincv::cuda::DeviceBinMat src(w, h);
    bincv::cuda::DeviceBinMat dst(w, h);
    bincv::cuda::DeviceBinMat scratch(w, h);
    if (bincv::cuda::upload(host.constView(), src.view()) != cudaSuccess) return;

    const auto rect3 = bincv::cuda::toDeviceElement(bincv::rect3x3());
    const auto ell5 = bincv::cuda::toDeviceElement(StructuringElement::ellipse(5, 5));

    std::printf("\n ABSOLUTE PER-FRAME COST, %s. [kernel-resident]\n", f.name);
    printArmVsFloor("erode rect3x3 BORDER_CONSTANT",
                    timeKernel([&] {
                        bincv::cuda::erode(src.constView(), dst.view(), rect3,
                                           bincv::BORDER_CONSTANT);
                    }, 50, 11),
                    floor, "kernel");
    printArmVsFloor("morphologyEx OPEN rect3x3 (2 launches)",
                    timeKernel([&] {
                        bincv::cuda::morphologyEx(src.constView(), dst.view(),
                                                  bincv::MORPH_OPEN, rect3, scratch.view(),
                                                  bincv::BORDER_CONSTANT);
                    }, 40, 11),
                    floor, "kernel");
    printArmVsFloor("erode ellipse5x5 BORDER_CONSTANT",
                    timeKernel([&] {
                        bincv::cuda::erode(src.constView(), dst.view(), ell5,
                                           bincv::BORDER_CONSTANT);
                    }, 30, 11),
                    floor, "kernel");
    printArmVsFloor("erode rect3x3 BORDER_REFLECT_101",
                    timeKernel([&] {
                        bincv::cuda::erode(src.constView(), dst.view(), rect3,
                                           bincv::BORDER_REFLECT_101);
                    }, 50, 11),
                    floor, "kernel");
    // toDeviceElement is a HOST function on the per-call path, and it evaluates
    // a sqrt per element row for MORPH_ELLIPSE. An entry point with no printed
    // cost makes no performance claim, which is the framing this project's own
    // benchmark-at-birth rule was written against -- so it gets a line, beside
    // the launch it precedes.
    {
        const int reps = 20000;
        const auto t0 = std::chrono::steady_clock::now();
        int sink = 0;
        for (int i = 0; i < reps; ++i) {
            const auto e = bincv::cuda::toDeviceElement(StructuringElement::ellipse(5, 5));
            sink += e.reachX;
        }
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0).count() /
                          reps;
        std::printf(" %-44s %9.4f us  [HOST, per call]  (sink %d)\n",
                    "toDeviceElement, ellipse5x5 (5 sqrt)", us, sink != 0);
        const auto t1 = std::chrono::steady_clock::now();
        sink = 0;
        for (int i = 0; i < reps; ++i) {
            const auto e = bincv::cuda::toDeviceElement(bincv::rect3x3());
            sink += e.reachX;
        }
        const double us2 = std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - t1).count() /
                           reps;
        std::printf(" %-44s %9.4f us  [HOST, per call]  (sink %d)\n",
                    "toDeviceElement, rect3x3", us2, sink != 0);
        std::printf("   It is host work on the call path, not a kernel, and the spans it\n"
                    "   resolves are why no kernel here evaluates a sqrt: the host header\n"
                    "   records that a shape query inside the word loop made a 5x5 erosion\n"
                    "   17x slower than cv::erode, and a device kernel would pay it once\n"
                    "   per output word.\n");
    }

    std::printf("   NO SHARE IS PRINTED, and that is deliberate: no in-repo device\n"
                "   pipeline calls morphology, so any share would be a percentage of a\n"
                "   pipeline that exists only because this benchmark inserted an OPEN\n"
                "   into it. These are the absolute numbers; the arithmetic over a real\n"
                "   pipeline is the caller's, over their own.\n");
}

// ---------------------------------------------------------------------------
// BAR 1 -- the role comparison, and the memory that goes with it
// ---------------------------------------------------------------------------

void memoryTable(const Frame& f) {
    const int w = static_cast<int>(f.width);
    const int h = static_cast<int>(f.height);
    bincv::cuda::DeviceBinMat probe(w, h);
    const size_t binStrideBytes = probe.getAlignedWidth() * sizeof(uint32_t);
    const size_t binBytes = 2 * f.height * binStrideBytes;  // src + dst

    printMemoryHeader(f.name);
    printAllocSum("binCV erode: src + dst", binBytes);
    printPitch("binCV plane", binStrideBytes, f.height, (f.width + 7) / 8);

#if defined(BINCV_CUDA_HAVE_OPENCV)
    cv::cuda::GpuMat cvSrc(h, w, CV_8UC1);
    cv::cuda::GpuMat cvDst(h, w, CV_8UC1);
    const size_t cvStep = cvSrc.step;
    const size_t cvBytes = 2 * f.height * cvStep;
    printAllocSum("cv::cuda CV_8UC1: src + dst", cvBytes);
    printPitch("cv::cuda GpuMat", cvStep, f.height, f.width);
    std::printf("   RATIO, meter 1 against meter 1 at the two MEASURED pitches:\n"
                "     2*%zu*%zu / 2*%zu*%zu = %.4fx smaller for binCV\n",
                f.height, cvStep, f.height, binStrideBytes,
                static_cast<double>(cvBytes) / static_cast<double>(binBytes));
    std::printf("   That is the formula evaluated at what the two containers actually\n"
                "   laid out, not a predicted pitch. The natural-pitch arithmetic for\n"
                "   comparison is %zu B/row against %zu B/row = %.4fx.\n",
                f.width, binStrideBytes, static_cast<double>(f.width) /
                                             static_cast<double>(binStrideBytes));

    // Meter 2 only where it can resolve. Its step on this driver is 2 MB.
    if (binBytes + cvBytes > 4u * 1024u * 1024u) {
        const size_t step = measureDriverMeterStep();
        DeviceMemMeter meter;
        {
            bincv::cuda::DeviceBinMat a(w, h), b(w, h);
            printDriverDelta("binCV src + dst", meter.deltaBytes(), step);
        }
        meter.reset();
        {
            cv::cuda::GpuMat a(h, w, CV_8UC1), b(h, w, CV_8UC1);
            printDriverDelta("cv::cuda src + dst", meter.deltaBytes(), step);
        }
    } else {
        std::printf("   [meter 2: cudaMemGetInfo] NOT TAKEN at this size. Its step on\n"
                    "                             this driver is measured in MB and this\n"
                    "                             whole working set is under one unit, so\n"
                    "                             a reading here would be the meter's\n"
                    "                             resolution rather than a footprint.\n");
    }
#else
    std::printf("   cv::cuda side: NOT MEASURED -- this binary was built without a\n"
                "   cudafilters-capable OpenCV. Point -DBINCV_CUDA_OPENCV_DIR at one.\n"
                "   Under the ship-on-both-axes rule an unmeasured speed axis is not a\n"
                "   pass, so the role table below says so rather than being omitted.\n");
#endif
}

#if defined(BINCV_CUDA_HAVE_OPENCV)
void roleComparison(const Frame& f) {
    const int w = static_cast<int>(f.width);
    const int h = static_cast<int>(f.height);
    const auto bytes = blobFrame(f.width, f.height, 0xC0FFEEu);
    // The SAME binary content on both sides: threshold once on the host, then
    // store it as bits for binCV and as {0,255} bytes for cv::cuda.
    std::vector<uint8_t> binary(bytes.size());
    for (size_t i = 0; i < bytes.size(); ++i) binary[i] = bytes[i] > 127 ? 255 : 0;

    BinMat<uint32_t> hostBits(w, h);
    bincv::packBits<bincv::PackRule::GreaterThan>(bytes.data(), f.width, f.height, f.width,
                                                  hostBits.view(), uint8_t{127});
    bincv::cuda::DeviceBinMat src(w, h), dst(w, h), scratch(w, h);
    if (bincv::cuda::upload(hostBits.constView(), src.view()) != cudaSuccess) return;

    cv::Mat hostByte(h, w, CV_8UC1, binary.data());
    cv::cuda::GpuMat cvSrc, cvDst;
    cvSrc.upload(hostByte);
    cvDst.create(h, w, CV_8UC1);

    struct Case {
        const char* name;
        cv::Mat kernel;
        StructuringElement se;
        int cvOp;
        bincv::MorphOp op;
    };
    std::vector<Case> cases = {
        {"erode rect3x3", cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)),
         bincv::rect3x3(), cv::MORPH_ERODE, bincv::MORPH_ERODE},
        {"morphologyEx OPEN rect3x3",
         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), bincv::rect3x3(),
         cv::MORPH_OPEN, bincv::MORPH_OPEN},
        {"erode ellipse5x5 -- the case the CPU arm LOSES 0.32x",
         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
         StructuringElement::ellipse(5, 5), cv::MORPH_ERODE, bincv::MORPH_ERODE},
    };

    std::printf("\n BAR 1 -- ROLE COMPARISON, %s. [kernel-resident, interleaved]\n", f.name);
    std::printf("   baseline: cv::cuda::createMorphologyFilter(op, CV_8UC1, kernel)\n"
                "   -> apply(), module cudafilters, NPP-backed. Same binary content,\n"
                "   same element, same anchor. ROLE-ONLY: cudafilters takes no\n"
                "   BorderType at all (NPP works on an ROI and OpenCV pads around it),\n"
                "   so the two sides are NOT asked to agree at the frame's rim.\n"
                "   Correctness is settled against the HOST kernel, which is itself\n"
                "   bit-exact against cv::erode on the CPU.\n");

    for (const auto& c : cases) {
        cv::Ptr<cv::cuda::Filter> filter =
            cv::cuda::createMorphologyFilter(c.cvOp, CV_8UC1, c.kernel);
        const auto el = bincv::cuda::toDeviceElement(c.se);
        const auto cvArm = [&] { filter->apply(cvSrc, cvDst); };
        const auto binArm = [&] {
            bincv::cuda::morphologyEx(src.constView(), dst.view(), c.op, el, scratch.view(),
                                      bincv::BORDER_CONSTANT);
        };
        const PairedTiming p = timeKernelPaired(cvArm, binArm, 30, 30, 11);
        char nameA[128], nameB[128];
        std::snprintf(nameA, sizeof(nameA), "cv::cuda  %s", c.name);
        std::snprintf(nameB, sizeof(nameB), "binCV     %s", c.name);
        printPaired(nameA, nameB, p, "kernel");
        const double gain = p.b.medianMs > 0.0 ? p.a.medianMs / p.b.medianMs : 0.0;
        // THE THREE DISPOSITIONS ARE THE SHIP RULE'S AND THEY ARE UNCHANGED;
        // what picks between them is now measure_util.hpp's rule on the
        // PER-ROUND RATIO rather than a margin against the two arms' summed
        // spreads. The old spelling charged the difference twice for the drift
        // the pairing had already divided out -- both arms are slow together in
        // a bad round, so both spreads grow while the ratio does not move --
        // and it read disjoint ranges as a second, independent veto.
        const double marginMs = p.a.medianMs - p.b.medianMs;
        const double summedSpreadMs = (p.a.maxMs - p.a.minMs) + (p.b.maxMs - p.b.minMs);
        const bool real = p.differenceClearsNoise(cudabench::runToRunScatterFactor());
        std::printf("   binCV runs in %.3f ms against cv::cuda's %.3f ms -- %.2fx.\n"
                    "   Margin %.4f ms; the two arms' summed spread is %.4f ms, printed\n"
                    "   as context and no longer as the bar.\n",
                    p.b.medianMs, p.a.medianMs, gain, marginMs, summedSpreadMs);
        if (real && p.ratioMedian < 1.0)
            std::printf("   VERDICT (a): ahead, and the difference clears the noise "
                        "-- leads on BOTH axes.\n");
        else if (real)
            std::printf("   VERDICT (c): BEHIND, and the difference clears the noise. The "
                        "ship rule\n   allows two dispositions and neither is silent "
                        "merging.\n");
        else
            std::printf("   VERDICT (b): NULL RESULT on time -- the two arms are the same "
                        "speed as far\n   as this run can tell, so it stands on the "
                        "memory axis, which memory wins.\n");
    }
}
#endif

// ---------------------------------------------------------------------------
// The occupancy arms that are NOT here, and the numbers that decided it
// ---------------------------------------------------------------------------
void occupancyDisposition() {
    std::printf(
        "\n OCCUPANCY -- NOT PORTED TO THE DEVICE, and this is the arithmetic.\n"
        "   There is no cv::cuda equivalent and no CPU OpenCV equivalent either\n"
        "   (cv::goodFeaturesToTrack's minDistance spaces ONE detection's corners\n"
        "   against each other, which occupancy.hpp is explicit is a different job),\n"
        "   so no role bar exists to measure against and stating one would be\n"
        "   inventing it.\n"
        "   The best existing option for the job the header exists for is the HOST's\n"
        "   own spaceCandidates: 3,333 ns at ZERO bytes. That is BELOW this host's\n"
        "   launch floor printed at the top of this run, so no device shape can clear\n"
        "   it -- one launch costs more than the whole host arm.\n"
        "   The alternative bar on offer was the host MASK arm (88,767 ns x86 /\n"
        "   380,629 ns aarch64), which spaceCandidates already beats by 26.6x. Making\n"
        "   a device arm pass against the option the header tells callers NOT to use\n"
        "   is measuring against a fallback nobody would use, which CLAUDE.md names\n"
        "   by name.\n"
        "   Dropped: markOccupiedBatch, occupiedBatch, clearOccupancy. Shipping a\n"
        "   mask producer and a mask reader with no device user between them would\n"
        "   also put two kernels in the bit-exactness budget for a consumer that does\n"
        "   not exist yet.\n");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return 77;
    }

    std::printf("============================================================\n");
    std::printf("  binCV CUDA -- the window family: morphology\n");
    std::printf("============================================================\n");
    printDevice();
    const Timing floor = measureLaunchFloor();
    printLaunchFloor(floor);

    for (const Frame& f : kFrames) {
        // Every pair emitted below is tagged with this frame, so a seven-run
        // aggregation does not pool two geometries under one arm-name key and
        // read the difference between them as run-to-run scatter.
        cudabench::pairedScope() = f.name;
        std::printf("\n------------------------------------------------------------\n");
        std::printf("  %s\n", f.name);
        std::printf("------------------------------------------------------------\n");
        memoryTable(f);
        internalArms(f, floor.medianMs);
        absoluteCost(f, floor);
#if defined(BINCV_CUDA_HAVE_OPENCV)
        roleComparison(f);
#else
        std::printf("\n BAR 1 -- ROLE COMPARISON: UNMEASURED in this build. An op whose\n"
                    " speed axis cannot be measured does not ship under the owner's\n"
                    " both-axes rule; rebuild with -DBINCV_CUDA_OPENCV_DIR pointing at\n"
                    " an OpenCV that has cudafilters.\n");
#endif
    }

    occupancyDisposition();
    std::printf("\n");
    return 0;
}
