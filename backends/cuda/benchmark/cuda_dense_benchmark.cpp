// The flagship, priced at birth: dense disparity on the device against the
// same operation on this machine's CPU, same frames, one binary.
//
// WHAT EACH NUMBER COVERS -- read the clock column:
//   kernel   CUDA events around the matcher alone: the per-frame cost of a
//            RESIDENT pipeline, the number the backend exists for.
//   e2e      host clock around upload + kernels + download + synchronize: the
//            cost when this is the ONLY thing the GPU does for the frame.
//   cpu      the host library on this machine's own CPU arm, measured by the
//            project's interleaved protocol. WSL2 CPU numbers carry spread;
//            it is printed.
//
// The cv::cuda::StereoBM denominator lives in cuda_stereobm_benchmark.cpp,
// which needs an OpenCV built with cudastereo.
//
// MEMORY, stated per arm on THREE NAMED METERS -- the allocation sum, the
// cudaMemGetInfo delta, and the read-back row pitch. They answer different
// questions, so each line names the meter it was read on and no ratio crosses
// two of them (docs/reports/cuda.md, "one meter per comparison"). The refused
// cost volume at this frame and D=64 would be 23 MB; neither backend
// materializes it.
//
// THE LAUNCH FLOOR is printed first, before any arm. The binary matcher runs
// at 0.069 ms here with a spread in the tens of percent, which is close enough
// to the floor that the floor is part of reading the number rather than a
// footnote under it.
//
// THE GATE-EXCLUDED ARM is the last section. Project rule (CLAUDE.md): a
// benchmark must carry a case the fast path's own gate REJECTS, and that case
// must read ~1.00x -- otherwise the switch positions above are not selecting
// what their lines claim.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/cuda/census.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/denseDisparity.hpp"
#include "bincv/cuda/pack.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/denseDisparity.hpp"
#include "bincv/ops/pack.hpp"
#include "cuda_bench_util.hpp"
#include "measure_util.hpp"

namespace {

constexpr size_t kW = 752, kH = 480;  // the reference frame
constexpr int kShift = 21;
constexpr size_t kK = 24;  // census 5x5

std::vector<uint8_t> smoothFrame(size_t w, size_t h) {
    std::vector<uint8_t> img(w * h);
    uint64_t st = 0xFEEDFACEULL;
    for (auto& v : img) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<uint8_t>(st >> 40);
    }
    std::vector<uint8_t> tmp(w * h);
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

double hostWallMs(const std::function<void()>& body, int repeats = 7) {
    using Clock = std::chrono::steady_clock;
    std::vector<double> samples;
    body();  // warm-up
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = Clock::now();
        body();
        samples.push_back(
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

} // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }

    std::printf("=== dense disparity: CUDA backend vs this machine's CPU ===\n");
    cudabench::printDevice();

    // The floor first, so every kernel number below can be read against it.
    const auto floor = cudabench::measureLaunchFloor();
    std::printf("\n");
    cudabench::printLaunchFloor(floor);

    const auto lw = smoothFrame(kW, kH);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kShift < kW; ++x)
            rw[y * kW + x] = lw[y * kW + x + kShift];

    bincv::DenseDisparityParams p;
    p.maxDisparity = 64;

    // Shared host-side objects.
    bincv::BinMat<uint64_t> lb(kW, kH), rb(kW, kH);
    bincv::packBits<bincv::PackRule::GreaterThan>(lw.data(), kW, kH, kW, lb.view(),
                                                  uint8_t{127});
    bincv::packBits<bincv::PackRule::GreaterThan>(rw.data(), kW, kH, kW, rb.view(),
                                                  uint8_t{127});
    std::vector<uint8_t> disp(kW * kH);

    // Device-side objects for the resident arms.
    bincv::cuda::DeviceBinMat dl(kW, kH), dr(kW, kH);
    bincv::cuda::upload(lb.constView(), dl.view());
    bincv::cuda::upload(rb.constView(), dr.view());
    bincv::cuda::DeviceImage<uint8_t> dDisp(kW, kH);
    bincv::cuda::DeviceImage<uint8_t> dLw(kW, kH), dRw(kW, kH);
    bincv::cuda::uploadImage<uint8_t>(lw.data(), kW, kH, kW, dLw.view());
    bincv::cuda::uploadImage<uint8_t>(rw.data(), kW, kH, kW, dRw.view());
    bincv::cuda::DeviceBinMat cenL(kW, kK * kH), cenR(kW, kK * kH);
    cudaDeviceSynchronize();

    const size_t binaryDeviceBytes =
        2 * kH * dl.getAlignedWidth() * 4 + kW * kH;  // two planes + the map
    // Two intermediates, two footprints: the plane block holds K bits per
    // pixel, the packed descriptor a whole 32-bit word. Both are printed, since
    // the 21% the packed layout costs is the other half of its 8.55x.
    const size_t censusPlaneBytes =
        2 * kK * kH * cenL.getAlignedWidth() * 4 + 2 * kW * kH + kW * kH;
    const size_t censusPackedBytes = 2 * kW * kH * 4 + 2 * kW * kH + kW * kH;

    // The driver meter's own granularity, MEASURED here rather than quoted, so
    // that every reading below can be read against it. It is the reason a
    // driver reading and an allocation sum never go into the same ratio.
    const size_t meterStep = cudabench::measureDriverMeterStep();

    std::printf("\n--- binary entry (pair already packed): D=64, 9x9 ---\n");
    cudabench::printMemoryHeader("binary entry");
    cudabench::printAllocSum("two packed planes + the map", binaryDeviceBytes);
    cudabench::printAllocSum("the cost volume this design REFUSES",
                             kW * kH * 65 /* D=64 plus d=0 */);
    {
        // The driver reading for exactly these shapes: a second set of the same
        // arrays, allocated inside the meter and freed again, so the number is
        // this working set's reservation and nothing else's.
        cudabench::DeviceMemMeter meter;
        bincv::cuda::DeviceBinMat l2(kW, kH), r2(kW, kH);
        bincv::cuda::DeviceImage<uint8_t> d2(kW, kH);
        cudabench::printDriverDelta("the same arrays, allocated again",
                                    meter.deltaBytes(), meterStep);
    }
    cudabench::printPitch("packed plane", dl.getAlignedWidth() * 4, kH, (kW + 7) / 8);
    cudabench::printPitch("disparity map", kW, kH, kW);
    std::printf("\n");

    // GPU, kernel-resident: the arm the launcher prefers.
    const auto tKernel = cudabench::timeKernel([&] {
        bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), p,
                                          dDisp.view());
    });
    cudabench::printArmVsFloor("GPU binary, resident (word-parallel arm)", tKernel,
                               floor, "kernel");

    // The two arms behind it, from the same binary, through the switches, each
    // INTERLEAVED with the arm it prices: one arm to completion and then the
    // other would put every bit of drift over the run onto the second of them,
    // and these ratios are the backend's headline claims.
    const auto wordVsSliding = cudabench::timeKernelPaired(
        [&] {
            bincv::cuda::impl::denseBitSlicedEnabled() = true;
            bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), p,
                                              dDisp.view());
        },
        [&] {
            bincv::cuda::impl::denseBitSlicedEnabled() = false;
            bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), p,
                                              dDisp.view());
        },
        40, 20, 9);
    bincv::cuda::impl::denseBitSlicedEnabled() = true;
    cudabench::printPaired("  A: word-parallel bit-sliced arm",
                           "  B: per-pixel sliding arm", wordVsSliding, "kernel");

    const auto wordVsRef = cudabench::timeKernelPaired(
        [&] {
            bincv::cuda::impl::denseFastArmEnabled() = true;
            bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), p,
                                              dDisp.view());
        },
        [&] {
            bincv::cuda::impl::denseFastArmEnabled() = false;
            bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), p,
                                              dDisp.view());
        },
        40, 8, 9);
    bincv::cuda::impl::denseFastArmEnabled() = true;
    cudabench::printPaired("  A: word-parallel bit-sliced arm",
                           "  B: reference arm (both switches off)", wordVsRef,
                           "kernel");

    // GPU, end-to-end: packed pair up, map down, synchronized.
    const double e2e = hostWallMs([&] {
        bincv::cuda::upload(lb.constView(), dl.view());
        bincv::cuda::upload(rb.constView(), dr.view());
        bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), p,
                                          dDisp.view());
        bincv::cuda::downloadImage<uint8_t>(dDisp.constView(), disp.data(), kW);
        cudaDeviceSynchronize();
    });
    std::printf(" %-44s %9.3f ms                [e2e]\n",
                "GPU binary, upload+kernel+download", e2e);

    // CPU arm, the project's own best on this machine (AVX2 at u64).
    {
        std::vector<uint64_t> sw(
            bincv::denseDisparityBinaryScratchWords<uint64_t>(kW, p));
        std::vector<uint16_t> sr(bincv::denseDisparityScratchRows(kW));
        std::vector<measure::Bench> arm = {
            {"cpu", [&](int) {
                 bincv::denseDisparityBinary<uint64_t>(lb.constView(), rb.constView(),
                                                       p, sw.data(), sw.size(),
                                                       sr.data(), sr.size(),
                                                       disp.data(), kW);
                 measure::g_sink += disp[kH / 2 * kW + kW / 2];
             }}};
        const auto t = measure::measureInterleaved(arm, 5, 60.0);
        std::printf(" %-44s %9.3f ms  spread %4.0f%%  [cpu]\n",
                    "CPU binary (host library, vector arm on)",
                    t[0].medianNs / 1e6, t[0].spreadPct());
        std::printf("\n resident speedup vs CPU: %.1fx   end-to-end: %.1fx\n",
                    t[0].medianNs / 1e6 / tKernel.medianMs, t[0].medianNs / 1e6 / e2e);
    }

    std::printf("\n--- census entry (wide 8-bit pair): D=64, 9x9, census 5x5 ---\n");
    cudabench::printMemoryHeader("census entry, two layouts");
    cudabench::printAllocSum("PLANE layout working set", censusPlaneBytes);
    cudabench::printAllocSum("PACKED layout working set", censusPackedBytes);
    {
        cudabench::DeviceMemMeter meter;
        bincv::cuda::DeviceImage<uint8_t> w1(kW, kH), w2(kW, kH);
        bincv::cuda::DeviceImage<uint32_t> d1(kW, kH), d2(kW, kH);
        bincv::cuda::DeviceImage<uint8_t> m1(kW, kH);
        cudabench::printDriverDelta("the PACKED arrays, allocated again",
                                    meter.deltaBytes(), meterStep);
    }
    cudabench::printPitch("census plane block", cenL.getAlignedWidth() * 4, kK * kH,
                          (kW + 7) / 8);
    cudabench::printPitch("packed descriptor", kW * 4, kH, kW * 4);
    std::printf("\n");

    // GPU census transform alone, then the matcher, kernel-resident.
    const auto tCen = cudabench::timeKernel([&] {
        bincv::cuda::censusTransform<kK>(dLw.constView(), bincv::kCensus5x5,
                                         cenL.view());
        bincv::cuda::censusTransform<kK>(dRw.constView(), bincv::kCensus5x5,
                                         cenR.view());
    });
    cudabench::printArm("GPU census transform, both frames", tCen, "kernel");
    const auto tMatch = cudabench::timeKernel(
        [&] {
            bincv::cuda::denseDisparityCensus(cenL.constView(), cenR.constView(), kK,
                                              kH, p, dDisp.view());
        },
        4, 7);
    cudabench::printArm("GPU census matcher (K=24, PLANE layout)", tMatch, "kernel");

    // The packed-descriptor layout: one word per pixel, so a pixel pair costs
    // one load and one popcount instead of K of each. Same map, byte for byte.
    bincv::cuda::DeviceImage<uint32_t> descL(kW, kH), descR(kW, kH);
    const auto tCenPacked = cudabench::timeKernel([&] {
        bincv::cuda::censusTransformPacked<kK>(dLw.constView(), bincv::kCensus5x5,
                                               descL.view());
        bincv::cuda::censusTransformPacked<kK>(dRw.constView(), bincv::kCensus5x5,
                                               descR.view());
    });
    cudabench::printArm("GPU census transform PACKED, both frames", tCenPacked,
                        "kernel");
    const auto tMatchPacked = cudabench::timeKernel(
        [&] {
            bincv::cuda::denseDisparityCensusPacked(descL.constView(), descR.constView(),
                                                    p, dDisp.view());
        },
        8, 9);
    std::printf(" %-44s %9.3f ms  spread %4.0f%%  [kernel]  (vs plane %.2fx)\n",
                "GPU census matcher, PACKED layout", tMatchPacked.medianMs,
                tMatchPacked.spreadPct(), tMatch.medianMs / tMatchPacked.medianMs);
    bincv::cuda::impl::denseFastArmEnabled() = false;
    const auto tMatchRef = cudabench::timeKernel(
        [&] {
            bincv::cuda::denseDisparityCensus(cenL.constView(), cenR.constView(), kK,
                                              kH, p, dDisp.view());
        },
        2, 5);
    bincv::cuda::impl::denseFastArmEnabled() = true;
    std::printf(" %-44s %9.3f ms  spread %4.0f%%  [kernel]  (fast arm %.2fx)\n",
                "GPU census matcher, reference arm", tMatchRef.medianMs,
                tMatchRef.spreadPct(), tMatchRef.medianMs / tMatch.medianMs);

    // End to end on the PACKED path -- the wide-input entry's shipped route.
    const double e2eCensus = hostWallMs(
        [&] {
            bincv::cuda::uploadImage<uint8_t>(lw.data(), kW, kH, kW, dLw.view());
            bincv::cuda::uploadImage<uint8_t>(rw.data(), kW, kH, kW, dRw.view());
            bincv::cuda::censusTransformPacked<kK>(dLw.constView(), bincv::kCensus5x5,
                                                   descL.view());
            bincv::cuda::censusTransformPacked<kK>(dRw.constView(), bincv::kCensus5x5,
                                                   descR.view());
            bincv::cuda::denseDisparityCensusPacked(descL.constView(),
                                                    descR.constView(), p, dDisp.view());
            bincv::cuda::downloadImage<uint8_t>(dDisp.constView(), disp.data(), kW);
            cudaDeviceSynchronize();
        },
        5);
    std::printf(" %-44s %9.3f ms                [e2e]\n",
                "GPU census, wide frames up to map down", e2eCensus);

    // CPU census path on the same frames.
    {
        std::vector<uint32_t> sw(
            bincv::denseDisparityScratchWords<kK, uint32_t>(kW, p));
        std::vector<uint16_t> sr(bincv::denseDisparityScratchRows(kW));
        std::vector<measure::Bench> arm = {
            {"cpu", [&](int) {
                 bincv::denseDisparity<kK, uint8_t, uint32_t>(
                     lw.data(), rw.data(), kW, kH, kW, kW, bincv::kCensus5x5, p,
                     sw.data(), sw.size(), sr.data(), sr.size(), disp.data(), kW);
                 measure::g_sink += disp[kH / 2 * kW + kW / 2];
             }}};
        const auto t = measure::measureInterleaved(arm, 3, 120.0);
        std::printf(" %-44s %9.3f ms  spread %4.0f%%  [cpu]\n",
                    "CPU census path (host library)", t[0].medianNs / 1e6,
                    t[0].spreadPct());
        const double gpuCensusTotal = tCenPacked.medianMs + tMatchPacked.medianMs;
        std::printf("\n resident (census+match) speedup vs CPU: %.1fx   end-to-end: %.1fx\n",
                    t[0].medianNs / 1e6 / gpuCensusTotal, t[0].medianNs / 1e6 / e2eCensus);
    }

    // ----------------------------------------------------------------------
    // THE GATE-EXCLUDED ARM: the case that MUST read ~1.00x.
    //
    // CLAUDE.md requires a benchmark to include a case where the fast path's
    // own gate excludes it, because that is the only cheap check that the
    // switch above is selecting anything at all. A mis-attached `#define` once
    // compiled a host vector block out entirely here and three consecutive
    // "improvements" were measured against nothing.
    //
    // The word-parallel arm's gate is winWidth <= 32 -- one 32-bit extraction
    // per window row is the whole premise -- so a winWidth of 33 leaves it and
    // BOTH switch positions run the same reference kernel. The in-gate control
    // at 31 differs in exactly one variable and must not read 1.00x.
    //
    // winHeight is 7 in both, not the 9 used above: the binary entry's own
    // contract is winWidth * winHeight <= 255, and 33 x 9 is outside it. Using
    // one height for both lines keeps the pair a one-variable comparison.
    // ----------------------------------------------------------------------
    std::printf("\n--- the gate's own exclusion: winWidth 33 (outside) vs 31 (inside) ---\n");
    {
        bincv::DenseDisparityParams pOut = p;
        pOut.winWidth = 33;
        pOut.winHeight = 7;
        bincv::DenseDisparityParams pIn = p;
        pIn.winWidth = 31;
        pIn.winHeight = 7;

        // Hazard 4 of the project's measurement protocol: what is compared must
        // AGREE before it is timed. At 33 the two switch positions are the same
        // kernel and agreement is trivial. At 31 they are two DIFFERENT kernels,
        // and the ratio below is only a comparison if their maps match.
        const auto mapOf = [&](const bincv::DenseDisparityParams& q, bool bitSliced) {
            bincv::cuda::impl::denseBitSlicedEnabled() = bitSliced;
            bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), q,
                                              dDisp.view());
            std::vector<uint8_t> out(kW * kH);
            bincv::cuda::downloadImage<uint8_t>(dDisp.constView(), out.data(), kW);
            cudaDeviceSynchronize();
            return out;
        };
        const bool agree31 = mapOf(pIn, true) == mapOf(pIn, false);
        bincv::cuda::impl::denseBitSlicedEnabled() = true;

        const auto outside = cudabench::timeKernelPaired(
            [&] {
                bincv::cuda::impl::denseBitSlicedEnabled() = true;
                bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), pOut,
                                                  dDisp.view());
            },
            [&] {
                bincv::cuda::impl::denseBitSlicedEnabled() = false;
                bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), pOut,
                                                  dDisp.view());
            },
            3, 3, 7);
        bincv::cuda::impl::denseBitSlicedEnabled() = true;
        std::printf(" winWidth 33 -- OUTSIDE the word-parallel gate (winWidth <= 32):\n");
        cudabench::printPaired("  A: switch ON  (gate rejects it anyway)",
                               "  B: switch OFF", outside, "kernel", /*expect1x=*/true);

        const auto inside = cudabench::timeKernelPaired(
            [&] {
                bincv::cuda::impl::denseBitSlicedEnabled() = true;
                bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), pIn,
                                                  dDisp.view());
            },
            [&] {
                bincv::cuda::impl::denseBitSlicedEnabled() = false;
                bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), pIn,
                                                  dDisp.view());
            },
            20, 6, 7);
        bincv::cuda::impl::denseBitSlicedEnabled() = true;
        std::printf(" winWidth 31 -- INSIDE it, the positive control. The two arms'"
                    " maps are %s.\n",
                    agree31 ? "IDENTICAL"
                            : "DIFFERENT, so the ratio below compares nothing");
        cudabench::printPaired("  A: switch ON  (word-parallel arm runs)",
                               "  B: switch OFF (per-pixel sliding arm)", inside,
                               "kernel");
        std::printf("   Read the two together: the 33 line at ~1.00x says the switch is\n"
                    "   real and the gate excludes what it claims; the 31 line away from\n"
                    "   1.00x says the arm above it is the one being timed.\n");
    }

    std::printf("\n sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
