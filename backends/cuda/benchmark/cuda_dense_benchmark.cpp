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
// MEMORY, stated per arm: the device working set is printed from the same
// dimensions the allocations use. The refused cost volume at this frame and
// D=64 would be 23 MB; neither backend materializes it.

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
    const size_t censusDeviceBytes =
        2 * kK * kH * cenL.getAlignedWidth() * 4 + 2 * kW * kH + kW * kH;

    std::printf("\n--- binary entry (pair already packed): D=64, 9x9 ---\n");
    std::printf(" device working set: %zu KB (the refused cost volume: 23 MB)\n\n",
                binaryDeviceBytes / 1024);

    // GPU, kernel-resident.
    const auto tKernel = cudabench::timeKernel([&] {
        bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), p,
                                          dDisp.view());
    });
    cudabench::printArm("GPU binary, resident (sliding arm)", tKernel, "kernel");

    // The reference arm from the same binary, through the switch. If the
    // ratio reads ~1.00x, the fast arm is not running where the line above
    // says it is -- the same check every host vector arm carries.
    bincv::cuda::impl::denseFastArmEnabled() = false;
    const auto tRef = cudabench::timeKernel([&] {
        bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), p,
                                          dDisp.view());
    });
    bincv::cuda::impl::denseFastArmEnabled() = true;
    std::printf(" %-44s %9.3f ms  spread %4.0f%%  [kernel]  (fast arm %.2fx)\n",
                "GPU binary, reference arm (switch off)", tRef.medianMs,
                tRef.spreadPct(), tRef.medianMs / tKernel.medianMs);

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
    std::printf(" device working set: %zu KB\n\n", censusDeviceBytes / 1024);

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
    cudabench::printArm("GPU census matcher (K=24, sliding arm)", tMatch, "kernel");
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

    const double e2eCensus = hostWallMs(
        [&] {
            bincv::cuda::uploadImage<uint8_t>(lw.data(), kW, kH, kW, dLw.view());
            bincv::cuda::uploadImage<uint8_t>(rw.data(), kW, kH, kW, dRw.view());
            bincv::cuda::censusTransform<kK>(dLw.constView(), bincv::kCensus5x5,
                                             cenL.view());
            bincv::cuda::censusTransform<kK>(dRw.constView(), bincv::kCensus5x5,
                                             cenR.view());
            bincv::cuda::denseDisparityCensus(cenL.constView(), cenR.constView(), kK,
                                              kH, p, dDisp.view());
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
        const double gpuCensusTotal = tCen.medianMs + tMatch.medianMs;
        std::printf("\n resident (census+match) speedup vs CPU: %.1fx   end-to-end: %.1fx\n",
                    t[0].medianNs / 1e6 / gpuCensusTotal, t[0].medianNs / 1e6 / e2eCensus);
    }

    std::printf("\n sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
