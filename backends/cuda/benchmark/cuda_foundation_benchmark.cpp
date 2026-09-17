// The foundation ops, priced at birth. MICROBENCHMARKS: each number is one
// kernel in a loop that does nothing else, next to the host library's arm for
// the same op on this machine's CPU -- shares of a real pipeline come from the
// dense benchmark, not from here. Transfers are priced too, because they are
// the tax every non-resident use pays.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/cuda/census.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/logic.hpp"
#include "bincv/cuda/pack.hpp"
#include "bincv/cuda/reduce.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/census.hpp"
#include "bincv/ops/logic.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/ops/reduce.hpp"
#include "cuda_bench_util.hpp"
#include "measure_util.hpp"

namespace {
constexpr size_t kW = 752, kH = 480;
constexpr size_t kK = 24;

std::vector<uint8_t> randomFrame(uint64_t seed) {
    std::vector<uint8_t> img(kW * kH);
    for (auto& v : img) v = static_cast<uint8_t>(measure::nextRandom(seed) >> 40);
    return img;
}

double cpuMedianMs(const char* name, std::function<void(int)> body) {
    std::vector<measure::Bench> arm = {{name, std::move(body)}};
    const auto t = measure::measureInterleaved(arm, 5, 25.0);
    return t[0].medianNs / 1e6;
}
} // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    std::printf("=== CUDA foundation ops, %zux%zu (microbenchmarks) ===\n", kW, kH);
    cudabench::printDevice();
    std::printf(" columns: GPU kernel-resident vs host library CPU arm, medians\n\n");

    const auto f1 = randomFrame(1), f2 = randomFrame(2);
    bincv::BinMat<uint32_t> a(kW, kH), b(kW, kH), hostDst(kW, kH);
    bincv::packBits<bincv::PackRule::GreaterThan>(f1.data(), kW, kH, kW, a.view(),
                                                  uint8_t{127});
    bincv::packBits<bincv::PackRule::GreaterThan>(f2.data(), kW, kH, kW, b.view(),
                                                  uint8_t{127});

    bincv::cuda::DeviceBinMat da(kW, kH), db(kW, kH), dd(kW, kH);
    bincv::cuda::upload(a.constView(), da.view());
    bincv::cuda::upload(b.constView(), db.view());
    bincv::cuda::DeviceImage<uint8_t> dImg(kW, kH);
    bincv::cuda::uploadImage<uint8_t>(f1.data(), kW, kH, kW, dImg.view());
    bincv::cuda::DeviceBinMat dBlock(kW, kK * kH);
    unsigned long long* dCount = nullptr;
    cudaMalloc(&dCount, sizeof(unsigned long long));
    cudaDeviceSynchronize();

    // ---- transfers: the non-resident tax, both directions, both shapes ----
    {
        const auto up = cudabench::timeKernel(
            [&] { bincv::cuda::upload(a.constView(), da.view()); }, 50, 9);
        cudabench::printArm("upload packed frame (45 KB)", up, "kernel");
        const auto upWide = cudabench::timeKernel(
            [&] { bincv::cuda::uploadImage<uint8_t>(f1.data(), kW, kH, kW, dImg.view()); },
            50, 9);
        cudabench::printArm("upload wide frame (361 KB)", upWide, "kernel");
        std::vector<uint8_t> back(kW * kH);
        const auto down = cudabench::timeKernel(
            [&] {
                bincv::cuda::downloadImage<uint8_t>(
                    bincv::cuda::DeviceImageConstView<uint8_t>(dImg.constView()),
                    back.data(), kW);
            },
            50, 9);
        cudabench::printArm("download wide frame (361 KB)", down, "kernel");
        measure::g_sink += back[0];
    }
    std::printf("\n");

    // ---- logic ----
    {
        const auto t = cudabench::timeKernel(
            [&] { bincv::cuda::bitwiseAnd(da.constView(), db.constView(), dd.view()); },
            100, 9);
        const double cpu = cpuMedianMs("and", [&](int) {
            bincv::bitwiseAnd<uint32_t>(a.constView(), b.constView(), hostDst.view());
            measure::g_sink += hostDst.data()[0];
        });
        cudabench::printArm("bitwiseAnd GPU", t, "kernel");
        std::printf(" %-44s %9.3f ms            [cpu]\n", "bitwiseAnd CPU", cpu);
    }

    // ---- reduce ----
    {
        const auto t = cudabench::timeKernel(
            [&] { bincv::cuda::countNonZeroAsync(da.constView(), dCount); }, 100, 9);
        const double cpu = cpuMedianMs("count", [&](int) {
            measure::g_sink += bincv::countNonZero<uint32_t>(a.constView());
        });
        cudabench::printArm("countNonZero GPU (async form)", t, "kernel");
        std::printf(" %-44s %9.3f ms            [cpu]\n", "countNonZero CPU", cpu);
    }

    // ---- sensor stage ----
    {
        const auto t = cudabench::timeKernel(
            [&] {
                bincv::cuda::packBits(dImg.constView(), dd.view(),
                                      bincv::PackRule::GreaterThan, uint8_t{127});
            },
            100, 9);
        const double cpu = cpuMedianMs("pack", [&](int) {
            bincv::packBits<bincv::PackRule::GreaterThan>(f1.data(), kW, kH, kW,
                                                          hostDst.view(), uint8_t{127});
            measure::g_sink += hostDst.data()[0];
        });
        cudabench::printArm("packBits GPU (GreaterThan)", t, "kernel");
        std::printf(" %-44s %9.3f ms            [cpu]\n", "packBits CPU (vector arm)", cpu);

        // The measured question the sensor stage exists to answer: wide frame
        // up + device pack, or CPU pack + bits up? Both enqueue-to-sync.
        bincv::cuda::DeviceImage<uint8_t> stageImg(kW, kH);
        const auto devPath = cudabench::timeKernel(
            [&] {
                bincv::cuda::uploadImage<uint8_t>(f1.data(), kW, kH, kW, stageImg.view());
                bincv::cuda::packBits(stageImg.constView(), dd.view(),
                                      bincv::PackRule::GreaterThan, uint8_t{127});
            },
            20, 9);
        cudabench::printArm("  path A: wide up + device pack", devPath, "kernel");
        const double cpuPath = cpuMedianMs("cpupack+up", [&](int) {
            bincv::packBits<bincv::PackRule::GreaterThan>(f1.data(), kW, kH, kW,
                                                          hostDst.view(), uint8_t{127});
            bincv::cuda::upload(hostDst.constView(), dd.view());
            cudaDeviceSynchronize();
        });
        std::printf(" %-44s %9.3f ms            [e2e]\n",
                    "  path B: CPU pack + bits up", cpuPath);
    }

    // ---- census transform ----
    {
        const auto t = cudabench::timeKernel(
            [&] {
                bincv::cuda::censusTransform<kK>(dImg.constView(), bincv::kCensus5x5,
                                                 dBlock.view());
            },
            20, 9);
        std::vector<bincv::BinMat<uint32_t>> planes;
        std::vector<bincv::BinMatView<uint32_t>> views;
        for (size_t k = 0; k < kK; ++k) planes.emplace_back(kW, kH);
        for (size_t k = 0; k < kK; ++k) views.push_back(planes[k].view());
        const double cpu = cpuMedianMs("census", [&](int) {
            bincv::censusTransform<kK, uint8_t, uint32_t>(f1.data(), kW, kH, kW,
                                                          bincv::kCensus5x5,
                                                          views.data());
            measure::g_sink += planes[0].data()[0];
        });
        cudabench::printArm("censusTransform GPU (24 planes, tiled)", t, "kernel");
        bincv::cuda::impl::censusTiledEnabled() = false;
        const auto tRef = cudabench::timeKernel(
            [&] {
                bincv::cuda::censusTransform<kK>(dImg.constView(), bincv::kCensus5x5,
                                                 dBlock.view());
            },
            20, 9);
        bincv::cuda::impl::censusTiledEnabled() = true;
        std::printf(" %-44s %9.3f ms  spread %4.0f%%  [kernel]  (tiled arm %.2fx)\n",
                    "censusTransform GPU, reference arm", tRef.medianMs,
                    tRef.spreadPct(), tRef.medianMs / t.medianMs);
        std::printf(" %-44s %9.3f ms            [cpu]\n",
                    "censusTransform CPU (vector arm)", cpu);
    }

    // ---- the covariance, and why the batched entry point exists ----
    //
    // THE MEASUREMENT THAT DECIDES THE SIGNATURE. 200 keypoints, 31x31 windows
    // -- the tracker's shape. Per-window launches pay ~5-10 us of launch each
    // against nanoseconds of work; the batch pays one. Both produce identical
    // counts (the tests pin that), so this is purely what the signature costs.
    {
        constexpr size_t kKeypoints = 200;
        constexpr int kWin = 31;
        const auto f3 = randomFrame(3), f4 = randomFrame(4);
        bincv::BinMat<uint32_t> magX(kW, kH), magY(kW, kH), sgnX(kW, kH), sgnY(kW, kH);
        bincv::packBits<bincv::PackRule::GreaterThan>(f1.data(), kW, kH, kW, magX.view(),
                                                      uint8_t{100});
        bincv::packBits<bincv::PackRule::GreaterThan>(f2.data(), kW, kH, kW, magY.view(),
                                                      uint8_t{100});
        bincv::packBits<bincv::PackRule::GreaterThan>(f3.data(), kW, kH, kW, sgnX.view(),
                                                      uint8_t{127});
        bincv::packBits<bincv::PackRule::GreaterThan>(f4.data(), kW, kH, kW, sgnY.view(),
                                                      uint8_t{127});
        bincv::cuda::DeviceBinMat dMagX(kW, kH), dMagY(kW, kH), dSgnX(kW, kH),
            dSgnY(kW, kH);
        bincv::cuda::upload(magX.constView(), dMagX.view());
        bincv::cuda::upload(magY.constView(), dMagY.view());
        bincv::cuda::upload(sgnX.constView(), dSgnX.view());
        bincv::cuda::upload(sgnY.constView(), dSgnY.view());

        std::vector<bincv::Rect> rects(kKeypoints);
        uint64_t seed = 0xC0FFEE;
        for (auto& r : rects) {
            const int x = static_cast<int>(measure::nextRandom(seed) % (kW - kWin));
            const int y = static_cast<int>(measure::nextRandom(seed) % (kH - kWin));
            r = bincv::Rect{x, y, kWin, kWin};
        }
        bincv::Rect* dRects = nullptr;
        bincv::cuda::DeviceCovarianceCount* dCov = nullptr;
        cudaMalloc(&dRects, kKeypoints * sizeof(bincv::Rect));
        cudaMalloc(&dCov, kKeypoints * sizeof(bincv::cuda::DeviceCovarianceCount));
        cudaMemcpy(dRects, rects.data(), kKeypoints * sizeof(bincv::Rect),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        std::printf("\n covariance over %zu keypoints, %dx%d windows (the tracker's shape)\n",
                    kKeypoints, kWin, kWin);
        const auto tBatch = cudabench::timeKernel(
            [&] {
                bincv::cuda::countCovarianceBatchAsync(dMagX.constView(),
                                                       dMagY.constView(),
                                                       dSgnX.constView(),
                                                       dSgnY.constView(), dRects,
                                                       kKeypoints, dCov);
            },
            20, 9);
        cudabench::printArm("  countCovariance BATCH (one launch)", tBatch, "kernel");

        // The same work through the single-region entry: 200 launches.
        const auto tLoop = cudabench::timeKernel(
            [&] {
                for (size_t i = 0; i < kKeypoints; ++i)
                    bincv::cuda::countCovarianceAsync(dMagX.constView(),
                                                      dMagY.constView(),
                                                      dSgnX.constView(),
                                                      dSgnY.constView(), rects[i],
                                                      dCov + i);
            },
            3, 7);
        std::printf(" %-44s %9.3f ms  spread %4.0f%%  [kernel]  (batch %.1fx)\n",
                    "  per-window loop (200 launches)", tLoop.medianMs,
                    tLoop.spreadPct(), tLoop.medianMs / tBatch.medianMs);

        const double cpu = cpuMedianMs("cov", [&](int) {
            for (size_t i = 0; i < kKeypoints; ++i) {
                const auto c = bincv::countCovariance<uint32_t>(
                    magX.constView(), magY.constView(), sgnX.constView(),
                    sgnY.constView(), rects[i]);
                measure::g_sink += c.xx;
            }
        });
        std::printf(" %-44s %9.3f ms            [cpu]\n",
                    "  host countCovariance x200", cpu);
        std::printf("  batch vs host CPU: %.1fx\n", cpu / tBatch.medianMs);
        cudaFree(dRects);
        cudaFree(dCov);
    }

    // ---- the N-bit ingestion path ----
    {
        constexpr size_t kN = 2;   // the shipped tracking depth
        bincv::cuda::DeviceBinMat dQuant(kW, kN * kH);
        const auto t = cudabench::timeKernel(
            [&] { bincv::cuda::packQuant(dImg.constView(), dQuant.view(), kN); }, 50, 9);
        std::vector<bincv::BinMat<uint32_t>> planes;
        for (size_t p = 0; p < kN; ++p) planes.emplace_back(kW, kH);
        bincv::BinMatView<uint32_t> views[kN];
        for (size_t p = 0; p < kN; ++p) views[p] = planes[p].view();
        const double cpu = cpuMedianMs("packQuant", [&](int) {
            bincv::packQuant<bincv::QuantRule::Scale, kN, uint8_t, uint32_t>(
                f1.data(), kW, kH, kW, views);
            measure::g_sink += planes[0].data()[0];
        });
        std::printf("\n");
        cudabench::printArm("packQuant N=2 GPU", t, "kernel");
        std::printf(" %-44s %9.3f ms            [cpu]\n",
                    "packQuant N=2 CPU (vector arm)", cpu);
    }

    cudaFree(dCount);
    std::printf("\n sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
