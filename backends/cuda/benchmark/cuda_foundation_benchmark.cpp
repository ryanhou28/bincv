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
        cudabench::printArm("censusTransform GPU (24 planes)", t, "kernel");
        std::printf(" %-44s %9.3f ms            [cpu]\n",
                    "censusTransform CPU (vector arm)", cpu);
    }

    cudaFree(dCount);
    std::printf("\n sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
