// The GPU-vs-GPU role comparison: binCV's dense disparity against
// cv::cuda::StereoBM on the same device and the same frames.
//
// ROLE, not equality: StereoBM matches SAD over prefiltered bytes with its own
// validity rules; binCV matches Hamming over census bits or packed binary. The
// test suite closes correctness against the HOST library; this file prices the
// role -- "a dense disparity map from a rectified pair, resident on device" --
// against the best existing GPU option a user could reach for instead.
//
// MEMORY is reported from cudaMemGetInfo deltas around each side's working-set
// allocation. That meter is crude -- it rounds, and an allocator may pool --
// but it is the ONLY one readable on both sides of a library boundary, which
// is why every cross-library figure in docs/reports/cuda.md uses it and no
// allocation sum appears beside one. Its own granularity is MEASURED here --
// one-byte allocations until the reading moves -- and printed with every
// reading, so a working set smaller than one unit cannot be read as a
// footprint. binCV's two entries are smaller than one unit; StereoBM's is not,
// which is the asymmetry that makes the memory lead a lower bound.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudastereo.hpp>

#include "bincv/binMat.hpp"
#include "bincv/cuda/census.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/denseDisparity.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/pack.hpp"
#include "cuda_bench_util.hpp"

namespace {
constexpr size_t kW = 752, kH = 480;
constexpr int kShift = 21;
constexpr size_t kK = 24;

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

} // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    std::printf("=== dense disparity role comparison on ONE GPU: binCV vs "
                "cv::cuda::StereoBM ===\n");
    cudabench::printDevice();
    std::printf(" both sides: %zux%zu pair, 64 disparities, 9x9 support, resident\n\n",
                kW, kH);
    const auto floor = cudabench::measureLaunchFloor();
    cudabench::printLaunchFloor(floor);
    const size_t meterStep = cudabench::measureDriverMeterStep();
    std::printf("\n");

    const auto lw = smoothFrame(kW, kH);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kShift < kW; ++x)
            rw[y * kW + x] = lw[y * kW + x + kShift];

    bincv::DenseDisparityParams p;
    p.maxDisparity = 64;

    // ---- binCV, binary entry (the premise-native operating point) ----
    {
        cudabench::DeviceMemMeter meter;
        bincv::BinMat<uint32_t> lb(kW, kH), rb(kW, kH);
        bincv::packBits<bincv::PackRule::GreaterThan>(lw.data(), kW, kH, kW, lb.view(),
                                                      uint8_t{127});
        bincv::packBits<bincv::PackRule::GreaterThan>(rw.data(), kW, kH, kW, rb.view(),
                                                      uint8_t{127});
        bincv::cuda::DeviceBinMat dl(kW, kH), dr(kW, kH);
        bincv::cuda::DeviceImage<uint8_t> dDisp(kW, kH);
        bincv::cuda::upload(lb.constView(), dl.view());
        bincv::cuda::upload(rb.constView(), dr.view());
        cudaDeviceSynchronize();
        const size_t used = meter.deltaBytes();
        const auto t = cudabench::timeKernel([&] {
            bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), p,
                                              dDisp.view());
        });
        cudabench::printArmVsFloor("binCV binary entry, resident", t, floor, "kernel");
        cudabench::printDriverDelta("binCV binary working set", used, meterStep);
    }

    // ---- binCV, census entry (wide-input operating point) ----
    {
        cudabench::DeviceMemMeter meter;
        bincv::cuda::DeviceImage<uint8_t> dLw(kW, kH), dRw(kW, kH);
        bincv::cuda::uploadImage<uint8_t>(lw.data(), kW, kH, kW, dLw.view());
        bincv::cuda::uploadImage<uint8_t>(rw.data(), kW, kH, kW, dRw.view());
        bincv::cuda::DeviceImage<uint32_t> descL(kW, kH), descR(kW, kH);
        bincv::cuda::DeviceImage<uint8_t> dDisp(kW, kH);
        cudaDeviceSynchronize();
        const size_t used = meter.deltaBytes();
        const auto t = cudabench::timeKernel(
            [&] {
                bincv::cuda::censusTransformPacked<kK>(dLw.constView(),
                                                       bincv::kCensus5x5, descL.view());
                bincv::cuda::censusTransformPacked<kK>(dRw.constView(),
                                                       bincv::kCensus5x5, descR.view());
                bincv::cuda::denseDisparityCensusPacked(descL.constView(),
                                                        descR.constView(), p,
                                                        dDisp.view());
            },
            8, 9);
        cudabench::printArmVsFloor("binCV census entry (transform + match)", t, floor,
                                   "kernel");
        cudabench::printDriverDelta("binCV census working set", used, meterStep);
    }

    // ---- cv::cuda::StereoBM, same frames resident as GpuMats ----
    {
        cudabench::DeviceMemMeter meter;
        cv::Mat lm(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                   const_cast<uint8_t*>(lw.data()));
        cv::Mat rm(static_cast<int>(kH), static_cast<int>(kW), CV_8UC1,
                   const_cast<uint8_t*>(rw.data()));
        cv::cuda::GpuMat gl, gr, gd;
        gl.upload(lm);
        gr.upload(rm);
        auto bm = cv::cuda::createStereoBM(64, 9);
        bm->compute(gl, gr, gd);  // first call allocates its internals
        cudaDeviceSynchronize();
        const size_t used = meter.deltaBytes();
        const auto t = cudabench::timeKernel([&] { bm->compute(gl, gr, gd); });
        cudabench::printArmVsFloor("cv::cuda::StereoBM(64, 9), resident", t, floor,
                                   "kernel");
        cudabench::printDriverDelta("StereoBM working set (GpuMat may pool)", used,
                                    meterStep);
    }

    std::printf("\n role only: SAD-on-bytes vs Hamming-on-bits give different maps;\n"
                " correctness is settled against the host library by the test suite.\n");
    return 0;
}
