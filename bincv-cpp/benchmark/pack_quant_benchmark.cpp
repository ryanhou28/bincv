// ===========================================================================
// T5.9 -- N-BIT INGESTION WITHOUT OpenCV, AND WHETHER IT COSTS ANYTHING.
//
// Before this, the only way into a `QuantMat<N>` was `fromCVMat`, which takes a
// `cv::Mat` -- so N-bit ingestion required linking OpenCV, which is the one thing the
// core-only build exists to avoid. `packQuant` is the core-only path.
//
// THREE ARMS, AND THE FIRST TWO ARE THE POINT:
//
//   (a) the PORTABLE path -- `transpose8x8`, eight pixels and all N planes at a time;
//   (b) the VECTOR path -- the scale as `MaxValue` byte compares, then one move-mask
//       per plane;
//   (c) `packQuantWith`, the arbitrary-map escape hatch, which cannot vectorise.
//
// (a) vs (b) IS ALSO A LIVENESS CHECK, and it is here deliberately. X-89 shipped a
// vector block that was compiled out by a mis-attached `#define`, measured three
// "improvements" against it, and only caught it by timing the kernel in isolation and
// noticing it did not respond to `-mavx2`. **A vector arm that cannot be switched off
// cannot be shown to be on.**
//
// This benchmark needs no OpenCV -- which is the claim it is measuring.
//
// Usage: pack_quant_benchmark
// ===========================================================================

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/ops/pack.hpp"
#include "bincv-cpp/quantMat.hpp"

namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kW = 752, kH = 480;

template <size_t N>
void runOne(const std::vector<uint8_t>& src) {
    bincv::QuantMat<N, uint32_t> dst(static_cast<int>(kW), static_cast<int>(kH));
    bincv::BinMatView<uint32_t> planes[N];
    for (size_t p = 0; p < N; ++p) planes[p] = dst.plane(p);

    uint8_t lut[256];
    for (unsigned v = 0; v < 256u; ++v) {
        lut[v] = static_cast<uint8_t>((v * ((1u << N) - 1u) + 127u) / 255u);
    }

    constexpr int kRounds = 10, kReps = 40;
    std::vector<double> tv, tp, tm;
    for (int r = 0; r < kRounds; ++r) {
        bincv::impl::packQuantSimdEnabled() = true;
        auto t0 = Clock::now();
        for (int i = 0; i < kReps; ++i) {
            bincv::packQuant<bincv::QuantRule::Scale, N, uint8_t, uint32_t>(
                src.data(), kW, kH, kW, planes);
        }
        tv.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count() /
                     kReps);

        bincv::impl::packQuantSimdEnabled() = false;
        t0 = Clock::now();
        for (int i = 0; i < kReps; ++i) {
            bincv::packQuant<bincv::QuantRule::Scale, N, uint8_t, uint32_t>(
                src.data(), kW, kH, kW, planes);
        }
        tp.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count() /
                     kReps);
        bincv::impl::packQuantSimdEnabled() = true;

        t0 = Clock::now();
        for (int i = 0; i < kReps; ++i) {
            bincv::packQuantWith<N, uint8_t, uint32_t>(src.data(), kW, kH, kW, planes,
                                                       [&](uint8_t v) { return lut[v]; });
        }
        tm.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count() /
                     kReps);
    }
    const double v = *std::min_element(tv.begin(), tv.end());
    const double p = *std::min_element(tp.begin(), tp.end());
    const double m = *std::min_element(tm.begin(), tm.end());
    std::printf("  N=%zu   vector %7.1f us   portable %7.1f us  (%4.2fx)   "
                "packQuantWith %7.1f us  (%4.2fx)\n",
                N, v, p, p / v, m, p / m);
}

}  // namespace

int main() {
    std::vector<uint8_t> src(kW * kH);
    uint64_t st = UINT64_C(20260828);
    for (auto& b : src) {
        st = st * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        b = static_cast<uint8_t>(st >> 41);
    }
    std::printf("=== T5.9: N-bit ingestion, %zux%zu, 10 interleaved rounds, minimum ===\n",
                kW, kH);
    std::printf("no OpenCV anywhere in this binary -- which is the claim\n\n");
    runOne<1>(src);
    runOne<2>(src);
    runOne<4>(src);
    std::printf("\n  (N=8 has MaxValue 255, so the compare-per-level form does not apply\n"
                "   and both arms take the transpose path -- expect ~1.00x.)\n");
    runOne<8>(src);
    return 0;
}
