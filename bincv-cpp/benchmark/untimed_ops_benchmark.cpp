// ===========================================================================
// THE AUDIT CLAUDE.md's NEW RULE CAME FROM.
//
// a measurement found `medianWide` and `edgeThreshold` at 78% of the frontend the day something
// first called them: written bit-exact against the reference, benchmarked by nobody, and
// therefore unoptimized. That prompted a sweep of every operation for the same state --
// correct, tested, and never timed.
//
// The sweep found exactly two on real paths:
//
// * `binarize<N>` -- the N-bit-to-1-bit reduction in ops/threshold.hpp. Word-wise
// already (it gathers N plane words and produces one output word), so the SHAPE was
// never in doubt; the NUMBER was simply unknown.
// * `unpackTo8Bit` -- the output path in ops/pack.hpp, and the only way to look at what
// binCV produced on a target with no OpenCV.
//
// `packBits` is here too. It has had a vector path since earlier work, but a measurement measured it as
// part of `fromCVMat`, and the core-only entry point had never been timed on its own.
//
// Everything else already had an arm. `readPgm` is deliberately absent: it parses a
// header and memcpies, runs once per file, and is not on any per-frame path -- stating
// that is the point, rather than leaving a reader to wonder.
//
// Usage: untimed_ops_benchmark
// ===========================================================================

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/ops/pack.hpp"
#include "bincv-cpp/ops/threshold.hpp"
#include "bincv-cpp/quantMat.hpp"

namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kW = 752, kH = 480;
constexpr int kRounds = 10, kReps = 60;

double minOf(std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }

template <size_t N>
void benchBinarize(const bincv::QuantMat<N, uint32_t>& src, bincv::BinMat<uint32_t>& dst) {
    std::vector<double> ts;
    for (int r = 0; r < kRounds; ++r) {
        auto t = Clock::now();
        for (int i = 0; i < kReps; ++i) {
            bincv::binarize<N, uint32_t>(src, dst.plane(0), 1u);
        }
        ts.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t).count() /
                     kReps);
    }
    const double m = minOf(ts);
    std::printf(" binarize<N=%zu> %8.1f us %5.2f ns/px\n", N, m,
                m * 1000.0 / static_cast<double>(kW * kH));
}

}  // namespace

int main() {
    std::vector<uint8_t> gray(kW * kH);
    uint64_t st = UINT64_C(20260828);
    for (auto& b : gray) {
        st = st * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        b = static_cast<uint8_t>(st >> 41);
    }
    std::printf("=== the ops that had no benchmark, %zux%zu, minimum of %d rounds ===\n\n",
                kW, kH, kRounds);

    bincv::QuantMat<2, uint32_t> q2(static_cast<int>(kW), static_cast<int>(kH));
    bincv::QuantMat<4, uint32_t> q4(static_cast<int>(kW), static_cast<int>(kH));
    {
        bincv::BinMatView<uint32_t> p2[2];
        for (size_t p = 0; p < 2; ++p) p2[p] = q2.plane(p);
        bincv::packQuant<bincv::QuantRule::Scale, 2, uint8_t, uint32_t>(gray.data(), kW, kH,
                                                                        kW, p2);
        bincv::BinMatView<uint32_t> p4[4];
        for (size_t p = 0; p < 4; ++p) p4[p] = q4.plane(p);
        bincv::packQuant<bincv::QuantRule::Scale, 4, uint8_t, uint32_t>(gray.data(), kW, kH,
                                                                        kW, p4);
    }
    bincv::BinMat<uint32_t> bits(static_cast<int>(kW), static_cast<int>(kH));

    benchBinarize<2>(q2, bits);
    benchBinarize<4>(q4, bits);

    {
        std::vector<double> ts;
        for (int r = 0; r < kRounds; ++r) {
            auto t = Clock::now();
            for (int i = 0; i < kReps; ++i) {
                bincv::packBits<bincv::PackRule::GreaterEqual, uint8_t, uint32_t>(
                    gray.data(), kW, kH, kW, bits.plane(0), uint8_t{17});
            }
            ts.push_back(
                std::chrono::duration<double, std::micro>(Clock::now() - t).count() / kReps);
        }
        const double m = minOf(ts);
        std::printf(" packBits<GreaterEqual> %8.1f us %5.2f ns/px\n", m,
                    m * 1000.0 / static_cast<double>(kW * kH));
    }
    {
        std::vector<uint8_t> out(kW * kH);
        std::vector<double> ts;
        for (int r = 0; r < kRounds; ++r) {
            auto t = Clock::now();
            for (int i = 0; i < kReps; ++i) {
                bincv::unpackTo8Bit<uint32_t>(bits.constPlane(0), out.data(), kW);
            }
            ts.push_back(
                std::chrono::duration<double, std::micro>(Clock::now() - t).count() / kReps);
        }
        const double m = minOf(ts);
        std::printf(" unpackTo8Bit %8.1f us %5.2f ns/px\n", m,
                    m * 1000.0 / static_cast<double>(kW * kH));
    }
    std::printf("\n (readPgm is deliberately not here: it parses a header and memcpies,\n"
                " runs once per file, and is on no per-frame path.)\n");
    return 0;
}
