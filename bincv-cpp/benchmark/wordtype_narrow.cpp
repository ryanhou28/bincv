// ===========================================================================
// X-97 / E-29 -- NATIVE 64-BIT KERNELS, OR NARROWING?
//
// Several kernels are gated on `sizeof(WordType) == 4`: the LK residual, the covariance
// lane kernel, `edgeThreshold` and `packQuant`. A caller holding 64-bit words takes the
// scalar fallback in all of them.
//
// Two ways out. Write native 64-bit kernels -- a second implementation of each -- or
// NARROW the view at the call, since on little-endian a 64-bit plane already IS a 32-bit
// plane with twice the stride.
//
// THE BASELINE THAT MATTERS IS NARROWING, NOT THE SCALAR FALLBACK. A native 64-bit
// kernel can at best match the 32-bit one running on the same bytes, so if narrowing
// recovers native 32-bit speed there is nothing left for it to win. Measuring a
// hypothetical native kernel against the fallback would be measuring the wrong thing --
// which is the whole reason X-97's rule was written down first.
//
// Usage: wordtype_narrow
// ===========================================================================

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/core/simd.hpp"
#include "bincv-cpp/ops/edge.hpp"
#include "bincv-cpp/quantMat.hpp"

using Clock = std::chrono::steady_clock;

namespace {
constexpr size_t kW = 640, kH = 480;
constexpr int kRounds = 15;

double minOf(std::vector<double> v) { return *std::min_element(v.begin(), v.end()); }

std::vector<uint8_t> makeGray() {
    std::vector<uint8_t> g(kW * kH);
    uint64_t s = 0x2545F4914F6CDD1Dull;
    for (size_t i = 0; i < g.size(); ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        g[i] = static_cast<uint8_t>(s >> 56);
    }
    return g;
}
}  // namespace

int main() {
    std::printf("=== X-97 / E-29: native 64-bit kernels, or narrowing? ===\n");
    std::printf("%zux%zu, %d interleaved rounds, minimum reported\n", kW, kH, kRounds);
    std::printf("%s\n\n", bincv::simdStatusString());

    const std::vector<uint8_t> gray = makeGray();

    // edgeThreshold: 8-bit in, 1-bit out, and its vector arm is gated on 32-bit words.
    bincv::BinMat<uint32_t> dst32(kW, kH);
    bincv::BinMat<uint64_t> dst64(kW, kH);

    std::vector<double> t32, t64, tNarrow;
    for (int r = 0; r < kRounds; ++r) {
        auto t0 = Clock::now();
        bincv::edgeThreshold<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                             bincv::EdgeSpatial::Wide>(gray.data(), kW, kH, kW, dst32.view(), static_cast<uint8_t>(17));
        t32.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count());

        t0 = Clock::now();
        bincv::edgeThreshold<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                             bincv::EdgeSpatial::Wide>(gray.data(), kW, kH, kW, dst64.view(), static_cast<uint8_t>(17));
        t64.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count());

        // ARM (C): the SAME 64-bit buffer, written through a narrowed view, so the
        // 32-bit vector arm applies to storage the caller still owns as uint64_t.
        t0 = Clock::now();
        bincv::edgeThreshold<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                             bincv::EdgeSpatial::Wide>(
            gray.data(), kW, kH, kW, bincv::narrowPlaneMutable<uint64_t>(dst64.view()), static_cast<uint8_t>(17));
        tNarrow.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count());
    }

    const double A = minOf(t32), B = minOf(t64), C = minOf(tNarrow);
    std::printf("edgeThreshold, 8-bit in, 1-bit out\n");
    std::printf("  (A) native uint32 buffer            %9.0f ns   1.00x\n", A);
    std::printf("  (B) uint64 buffer, scalar fallback  %9.0f ns   %5.2fx\n", B, A / B);
    std::printf("  (C) uint64 buffer, narrowed view    %9.0f ns   %5.2fx\n", C, A / C);

    // CORRECTNESS BEFORE SPEED: (C) must produce the same bits as (A), or it is not an
    // arm, it is a different operation that happens to be quicker.
    bincv::edgeThreshold<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                         bincv::EdgeSpatial::Wide>(gray.data(), kW, kH, kW, dst32.view(), static_cast<uint8_t>(17));
    bincv::edgeThreshold<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                         bincv::EdgeSpatial::Wide>(
        gray.data(), kW, kH, kW, bincv::narrowPlaneMutable<uint64_t>(dst64.view()), static_cast<uint8_t>(17));
    size_t differ = 0;
    for (size_t y = 0; y < kH; ++y) {
        for (size_t x = 0; x < kW; ++x) {
            if (dst32.at(static_cast<int>(y), static_cast<int>(x)) !=
                dst64.at(static_cast<int>(y), static_cast<int>(x))) ++differ;
        }
    }
    std::printf("\n  (A) vs (C), pixel by pixel: %zu of %zu differ\n", differ, kW * kH);

    // MEMORY, AND AT THE SIZE WHERE IT SHOWS. 640 is divisible by 64, so a full frame
    // costs both word types exactly the same and reports nothing -- the first version of
    // this line printed "38400 B against 38400 B" under the words "the gap widens".
    // The penalty is a ROW-STRIDE ROUNDING, so it appears at the upper pyramid levels,
    // which is where D-14 measured it and where an embedded target is tightest.
    std::printf("\n  memory (row-stride rounding, the reason D-14 chose 32):\n");
    const struct { size_t w, h; const char* what; } kSizes[] = {
        {640, 480, "level 0"}, {320, 240, "level 1"}, {160, 120, "level 2"},
        {80, 60, "level 3"}, {94, 60, "D-14's case"},
    };
    for (const auto& s : kSizes) {
        const bincv::BinMat<uint32_t> a(static_cast<int>(s.w), static_cast<int>(s.h));
        const bincv::BinMat<uint64_t> b(static_cast<int>(s.w), static_cast<int>(s.h));
        const size_t ab = a.sizeInWords() * sizeof(uint32_t);
        const size_t bb = b.sizeInWords() * sizeof(uint64_t);
        std::printf("    %-12s %4zux%-4zu  uint32 %7zu B   uint64 %7zu B   %+.1f%%\n", s.what,
                    s.w, s.h, ab, bb,
                    100.0 * (static_cast<double>(bb) - static_cast<double>(ab)) /
                        static_cast<double>(ab));
    }

    std::printf("\nIf (C) reaches (A), a native 64-bit kernel can at best MATCH it --\n"
                "same kernel, same bytes -- so there is nothing left for it to win, and\n"
                "the answer is guidance rather than a second implementation of every\n"
                "gated kernel.\n");
    return 0;
}
