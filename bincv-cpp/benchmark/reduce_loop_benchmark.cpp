/// @file reduce_loop_benchmark.cpp
/// @brief Four loop shapes for a bulk bit-count, timed against each other on the
/// host. The same four arms are timed on the Cortex-M7 by
/// embedded/stm32h753/main.cpp, so the two targets are comparable.
///
/// **Run this build twice** -- once with `-DBINCV_X86_POPCNT=ON` (the default) and
/// once with `OFF`. Those are the two architecture families the arms behave
/// differently in, and the banner says which one this binary is:
///
/// - popcount=hardware: one instruction per word, nothing to amortize. This is the
///   family **aarch64 belongs to**, and the only proxy for it available without the
///   reference device.
/// - popcount=software: a ~10-operation SWAR per word ending in a horizontal
///   collapse. This is Cortex-M's family, and RISC-V without Zbb.
///
/// No arm uses an intrinsic or a target `#if`, so a win here is a win for a family
/// rather than for a part.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/core/simd.hpp"
#include "bincv-cpp/ops/reduce.hpp"
#include "measure_util.hpp"
#include "reduce_loop_arms.hpp"

namespace {

// The reference dataset's frame, at one bit per pixel: 752x480 packs to 24 words a
// row. Sized to the real thing so the working set is the real one -- 46 KB does not
// fit a typical L1, which is part of what the loop is up against.
constexpr size_t kWidth = 752;
constexpr size_t kHeight = 480;
constexpr size_t kStrideWords = (kWidth + 31) / 32;
constexpr size_t kWords = kStrideWords * kHeight;

// Four inputs, rotated across calls. Densities differ because a reader will ask
// whether any arm is data-dependent; none of these four is, and rotating shows it
// rather than asserting it.
constexpr int kInputs = 4;
constexpr int kRepeats = 15;
constexpr double kTargetMs = 40.0;

struct Fixture {
    std::vector<uint32_t> words[kInputs];
};

void build(Fixture& f) {
    for (int k = 0; k < kInputs; ++k) {
        f.words[k].resize(kWords);
        uint64_t state = UINT64_C(0x9E3779B9) + static_cast<uint64_t>(k);
        // k = 0 dense, k = 3 sparse: AND-ing k random words together thins the field.
        for (size_t i = 0; i < kWords; ++i) {
            uint32_t w = static_cast<uint32_t>(measure::nextRandom(state));
            for (int extra = 0; extra < k; ++extra) {
                w &= static_cast<uint32_t>(measure::nextRandom(state));
            }
            f.words[k][i] = w;
        }
        // Padding bits past `width` are zero in every real BinMat, and a reduction
        // counts them if they are not (ARCHITECTURE 1). Keep the fixture honest.
        constexpr uint32_t kTail =
            (kWidth % 32 == 0) ? 0xFFFFFFFFu : ((1u << (kWidth % 32)) - 1u);
        for (size_t y = 0; y < kHeight; ++y) {
            f.words[k][y * kStrideWords + (kStrideWords - 1)] &= kTail;
        }
    }
}

/// Every arm on every input, plus the library's own entry point. A faster arm that
/// computes something else is not a result, and this is what says so.
bool agree(const Fixture& f) {
    size_t armCount = 0;
    const loopbench::Arm* arms = loopbench::arms(armCount);
    bool ok = true;

    for (int k = 0; k < kInputs; ++k) {
        const uint64_t ref = arms[0].run(f.words[k].data(), kWords);
        for (size_t a = 1; a < armCount; ++a) {
            const uint64_t got = arms[a].run(f.words[k].data(), kWords);
            if (got != ref) {
                std::printf("  DISAGREE input %d: %s = %llu, expected %llu\n", k, arms[a].name,
                            static_cast<unsigned long long>(got),
                            static_cast<unsigned long long>(ref));
                ok = false;
            }
        }
        const bincv::BinMatConstView<uint32_t> view{f.words[k].data(), kWidth, kHeight,
                                                    kStrideWords};
        const uint64_t lib = static_cast<uint64_t>(bincv::countNonZero(view));
        if (lib != ref) {
            std::printf("  DISAGREE input %d: bincv::countNonZero = %llu, arms say %llu\n", k,
                        static_cast<unsigned long long>(lib),
                        static_cast<unsigned long long>(ref));
            ok = false;
        }
    }
    return ok;
}

}  // namespace

int main() {
    const bincv::SimdStatus s = bincv::simdStatus();

    std::printf("binCV -- four loop shapes for a bulk bit-count\n");
    std::printf("%s\n", bincv::simdStatusString());
    std::printf("FAMILY: popcount=%s -- %s\n", s.hardwarePopcount ? "HARDWARE" : "SOFTWARE",
                s.hardwarePopcount
                    ? "one instruction per word; L2/L3 have no collapse to amortize and "
                      "should LOSE. This is aarch64's family."
                    : "SWAR per word; L2/L3 amortize its horizontal tail. This is "
                      "Cortex-M's family, and RISC-V without Zbb.");
    std::printf("frame %zux%zu @ 1bpp = %zu words (%zu bytes); %d inputs, %d batches, "
                "%.0f ms budget\n\n",
                kWidth, kHeight, kWords, kWords * sizeof(uint32_t), kInputs, kRepeats,
                kTargetMs);

    Fixture f;
    build(f);

    std::printf("-- correctness --\n");
    if (!agree(f)) {
        std::printf("\nARMS DISAGREE -- nothing below would be a measurement\n");
        return 1;
    }
    std::printf("  all arms and bincv::countNonZero agree on all %d inputs\n", kInputs);

    size_t armCount = 0;
    const loopbench::Arm* arms = loopbench::arms(armCount);

    std::vector<measure::Bench> benches;
    for (size_t a = 0; a < armCount; ++a) {
        const loopbench::Arm& arm = arms[a];
        const Fixture* pf = &f;
        benches.push_back({arm.name, [pf, &arm](int i) {
                               const std::vector<uint32_t>& in = pf->words[i % kInputs];
                               measure::g_sink += static_cast<size_t>(
                                   arm.run(in.data(), kWords));
                           }});
    }

    const std::vector<measure::Timing> t =
        measure::measureInterleaved(benches, kRepeats, kTargetMs);

    std::printf("\n %-32s %26s %9s %10s\n", "arm", "ns/word min/med/max", "spread", "vs L0");
    const double base = t[0].medianNs;
    for (size_t a = 0; a < armCount; ++a) {
        const double w = static_cast<double>(kWords);
        std::printf(" %-32s %7.4f/%7.4f/%7.4f %8.1f%% %9.2fx\n", arms[a].name,
                    t[a].minNs / w, t[a].medianNs / w, t[a].maxNs / w, t[a].spreadPct(),
                    t[a].medianNs / base);
    }

    // The spread is printed per row and repeated here because on a loaded desktop it
    // is routinely larger than the differences being compared, and a ratio quoted
    // without it invites a conclusion the data does not carry.
    double worst = 0.0;
    for (size_t a = 0; a < armCount; ++a) worst = std::max(worst, t[a].spreadPct());
    std::printf("\n worst spread %.1f%% -- treat any difference smaller than this as noise\n",
                worst);

    std::printf("\nsink=%llu\n", static_cast<unsigned long long>(measure::g_sink));
    return 0;
}
