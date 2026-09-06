/// @file main.cpp
/// @brief binCV on a Cortex-M7: a smoke test that it runs at all, and the population
/// count comparison the README's decision rule governs.
///
/// The rule was written before this ran. Read README.md first; the numbers this
/// prints mean nothing without it.

#include <cstddef>
#include <cstdint>

#include "board.h"
#include "popcount_arms.hpp"
#include "reduce_loop_arms.hpp"

#include "bincv/core/simd.hpp"
#include "bincv/ops/opticalFlow.hpp"
#include "bincv/ops/reduce.hpp"

namespace {

// A 752x480 frame, the reference dataset's size, at one bit per pixel. 46 080 bytes
// -- the whole point of the library, and the reason this fits a part whose RAM would
// not hold two 8-bit frames.
constexpr size_t kWidth = 752;
constexpr size_t kHeight = 480;
constexpr size_t kStrideWords = (kWidth + 31) / 32;  // 24
constexpr size_t kTotalWords = kStrideWords * kHeight;

// .frames rather than .bss so the map file states the frame footprint separately
// from everything else the image holds.
__attribute__((section(".frames"), aligned(32))) uint32_t g_frame[kTotalWords];

constexpr size_t kRepeats = 7;

/// Deterministic fill. A constant pattern would let the counter's cost depend on the
/// data in a way that differs between arms -- none of these four are data-dependent,
/// but that is a property to demonstrate rather than assume.
void fillFrame() {
    uint32_t s = 0x12345678u;
    for (size_t i = 0; i < kTotalWords; ++i) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        g_frame[i] = s;
    }
    // Padding bits past `width` must be zero or every word-wise reduction
    // over-counts (ARCHITECTURE 1). 752 pixels is 23.5 words, so 16 bits of each
    // row's last word are padding.
    constexpr uint32_t kTailMask = (kWidth % 32 == 0) ? 0xFFFFFFFFu : ((1u << (kWidth % 32)) - 1u);
    for (size_t y = 0; y < kHeight; ++y) g_frame[y * kStrideWords + (kStrideWords - 1)] &= kTailMask;
}

uint32_t medianOf(uint32_t* v, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        const uint32_t k = v[i];
        size_t j = i;
        while (j > 0 && v[j - 1] > k) {
            v[j] = v[j - 1];
            --j;
        }
        v[j] = k;
    }
    return v[n / 2];
}

struct Arm {
    const char* name;
    uint64_t (*run)(const uint32_t*, size_t);
};

const Arm kArms[] = {
    {"A shipped  (__builtin_popcountll)", &armbench::sumWords<armbench::popcountA>},
    {"B portable (64-bit SWAR)         ", &armbench::sumWords<armbench::popcountB>},
    {"C SWAR32                         ", &armbench::sumWords<armbench::popcountC>},
    {"D SWAR32 + USAD8 (DSP)           ", &armbench::sumWords<armbench::popcountD>},
};
constexpr size_t kArmCount = sizeof(kArms) / sizeof(kArms[0]);

void printBanner() {
    boardPuts("\n=== binCV on STM32H753ZI (Cortex-M7) ===\n");
    boardPuts("clock      : ");
    boardPutU32(boardClockHz());
    boardPuts(" Hz (HSI, no PLL)\n");
    boardPuts("caches     : I+D enabled\n");
    boardPuts("simd       : ");
    boardPuts(bincv::simdStatusString());
    boardPuts("\nframe      : 752x480 @ 1bpp = ");
    boardPutU32(static_cast<uint32_t>(sizeof(g_frame)));
    boardPuts(" bytes\n");
    boardPuts("staging    : ");
    boardPutU32(static_cast<uint32_t>(bincv::stagingStackBytes<2, uint32_t>()));
    boardPuts(" bytes at N=2 (budget ");
    boardPutU32(static_cast<uint32_t>(BINCV_STAGING_BUDGET_BYTES));
    boardPuts(")\n\n");
}

/// All four arms must agree, on every word, before any timing is reported. A faster
/// arm that computes something else is not a result.
bool checkArmsAgree() {
    boardPuts("-- correctness ------------------------------------\n");
    const uint64_t ref = kArms[0].run(g_frame, kTotalWords);
    bool ok = true;
    for (size_t a = 0; a < kArmCount; ++a) {
        const uint64_t got = kArms[a].run(g_frame, kTotalWords);
        boardPuts("  ");
        boardPuts(kArms[a].name);
        boardPuts(" = ");
        boardPutU64(got);
        if (got != ref) {
            boardPuts("   MISMATCH");
            ok = false;
        }
        boardPutc('\n');
    }

    // Cross-check against the library's own entry point, which is what a caller
    // actually invokes. Same answer, or the harness is measuring the wrong thing.
    const bincv::BinMatConstView<uint32_t> view{g_frame, kWidth, kHeight, kStrideWords};
    const uint64_t lib = static_cast<uint64_t>(bincv::countNonZero(view));
    boardPuts("  bincv::countNonZero               = ");
    boardPutU64(lib);
    if (lib != ref) {
        boardPuts("   MISMATCH");
        ok = false;
    }
    boardPuts(ok ? "\n  all agree\n\n" : "\n  FAILED\n\n");
    return ok;
}

void runBenchmark() {
    boardPuts("-- cycles over ");
    boardPutU32(static_cast<uint32_t>(kTotalWords));
    boardPuts(" words, median of ");
    boardPutU32(kRepeats);
    boardPuts(" --\n");

    uint32_t samples[kArmCount][kRepeats];
    uint64_t sink = 0;

    // Interleaved: every repeat runs all four arms before the next repeat starts, so
    // any drift lands on all of them rather than on whichever ran last.
    for (size_t r = 0; r < kRepeats; ++r) {
        for (size_t a = 0; a < kArmCount; ++a) {
            const uint32_t t0 = boardCycles();
            sink += kArms[a].run(g_frame, kTotalWords);
            const uint32_t t1 = boardCycles();
            samples[a][r] = t1 - t0;
        }
    }

    uint32_t baseline = 0;
    for (size_t a = 0; a < kArmCount; ++a) {
        const uint32_t med = medianOf(samples[a], kRepeats);
        if (a == 0) baseline = med;
        boardPuts("  ");
        boardPuts(kArms[a].name);
        boardPuts(" ");
        boardPutU32(med);
        boardPuts(" cyc  ");
        // Ratio against arm A in hundredths, printed as an integer pair: no floats,
        // so no soft-float formatting in the image.
        const uint32_t pct = baseline ? static_cast<uint32_t>((uint64_t)med * 100u / baseline) : 0u;
        boardPutU32(pct / 100u);
        boardPutc('.');
        if ((pct % 100u) < 10u) boardPutc('0');
        boardPutU32(pct % 100u);
        boardPuts("x A\n");
    }

    boardPuts("\n  (checksum ");
    boardPutU64(sink);
    boardPuts(")\n");
}

}  // namespace

/// The four LOOP shapes, timed the same way as the four per-word counters above.
/// These are the arms that matter: they are plain C++ with no intrinsic and no target
/// `#if`, so what they show here applies to every target with no population count
/// instruction rather than to this part.
void runLoopBenchmark() {
    size_t armCount = 0;
    const loopbench::Arm* arms = loopbench::arms(armCount);

    boardPuts("-- loop shapes, same ");
    boardPutU32(static_cast<uint32_t>(kTotalWords));
    boardPuts(" words --\n");

    // Correctness first, as above: all four loops must return the same count.
    const uint64_t ref = arms[0].run(g_frame, kTotalWords);
    bool ok = true;
    for (size_t a = 0; a < armCount; ++a) {
        if (arms[a].run(g_frame, kTotalWords) != ref) {
            boardPuts("  MISMATCH: ");
            boardPuts(arms[a].name);
            boardPutc('\n');
            ok = false;
        }
    }
    if (!ok) {
        boardPuts("  loop arms disagree -- no timing below\n");
        return;
    }

    uint32_t samples[4][kRepeats];
    uint64_t sink = 0;
    for (size_t r = 0; r < kRepeats; ++r) {
        for (size_t a = 0; a < armCount && a < 4; ++a) {
            const uint32_t t0 = boardCycles();
            sink += arms[a].run(g_frame, kTotalWords);
            const uint32_t t1 = boardCycles();
            samples[a][r] = t1 - t0;
        }
    }

    uint32_t base = 0;
    for (size_t a = 0; a < armCount && a < 4; ++a) {
        const uint32_t med = medianOf(samples[a], kRepeats);
        if (a == 0) base = med;
        boardPuts("  ");
        boardPuts(arms[a].name);
        boardPuts(" ");
        boardPutU32(med);
        boardPuts(" cyc  ");
        const uint32_t pct = base ? static_cast<uint32_t>((uint64_t)med * 100u / base) : 0u;
        boardPutU32(pct / 100u);
        boardPutc('.');
        if ((pct % 100u) < 10u) boardPutc('0');
        boardPutU32(pct % 100u);
        boardPuts("x L0\n");
    }
    boardPuts("  (checksum ");
    boardPutU64(sink);
    boardPuts(")\n");
}

/// Busy-wait on the cycle counter. No SysTick, because the only thing this needs
/// timing for is a pause between reports and the counter is already running.
void delayCycles(uint32_t n) {
    const uint32_t t0 = boardCycles();
    while ((boardCycles() - t0) < n) {
    }
}

int main() {
    boardEnableCaches();
    boardSerialInit();

    if (!boardCycleCounterInit()) {
        boardPuts("\nFATAL: DWT cycle counter does not advance; no timing is possible.\n");
        for (;;) {
        }
    }

    fillFrame();

    // The whole report repeats rather than running once. Flashing over the ST-LINK's
    // mass-storage interface resets the part as the copy completes, so a run-once
    // firmware prints its only report into a serial port nobody has opened yet --
    // and this board exposes no reset this harness can drive to get a second one.
    // Repeating also means the reported medians are not a single power-on sample.
    for (;;) {
        printBanner();
        if (checkArmsAgree()) runBenchmark();
        boardPutc('\n');
        runLoopBenchmark();
        boardPuts("\n=== end of report ===\n\n");
        delayCycles(3u * BINCV_M7_CLOCK_HZ);
    }
}
