/// @file dense_m7.hpp
/// @brief `denseDisparityBinary` executed on the M7, at both word types.
///
/// The README's dense section is the rule for this file and was written first.
/// 320x240, D=32, 9x9, a known constant shift: the supported region must answer
/// with exactly that constant at BOTH word types, and the two maps must agree
/// byte for byte, before any cycle count is reported.

#pragma once

#include <cstdint>
#include <cstring>

#include "board.h"

#include "bincv/binMat.hpp"
#include "bincv/ops/denseDisparity.hpp"

namespace densebench {

constexpr size_t kDW = 320, kDH = 240;
constexpr int kTrueDisp = 13;
constexpr size_t kW32 = kDW / 32;   // words per row; 320 has no padding at either width
constexpr size_t kW64 = kDW / 64;
constexpr size_t kDenseRepeats = 3;

// .frames: the map file states these apart from the rest of the image, as with
// the reduction frame.
__attribute__((section(".frames"), aligned(32))) static uint32_t gL32[kW32 * kDH];
__attribute__((section(".frames"), aligned(32))) static uint32_t gR32[kW32 * kDH];
__attribute__((section(".frames"), aligned(32))) static uint64_t gL64[kW64 * kDH];
__attribute__((section(".frames"), aligned(32))) static uint64_t gR64[kW64 * kDH];
__attribute__((section(".frames"), aligned(32))) static uint8_t gDispA[kDW * kDH];
__attribute__((section(".frames"), aligned(32))) static uint8_t gDispB[kDW * kDH];
__attribute__((section(".frames"), aligned(32))) static uint32_t gScratch32[4096];
__attribute__((section(".frames"), aligned(32))) static uint64_t gScratch64[4096];
static uint16_t gRows[3 * kDW];

/// The pair: random left, right = left with every lane reading lane `x + d` --
/// a rigid constant-disparity scene with zero-extension past the edge, the same
/// construction the host tests hold to exactness.
inline void fillDensePair() {
    uint32_t s = 0x2468ACE1u;
    for (size_t i = 0; i < kW32 * kDH; ++i) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        gL32[i] = s;
    }
    constexpr unsigned r = static_cast<unsigned>(kTrueDisp);
    for (size_t y = 0; y < kDH; ++y) {
        const uint32_t* l = gL32 + y * kW32;
        uint32_t* rr = gR32 + y * kW32;
        for (size_t i = 0; i < kW32; ++i) {
            const uint32_t hi = i + 1 < kW32 ? l[i + 1] : 0u;
            rr[i] = (l[i] >> r) | (hi << (32u - r));
        }
    }
    // Same bits at the other width: rows are identical byte sequences on this
    // little-endian core, so the copy IS the reinterpretation.
    std::memcpy(gL64, gL32, sizeof(gL32));
    std::memcpy(gR64, gR32, sizeof(gR32));
}

template <typename W>
inline bool runDenseOnce(const W* l, const W* r, size_t strideWords, uint8_t* out) {
    const bincv::BinMatConstView<W> lv{l, kDW, kDH, strideWords};
    const bincv::BinMatConstView<W> rv{r, kDW, kDH, strideWords};
    bincv::DenseDisparityParams p;
    p.maxDisparity = 32;
    const size_t need = bincv::denseDisparityBinaryScratchWords<W>(kDW, p);
    if (sizeof(W) == 8) {
        if (need > 4096) return false;
        bincv::denseDisparityBinary<W>(lv, rv, p, reinterpret_cast<W*>(gScratch64),
                                       4096, gRows, 3 * kDW, out, kDW);
    } else {
        if (need > 4096) return false;
        bincv::denseDisparityBinary<W>(lv, rv, p, reinterpret_cast<W*>(gScratch32),
                                       4096, gRows, 3 * kDW, out, kDW);
    }
    return true;
}

/// Supported-region exactness: every pixel whose window and search range are
/// fully inside both frames must answer kTrueDisp.
inline bool checkDenseMap(const uint8_t* d) {
    const size_t hw = 4, hh = 4;
    size_t bad = 0;
    for (size_t y = hh; y + hh < kDH; ++y)
        for (size_t x = 32 + hw; x + kTrueDisp + hw < kDW; ++x)
            if (d[y * kDW + x] != kTrueDisp) ++bad;
    if (bad != 0) {
        boardPuts("  MISMATCH: ");
        boardPutU32(static_cast<uint32_t>(bad));
        boardPuts(" supported pixels off the constant shift\n");
    }
    return bad == 0;
}

inline uint32_t medianOf3(uint32_t a, uint32_t b, uint32_t c) {
    if (a > b) { const uint32_t t = a; a = b; b = t; }
    if (b > c) { b = c; }
    return a > b ? a : b;
}

inline void runDenseBenchmark() {
    boardPuts("-- denseDisparityBinary, 320x240, D=32, 9x9 --\n");
    fillDensePair();

    // Correctness gates the timing.
    if (!runDenseOnce<uint32_t>(gL32, gR32, kW32, gDispA) ||
        !runDenseOnce<uint64_t>(gL64, gR64, kW64, gDispB)) {
        boardPuts("  scratch caps too small -- no timing\n");
        return;
    }
    bool ok = checkDenseMap(gDispA) && checkDenseMap(gDispB);
    if (std::memcmp(gDispA, gDispB, sizeof(gDispA)) != 0) {
        boardPuts("  MISMATCH: u32 and u64 maps differ\n");
        ok = false;
    }
    if (!ok) {
        boardPuts("  FAILED -- no timing below\n");
        return;
    }
    boardPuts("  both word types exact on the supported region, maps identical\n");

    uint32_t s32[kDenseRepeats], s64[kDenseRepeats];
    for (size_t rep = 0; rep < kDenseRepeats; ++rep) {
        uint32_t t0 = boardCycles();
        runDenseOnce<uint32_t>(gL32, gR32, kW32, gDispA);
        s32[rep] = boardCycles() - t0;
        t0 = boardCycles();
        runDenseOnce<uint64_t>(gL64, gR64, kW64, gDispB);
        s64[rep] = boardCycles() - t0;
    }
    const uint32_t m32 = medianOf3(s32[0], s32[1], s32[2]);
    const uint32_t m64 = medianOf3(s64[0], s64[1], s64[2]);

    bincv::DenseDisparityParams p;
    p.maxDisparity = 32;
    const auto line = [&](const char* name, uint32_t cyc, uint32_t scratchBytes) {
        boardPuts("  ");
        boardPuts(name);
        boardPuts(" ");
        boardPutU32(cyc);
        boardPuts(" cyc = ");
        boardPutU32(cyc / (BINCV_M7_CLOCK_HZ / 1000u));
        boardPuts(" ms   scratch ");
        boardPutU32(scratchBytes);
        boardPuts(" B\n");
    };
    line("u32", m32,
         static_cast<uint32_t>(
             bincv::denseDisparityBinaryScratchWords<uint32_t>(kDW, p) * 4));
    line("u64", m64,
         static_cast<uint32_t>(
             bincv::denseDisparityBinaryScratchWords<uint64_t>(kDW, p) * 8));

    const uint32_t pct =
        m32 ? static_cast<uint32_t>(static_cast<uint64_t>(m64) * 100u / m32) : 0u;
    boardPuts("  u64 is ");
    boardPutU32(pct / 100u);
    boardPutc('.');
    if ((pct % 100u) < 10u) boardPutc('0');
    boardPutU32(pct % 100u);
    boardPuts("x the u32 time (the header's 64-bit guidance, tested at 32-bit)\n");
}

}  // namespace densebench
