// X-44 / E-26 -- is INTERLEAVED a layout binCV should support?
//
// X-43 found the LK extraction 1.638x faster once its eight words are contiguous,
// measured against a FABRICATED buffer. This measures the real thing, and both
// sides of it: what interleaving buys the operation that re-reads, and what it
// costs the operations that stream.
//
// LAYOUT UNDER TEST. Eight planes of one N=2 LK level -- prev x2, dxMag x2,
// dyMag x2, dxSign, dySign -- stored as TWO groups of four, because four 32-bit
// words is one NEON register and aarch64's vld4q/vst4q do a 4-way interleave for
// free. Group g, row y, word i lives at buf[g][(y * words + i) * 4 + p].
//
// The conversion is per LEVEL PER FRAME; the extractions that benefit are per
// KEYPOINT PER ITERATION. The crossover -- how many re-reads a word needs before
// the conversion pays -- is reported, because that number is what the NEXT
// operation wanting this layout will need.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "measure_util.hpp"

#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#define IL_NEON 1
#else
#define IL_NEON 0
#endif

using W = uint32_t;
namespace {

constexpr size_t kPlanes = 8;   ///< the eight words one window row needs
constexpr size_t kGroup = 4;    ///< one NEON register
constexpr size_t kGroups = kPlanes / kGroup;

/// Plane-major (today) -> group-interleaved. One pass over the level.
void convert(const W* const* planes, size_t rows, size_t words, size_t stride,
             std::vector<W>& out) {
    out.resize(kGroups * rows * words * kGroup);
    for (size_t g = 0; g < kGroups; ++g) {
        W* dst = out.data() + g * rows * words * kGroup;
        for (size_t y = 0; y < rows; ++y) {
            const W* p0 = planes[g * kGroup + 0] + y * stride;
            const W* p1 = planes[g * kGroup + 1] + y * stride;
            const W* p2 = planes[g * kGroup + 2] + y * stride;
            const W* p3 = planes[g * kGroup + 3] + y * stride;
            W* row = dst + y * words * kGroup;
            size_t i = 0;
#if IL_NEON
            // vst4q is a 4-way interleaving store: four registers in, one
            // contiguous run out. The transpose this layout needs is free.
            for (; i + 4 <= words; i += 4) {
                uint32x4x4_t q;
                q.val[0] = vld1q_u32(p0 + i); q.val[1] = vld1q_u32(p1 + i);
                q.val[2] = vld1q_u32(p2 + i); q.val[3] = vld1q_u32(p3 + i);
                vst4q_u32(row + i * kGroup, q);
            }
#endif
            for (; i < words; ++i) {
                row[i * kGroup + 0] = p0[i]; row[i * kGroup + 1] = p1[i];
                row[i * kGroup + 2] = p2[i]; row[i * kGroup + 3] = p3[i];
            }
        }
    }
}

/// Arm 4's cost side: stream ONE plane, the access pattern every other kernel has.
/// Planar reads consecutive words; interleaved strides by four.
W streamPlanar(const W* plane, size_t rows, size_t words, size_t stride) {
    W acc = 0;
    for (size_t y = 0; y < rows; ++y) {
        const W* row = plane + y * stride;
        for (size_t i = 0; i < words; ++i) acc = static_cast<W>(acc ^ row[i]);
    }
    return acc;
}
W streamInterleaved(const W* buf, size_t rows, size_t words, size_t lane) {
    W acc = 0;
    for (size_t y = 0; y < rows; ++y) {
        const W* row = buf + y * words * kGroup;
        for (size_t i = 0; i < words; ++i) acc = static_cast<W>(acc ^ row[i * kGroup + lane]);
    }
    return acc;
}

/// Arms 2 and 3: one window row's eight words, from each layout.
/// Planar -- eight scalar loads from eight unrelated plane rows, then the shift.
W extractPlanar(const W* const* planes, size_t y, size_t stride, size_t w0, size_t sh,
                bool hasHi) {
    W acc = 0;
    for (size_t p = 0; p < kPlanes; ++p) {
        const W* row = planes[p] + y * stride;
        const W lo = row[w0];
        const W hi = hasHi ? row[w0 + 1] : static_cast<W>(0);
        acc = static_cast<W>(acc ^ (sh == 0 ? lo : static_cast<W>((lo >> sh) | (hi << (32 - sh)))));
    }
    return acc;
}
/// Interleaved -- two vector loads per group for `lo`, two for `hi`, one shift pair.
W extractInterleaved(const std::vector<W>& buf, size_t rows, size_t words, size_t y,
                     size_t w0, size_t sh, bool hasHi) {
    W acc = 0;
    for (size_t g = 0; g < kGroups; ++g) {
        const W* row = buf.data() + g * rows * words * kGroup + y * words * kGroup;
        const W* lo = row + w0 * kGroup;
        const W* hi = hasHi ? row + (w0 + 1) * kGroup : nullptr;
#if IL_NEON
        const uint32x4_t vlo = vld1q_u32(lo);
        uint32x4_t v;
        if (sh == 0) {
            v = vlo;
        } else {
            const uint32x4_t vhi = hi ? vld1q_u32(hi) : vdupq_n_u32(0);
            v = vorrq_u32(vshlq_u32(vlo, vdupq_n_s32(-static_cast<int32_t>(sh))),
                          vshlq_u32(vhi, vdupq_n_s32(static_cast<int32_t>(32 - sh))));
        }
        W tmp[4];
        vst1q_u32(tmp, v);
        acc = static_cast<W>(acc ^ tmp[0] ^ tmp[1] ^ tmp[2] ^ tmp[3]);
#else
        for (size_t p = 0; p < kGroup; ++p) {
            const W l = lo[p];
            const W h = hi ? hi[p] : static_cast<W>(0);
            acc = static_cast<W>(acc ^ (sh == 0 ? l : static_cast<W>((l >> sh) | (h << (32 - sh)))));
        }
#endif
    }
    return acc;
}

/// Arm 2c: the same interleaved extraction with the result kept IN VECTOR FORM.
/// Arms 2b and X-43's arm C both end each extraction with vst1q + scalar XOR,
/// which a real kernel would not do -- its next step (the popcount) is also
/// vector. Without this arm the interleaved side is charged a domain crossing
/// per row that only the benchmark's accumulator needs.
#if IL_NEON
uint32x4_t extractInterleavedV(const std::vector<W>& buf, size_t rows, size_t words,
                               size_t y, size_t w0, size_t sh, bool hasHi, uint32x4_t acc) {
    for (size_t g = 0; g < kGroups; ++g) {
        const W* row = buf.data() + g * rows * words * kGroup + y * words * kGroup;
        const uint32x4_t vlo = vld1q_u32(row + w0 * kGroup);
        uint32x4_t v;
        if (sh == 0) {
            v = vlo;
        } else {
            const uint32x4_t vhi = hasHi ? vld1q_u32(row + (w0 + 1) * kGroup) : vdupq_n_u32(0);
            v = vorrq_u32(vshlq_u32(vlo, vdupq_n_s32(-static_cast<int32_t>(sh))),
                          vshlq_u32(vhi, vdupq_n_s32(static_cast<int32_t>(32 - sh))));
        }
        acc = veorq_u32(acc, v);
    }
    return acc;
}
#endif

} // namespace

int main() {
    // One N=2 level of the shipped ladder at the frontend's geometry: level 1 of a
    // 752x480 frame. Levels 2 and 3 are a quarter and a sixteenth of it.
    const size_t cols = 376, rows = 240;
    const size_t words = (cols + 31) / 32;
    const size_t stride = words;
    std::vector<W> storage(kPlanes * rows * stride);
    uint64_t st = 0xBEEFULL;
    for (W& v : storage) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<W>(st >> 33);
    }
    const W* planes[kPlanes];
    for (size_t p = 0; p < kPlanes; ++p) planes[p] = storage.data() + p * rows * stride;

    std::vector<W> inter;
    convert(planes, rows, words, stride, inter);

    std::printf("=== X-44 / E-26: interleaved layout, %zux%zu level, %zu planes ===\n\n",
                cols, rows, kPlanes);

    // ---- equality: the two extractions must agree everywhere ----
    size_t bad = 0, checked = 0;
    for (size_t y = 0; y < rows; y += 7) {
        for (size_t w0 = 0; w0 + 1 < words; ++w0) {
            for (size_t sh = 0; sh < 32; sh += 5) {
                if (extractPlanar(planes, y, stride, w0, sh, true) !=
                    extractInterleaved(inter, rows, words, y, w0, sh, true)) ++bad;
                ++checked;
            }
        }
    }
    std::printf("  EQUALITY: %zu of %zu extractions differ\n", bad, checked);
    if (bad) return 1;

    // ---- arm 1: the conversion ----
    std::vector<W> scratch;
    // ---- arms 2/4: extraction and streaming ----
    const size_t kReads = 2000;   // stand-in for "many windows re-reading the level"
    std::vector<measure::Bench> b = {
        {"1  convert planar -> interleaved", [&](int) {
             convert(planes, rows, words, stride, scratch);
             measure::g_sink += scratch.size() ? scratch[0] : 0u; }},
        {"2a extract x2000, planar (gather)", [&](int) { W a = 0;
             for (size_t k = 0; k < kReads; ++k)
                 a = static_cast<W>(a ^ extractPlanar(planes, k % rows, stride,
                                                      k % (words - 1), k % 32, true));
             measure::g_sink += a; }},
        {"2b extract x2000, interleaved", [&](int) { W a = 0;
             for (size_t k = 0; k < kReads; ++k)
                 a = static_cast<W>(a ^ extractInterleaved(inter, rows, words, k % rows,
                                                           k % (words - 1), k % 32, true));
             measure::g_sink += a; }},
#if IL_NEON
        {"2c extract x2000, interleaved+vector", [&](int) {
             uint32x4_t a = vdupq_n_u32(0);
             for (size_t k = 0; k < kReads; ++k)
                 a = extractInterleavedV(inter, rows, words, k % rows, k % (words - 1),
                                         k % 32, true, a);
             measure::g_sink += vgetq_lane_u32(a, 0); }},
#endif
        {"4a stream one plane, planar", [&](int) {
             measure::g_sink += streamPlanar(planes[3], rows, words, stride); }},
        {"4b stream one plane, interleaved", [&](int) {
             measure::g_sink += streamInterleaved(inter.data(), rows, words, 3); }},
    };
    const auto t = measure::measureInterleaved(b, 9, 60.0);
    std::printf("\n  %-36s %12s\n", "arm", "us");
    for (size_t i = 0; i < b.size(); ++i)
        std::printf("  %-36s %12.3f\n", b[i].name.c_str(), t[i].medianNs / 1000.0);

    const double convUs = t[0].medianNs / 1000.0;
    const double planarUs = t[1].medianNs / 1000.0;
    const double interUs = t[2].medianNs / 1000.0;
#if IL_NEON
    // Arm 2c is the honest one: no domain crossing the real kernel would not pay.
    const double interVUs = t[3].medianNs / 1000.0;
    std::printf("\n  extraction  planar %.3f / interleaved+vector %.3f = %.3fx  (2c, the honest arm)\n",
                planarUs, interVUs, planarUs / interVUs);
    const double perReadV = (planarUs - interVUs) / static_cast<double>(kReads);
    if (perReadV > 0.0)
        std::printf("  CROSSOVER (2c): conversion %.3f us / %.6f us saved per re-read"
                    " = pays after %.0f re-reads\n", convUs, perReadV, convUs / perReadV);
#endif
    const double perRead = (planarUs - interUs) / static_cast<double>(kReads);
    std::printf("\n  extraction  planar %.3f us / interleaved %.3f us  = %.3fx over %zu reads\n",
                planarUs, interUs, planarUs / interUs, kReads);
    const size_t sIdx = b.size() - 2;   // the two streaming arms are last
    std::printf("  streaming   planar %.3f us / interleaved %.3f us  = %.3fx COST\n",
                t[sIdx].medianNs / 1000.0, t[sIdx + 1].medianNs / 1000.0,
                t[sIdx + 1].medianNs / t[sIdx].medianNs);
    if (perRead > 0.0) {
        std::printf("\n  CROSSOVER: conversion costs %.3f us and each re-read saves %.6f us,\n"
                    "             so it pays after %.0f re-reads of this level.\n",
                    convUs, perRead, convUs / perRead);
    } else {
        std::printf("\n  CROSSOVER: none -- interleaved extraction is not faster here.\n");
    }
    std::printf("\n  FOOTPRINT: interleaved copy of this level = %zu B\n",
                inter.size() * sizeof(W));
    return 0;
}
