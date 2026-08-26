// THE CEILING FOR BATCHING ACROSS KEYPOINTS (E-33, after X-60).
//
// X-59 priced a vector popcount on a bulk contiguous array at 7.9x. X-60 applied
// it INSIDE one keypoint's window row and got 0.53x -- because the eight words
// were computed in registers, so packing and unpacking cost more than the eight
// POPCNTs saved. Both were the wrong granularity, and this file measures the one
// that is left.
//
// A window row is 31 pixels: ONE uint32 word. binCV's packing can hold 256 pixels
// in an AVX2 register, so at that granularity 8/9ths of the capacity is unused --
// and no amount of reshaping inside a row can recover it. But LK tracks 150-200
// KEYPOINTS doing the IDENTICAL computation on different windows, so eight
// keypoints fill the register exactly.
//
//   arm A  the shipped shape: 8 keypoints, one after another, scalar POPCNT.
//   arm B  8 keypoints at once, one vector lane each, gathered per row.
//
// The gather is the thing in question. X-43 measured gathering to be a net loss
// when it fed ONE popcount; here it feeds the whole inner computation -- 4 plane
// pairs x 2 components x 2 -- so it has 16x more work to amortise against. That
// is the entire hypothesis and it is what this arm exists to test.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "measure_util.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#define KP_AVX2 1
#else
#define KP_AVX2 0
#endif

namespace {
constexpr size_t kRows = 31;      // the shipped 31x31 window
constexpr size_t kKp = 8;         // one AVX2 register's worth of keypoints
constexpr size_t kStride = 24;    // words per row of a 752-px plane

#if KP_AVX2
/// Arm A: the shipped shape. Eight keypoints, sequentially, scalar popcount.
/// N = 2, so four plane pairs, and total - 2*opposing per pair.
long long scalarEight(const uint32_t* prev, const uint32_t* mag0, const uint32_t* mag1,
                      const uint32_t* sgn, const size_t* off) {
    long long acc = 0;
    for (size_t k = 0; k < kKp; ++k) {
        for (size_t y = 0; y < kRows; ++y) {
            const size_t o = off[k] + y * kStride;
            const uint32_t v0 = prev[o], v1 = prev[o + 1];
            const uint32_t m0 = mag0[o], m1 = mag1[o], sg = sgn[o];
            const uint32_t p[4] = {v0 & m0, v1 & m0, v0 & m1, v1 & m1};
            const long long w[4] = {1, 2, 2, 4};
            for (int j = 0; j < 4; ++j) {
                const long long tot = __builtin_popcount(p[j]);
                const long long opp = __builtin_popcount(p[j] & sg);
                acc += (tot - 2 * opp) * w[j];
            }
        }
    }
    return acc;
}

/// Arm B: eight keypoints in eight lanes. One gather per array per row, then the
/// whole inner computation once, in vector form, with the accumulator carried
/// across the window so nothing is unpacked until the end.
long long vectorEight(const uint32_t* prev, const uint32_t* mag0, const uint32_t* mag1,
                      const uint32_t* sgn, const size_t* off) {
    __m256i acc = _mm256_setzero_si256();
    alignas(32) int32_t idx[kKp];
    for (size_t k = 0; k < kKp; ++k) idx[k] = static_cast<int32_t>(off[k]);
    const __m256i base = _mm256_load_si256(reinterpret_cast<const __m256i*>(idx));
    const __m256i lut = _mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
                                         0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i lowMask = _mm256_set1_epi8(0x0f);
    auto popcnt = [&](__m256i v) {
        const __m256i lo = _mm256_and_si256(v, lowMask);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lowMask);
        const __m256i by = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                                           _mm256_shuffle_epi8(lut, hi));
        return _mm256_madd_epi16(_mm256_maddubs_epi16(by, _mm256_set1_epi8(1)),
                                 _mm256_set1_epi16(1));
    };
    for (size_t y = 0; y < kRows; ++y) {
        const __m256i o = _mm256_add_epi32(base, _mm256_set1_epi32(
                                               static_cast<int32_t>(y * kStride)));
        const __m256i v0 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(prev), o, 4);
        const __m256i v1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(prev + 1), o, 4);
        const __m256i m0 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(mag0), o, 4);
        const __m256i m1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(mag1), o, 4);
        const __m256i sg = _mm256_i32gather_epi32(reinterpret_cast<const int*>(sgn), o, 4);
        const __m256i p[4] = {_mm256_and_si256(v0, m0), _mm256_and_si256(v1, m0),
                              _mm256_and_si256(v0, m1), _mm256_and_si256(v1, m1)};
        const int wts[4] = {1, 2, 2, 4};
        for (int j = 0; j < 4; ++j) {
            const __m256i tot = popcnt(p[j]);
            const __m256i opp = popcnt(_mm256_and_si256(p[j], sg));
            const __m256i d = _mm256_sub_epi32(tot, _mm256_slli_epi32(opp, 1));
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(d, _mm256_set1_epi32(wts[j])));
        }
    }
    alignas(32) int32_t out[kKp];
    _mm256_store_si256(reinterpret_cast<__m256i*>(out), acc);
    long long s = 0;
    for (size_t k = 0; k < kKp; ++k) s += out[k];
    return s;
}
#endif
} // namespace

int main() {
    std::printf("=== CEILING: batching across KEYPOINTS, not within a window ===\n");
#if !KP_AVX2
    std::printf("  needs -mavx2.\n");
    return 0;
#else
    const size_t words = kStride * 512;
    std::vector<uint32_t> prev(words + 8), mag0(words + 8), mag1(words + 8), sgn(words + 8);
    uint64_t st = 99;
    auto rnd = [&st]{ st = st*6364136223846793005ULL + 1442695040888963407ULL;
                      return static_cast<uint32_t>(st >> 33); };
    for (size_t i = 0; i < words + 8; ++i) {
        prev[i] = rnd(); mag0[i] = rnd() & rnd(); mag1[i] = rnd() & rnd(); sgn[i] = rnd();
    }
    size_t off[kKp];
    for (size_t k = 0; k < kKp; ++k) off[k] = (k * 37 + 11) * kStride + (k % 5);

    const long long a = scalarEight(prev.data(), mag0.data(), mag1.data(), sgn.data(), off);
    const long long b = vectorEight(prev.data(), mag0.data(), mag1.data(), sgn.data(), off);
    std::printf("  EQUALITY: %lld vs %lld -- %s\n", a, b, a == b ? "identical" : "MISMATCH");
    if (a != b) return 1;

    std::vector<measure::Bench> bs = {
        {"A  8 keypoints, scalar POPCNT", [&](int){
            measure::g_sink += static_cast<size_t>(
                scalarEight(prev.data(), mag0.data(), mag1.data(), sgn.data(), off)); }},
        {"B  8 keypoints in 8 lanes, gathered", [&](int){
            measure::g_sink += static_cast<size_t>(
                vectorEight(prev.data(), mag0.data(), mag1.data(), sgn.data(), off)); }},
    };
    const auto t = measure::measureInterleaved(bs, 9, 60.0);
    std::printf("\n  %-38s %10s %9s %8s\n", "arm", "ns", "vs A", "spread");
    for (size_t i = 0; i < bs.size(); ++i)
        std::printf("  %-38s %10.1f %8.2fx %7.1f%%\n", bs[i].name.c_str(), t[i].medianNs,
                    t[0].medianNs / t[i].medianNs, t[i].spreadPct());
    std::printf("\n  The gather feeds the WHOLE inner computation here -- 16 popcounts per\n"
                "  row -- where X-43's gather fed one. That amortisation is the hypothesis.\n");
    return 0;
#endif
}
