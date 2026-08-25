// E-33's CEILING (X-58). Before any kernel is restructured for AVX2, price the two
// operations a restructured kernel would rest on -- against the scalar forms binCV
// runs today. Ceiling before arm: X-33's discipline, which X-52 forgot at a cost.
//
// ARM 1, THE ADDER. Bit-sliced arithmetic is pure AND/XOR/OR, so a ripple add over
// N planes vectorises trivially ONCE the loop order is fixed -- 8 words per AVX2
// register against 1. This measures boxSum4's shape: three ripple adds of 1-bit
// operands into a 3-plane sum, scalar against 256-bit.
//
// ARM 2, THE POPCOUNT. residualSums is 67% of the frontend and counts bits in
// 32-bit words. AVX2 has no popcount instruction; the standard answer is Mula's
// pshufb nibble-table, which counts 32 BYTES per pass. Against hardware POPCNT --
// which x86 does have and which binCV now uses (X-57) -- that is not obviously a
// win, and asserting it either way without measuring is what this file prevents.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "measure_util.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CEIL_X86 1
#else
#define CEIL_X86 0
#endif

namespace {
#if CEIL_X86
constexpr size_t kWords = 1 << 14;   // 64 KiB per plane, L1-resident; a power of two,
                                     // so the vector loops need no tail (see adderAvx2)

/// boxSum4's shape, scalar: sum = a+b+c+d over 1-bit planes into 3 planes.
void adderScalar(const uint32_t* a, const uint32_t* b, const uint32_t* c, const uint32_t* d,
                 uint32_t* s0, uint32_t* s1, uint32_t* s2, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t x = a[i], y = b[i], z = c[i], w = d[i];
        const uint32_t l0 = x ^ y, l1 = x & y;          // a+b
        const uint32_t r0 = z ^ w, r1 = z & w;          // c+d
        s0[i] = l0 ^ r0;                                 // and the tree
        const uint32_t car = l0 & r0;
        s1[i] = l1 ^ r1 ^ car;
        s2[i] = (l1 & r1) | (car & (l1 ^ r1));
    }
}

__attribute__((target("avx2")))
void adderAvx2(const uint32_t* a, const uint32_t* b, const uint32_t* c, const uint32_t* d,
               uint32_t* s0, uint32_t* s1, uint32_t* s2, size_t n) {
    const size_t step = 8;
    size_t i = 0;
    for (; i + step <= n; i += step) {
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i z = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + i));
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + i));
        const __m256i l0 = _mm256_xor_si256(x, y), l1 = _mm256_and_si256(x, y);
        const __m256i r0 = _mm256_xor_si256(z, w), r1 = _mm256_and_si256(z, w);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(s0 + i), _mm256_xor_si256(l0, r0));
        const __m256i car = _mm256_and_si256(l0, r0);
        const __m256i x1 = _mm256_xor_si256(l1, r1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(s1 + i), _mm256_xor_si256(x1, car));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(s2 + i),
                            _mm256_or_si256(_mm256_and_si256(l1, r1), _mm256_and_si256(car, x1)));
    }
    // No tail: kWords is a power of two. A real kernel needs one; a ceiling that
    // measures the steady state does not, and adding it here only obscures the
    // number being priced.
}

uint64_t popcntScalar(const uint32_t* p, size_t n) {
    uint64_t acc = 0;
    for (size_t i = 0; i < n; ++i) acc += static_cast<uint64_t>(__builtin_popcount(p[i]));
    return acc;
}

/// Mula's nibble-table popcount: pshufb over 32 bytes per pass.
__attribute__((target("avx2")))
uint64_t popcntAvx2(const uint32_t* p, size_t n) {
    const __m256i lut = _mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
                                         0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i lowMask = _mm256_set1_epi8(0x0f);
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
        const __m256i lo = _mm256_and_si256(v, lowMask);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lowMask);
        const __m256i cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                                            _mm256_shuffle_epi8(lut, hi));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(cnt, _mm256_setzero_si256()));
    }
    uint64_t out[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), acc);
    return out[0] + out[1] + out[2] + out[3];   // no tail; see adderAvx2
}
#endif
} // namespace

int main() {
    std::printf("=== E-33 CEILING (X-58): what AVX2 would buy a restructured kernel ===\n");
#if !CEIL_X86
    std::printf("  x86 only.\n");
    return 0;
#else
    std::vector<uint32_t> a(kWords), b(kWords), c(kWords), d(kWords);
    std::vector<uint32_t> s0(kWords), s1(kWords), s2(kWords), t0(kWords), t1(kWords), t2(kWords);
    uint64_t st = 12345;
    auto rnd = [&st] { st = st * 6364136223846793005ULL + 1442695040888963407ULL;
                       return static_cast<uint32_t>(st >> 33); };
    for (size_t i = 0; i < kWords; ++i) { a[i]=rnd(); b[i]=rnd(); c[i]=rnd(); d[i]=rnd(); }

    // EQUALITY FIRST, both arms, before anything is timed.
    adderScalar(a.data(),b.data(),c.data(),d.data(),s0.data(),s1.data(),s2.data(),kWords);
    adderAvx2  (a.data(),b.data(),c.data(),d.data(),t0.data(),t1.data(),t2.data(),kWords);
    size_t bad = 0;
    for (size_t i = 0; i < kWords; ++i)
        if (s0[i]!=t0[i] || s1[i]!=t1[i] || s2[i]!=t2[i]) ++bad;
    const uint64_t ps = popcntScalar(a.data(), kWords), pv = popcntAvx2(a.data(), kWords);
    std::printf("  EQUALITY: adder %zu of %zu words differ; popcount %llu vs %llu -- %s\n",
                bad, kWords, (unsigned long long)ps, (unsigned long long)pv,
                (bad == 0 && ps == pv) ? "identical" : "MISMATCH");
    if (bad || ps != pv) return 1;

    std::vector<measure::Bench> bs = {
        {"adder  scalar (boxSum4 shape)", [&](int){
            adderScalar(a.data(),b.data(),c.data(),d.data(),s0.data(),s1.data(),s2.data(),kWords);
            measure::g_sink += s0[0]; }},
        {"adder  AVX2 (8 words at once)", [&](int){
            adderAvx2(a.data(),b.data(),c.data(),d.data(),t0.data(),t1.data(),t2.data(),kWords);
            measure::g_sink += t0[0]; }},
        {"popcnt scalar (hardware POPCNT)", [&](int){
            measure::g_sink += static_cast<size_t>(popcntScalar(a.data(), kWords)); }},
        {"popcnt AVX2 (Mula pshufb)", [&](int){
            measure::g_sink += static_cast<size_t>(popcntAvx2(a.data(), kWords)); }},
    };
    const auto t = measure::measureInterleaved(bs, 9, 60.0);
    std::printf("\n  %-34s %10s %9s %8s\n", "arm", "ns", "vs scalar", "spread");
    for (size_t i = 0; i < bs.size(); ++i) {
        const double base = (i < 2) ? t[0].medianNs : t[2].medianNs;
        std::printf("  %-34s %10.1f %8.2fx %7.1f%%\n", bs[i].name.c_str(), t[i].medianNs,
                    base / t[i].medianNs, t[i].spreadPct());
    }
    std::printf("\n  The adder is what pyrDown and every bit-sliced sum rest on.\n"
                "  The popcount is what residualSums rests on, and x86 already has a\n"
                "  hardware instruction for it -- so a win there is NOT a given.\n");
    return 0;
#endif
}
