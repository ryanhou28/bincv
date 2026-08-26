// X-71's arms for E-40: the CV_8U -> bit-plane conversion.
//
// After X-69 and X-70 shrank `track`, `fromCVMat` is 33% of the x86 frontend at one
// thread and 53% at twelve, and it does not scale. It is the largest single item.
//
// WHAT IT DOES TODAY, at N == 1 -- the hot case, because level 0 of the shipped
// 1/2/2/2 ladder is a QuantMat<1>:
//
//     for (x) if (rowIn[x]) rowOut[wordIndex(x)] |= bitMask(x);
//
// A data-dependent branch per pixel and a read-modify-write per SET pixel. On a
// sparse edge map the branch is unpredictable by construction -- that is what an
// edge map IS -- so this is close to a worst case for a scalar loop.
//
// THE OBSERVATION. `bitMask(x)` is `1 << (x % WordBits)`, so pixel x lands in bit x
// of its word, LSB first. `_mm256_movemask_epi8` takes the top bit of each of 32
// bytes and returns them as 32 bits, byte i in bit i, LSB first. THE TWO ORDERS ARE
// THE SAME, so one plane of 32 pixels is one instruction and no shuffle is needed.
//
// ARMS -- all produce IDENTICAL BITS to A, checked before timing. This is a
// repacking, not an approximation: unlike X-62 there is no trade to weigh here and a
// mismatch is a bug.
//
//   A   shipped: per-pixel branch, read-modify-write
//   A'  shipped packing with the ALLOCATION HOISTED -- see below
//   B   portable branchless: accumulate a whole word, store once, no intrinsics
//   C   x86 movemask
//   D   aarch64 NEON bitmask (AND bit weights, pairwise add)
//
// WHY A' EXISTS AND IS REPORTED WHATEVER THE BANDS SAY (X-71's rule). `fromCVMat`
// ALLOCATES A FRESH BUFFER EVERY CALL -- commit-last, for the exception-safety reason
// its comment gives -- and the frontend calls it twice per frame across 1710 frames.
// An arm that fixed the packing and left the allocation would land in Band D and look
// like a failure of the IDEA rather than of the SCOPE. A' separates them.
//
// Rule (EXPERIMENTS.md X-71, committed before this file): >=5x AND >=1.10x on the
// frontend -> ship the platform paths; >=5x with a flat frontend -> the conversion
// was not the constraint and that is the finding; 2-5x -> ship what is bit-exact and
// report arm B separately, because if the portable arm gets most of it the intrinsics
// are not worth their maintenance; <2x -> the packing was never the problem.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "measure_util.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#define CVT_AVX2 1
#else
#define CVT_AVX2 0
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#define CVT_NEON 1
#else
#define CVT_NEON 0
#endif

namespace {

using Word = uint32_t;
constexpr size_t kWordBits = 32;

/// ARM A -- the shipped loop, verbatim.
void armPerPixel(const uint8_t* in, size_t w, size_t h, size_t inPitch, Word* out,
                 size_t outWords) {
    std::memset(out, 0, outWords * h * sizeof(Word));
    for (size_t y = 0; y < h; ++y) {
        const uint8_t* rowIn = in + y * inPitch;
        Word* rowOut = out + y * outWords;
        for (size_t x = 0; x < w; ++x) {
            if (rowIn[x]) rowOut[x / kWordBits] |= static_cast<Word>(Word{1} << (x % kWordBits));
        }
    }
}

/// ARM B -- portable, branchless, one store per word. No intrinsics anywhere.
/// @note This is the arm that decides whether the intrinsics below are worth their
///       maintenance. If it gets most of the win, they are not.
void armBranchless(const uint8_t* in, size_t w, size_t h, size_t inPitch, Word* out,
                   size_t outWords) {
    for (size_t y = 0; y < h; ++y) {
        const uint8_t* rowIn = in + y * inPitch;
        Word* rowOut = out + y * outWords;
        size_t x = 0;
        for (; x + kWordBits <= w; x += kWordBits) {
            Word acc = 0;
            for (size_t i = 0; i < kWordBits; ++i) {
                // `!= 0` is a set-flag, not a branch: no data-dependent control flow.
                acc |= static_cast<Word>(static_cast<Word>(rowIn[x + i] != 0) << i);
            }
            rowOut[x / kWordBits] = acc;
        }
        if (x < w) {
            Word acc = 0;
            for (size_t i = 0; x + i < w; ++i)
                acc |= static_cast<Word>(static_cast<Word>(rowIn[x + i] != 0) << i);
            rowOut[x / kWordBits] = acc;
        }
    }
}

#if CVT_AVX2
/// ARM C -- one 32-pixel plane per compare + movemask.
void armMovemask(const uint8_t* in, size_t w, size_t h, size_t inPitch, Word* out,
                 size_t outWords) {
    const __m256i zero = _mm256_setzero_si256();
    for (size_t y = 0; y < h; ++y) {
        const uint8_t* rowIn = in + y * inPitch;
        Word* rowOut = out + y * outWords;
        size_t x = 0;
        for (; x + 32 <= w; x += 32) {
            const __m256i v =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowIn + x));
            // cmpeq gives 0xFF where the pixel is ZERO, so the mask is inverted once
            // at the end rather than the comparison being done the other way round --
            // AVX2 has no "not equal".
            const int isZero = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, zero));
            rowOut[x / kWordBits] = static_cast<Word>(~static_cast<Word>(isZero));
        }
        for (; x < w; ++x) {
            if (rowIn[x]) rowOut[x / kWordBits] |= static_cast<Word>(Word{1} << (x % kWordBits));
            else rowOut[x / kWordBits] &= static_cast<Word>(~(Word{1} << (x % kWordBits)));
        }
    }
}
#endif

#if CVT_NEON
/// ARM D -- aarch64 has no movemask. AND with per-lane bit weights, then three
/// pairwise adds fold 16 bytes into a 16-bit mask.
void armNeonMask(const uint8_t* in, size_t w, size_t h, size_t inPitch, Word* out,
                 size_t outWords) {
    const uint8x16_t weights = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    for (size_t y = 0; y < h; ++y) {
        const uint8_t* rowIn = in + y * inPitch;
        Word* rowOut = out + y * outWords;
        size_t x = 0;
        for (; x + 32 <= w; x += 32) {
            Word acc = 0;
            for (size_t half = 0; half < 2; ++half) {
                const uint8x16_t v = vld1q_u8(rowIn + x + half * 16);
                // 0xFF per lane where the pixel is non-zero.
                const uint8x16_t nz = vmvnq_u8(vceqzq_u8(v));
                const uint8x16_t m = vandq_u8(nz, weights);
                uint8x8_t f = vpadd_u8(vget_low_u8(m), vget_high_u8(m));
                f = vpadd_u8(f, f);
                f = vpadd_u8(f, f);
                const uint16_t mask = vget_lane_u16(vreinterpret_u16_u8(f), 0);
                acc |= static_cast<Word>(static_cast<Word>(mask) << (16 * half));
            }
            rowOut[x / kWordBits] = acc;
        }
        for (; x < w; ++x) {
            if (rowIn[x]) rowOut[x / kWordBits] |= static_cast<Word>(Word{1} << (x % kWordBits));
            else rowOut[x / kWordBits] &= static_cast<Word>(~(Word{1} << (x % kWordBits)));
        }
    }
}
#endif

} // namespace

int main() {
    // A real binarised frame's shape and density: 752x480 at ~10% set (D-31's
    // workload). A dense or empty image would make arm A's branch predictable and
    // flatter it -- an edge map is exactly the case where it is not.
    const size_t w = 752, h = 480;
    const size_t inPitch = w;
    const size_t outWords = (w + kWordBits - 1) / kWordBits;
    std::vector<uint8_t> img(inPitch * h);
    uint64_t st = 0x9E3779B97F4A7C15ULL;
    auto next = [&st]() {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(st >> 33);
    };
    size_t set = 0;
    for (size_t i = 0; i < img.size(); ++i) {
        img[i] = (next() % 100u < 10u) ? static_cast<uint8_t>(255) : uint8_t{0};
        if (img[i]) ++set;
    }
    std::printf("=== X-71 arms for E-40: CV_8U -> 1 bit/pixel ===\n");
    std::printf("  %zux%zu, %.1f%% set, word=uint32_t\n\n", w, h,
                100.0 * static_cast<double>(set) / static_cast<double>(img.size()));

    std::vector<Word> ref(outWords * h), got(outWords * h);
    armPerPixel(img.data(), w, h, inPitch, ref.data(), outWords);

    auto check = [&](const char* name, void (*fn)(const uint8_t*, size_t, size_t, size_t,
                                                  Word*, size_t)) {
        std::memset(got.data(), 0, got.size() * sizeof(Word));
        fn(img.data(), w, h, inPitch, got.data(), outWords);
        const bool eq = std::memcmp(ref.data(), got.data(), got.size() * sizeof(Word)) == 0;
        std::printf("  %-28s bits identical to A: %s\n", name, eq ? "YES" : "*** NO ***");
        return eq;
    };
    bool ok = check("B  portable branchless", armBranchless);
#if CVT_AVX2
    ok = check("C  x86 movemask", armMovemask) && ok;
#endif
#if CVT_NEON
    ok = check("D  aarch64 NEON bitmask", armNeonMask) && ok;
#endif
    if (!ok) {
        std::printf("\n  MISMATCH -- not timing arms that do not agree.\n");
        return 1;
    }

    std::vector<measure::Bench> bs;
    bs.push_back({"A  shipped: per-pixel branch", [&](int) {
                      armPerPixel(img.data(), w, h, inPitch, got.data(), outWords);
                      measure::g_sink += got[0];
                  }});
    bs.push_back({"B  portable branchless", [&](int) {
                      armBranchless(img.data(), w, h, inPitch, got.data(), outWords);
                      measure::g_sink += got[0];
                  }});
#if CVT_AVX2
    bs.push_back({"C  x86 movemask", [&](int) {
                      armMovemask(img.data(), w, h, inPitch, got.data(), outWords);
                      measure::g_sink += got[0];
                  }});
#endif
#if CVT_NEON
    bs.push_back({"D  aarch64 NEON bitmask", [&](int) {
                      armNeonMask(img.data(), w, h, inPitch, got.data(), outWords);
                      measure::g_sink += got[0];
                  }});
#endif
    // A' -- the ALLOCATION, priced on its own. X-71 requires this number whatever the
    // bands say, so a packing win cannot be mistaken for a scope failure.
    bs.push_back({"   (allocate + zero, no packing)", [&](int) {
                      std::vector<Word> fresh(outWords * h);
                      measure::g_sink += fresh[0];
                  }});
    const auto tt = measure::measureInterleaved(bs, 9, 50.0);
    std::printf("\n  %-32s %14s %10s %12s\n", "arm", "ns/frame", "vs A", "ns/px");
    for (size_t i = 0; i < bs.size(); ++i)
        std::printf("  %-32s %14.1f %9.3fx %11.4f\n", bs[i].name.c_str(), tt[i].medianNs,
                    tt[0].medianNs / tt[i].medianNs,
                    tt[i].medianNs / static_cast<double>(w * h));
    std::printf("\n  X-71's rule: >=5x AND >=1.10x on the frontend -> ship the platform\n"
                "  paths; >=5x with a flat frontend -> the conversion was not the\n"
                "  constraint; 2-5x -> ship what is bit-exact and report B separately;\n"
                "  <2x -> the packing was never the problem, look at the allocation.\n");
    std::printf("\n  sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
