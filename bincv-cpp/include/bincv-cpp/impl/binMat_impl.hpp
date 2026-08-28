#pragma once

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <utility>

// BINCV_THROW / BINCV_ASSERT. Named here rather than left to binMat.hpp, which
// is the only file that includes this one: every validation check below is
// written in terms of these two macros, so the dependency is real. It also
// carries <stdexcept> in exactly the configuration whose expansion needs it, and
// BINCV_ABI_NAMESPACE.
// X-71: the AVX2 row packer is selected at RUN TIME, so the library's baseline ISA is
// unchanged and no -mavx2 build is required. Guarded on the compiler supporting both
// the target attribute and the cpu probe.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define BINCV_X86_RUNTIME_AVX2 1
#include <immintrin.h>
#endif
#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "../core/error.hpp"

// <ostream> is here for operator<< alone, which cannot be expressed without a
// stream type. It declares no global objects, so it costs no static initializer
// -- unlike <iostream>, which is why the two print helpers below use std::fprintf
// instead of std::cout. See the include block in binMat.hpp.

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

namespace impl {

// These helpers are templated directly on the storage word type, so they never
// need to reach back into BinMat to recover it.

/// @brief Number of pixels packed into one word of the given type.
template <typename WordType>
constexpr size_t bitsPerWord() {
    return sizeof(WordType) * 8;
}

/// @brief Index of the word within a row that holds the pixel at column x.
template <typename WordType>
inline size_t wordIndex(size_t x) {
    return x / bitsPerWord<WordType>();
}

/// @brief Mask with a single 1 at the bit corresponding to column x.
/// @note Indexing runs LSB->MSB, so column 0 is the least significant bit.
/// @note The outer cast is not redundant at narrow word widths. Integer
///       promotion runs the shift in `int` for uint8_t and uint16_t, so the
///       return narrows implicitly -- harmless for a single bit, which always
///       fits, and exactly the shape that stops being harmless the moment a mask
///       is built from more than one. Written explicitly so the truncation is a
///       decision rather than a side effect (clang's -Wimplicit-int-conversion
///       reports the implicit form; GCC's -Wconversion does not).
template <typename WordType>
inline WordType bitMask(size_t x) {
    return static_cast<WordType>(static_cast<WordType>(1) << (x % bitsPerWord<WordType>()));
}

/// @brief Mask covering bits [0, n) of a word; n == bitsPerWord yields all ones.
template <typename WordType>
inline WordType lowBitsMask(size_t n) {
    if (n == 0) return static_cast<WordType>(0);
    if (n >= bitsPerWord<WordType>()) return static_cast<WordType>(~static_cast<WordType>(0));
    return static_cast<WordType>((static_cast<WordType>(1) << n) - 1);
}

/// @brief Words a row of `widthPixels` inherently needs: ceil(width / WordBits).
/// @note This is the stride at the default alignment (D-4), and the floor below
///       which a caller-supplied stride cannot go without rows overlapping.
template <typename WordType>
inline size_t minRowWords(size_t widthPixels) {
    constexpr size_t bitsPerWordV = bitsPerWord<WordType>();
    return (widthPixels + bitsPerWordV - 1) / bitsPerWordV;
}

/// @brief Words per row so that the row stride meets the requested byte alignment.
/// @note With the default alignment of one word this returns minRowWords()
///       unchanged -- the row already occupies a whole number of words.
template <typename WordType>
inline size_t calcAlignedWidth(size_t widthPixels, size_t alignmentBytes) {
    constexpr size_t bytesPerWord = sizeof(WordType);

    // Words needed to hold the row's pixels (one bit each), rounded up
    size_t rowBytes = minRowWords<WordType>(widthPixels) * bytesPerWord;

    // Round the row stride up to the requested byte alignment
    size_t alignedBytes = ((rowBytes + alignmentBytes - 1) / alignmentBytes) * alignmentBytes;

    return alignedBytes / bytesPerWord;
}

/// @brief Validates a rowAlignment argument, throwing on the two invalid shapes.
/// @note Shared by every entry point that accepts one, so the diagnostics cannot
///       drift apart between constructors.
template <typename WordType>
inline void checkRowAlignment(size_t alignmentBytes) {
    if (alignmentBytes == 0 || (alignmentBytes & (alignmentBytes - 1)) != 0) {
        BINCV_THROW(std::invalid_argument,
                    "BinMat rowAlignment must be a positive power of two");
    }
    if (alignmentBytes % sizeof(WordType) != 0) {
        BINCV_THROW(std::invalid_argument,
                    "BinMat rowAlignment must be a multiple of the word size");
    }
}

} // namespace impl

// Constructors
template <typename WordType_>
QuantMat<1, WordType_>::QuantMat()
    : width(0), height(0), rowAlignment(DefaultRowAlignment), alignedWidth(0), storage() {}

template <typename WordType_>
QuantMat<1, WordType_>::QuantMat(int w, int h, size_t rowAlign)
    : width(0), height(0), rowAlignment(rowAlign), alignedWidth(0), storage() {

    if (w < 0 || h < 0) {
        BINCV_THROW(std::invalid_argument, "BinMat dimensions must be non-negative");
    }
    impl::checkRowAlignment<WordType>(rowAlign);

    width = static_cast<size_t>(w);
    height = static_cast<size_t>(h);
    alignedWidth = impl::calcAlignedWidth<WordType>(width, rowAlignment);

    // Allocate zero-initialized storage. Storage zero-fills, which is what keeps
    // the padding bits of a fresh matrix clear.
    storage = Storage<WordType>(height * alignedWidth);
}

template <typename WordType_>
QuantMat<1, WordType_>::QuantMat(WordType* dataPtr, int w, int h, size_t strideWords)
    : width(0), height(0), rowAlignment(DefaultRowAlignment), alignedWidth(0), storage() {

    if (w < 0 || h < 0) {
        BINCV_THROW(std::invalid_argument, "BinMat dimensions must be non-negative");
    }

    const size_t wrapWidth = static_cast<size_t>(w);
    const size_t wrapHeight = static_cast<size_t>(h);

    // A stride shorter than the row needs would make consecutive rows overlap,
    // and every row pointer past row 0 would be wrong rather than merely tight.
    if (strideWords < impl::minRowWords<WordType>(wrapWidth)) {
        BINCV_THROW(std::invalid_argument,
                    "BinMat strideWords must be at least ceil(width / WordBits) words");
    }
    if (dataPtr == nullptr && wrapWidth > 0 && wrapHeight > 0) {
        BINCV_THROW(std::invalid_argument,
                    "BinMat cannot wrap a null pointer as a non-empty matrix");
    }

    width = wrapWidth;
    height = wrapHeight;
    alignedWidth = strideWords;

    // Non-owning: no allocation, and the buffer is used exactly as handed over.
    storage = Storage<WordType>(dataPtr, height * alignedWidth);
}

// Special members
template <typename WordType_>
QuantMat<1, WordType_>::QuantMat(const QuantMat& other)
    : width(other.width),
      height(other.height),
      rowAlignment(other.rowAlignment),
      alignedWidth(other.alignedWidth),
      storage() {
    // The copy owns its memory whether or not the source did -- D-8 applies to the
    // source's *contents*, not to how the source happened to be constructed.
    if (other.storage.ownsMemory()) {
        // Storage's own copy deep-copies an owning source, allocating and filling
        // in a single pass. Routing this through Storage(size_t) instead would
        // zero the whole buffer and then immediately overwrite every word of it.
        storage = other.storage;
    } else if (!other.storage.empty()) {
        // A non-owning source is the one case Storage's copy aliases rather than
        // duplicates, so the deep copy is made here instead.
        storage = Storage<WordType>(other.storage.size());
        std::memcpy(storage.data(), other.storage.data(), storage.size() * sizeof(WordType));

        // The wrap constructor leaves the padding-bit invariant to the caller, so
        // a wrapped source may carry set bits past `width`. This object owns its
        // storage from here on, and an owning BinMat is expected to keep those bits
        // zero -- a word-wise reduction over the copy would otherwise count phantom
        // pixels the source's own per-pixel count never showed. Re-establish the
        // invariant at the moment BinMat takes ownership.
        clearTrailingBits();
    }
}

template <typename WordType_>
QuantMat<1, WordType_>& QuantMat<1, WordType_>::operator=(const QuantMat& other) {
    if (this == &other) return *this;

    // Copy first, then move into place. The copy reads `other` while this object
    // still holds its own buffer, which is what makes the case of `other` wrapping
    // this object's storage a live read rather than a use-after-free.
    QuantMat copy(other);
    *this = std::move(copy);
    return *this;
}

template <typename WordType_>
QuantMat<1, WordType_>::QuantMat(QuantMat&& other) noexcept
    : width(other.width),
      height(other.height),
      rowAlignment(other.rowAlignment),
      alignedWidth(other.alignedWidth),
      storage(std::move(other.storage)) {
    // Storage's move leaves the source with no buffer, so the source's dimensions
    // have to go with it. Otherwise a moved-from matrix would report empty()
    // == false while data() is null, and at()/ptr() would walk a null pointer.
    // @note Unlike move-assignment below, this can adopt `other`'s dimensions
    //       unconditionally: a freshly built object owns no block for `other` to
    //       alias, so Storage's move constructor has no refusal path.
    other.width = 0;
    other.height = 0;
    other.alignedWidth = 0;
}

template <typename WordType_>
QuantMat<1, WordType_>& QuantMat<1, WordType_>::operator=(QuantMat&& other) noexcept {
    if (this == &other) return *this;

    // Read the source's descriptor before the storage move. Storage refuses to
    // move from a NON-OWNING source that wraps this object's own block -- it can
    // neither free the block nor adopt a pointer into it -- and answers by leaving
    // this object's buffer untouched. The dimensions must therefore not be
    // committed until the transfer is known to have happened: adopting `other`'s
    // shape over an unchanged buffer would describe memory this matrix does not
    // have, breaking sizeInWords() == height * alignedWidth and making every row
    // pointer address the wrong row.
    const WordType* const otherPtr = other.storage.data();
    const size_t otherWords = other.storage.size();
    const size_t otherWidth = other.width;
    const size_t otherHeight = other.height;
    const size_t otherRowAlignment = other.rowAlignment;
    const size_t otherAlignedWidth = other.alignedWidth;

    storage = std::move(other.storage);

    // The transfer happened iff this object's buffer is now the one `other` named.
    // (When a refused move leaves an identical descriptor behind -- same base, same
    // word count -- the two shapes describe the same memory, so adopting is correct
    // there too.)
    if (storage.data() == otherPtr && storage.size() == otherWords) {
        width = otherWidth;
        height = otherHeight;
        rowAlignment = otherRowAlignment;
        alignedWidth = otherAlignedWidth;
    }

    // `other` is emptied either way: Storage empties a moved-from source even when
    // it refuses the transfer, so a moved-from BinMat is always a valid empty one.
    other.width = 0;
    other.height = 0;
    other.alignedWidth = 0;
    return *this;
}


// ---------------------------------------------------------------------------
// THE ROW PACKER LIVES ABOVE THE OpenCV GUARD, AND THAT PLACEMENT IS THE POINT.
//
// It was written inside `#ifdef BINCV_WITH_OPENCV` because its first caller was
// `fromCVMat`. That made the one path a core-only build needs -- turning a plain
// pixel array into bits -- available only when OpenCV was present, which is the
// exact gap ops/pack.hpp exists to close. Caught by a core-only compile, since
// every OpenCV-enabled build hid it.
// ---------------------------------------------------------------------------
namespace impl {

/// @brief Scatters a 32-pixel mask into `WordType` words. **INTERNAL** (X-71).
/// @note `bitsPerWord` is a power of two, so this is one store, a shifted OR, or a
///       split -- a 32-pixel group can never straddle a word boundary.
template <typename WordType>
inline void scatter32(uint32_t m, size_t x, WordType* rowOut) {
    constexpr size_t kBits = bitsPerWord<WordType>();
    if constexpr (kBits >= 32) {
        rowOut[x / kBits] |= static_cast<WordType>(static_cast<WordType>(m) << (x % kBits));
    } else {
        for (size_t c = 0; c < 32 / kBits; ++c) {
            rowOut[(x / kBits) + c] |=
                static_cast<WordType>((m >> (c * kBits)) & lowBitsMask<WordType>(kBits));
        }
    }
}

/// @brief Packs `[from, width)` of a row, branchless and portable. **INTERNAL** (X-71).
/// @note **Not merely a tail handler.** This is the whole path on any target without a
///       vector one, and on its own it is **10.3×** the per-pixel loop it replaces
///       (0.108 ns/px against 1.112). `!= 0` is a set-flag, not a jump.
template <typename WordType>
inline void packRowNonZeroPortable(const uint8_t* rowIn, size_t from, size_t width,
                                   WordType* rowOut) {
    constexpr size_t kBits = bitsPerWord<WordType>();
    // The vector paths consume whole WORDS precisely so this holds; `uint64_t` is
    // where it first mattered, and nothing narrower could have exposed it.
    BINCV_ASSERT(from % kBits == 0, "packRowNonZeroPortable: `from` must be word-aligned");
    size_t x = from;
    for (; x + kBits <= width; x += kBits) {
        WordType acc = 0;
        for (size_t i = 0; i < kBits; ++i)
            acc |= static_cast<WordType>(static_cast<WordType>(rowIn[x + i] != 0) << i);
        rowOut[x / kBits] |= acc;
    }
    if (x < width) {
        WordType acc = 0;
        for (size_t i = 0; x + i < width; ++i)
            acc |= static_cast<WordType>(static_cast<WordType>(rowIn[x + i] != 0) << i);
        rowOut[x / kBits] |= acc;
    }
}

/// @brief How a source pixel becomes a bit. **INTERNAL mirror of `PackRule`** --
///        ops/pack.hpp holds the public enum, and this header cannot include it
///        (pack.hpp includes binMat.hpp, which includes this file).
enum class PackCmp { NonZero, GreaterThan, GreaterEqual };

/// @brief One pixel's bit, scalar.
template <PackCmp R, typename SrcT>
inline bool packCmp(SrcT v, SrcT t) {
    if constexpr (R == PackCmp::NonZero) {
        (void)t;
        return v != SrcT{0};
    } else if constexpr (R == PackCmp::GreaterThan) {
        return v > t;
    } else {
        return v >= t;
    }
}

#if defined(BINCV_X86_RUNTIME_AVX2)
/// @brief 32 source pixels' rule result, as 32 bits. **INTERNAL** (X-71).
///
/// **ONE COARSE ENTRY POINT, NEVER LEAF HELPERS -- and this is the coarse one.**
/// `__attribute__((target(...)))` stops a compiler inlining a function into a caller
/// whose feature set is smaller: that is what the attribute is FOR, since the caller
/// may run where the callee's instructions do not exist.
/// [X-60](../../../EXPERIMENTS.md) marked leaf helpers *inside* a hot loop and paid
/// **310 real calls per window**. Here the call IS the unit of work -- loads, a
/// compare and a movemask for 32 pixels -- not a helper inside one.
///
/// This is why binCV uses AVX2 with **no `-mavx2` build**: the baseline ISA is
/// unchanged and the fast path is chosen at run time.
///
/// @note **The comparisons are BIASED because the SSE/AVX integer compares are
///       SIGNED and binCV's pixels are not.** `cmpgt_epi8` on `0xFF` against `0x01`
///       answers "is -1 > 1", which is false and wrong. XOR-ing both sides with the
///       sign bit maps unsigned order onto signed order exactly.
template <PackCmp R, typename SrcT>
__attribute__((target("avx2"))) inline uint32_t movemask32(const SrcT* p, SrcT t) {
    if constexpr (sizeof(SrcT) == 1) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        if constexpr (R == PackCmp::NonZero) {
            (void)t;
            // AVX2 has no "compare not equal": compare against zero and invert once.
            return ~static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_setzero_si256())));
        } else {
            const __m256i bias = _mm256_set1_epi8(static_cast<char>(0x80));
            const __m256i vb = _mm256_xor_si256(v, bias);
            const __m256i tb =
                _mm256_xor_si256(_mm256_set1_epi8(static_cast<char>(t)), bias);
            const uint32_t gt = static_cast<uint32_t>(_mm256_movemask_epi8(
                _mm256_cmpgt_epi8(vb, tb)));
            if constexpr (R == PackCmp::GreaterThan) return gt;
            // `>= t` is `> t` OR `== t`.
            return gt | static_cast<uint32_t>(_mm256_movemask_epi8(
                       _mm256_cmpeq_epi8(v, _mm256_set1_epi8(static_cast<char>(t)))));
        }
    } else {
        // 16-bit source: two vectors of sixteen, narrowed to bytes before the mask.
        // `packs_epi16` saturates, which is exactly right here -- the operands are
        // already 0 or -1, and it preserves the LANE ORDER within each 128-bit half,
        // which `permute4x64` then puts back in sequence.
        const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 16));
        __m256i mlo, mhi;
        if constexpr (R == PackCmp::NonZero) {
            (void)t;
            const __m256i z = _mm256_setzero_si256();
            mlo = ~_mm256_cmpeq_epi16(lo, z);
            mhi = ~_mm256_cmpeq_epi16(hi, z);
            mlo = _mm256_xor_si256(_mm256_cmpeq_epi16(lo, z), _mm256_set1_epi16(-1));
            mhi = _mm256_xor_si256(_mm256_cmpeq_epi16(hi, z), _mm256_set1_epi16(-1));
        } else {
            const __m256i bias = _mm256_set1_epi16(static_cast<short>(0x8000));
            const __m256i tb =
                _mm256_xor_si256(_mm256_set1_epi16(static_cast<short>(t)), bias);
            mlo = _mm256_cmpgt_epi16(_mm256_xor_si256(lo, bias), tb);
            mhi = _mm256_cmpgt_epi16(_mm256_xor_si256(hi, bias), tb);
            if constexpr (R == PackCmp::GreaterEqual) {
                const __m256i te = _mm256_set1_epi16(static_cast<short>(t));
                mlo = _mm256_or_si256(mlo, _mm256_cmpeq_epi16(lo, te));
                mhi = _mm256_or_si256(mhi, _mm256_cmpeq_epi16(hi, te));
            }
        }
        const __m256i packed = _mm256_permute4x64_epi64(_mm256_packs_epi16(mlo, mhi), 0xD8);
        return static_cast<uint32_t>(_mm256_movemask_epi8(packed));
    }
}

/// @brief Is AVX2 present on this machine? Asked once, not once per row.
inline bool hasVectorPack() {
    static const bool kYes = __builtin_cpu_supports("avx2");
    return kYes;
}
#define BINCV_HAVE_VECTOR_PACK 1
#elif defined(BINCV_HAVE_NEON) && defined(__aarch64__)
/// @brief The same 32-bit mask, without a move-mask instruction. **INTERNAL** (X-71).
/// @note aarch64 has none, so AND per-lane bit weights and let three pairwise adds
///       fold sixteen bytes into a sixteen-bit mask. NEON is baseline on aarch64, so
///       unlike the AVX2 path there is nothing to dispatch on. NEON's compares ARE
///       unsigned (`vcgtq_u8`), so no bias is needed here.
template <PackCmp R, typename SrcT>
inline uint32_t movemask32(const SrcT* p, SrcT t) {
    const uint8x16_t weights = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint32_t m = 0;
    for (size_t half = 0; half < 2; ++half) {
        uint8x16_t nz;
        if constexpr (sizeof(SrcT) == 1) {
            const uint8x16_t v = vld1q_u8(reinterpret_cast<const uint8_t*>(p) + half * 16);
            if constexpr (R == PackCmp::NonZero) {
                (void)t;
                nz = vmvnq_u8(vceqzq_u8(v));
            } else if constexpr (R == PackCmp::GreaterThan) {
                nz = vcgtq_u8(v, vdupq_n_u8(static_cast<uint8_t>(t)));
            } else {
                nz = vcgeq_u8(v, vdupq_n_u8(static_cast<uint8_t>(t)));
            }
        } else {
            const uint16_t* q = reinterpret_cast<const uint16_t*>(p) + half * 16;
            const uint16x8_t a = vld1q_u16(q), b = vld1q_u16(q + 8);
            uint16x8_t ma, mb;
            if constexpr (R == PackCmp::NonZero) {
                (void)t;
                ma = vmvnq_u16(vceqzq_u16(a));
                mb = vmvnq_u16(vceqzq_u16(b));
            } else if constexpr (R == PackCmp::GreaterThan) {
                ma = vcgtq_u16(a, vdupq_n_u16(static_cast<uint16_t>(t)));
                mb = vcgtq_u16(b, vdupq_n_u16(static_cast<uint16_t>(t)));
            } else {
                ma = vcgeq_u16(a, vdupq_n_u16(static_cast<uint16_t>(t)));
                mb = vcgeq_u16(b, vdupq_n_u16(static_cast<uint16_t>(t)));
            }
            nz = vcombine_u8(vmovn_u16(ma), vmovn_u16(mb));
        }
        const uint8x16_t w = vandq_u8(nz, weights);
        uint8x8_t f = vpadd_u8(vget_low_u8(w), vget_high_u8(w));
        f = vpadd_u8(f, f);
        f = vpadd_u8(f, f);
        m |= static_cast<uint32_t>(vget_lane_u16(vreinterpret_u16_u8(f), 0)) << (16 * half);
    }
    return m;
}
inline bool hasVectorPack() { return true; }
#define BINCV_HAVE_VECTOR_PACK 1
#endif

/// @brief Packs one row to 1 bit per pixel under `R`. **INTERNAL** (X-71).
/// @param rowOut Only bits `[0, width)` are written and whole words are STORED, so a
///        row's PADDING BITS ARE ZERO on return ([CLAUDE.md](../../../CLAUDE.md)'s
///        hard rule -- word-wise reductions over-count otherwise).
///
/// **WHY THE BIT ORDERS LINE UP AND NO SHUFFLE IS NEEDED.** `bitMask(x)` is
/// `1 << (x % WordBits)`, so pixel `x` lands in bit `x` of its word, LSB first.
/// `_mm256_movemask_epi8` returns the top bit of each of 32 bytes as 32 bits, byte
/// `i` in bit `i`, LSB first. **The two conventions are the same one.**
template <PackCmp R, typename SrcT, typename WordType>
inline void packRowCmp(const SrcT* rowIn, size_t width, SrcT t, WordType* rowOut) {
    size_t x = 0;
#if defined(BINCV_HAVE_VECTOR_PACK)
    // WHOLE WORDS, NOT WHOLE 32-PIXEL GROUPS. At 64-bit words a 32-pixel group is
    // half a word, and letting the scalar tail start mid-word put its bits at the
    // wrong offsets -- caught by the `uint64_t` arm of tests/test_opencv_interop.cpp
    // and by nothing narrower.
    constexpr size_t kGroup =
        bitsPerWord<WordType>() > 32 ? bitsPerWord<WordType>() : size_t{32};
    if (hasVectorPack()) {
        for (; x + kGroup <= width; x += kGroup) {
            for (size_t g = 0; g < kGroup / 32; ++g)
                scatter32<WordType>(movemask32<R, SrcT>(rowIn + x + g * 32, t),
                                    x + g * 32, rowOut);
        }
    }
#endif
    constexpr size_t kBits = bitsPerWord<WordType>();
    for (; x + kBits <= width; x += kBits) {
        WordType acc = 0;
        for (size_t i = 0; i < kBits; ++i)
            acc |= static_cast<WordType>(static_cast<WordType>(packCmp<R, SrcT>(rowIn[x + i], t))
                                         << i);
        rowOut[x / kBits] |= acc;
    }
    if (x < width) {
        WordType acc = 0;
        for (size_t i = 0; x + i < width; ++i)
            acc |= static_cast<WordType>(static_cast<WordType>(packCmp<R, SrcT>(rowIn[x + i], t))
                                         << i);
        rowOut[x / kBits] |= acc;
    }
}

/// @brief Unpacks one bit per pixel to one byte per pixel. **INTERNAL** (T5.7).
/// @note Lives here rather than in ops/pack.hpp because impl/binMat_impl.hpp cannot
///       include that file -- pack.hpp includes binMat.hpp, which includes this one.
///       ops/pack.hpp's `unpackTo8Bit` is the public spelling and calls straight
///       through.
// GUARDED ON THE ISA, NOT ON `BINCV_HAVE_VECTOR_PACK`. That macro means "a vector row
// packer exists" and BOTH backends define it, so using it here compiled the AVX2 branch
// on aarch64 -- caught by `check_arm_syntax.sh`, which is the third time this session
// that a guard has been wrong in code x86 never compiles.
#if defined(BINCV_X86_RUNTIME_AVX2)
inline bool unpackVectorReady() { return hasVectorPack(); }

/// @brief Thirty-two bits into thirty-two bytes. **INTERNAL.**
/// @note The shuffle mask replicates byte `k` of the word eight times, and
///       `vpshufb` indexes WITHIN each 128-bit lane -- so lane 0 asks for the word's
///       bytes 0 and 1 and lane 1 asks for bytes 2 and 3, which is why the two halves
///       of the mask differ.
__attribute__((target("avx2"))) inline void unpackWord32(uint32_t w, uint8_t* out,
                                                         uint8_t onValue,
                                                         uint8_t zeroValue) {
    const __m256i spread = _mm256_shuffle_epi8(
        _mm256_set1_epi32(static_cast<int>(w)),
        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                         2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32,
                                         64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8,
                                         16, 32, 64, -128);
    const __m256i m = _mm256_cmpeq_epi8(_mm256_and_si256(spread, sel), sel);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out),
                        _mm256_blendv_epi8(_mm256_set1_epi8(static_cast<char>(zeroValue)),
                                           _mm256_set1_epi8(static_cast<char>(onValue)), m));
}
#elif defined(BINCV_HAVE_NEON) && defined(__aarch64__)
inline bool unpackVectorReady() { return true; }

inline void unpackWord32(uint32_t w, uint8_t* out, uint8_t onValue, uint8_t zeroValue) {
    static const uint8_t kSel[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                     1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t sel = vld1q_u8(kSel);
    const uint8x16_t on = vdupq_n_u8(onValue);
    const uint8x16_t off = vdupq_n_u8(zeroValue);
    for (int half = 0; half < 2; ++half) {
        const uint8_t b0 = static_cast<uint8_t>(w >> (half * 16));
        const uint8_t b1 = static_cast<uint8_t>(w >> (half * 16 + 8));
        const uint8x16_t spread = vcombine_u8(vdup_n_u8(b0), vdup_n_u8(b1));
        const uint8x16_t m = vceqq_u8(vandq_u8(spread, sel), sel);
        vst1q_u8(out + static_cast<size_t>(half) * 16, vbslq_u8(m, on, off));
    }
}
#endif

template <typename WordType>
inline void unpackTo8BitRaw(const WordType* src, size_t srcStride, size_t width,
                            size_t height, uint8_t* dst, size_t dstStride, uint8_t onValue,
                            uint8_t zeroValue) {
    constexpr size_t kBits = bitsPerWord<WordType>();
    for (size_t y = 0; y < height; ++y) {
        const WordType* rowIn = src + y * srcStride;
        uint8_t* rowOut = dst + y * dstStride;
        for (size_t x = 0; x < width; x += kBits) {
            const size_t n = (width - x < kBits) ? (width - x) : kBits;
            // ONE word load per 32 pixels instead of one per pixel: the shift and
            // mask stay in a register where they used to be recomputed from `x`.
            WordType w = rowIn[x / kBits];
#if defined(BINCV_X86_RUNTIME_AVX2) || (defined(BINCV_HAVE_NEON) && defined(__aarch64__))
            // THE INVERSE OF A MOVE-MASK, AND IT VECTORISES THE SAME WAY. Broadcast the
            // word so byte i holds the byte containing bit i, AND with the per-lane bit
            // weights, compare, and select. Six operations for what was thirty-two
            // BRANCHES -- and the branch is the cost, not the shift.
            //
            // This is the audit CLAUDE.md's benchmark rule produced: `unpackTo8Bit` was
            // correct, word-wise, and 4x slower than `packBits` doing the same job in the
            // other direction, because nothing had ever timed it.
            if (n == 32 && sizeof(WordType) == 4 && unpackVectorReady()) {
                unpackWord32(static_cast<uint32_t>(w), rowOut + x, onValue, zeroValue);
                continue;
            }
#endif
            for (size_t i = 0; i < n; ++i) {
                rowOut[x + i] = (w & WordType{1}) ? onValue : zeroValue;
                w = static_cast<WordType>(w >> 1);
            }
        }
    }
}

} // namespace impl

#ifdef BINCV_WITH_OPENCV
// OpenCV interoperability implementations


template <typename WordType_>
void QuantMat<1, WordType_>::fromCVMat(const cv::Mat& input) {
    if (input.empty()) {
        BINCV_THROW(std::invalid_argument, "Input cv::Mat is empty");
    }
    if (input.type() != CV_8UC1) {
        BINCV_THROW(std::invalid_argument, "Input cv::Mat must be of type CV_8UC1");
    }

    const size_t newWidth = static_cast<size_t>(input.cols);
    const size_t newHeight = static_cast<size_t>(input.rows);
    const size_t newAlignedWidth = impl::calcAlignedWidth<WordType>(newWidth, rowAlignment);

    // Allocate and fill zero-initialized storage BEFORE touching this object's
    // dimensions, the same commit-last shape resize() uses. Committing first would
    // leave a failed allocation behind a matrix that describes a buffer it does not
    // have, and every later read would trust those dimensions -- at() cannot catch
    // it, since T1.4 made the bounds check debug-only.
    Storage<WordType> newData(newHeight * newAlignedWidth);

    for (size_t y = 0; y < newHeight; ++y) {
        const uint8_t* rowIn = input.ptr<uint8_t>(static_cast<int>(y));
        WordType* rowOut = newData.data() + y * newAlignedWidth;
        impl::packRowCmp<impl::PackCmp::NonZero, uint8_t, WordType>(rowIn, newWidth,
                                                                    uint8_t{0}, rowOut);
    }

    width = newWidth;
    height = newHeight;
    alignedWidth = newAlignedWidth;
    storage = std::move(newData);
}

// Shared unpacking loop for the two cv::Mat conversions; `transform` maps a bit
// to the output pixel value.
/// @note ONE IMPLEMENTATION. The unpacking lives in ops/pack.hpp, in CORE, so the
///       `cv::Mat` wrapper is a shape adapter and nothing more. Before this split it
///       was a per-pixel loop recomputing `wordIndex` and `bitMask` for every pixel --
///       the exact shape `fromCVMat` had before [X-71](../../../EXPERIMENTS.md), and
///       the slowest thing in the library.
template <typename WordType, typename PixelTransform>
inline void toCVMatHelper(const BinMat<WordType>& binmat, cv::Mat& output,
                          PixelTransform transform) {
    if (binmat.empty()) {
        output = cv::Mat();
        return;
    }
    output = cv::Mat::zeros(binmat.rows(), binmat.cols(), CV_8U);
    impl::unpackTo8BitRaw<WordType>(binmat.data(), binmat.getAlignedWidth(),
                              static_cast<size_t>(binmat.cols()),
                              static_cast<size_t>(binmat.rows()), output.ptr<uint8_t>(0),
                              output.step, transform(true), transform(false));
}

// @todo: could an approach using OpenCV's resize and scaling functions be more efficient?
//        need to think more about how to do this efficiently
template <typename WordType_>
void QuantMat<1, WordType_>::toCVMat(cv::Mat& output) const {
    toCVMatHelper(*this, output, [](bool value) -> uint8_t { return value ? 1 : 0; });
}

template <typename WordType_>
void QuantMat<1, WordType_>::toCVMatNormalized(cv::Mat& output) const {
    toCVMatHelper(*this, output, [](bool value) -> uint8_t { return value ? 255 : 0; });
}

#endif // BINCV_WITH_OPENCV

// clearTrailingBits
template <typename WordType_>
void QuantMat<1, WordType_>::clearTrailingBits() {
    if (empty()) return;

    // Bits [width, alignedWidth * WordBits) are padding and must stay zero.
    size_t lastWord = impl::wordIndex<WordType>(width == 0 ? 0 : width - 1);
    size_t validBitsInLastWord = width - lastWord * WordBits;
    WordType keepMask = impl::lowBitsMask<WordType>(validBitsInLastWord);

    for (size_t y = 0; y < height; ++y) {
        WordType* row = storage.data() + y * alignedWidth;
        row[lastWord] &= keepMask;
        // Any whole words past the last partially-used one are pure padding. At the
        // default alignment there are none; opt-in alignment is the case this covers.
        std::fill(row + lastWord + 1, row + alignedWidth, static_cast<WordType>(0));
    }
}

// at and set
//
// Debug-checked, unchecked in release (ARCHITECTURE 5.3, and the behaviour
// change D-7 sanctions). These are the two functions on the per-pixel path, and
// a throw here would sit inside every loop that reads an image. In a release
// build the checks are gone entirely -- what remains is the row offset, a shift
// and a mask -- and an out-of-range index is undefined behaviour, exactly as it
// is for cv::Mat::at. Callers that cannot guarantee their indices should clamp
// before calling, not rely on the container to report it.
template <typename WordType_>
bool QuantMat<1, WordType_>::at(int row, int col) const {
    BINCV_ASSERT(row >= 0 && row < static_cast<int>(height) &&
                     col >= 0 && col < static_cast<int>(width),
                 "BinMat::at: index out of range");
    const WordType* rowPtr = storage.data() + static_cast<size_t>(row) * alignedWidth;
    size_t x = static_cast<size_t>(col);
    return (rowPtr[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) != 0;
}

template <typename WordType_>
void QuantMat<1, WordType_>::set(int row, int col, bool value) {
    BINCV_ASSERT(row >= 0 && row < static_cast<int>(height) &&
                     col >= 0 && col < static_cast<int>(width),
                 "BinMat::set: index out of range");
    WordType* rowPtr = storage.data() + static_cast<size_t>(row) * alignedWidth;
    size_t x = static_cast<size_t>(col);
    WordType& word = rowPtr[impl::wordIndex<WordType>(x)];
    WordType mask = impl::bitMask<WordType>(x);

    if (value) {
        word |= mask;   // set bit
    } else {
        word &= ~mask;  // clear bit
    }
}

// ptr
template <typename WordType_>
const typename QuantMat<1, WordType_>::WordType* QuantMat<1, WordType_>::ptr(int row) const {
    return storage.data() + static_cast<size_t>(row) * alignedWidth;
}

template <typename WordType_>
typename QuantMat<1, WordType_>::WordType* QuantMat<1, WordType_>::ptr(int row) {
    return storage.data() + static_cast<size_t>(row) * alignedWidth;
}

// resize
// @todo: This could potentially be optimized (word-wise copy when alignment permits)
template <typename WordType_>
void QuantMat<1, WordType_>::resize(int newWidth, int newHeight) {
    if (newWidth < 0 || newHeight < 0)
        BINCV_THROW(std::invalid_argument, "BinMat dimensions must be non-negative");

    size_t nw = static_cast<size_t>(newWidth);
    size_t nh = static_cast<size_t>(newHeight);
    size_t newAlignedWidth = impl::calcAlignedWidth<WordType>(nw, rowAlignment);

    // Create new zero-initialized storage
    Storage<WordType> newData(nh * newAlignedWidth);

    // Determine region to copy from old matrix
    size_t minWidth = std::min(width, nw);
    size_t minHeight = std::min(height, nh);

    for (size_t y = 0; y < minHeight; ++y) {
        const WordType* oldRow = storage.data() + y * alignedWidth;
        WordType* newRow = newData.data() + y * newAlignedWidth;

        for (size_t x = 0; x < minWidth; ++x) {
            if (oldRow[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) {
                newRow[impl::wordIndex<WordType>(x)] |= impl::bitMask<WordType>(x);
            }
        }
    }

    // Update internal state
    width = nw;
    height = nh;
    alignedWidth = newAlignedWidth;
    storage = std::move(newData);
}

// pad
// @todo: This could potentially be optimized further
// @todo: OpenCV has a copyMakeBorder function, however alignment with our packed
//        representation becomes complicated when padding towards the left.
//        For now we implement our own padding.
template <typename WordType_>
void QuantMat<1, WordType_>::pad(int top, int bottom, int left, int right, bool value) {
    if (top < 0 || bottom < 0 || left < 0 || right < 0) {
        BINCV_THROW(std::invalid_argument, "Padding values must be non-negative");
    }

    // Calculate new dimensions
    size_t newWidth = width + static_cast<size_t>(left) + static_cast<size_t>(right);
    size_t newHeight = height + static_cast<size_t>(top) + static_cast<size_t>(bottom);
    size_t newAlignedWidth = impl::calcAlignedWidth<WordType>(newWidth, rowAlignment);

    // Start from a buffer filled with the padding value, then overwrite the
    // interior with the original contents. Storage already zero-fills what it
    // allocates, so only a `true` pad has anything left to write -- filling
    // unconditionally would write the whole destination twice.
    Storage<WordType> newData(newHeight * newAlignedWidth);
    if (value) {
        const WordType fillValue = static_cast<WordType>(~static_cast<WordType>(0));
        std::fill(newData.data(), newData.data() + newData.size(), fillValue);
    }

    for (size_t y = 0; y < height; ++y) {
        const WordType* oldRow = storage.data() + y * alignedWidth;
        WordType* newRow = newData.data() + (y + static_cast<size_t>(top)) * newAlignedWidth;

        for (size_t x = 0; x < width; ++x) {
            size_t newX = x + static_cast<size_t>(left);
            bool bit = (oldRow[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) != 0;
            if (bit) {
                newRow[impl::wordIndex<WordType>(newX)] |= impl::bitMask<WordType>(newX);
            } else {
                // The cast is not cosmetic at narrow word widths: `~` promotes a
                // uint8_t/uint16_t mask to int, so the compound assignment
                // narrows back. Harmless for `&`, but it is the same shape as a
                // genuine truncation, and -Wconversion cannot tell them apart.
                newRow[impl::wordIndex<WordType>(newX)] &=
                    static_cast<WordType>(~impl::bitMask<WordType>(newX));
            }
        }
    }

    // Update internal state
    width = newWidth;
    height = newHeight;
    alignedWidth = newAlignedWidth;
    storage = std::move(newData);

    // A `true` fill writes ones across the whole stride, including padding bits
    if (value) clearTrailingBits();
}

// transposed
// @todo: naive pixel-by-pixel transpose; replace with a cache-blocked / bit-parallel
//        version (see ARCHITECTURE.md 6.4). This is currently the slowest operation.
template <typename WordType_>
QuantMat<1, WordType_> QuantMat<1, WordType_>::transposed() const {
    // An empty matrix still has a shape to transpose: a 640x0 matrix transposes to
    // 0x640, not to 0x0, and it keeps its row alignment like any other result. The
    // constructor allocates nothing when either dimension is zero, so this costs
    // nothing; the pixel loops below simply do not run.
    QuantMat result(static_cast<int>(height), static_cast<int>(width), rowAlignment);
    if (empty()) {
        return result;
    }

    for (size_t y = 0; y < height; ++y) {
        const WordType* row = storage.data() + y * alignedWidth;
        for (size_t x = 0; x < width; ++x) {
            if (row[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) {
                result.set(static_cast<int>(x), static_cast<int>(y), true);
            }
        }
    }
    return result;
}

// transpose
// @todo: this could avoid the copy made by transposed() for the square case
template <typename WordType_>
void QuantMat<1, WordType_>::transpose() {
    *this = this->transposed();
}

// forEachNonZero
template <typename WordType_>
template <typename Func>
void QuantMat<1, WordType_>::forEachNonZero(Func callback) const {
    if (empty()) {
        BINCV_THROW(std::runtime_error,
                    "BinMat is empty, cannot iterate over non-zero pixels");
    }
    for (size_t y = 0; y < height; ++y) {
        const WordType* row = storage.data() + y * alignedWidth;
        for (size_t x = 0; x < width; ++x) {
            if (row[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) {
                callback(static_cast<int>(y), static_cast<int>(x));
            }
        }
    }
}

// printMatrix
template <typename WordType_>
void QuantMat<1, WordType_>::printMatrix() const {
    if (empty()) {
        return;
    }

    for (size_t y = 0; y < height; ++y) {
        const WordType* row = storage.data() + y * alignedWidth;
        for (size_t x = 0; x < width; ++x) {
            std::fputc((row[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) ? '1' : '0',
                       stdout);
        }
        std::fputc('\n', stdout);
    }
}

// operator<<
// @todo: consider std::ostringstream or wrapping std::ostream& for reusable logging.
template <typename WordType>
std::ostream& operator<<(std::ostream& os, const BinMat<WordType>& binmat) {
    if (binmat.empty()) {
        return os; // Nothing to print for empty matrix
    }

    for (int y = 0; y < binmat.rows(); ++y) {
        const WordType* row = binmat.ptr(y);
        for (size_t x = 0; x < binmat.getWidth(); ++x) {
            os << ((row[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) ? '1' : '0');
        }
        os << '\n';
    }
    return os;
}

// printInternalData
template <typename WordType_>
void QuantMat<1, WordType_>::printInternalData(bool hex) const {
    if (empty()) {
        return;
    }

    for (size_t y = 0; y < height; ++y) {
        const WordType* row = storage.data() + y * alignedWidth;
        for (size_t b = 0; b < alignedWidth; ++b) {
            // Widened to the largest word type so one format string covers all
            // four. Nothing to reset afterwards, unlike the std::hex this used to
            // leave stuck on std::cout.
            std::fprintf(stdout, hex ? "%llx " : "%llu ",
                         static_cast<unsigned long long>(row[b]));
        }
        std::fputc('\n', stdout);
    }
}

// fill
template <typename WordType_>
void QuantMat<1, WordType_>::fill(bool value) {
    if (empty())
        return;

    WordType fillWord = value ? static_cast<WordType>(~static_cast<WordType>(0))
                              : static_cast<WordType>(0);

    std::fill(storage.data(), storage.data() + storage.size(), fillWord);

    // fill(true) sets the row-padding bits too; clear them so that word-wise
    // consumers (countNonZero, future bitwise ops) don't see phantom pixels.
    if (value) clearTrailingBits();
}

// countNonZero
//
// STILL A PER-PIXEL LOOP, and deliberately so since T2.5. The bulk reduction is
// `bincv::countNonZero(m.constView())` in ops/reduce.hpp, which is 6x faster here
// and 35x faster where the popcount lowers to an instruction
// (bincv-cpp/results/reduce_benchmark.log). This member cannot simply forward to
// it: ops/reduce.hpp includes binMat.hpp, which includes this file, so the call
// would close a cycle. It stays for two reasons -- it is the container-shaped
// spelling callers already use, and it is the "before" the T2.5 benchmark
// measures against, which stops being true the moment it becomes a wrapper.
//
// It does NOT rely on padding bits being zero (it never reads one), which is the
// same guarantee the bulk kernels now make by masking; see D-13.
template <typename WordType_>
int QuantMat<1, WordType_>::countNonZero() const {
    if (empty())
        return 0;

    int count = 0;
    for (size_t y = 0; y < height; ++y) {
        const WordType* row = storage.data() + y * alignedWidth;
        for (size_t x = 0; x < width; ++x) {
            if (row[impl::wordIndex<WordType>(x)] & impl::bitMask<WordType>(x)) {
                ++count;
            }
        }
    }
    return count;
}

// sparsity
template <typename WordType_>
float QuantMat<1, WordType_>::sparsity() const {
    size_t totalPixels = width * height;
    if (totalPixels == 0)
        BINCV_THROW(std::runtime_error, "Sparsity is undefined for empty BinMat");

    int nonzero = countNonZero();
    return 1.0f - static_cast<float>(nonzero) / static_cast<float>(totalPixels);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
