#pragma once

/// @file edge.hpp
/// @brief Gradient-magnitude edge extraction, 8- or 16-bit in, 1 bit out.
/// **API TIER 3** -- there is no `cv::` equivalent; see the tier note.
///
/// ---------------------------------------------------------------------------
/// READ OUT OF THE REFERENCE, NOT INFERRED
///
/// The edge filter of the reference pipeline (the visual-inertial odometry system,
/// not in this repository, that binCV was built to serve stage by stage; see
/// docs/ARCHITECTURE.md), in its own spelling:
///
/// kernel_x = [-1 0 1] diff_x = |filter2D(img, kernel_x)|
/// kernel_y = [-1 0 1]^T diff_y = |filter2D(img, kernel_y)|
/// mask = (diff_x >= t) OR (diff_y >= t)
///
/// THREE DETAILS THAT ARE EASY TO GET WRONG, AND ALL THREE ARE THE DEFAULTS:
///
/// * the combination is **OR**, not AND;
/// * the relation is **`>=`**, not `>`;
/// * **"wide" is the CENTRAL difference** `[-1, 0, 1]` -- the left neighbour
/// against the RIGHT neighbour, spanning two pixels -- not an adjacent `[-1, 1]`.
///
/// A caller who says nothing gets exactly the reference's operation.
///
/// ---------------------------------------------------------------------------
/// ALL TWELVE COMBINATIONS SHIP
///
/// combine {Or, And} x relation {Ge, Gt} x spatial {Wide, Forward, Backward}. They
/// are compile-time parameters, so a caller pays only for the one instantiated and
/// the comparison folds to a single predicate -- the same requirement ops/pack.hpp
/// puts on its rules, for the same measured reason (a runtime branch in the
/// packer's inner loop measured 1.17× the cost of the compile-time rule).
///
/// The point of the operation is that these choices are cheap. A caller wanting AND
/// instead of OR, or an adjacent difference instead of a central one, should not have
/// to fork the kernel.
///
/// ---------------------------------------------------------------------------
/// WHY `SrcT` AND NOT JUST `uint8_t`
///
/// **This operation is why the input contract in ops/pack.hpp reaches past
/// 8 bits.** "Downconvert 12->8 yourself, then
/// call binCV" is `v >> 4`, which truncates the OPERANDS before they are differenced:
/// a genuine 12-bit gradient of 15 counts becomes **exactly zero**, and the edge is
/// gone before binCV sees the pixel. Low contrast -- indoors, at night, on untextured
/// walls -- is where a VIO frontend needs every edge it can get, so the workaround
/// fails hardest exactly where it matters.
///
/// ---------------------------------------------------------------------------
/// TIER 3, AND THE NAME IS NOT OPENCV'S
///
/// `cv::Sobel` + `cv::threshold` is a DIFFERENT computation: a 3x3 separable kernel
/// with smoothing, against this operation's single-axis difference. A Tier 3
/// operation does not borrow an OpenCV name it cannot honour.
///
/// The border rule IS OpenCV's, though: `cv::filter2D` defaults to
/// `BORDER_REFLECT_101`, so index -1 reads index 1 and index `w` reads index `w-2`.
/// That is what makes the shipped defaults reproduce the reference exactly rather
/// than approximately.
// BEFORE THE GATE, NOT AFTER. This header defines BINCV_HAVE_NEON from the
// compiler's own macros on aarch64, so an include-only integration still gets the
// NEON kernels. Relying on transitive inclusion would not do -- this file evaluates
// its gate before its first core include.
#include "../core/simd.hpp"


#include <cstddef>
#include <cstdint>
#include <initializer_list>   // the two border columns, as a braced list
#include <type_traits>

#include "../binMat.hpp"
#include "../impl/kernel_util.hpp"

// !__CUDACC__: a CUDA translation unit takes the portable arm, matching every
// other x86 arm's gate (impl/binMat_impl.hpp has the full note).
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
#define BINCV_EDGE_AVX2 1
#include <immintrin.h>
#elif defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#define BINCV_EDGE_NEON 1
#include <arm_neon.h>
#elif defined(__ARM_FEATURE_SIMD32) && !defined(__CUDACC__)
// The ARMv7E-M / ARMv8-M DSP extension: four byte lanes in a general register, which
// on a core with no vector unit is the widest arithmetic there is.
#define BINCV_EDGE_DSP 1
#include <arm_acle.h>
#include <cstring>
#endif

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief How the two axes' results are combined. `Or` is the reference's.
/// **API TIER 3.**
enum class EdgeCombine { Or, And };

/// @brief How a gradient is compared with the threshold. `Ge` is the reference's.
/// **API TIER 3.**
enum class EdgeRelation { Ge, Gt };

/// @brief Which pixels are differenced. `Wide` is the reference's. **API TIER 3.**
enum class EdgeSpatial {
    Wide,      ///< `|v[x+1] - v[x-1]|` -- the central difference, spanning two pixels
    Forward,   ///< `|v[x+1] - v[x]|`
    Backward,  ///< `|v[x] - v[x-1]|`
};

namespace impl {

/// @brief `BORDER_REFLECT_101`: index -1 reads 1, index n reads n-2. **INTERNAL.**
/// @note OpenCV's `filter2D` default, and therefore the reference's. A one-pixel
/// extent reflects to itself.
/// @note BINCV_HOST_DEVICE. One index in, one index out, no memory: the loop
/// folds the coordinate, it does not traverse an image. It is iterative
/// rather than closed-form (ops/shift.hpp's borderIndex is the closed form,
/// and is the one to reach for if an index far outside the extent ever
/// becomes reachable) -- every caller here is a single tap, so `i` is at most
/// one pixel outside and the fold runs once.
BINCV_HOST_DEVICE inline size_t reflect101Edge(long long i, size_t n) {
    if (n <= 1) return 0;
    const long long last = static_cast<long long>(n) - 1;
    while (i < 0 || i > last) {
        if (i < 0) i = -i;
        if (i > last) i = 2 * last - i;
    }
    return static_cast<size_t>(i);
}

/// @brief The difference type for a source type: `int` where it holds every
/// difference of two samples exactly, `long long` otherwise. **INTERNAL.**
/// @note On a 32-bit core every `long long` is a register pair, and the 8- and
/// 16-bit sources the sensor stage takes never need one.
template <typename SrcT>
using EdgeDiff = typename std::conditional<(sizeof(SrcT) < sizeof(int)), int, long long>::type;

template <EdgeRelation R, typename D>
inline bool passesEdge(D d, D t) {
    if constexpr (R == EdgeRelation::Ge) return d >= t;
    return d > t;
}

/// @brief The edge predicate at column `x` of one row. **INTERNAL.**
/// @param up,mid,dn The rows above, at and below the output row, **with the
/// vertical border already applied** -- the caller passes the reflected row --
/// so only the horizontal border is this function's. `S` picks which of the
/// three the vertical difference reads.
template <EdgeCombine C, EdgeRelation R, EdgeSpatial S, typename SrcT>
inline bool edgeAt(const SrcT* up, const SrcT* mid, const SrcT* dn, size_t width, size_t x,
                   EdgeDiff<SrcT> t) {
    using D = EdgeDiff<SrcT>;
    constexpr size_t offA = (S == EdgeSpatial::Backward) ? 0 : 1;  // the `+` sample
    constexpr size_t offB = (S == EdgeSpatial::Forward) ? 0 : 1;   // the `-` sample
    size_t a = x + offA, b = x - offB;
    if (x < offB || x + offA >= width) {
        a = reflect101Edge(static_cast<long long>(x) + static_cast<long long>(offA), width);
        b = reflect101Edge(static_cast<long long>(x) - static_cast<long long>(offB), width);
    }
    const D dh = static_cast<D>(mid[a]) - static_cast<D>(mid[b]);
    const bool ph = passesEdge<R, D>(dh < 0 ? -dh : dh, t);
    // SHORT-CIRCUITING IS CORRECT AND WORTH HAVING: on a sparse edge map most pixels
    // fail both tests, so `And` skips the vertical difference for nearly all of them.
    if constexpr (C == EdgeCombine::Or) {
        if (ph) return true;
    } else {
        if (!ph) return false;
    }
    const SrcT* va = (S == EdgeSpatial::Backward) ? mid : dn;
    const SrcT* vb = (S == EdgeSpatial::Forward) ? mid : up;
    const D dv = static_cast<D>(va[x]) - static_cast<D>(vb[x]);
    return passesEdge<R, D>(dv < 0 ? -dv : dv, t);
}

/// @brief The predicate at a column whose horizontal neighbours are inside the row.
/// **INTERNAL.** Branch-free: both axes are tested and combined bitwise, where
/// `edgeAt` short-circuits -- on a core without a vector unit this IS the inner
/// loop, and the short circuit is a data-dependent branch that mispredicts on
/// every edge pixel. Same function, so the same bit.
template <EdgeCombine C, EdgeRelation R, EdgeSpatial S, typename SrcT>
inline bool edgeInteriorAt(const SrcT* up, const SrcT* mid, const SrcT* dn, size_t x,
                           EdgeDiff<SrcT> t) {
    using D = EdgeDiff<SrcT>;
    constexpr size_t offA = (S == EdgeSpatial::Backward) ? 0 : 1;
    constexpr size_t offB = (S == EdgeSpatial::Forward) ? 0 : 1;
    const SrcT* va = (S == EdgeSpatial::Backward) ? mid : dn;
    const SrcT* vb = (S == EdgeSpatial::Forward) ? mid : up;
    const D dh = static_cast<D>(mid[x + offA]) - static_cast<D>(mid[x - offB]);
    const D dv = static_cast<D>(va[x]) - static_cast<D>(vb[x]);
    const bool ph = passesEdge<R, D>(dh < 0 ? -dh : dh, t);
    const bool pv = passesEdge<R, D>(dv < 0 ? -dv : dv, t);
    return (C == EdgeCombine::Or) ? (ph | pv) : (ph & pv);
}

/// @brief One output row, one pixel at a time. **INTERNAL** -- the scalar body:
/// the border rule, the general `SrcT`, the general `EdgeSpatial`, and the
/// oracle `tests/test_edge.cpp` holds the vector arm to.
/// @note The output word is built in a register and STORED, not OR-ed: the bits
/// past `width` are never set, which is the padding invariant with no mask.
/// @note The two columns whose neighbour is outside the row take `edgeAt`, which
/// reflects; every other column takes the branch-free interior form.
template <EdgeCombine C, EdgeRelation R, EdgeSpatial S, typename SrcT, typename WordType>
inline void edgeRowScalar(const SrcT* up, const SrcT* mid, const SrcT* dn, size_t width,
                          WordType* out, EdgeDiff<SrcT> t) {
    constexpr size_t kBits = bitsPerWord<WordType>();
    constexpr size_t offA = (S == EdgeSpatial::Backward) ? 0 : 1;
    constexpr size_t offB = (S == EdgeSpatial::Forward) ? 0 : 1;
    // Columns [lo, hi) have both horizontal neighbours inside the row.
    const size_t lo = offB < width ? offB : width;
    const size_t hi = width > offA ? width - offA : 0;
    for (size_t x = 0; x < width; x += kBits) {
        const size_t n = (width - x < kBits) ? (width - x) : kBits;
        WordType acc = 0;
        for (size_t i = 0; i < n; ++i) {
            const size_t xi = x + i;
            const bool on = (xi >= lo && xi < hi)
                                ? edgeInteriorAt<C, R, S, SrcT>(up, mid, dn, xi, t)
                                : edgeAt<C, R, S, SrcT>(up, mid, dn, width, xi, t);
            acc = static_cast<WordType>(acc | (static_cast<WordType>(on) << i));
        }
        out[x / kBits] = acc;
    }
}

// ==================================================================
// THE EDGE PREDICATE, THIRTY-TWO PIXELS AT A TIME.
//
// `|a - b| >= t` on unsigned bytes is `subs_epu8(a,b) | subs_epu8(b,a)` for the absolute
// difference and `subs_epu8(t, d) == 0` for the comparison — **no widening, no sign, no
// branch.** The result is a byte mask, and `movemask_epi8` turns thirty-two of those into
// the thirty-two bits of an output word directly, LSB first, which is exactly binCV's bit
// order. **The 8-bit intermediate this operation exists to avoid never appears
// even inside the kernel.**
//
// aarch64 has no move-mask, so bit weights and three pairwise adds fold sixteen byte
// masks into sixteen bits — the same substitute measured for the row packer.
//
// The scalar body above stays: it is the border rule, the general `SrcT`, the general
// `EdgeSpatial`, and the oracle `tests/test_edge.cpp` compares against.

#if defined(BINCV_EDGE_AVX2)
/// @brief Force the portable arm, for the benchmark and the tests: the rule is
/// that a vector arm is switchable off and the benchmark shows it is on.
/// **INTERNAL.**
inline bool& edgeSimdEnabled() {
    static bool on = true;
    return on;
}

inline bool hasEdgeSimd() {
    static const bool kYes = __builtin_cpu_supports("avx2");
    return kYes && edgeSimdEnabled();
}

/// @brief `|a - b| >= tp` for thirty-two unsigned bytes, as a byte mask.
__attribute__((target("avx2"))) inline __m256i edgePass32(__m256i a, __m256i b, int tp) {
    const __m256i d = _mm256_or_si256(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a));
    return _mm256_cmpeq_epi8(_mm256_subs_epu8(_mm256_set1_epi8(static_cast<char>(tp)), d),
                             _mm256_setzero_si256());
}

/// @brief Thirty-two pixels of the edge predicate, as thirty-two bits.
/// @param tp The threshold ALREADY adjusted for the relation: `t` for `Ge`, `t + 1` for
/// `Gt`, so the comparison itself has only one spelling.
template <EdgeCombine C>
__attribute__((target("avx2"))) inline uint32_t edgeMask32(const uint8_t* rowUp,
                                                           const uint8_t* row,
                                                           const uint8_t* rowDn,
                                                           const uint8_t* colA,
                                                           const uint8_t* colB, int tp) {
    const __m256i px = edgePass32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(colA)),
                                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(colB)),
                                  tp);
    const __m256i py = edgePass32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowDn)),
                                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowUp)),
                                  tp);
    (void)row;
    const __m256i m = (C == EdgeCombine::Or) ? _mm256_or_si256(px, py)
                                             : _mm256_and_si256(px, py);
    return static_cast<uint32_t>(_mm256_movemask_epi8(m));
}
#elif defined(BINCV_EDGE_NEON)
/// @brief Force the portable arm, for the benchmark and the tests. **INTERNAL.**
inline bool& edgeSimdEnabled() {
    static bool on = true;
    return on;
}

inline bool hasEdgeSimd() { return edgeSimdEnabled(); }

inline uint8x16_t edgePass16(uint8x16_t a, uint8x16_t b, int tp) {
    const uint8x16_t d = vorrq_u8(vqsubq_u8(a, b), vqsubq_u8(b, a));
    return vceqq_u8(vqsubq_u8(vdupq_n_u8(static_cast<uint8_t>(tp)), d), vdupq_n_u8(0));
}

/// @brief Sixteen byte masks into sixteen bits, LSB first. No move-mask on aarch64.
inline uint32_t edgeFold16(uint8x16_t m) {
    static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                   1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t w = vandq_u8(m, vld1q_u8(kW));
    const uint8x8_t lo = vget_low_u8(w), hi = vget_high_u8(w);
    const uint8_t a = vaddv_u8(lo), b = vaddv_u8(hi);
    return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 8);
}

template <EdgeCombine C>
inline uint32_t edgeMask32(const uint8_t* rowUp, const uint8_t* row, const uint8_t* rowDn,
                           const uint8_t* colA, const uint8_t* colB, int tp) {
    (void)row;
    uint32_t out = 0;
    for (int half = 0; half < 2; ++half) {
        const size_t o = static_cast<size_t>(half) * 16;
        const uint8x16_t px = edgePass16(vld1q_u8(colA + o), vld1q_u8(colB + o), tp);
        const uint8x16_t py = edgePass16(vld1q_u8(rowDn + o), vld1q_u8(rowUp + o), tp);
        const uint8x16_t m = (C == EdgeCombine::Or) ? vorrq_u8(px, py) : vandq_u8(px, py);
        out |= edgeFold16(m) << (half * 16);
    }
    return out;
}
#elif defined(BINCV_EDGE_DSP)
/// @brief Force the portable arm, for the board's arm comparison and the tests.
/// **INTERNAL.**
inline bool& edgeSimdEnabled() {
    static bool on = true;
    return on;
}

inline bool hasEdgeSimd() { return edgeSimdEnabled(); }

inline uint32_t loadBytes4(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof v);  // one unaligned LDR on ARMv7-M
    return v;
}

/// @brief Thirty-two pixels of the edge predicate as thirty-two bits, four byte lanes
/// at a time. **INTERNAL.**
/// @note `|a - b|` is `uqsub8(a, b) | uqsub8(b, a)`, as on the vector arms. The two
/// axes are combined by a byte max (`Or`) or min (`And`) -- `usub8` then `sel` -- and
/// `usub8` against the threshold leaves each lane's `>= t` in its GE flag, which
/// `sel` turns into a byte mask and one multiply gathers into four bits.
template <EdgeCombine C>
inline uint32_t edgeMask32(const uint8_t* rowUp, const uint8_t* row, const uint8_t* rowDn,
                           const uint8_t* colA, const uint8_t* colB, int tp) {
    (void)row;
    const uint32_t t4 = 0x01010101u * static_cast<uint32_t>(tp);
    uint32_t out = 0;
    for (size_t g = 0; g < 8; ++g) {
        const size_t o = g * 4;
        const uint32_t a = loadBytes4(colA + o), b = loadBytes4(colB + o);
        const uint32_t u = loadBytes4(rowUp + o), d = loadBytes4(rowDn + o);
        const uint32_t dh = __uqsub8(a, b) | __uqsub8(b, a);
        const uint32_t dv = __uqsub8(d, u) | __uqsub8(u, d);
        (void)__usub8(dh, dv);  // GE[i]: dh_i >= dv_i
        const uint32_t m = (C == EdgeCombine::Or) ? __sel(dh, dv) : __sel(dv, dh);
        (void)__usub8(m, t4);   // GE[i]: m_i >= t
        const uint32_t mask = __sel(0x01010101u, 0u);
        out |= ((mask * 0x01020408u) >> 24) << o;
    }
    return out;
}

// ==================================================================
// ON ALIGNED ROWS, NO UNALIGNED LOAD AT ALL.
//
// `edgeMask32` reads the horizontal neighbours at `x - 1` and `x + 1`, and on this
// core an unaligned word load takes 3 cycles where an aligned one issues two to a
// cycle: two of them per four pixels were a quarter of the row. When the three rows
// are 4-byte aligned, the neighbours are funnel-shifted out of the middle row's own
// aligned words instead -- `(c >> 8) | (q << 24)` is the word one byte on, and each
// half is one shift that pairs with a DSP instruction.
//
// The rest is shaped by the same core. A DSP SIMD instruction issues one a cycle and
// `uqsub8` takes two to deliver, so the thresholds stay on the GE flags, whose
// `usub8` and `sel` deliver in one. `sel` picks the lane's bit out of a constant
// directly -- bit `i` for the first four pixels, bit `4 + i` for the next four -- and
// one `usad8` against zero sums the four lanes into the eight pixels' byte.

/// @brief Force the unaligned arm on rows the aligned one could serve, so the board
/// can time the two in one image. **INTERNAL.**
inline bool& edgeAlignedRowEnabled() {
    static bool on = true;
    return on;
}

/// @brief Eight pixels of the predicate from aligned words, as eight bits. **INTERNAL.**
/// @param p,c,q,n The middle row's words at `x - 4`, `x`, `x + 4`, `x + 8`.
/// @param u0,u1,d0,d1 The rows above and below, at `x` and `x + 4`.
/// @param t4 The relation-adjusted threshold in every byte lane.
template <EdgeCombine C>
inline uint32_t edgeByte8(uint32_t p, uint32_t c, uint32_t q, uint32_t n, uint32_t u0,
                          uint32_t u1, uint32_t d0, uint32_t d1, uint32_t t4) {
    const uint32_t a0 = (c >> 8) | (q << 24), b0 = (c << 8) | (p >> 24);
    const uint32_t a1 = (q >> 8) | (n << 24), b1 = (q << 8) | (c >> 24);
    const uint32_t h0 = __uqsub8(a0, b0) | __uqsub8(b0, a0);
    const uint32_t v0 = __uqsub8(d0, u0) | __uqsub8(u0, d0);
    const uint32_t h1 = __uqsub8(a1, b1) | __uqsub8(b1, a1);
    const uint32_t v1 = __uqsub8(d1, u1) | __uqsub8(u1, d1);
    constexpr uint32_t kLo = 0x08040201u, kHi = 0x80402010u;
    (void)__usub8(h0, t4);  // GE[i]: h0_i >= t
    uint32_t s = __sel(kLo, 0u);
    (void)__usub8(v0, t4);
    s = (C == EdgeCombine::Or) ? __sel(kLo, s) : __sel(s, 0u);
    const uint32_t sHi = s | kHi;
    (void)__usub8(h1, t4);
    uint32_t r = __sel(sHi, s);
    (void)__usub8(v1, t4);
    r = (C == EdgeCombine::Or) ? __sel(sHi, r) : __sel(r, s);
    return __usad8(r, 0u);
}

// GCC's scheduling pass before register allocation hoists the loads of this loop
// until its row pointers and constants no longer fit the core's registers, and the
// spills then cost more than the scheduling gains: the row measured 1.19x faster with
// the pass off. Clang has no per-function form of the switch.
#if defined(__GNUC__) && !defined(__clang__)
#define BINCV_EDGE_ALIGNED_ROW __attribute__((noinline, optimize("no-schedule-insns")))
#else
#define BINCV_EDGE_ALIGNED_ROW __attribute__((noinline))
#endif

/// @brief One output row from 4-byte-aligned rows, eight pixels at a time. **INTERNAL.**
/// @pre The three rows are 4-byte aligned, `width` is a multiple of 8 and at least 34,
/// and `tp <= 255`.
/// @note Every load is inside the three rows. The words before column 0 and after the
/// last are taken as zero, and only the two border columns read them, which are then
/// set from the scalar predicate as in `edgeRowVector`.
template <EdgeCombine C, EdgeRelation R, typename SrcT, typename WordType>
BINCV_EDGE_ALIGNED_ROW void edgeRowAligned(const SrcT* up, const SrcT* mid, const SrcT* dn,
                                           size_t width, WordType* out, EdgeDiff<SrcT> t,
                                           int tp) {
    const uint8_t* u = reinterpret_cast<const uint8_t*>(up);
    const uint8_t* m = reinterpret_cast<const uint8_t*>(mid);
    const uint8_t* d = reinterpret_cast<const uint8_t*>(dn);
    const uint32_t t4 = 0x01010101u * static_cast<uint32_t>(tp);
    const size_t last = width / 8 - 1;  // the chunk whose right neighbour is outside
    uint32_t p = 0, c = loadBytes4(m);
    const auto chunk = [&](size_t j, bool final) {
        const size_t o = j * 8;
        const uint32_t q = loadBytes4(m + o + 4);
        const uint32_t n = final ? 0u : loadBytes4(m + o + 8);
        const uint32_t b = edgeByte8<C>(p, c, q, n, loadBytes4(u + o), loadBytes4(u + o + 4),
                                        loadBytes4(d + o), loadBytes4(d + o + 4), t4);
        p = q;
        c = n;
        return b;
    };
    size_t w = 0;
    for (; 4 * w + 4 <= last; ++w) {
        const uint32_t b0 = chunk(4 * w, false);
        const uint32_t b1 = chunk(4 * w + 1, false);
        const uint32_t b2 = chunk(4 * w + 2, false);
        const uint32_t b3 = chunk(4 * w + 3, false);
        out[w] = static_cast<WordType>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
    }
    // The last word: its chunks past the row are never computed, so its padding bytes
    // are zero.
    uint32_t acc = 0;
    for (size_t k = 0; 4 * w + k <= last; ++k) {
        acc |= chunk(4 * w + k, 4 * w + k == last) << (8 * k);
    }
    out[w] = static_cast<WordType>(acc);
    for (const size_t bx : {size_t{0}, width - 1}) {
        const bool on = edgeAt<C, R, EdgeSpatial::Wide, SrcT>(up, mid, dn, width, bx, t);
        WordType& wd = out[bx / 32];
        const WordType bit = static_cast<WordType>(WordType{1} << (bx % 32));
        wd = on ? static_cast<WordType>(wd | bit) : static_cast<WordType>(wd & ~bit);
    }
}
#endif

#if defined(BINCV_EDGE_AVX2) || defined(BINCV_EDGE_NEON) || defined(BINCV_EDGE_DSP)
/// @brief Whether the vector row serves this call. **INTERNAL.** Only the shipped
/// shape -- 8-bit source, 32-bit words, the WIDE central difference -- because that is
/// what the sensor stage runs; every other combination keeps the scalar body.
template <EdgeSpatial S, typename SrcT, typename WordType>
inline constexpr bool kEdgeVectorShape =
    S == EdgeSpatial::Wide && sizeof(SrcT) == 1 && sizeof(WordType) == 4;

/// @brief One output row, thirty-two pixels at a time. **INTERNAL.**
/// @param tp The threshold ALREADY adjusted for the relation, as `edgeMask32` takes it.
/// @pre `width >= 34`, and the shape is `kEdgeVectorShape`.
///
/// **NOTHING OUTSIDE THE THREE ROWS IS READ**, so the rows may be anywhere -- three
/// rows of a frame, or three slots of a ring that holds nothing else. A word's mask
/// reads columns `x - 1` through `x + 32`, so two words are taken from a shifted
/// anchor instead: the first from column 1, shifted up one, and any word too close
/// to the right edge from column `width - 33`, shifted down into place. The two
/// columns whose horizontal neighbour is outside the row -- 0 and `width - 1` --
/// are then set from the scalar predicate, two pixels a row against the sixty-four
/// that falling back for whole words would cost.
template <EdgeCombine C, EdgeRelation R, typename SrcT, typename WordType>
inline void edgeRowVector(const SrcT* up, const SrcT* mid, const SrcT* dn, size_t width,
                          WordType* out, EdgeDiff<SrcT> t, int tp) {
    constexpr size_t kBits = bitsPerWord<WordType>();
    const uint8_t* u = reinterpret_cast<const uint8_t*>(up);
    const uint8_t* m = reinterpret_cast<const uint8_t*>(mid);
    const uint8_t* d = reinterpret_cast<const uint8_t*>(dn);
    const auto mask32 = [&](size_t xs) {
        return edgeMask32<C>(u + xs, m + xs, d + xs, m + xs + 1, m + xs - 1, tp);
    };
    const size_t words = (width + kBits - 1) / kBits;
    out[0] = static_cast<WordType>(mask32(1) << 1);
    const size_t tailAnchor = width - (kBits + 1);
    for (size_t w = 1; w < words; ++w) {
        const size_t x = w * kBits;
        if (x + kBits + 1 <= width) {
            out[w] = static_cast<WordType>(mask32(x));
        } else {
            // The padding bits shift in as zero, which is the invariant that would
            // otherwise need a mask.
            const size_t shift = x - tailAnchor;
            out[w] = shift >= kBits ? WordType{0}
                                    : static_cast<WordType>(mask32(tailAnchor) >> shift);
        }
    }
    for (const size_t bx : {size_t{0}, width - 1}) {
        const bool on = edgeAt<C, R, EdgeSpatial::Wide, SrcT>(up, mid, dn, width, bx, t);
        WordType& wd = out[bx / kBits];
        const WordType bit = static_cast<WordType>(WordType{1} << (bx % kBits));
        wd = on ? static_cast<WordType>(wd | bit) : static_cast<WordType>(wd & ~bit);
    }
}
#endif

/// @brief One output row of `edgeThreshold`, from the three source rows it reads.
/// **INTERNAL** -- the body every row of every entry point goes through, so the
/// whole-frame call and the streaming one (ops/medianEdgeStream.hpp) cannot differ.
/// @param up,mid,dn With the vertical border already applied: for `BORDER_REFLECT_101`
/// the row above row 0 is row 1, and the row below the last is the one before it.
template <EdgeCombine C, EdgeRelation R, EdgeSpatial S, typename SrcT, typename WordType>
inline void edgeRow(const SrcT* up, const SrcT* mid, const SrcT* dn, size_t width,
                    WordType* out, SrcT threshold) {
    const EdgeDiff<SrcT> t = static_cast<EdgeDiff<SrcT>>(threshold);
#if defined(BINCV_EDGE_AVX2) || defined(BINCV_EDGE_NEON) || defined(BINCV_EDGE_DSP)
    if constexpr (kEdgeVectorShape<S, SrcT, WordType>) {
        const int tp = static_cast<int>(t) + (R == EdgeRelation::Ge ? 0 : 1);
        if (hasEdgeSimd() && width >= bitsPerWord<WordType>() + 2 && tp <= 255) {
#if defined(BINCV_EDGE_DSP)
            const uintptr_t rows = reinterpret_cast<uintptr_t>(up) |
                                   reinterpret_cast<uintptr_t>(mid) |
                                   reinterpret_cast<uintptr_t>(dn);
            if (edgeAlignedRowEnabled() && (rows & 3u) == 0 && width % 8 == 0) {
                edgeRowAligned<C, R, SrcT, WordType>(up, mid, dn, width, out, t, tp);
                return;
            }
#endif
            edgeRowVector<C, R, SrcT, WordType>(up, mid, dn, width, out, t, tp);
            return;
        }
    }
#endif
    edgeRowScalar<C, R, S, SrcT, WordType>(up, mid, dn, width, out, t);
}

} // namespace impl

/// @brief Gradient-magnitude edge extraction straight into bits. **API TIER 3.**
///
/// @param src Row-major, `srcStride` ELEMENTS between rows (not bytes).
/// @param t Threshold, in source units.
///
/// **THE DEFAULTS ARE THE REFERENCE.** `edgeThreshold(src, w, h, stride, dst, 17)`
/// with no template arguments is the reference pipeline's edge filter at threshold 17.
///
/// @note 8-bit-in, 1-bit-out, and **the byte never exists**: the comparison yields a
/// boolean per pixel which goes straight into a word. Computing an 8-bit edge
/// image and packing it afterwards would be two passes and an intermediate.
/// @note `dst`'s padding bits are zero on return.
/// @note Never allocates and never throws.
template <EdgeCombine C = EdgeCombine::Or, EdgeRelation R = EdgeRelation::Ge,
          EdgeSpatial S = EdgeSpatial::Wide, typename SrcT = uint8_t,
          typename WordType = uint32_t>
inline void edgeThreshold(const SrcT* src, size_t width, size_t height, size_t srcStride,
                          BinMatView<WordType> dst, SrcT t) {
    BINCV_ASSERT(width == dst.width && height == dst.height,
                 "edgeThreshold: src and dst must have the same dimensions");
    BINCV_ASSERT(impl::strideCoversARow<WordType>(dst.width, dst.height, dst.stride),
                 "edgeThreshold: dst's stride must cover a whole row");
    if (dst.width == 0 || dst.height == 0) return;
    BINCV_ASSERT(src != nullptr && dst.ptr != nullptr,
                 "edgeThreshold: a non-empty image needs non-null pointers");

    // Row by row through the one body every entry point shares. The vertical border
    // is applied here, by which rows are handed over; the horizontal one is the row
    // body's.
    for (size_t y = 0; y < height; ++y) {
        const long long yy = static_cast<long long>(y);
        const SrcT* up = src + impl::reflect101Edge(yy - 1, height) * srcStride;
        const SrcT* dn = src + impl::reflect101Edge(yy + 1, height) * srcStride;
        impl::edgeRow<C, R, S, SrcT, WordType>(up, src + y * srcStride, dn, width, dst.row(y),
                                               t);
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
