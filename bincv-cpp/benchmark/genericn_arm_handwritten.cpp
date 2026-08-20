// T3.9 / E-4 (X-21) -- ARM 3: THE HAND-WRITTEN BINARY-ONLY REFERENCE.
//
// This is the arm the experiment turns on. Arms 1 and 2 are both binCV, so
// comparing them shows only whether the specialization is SELECTED. Only a
// control written with no genericity at all shows whether genericity COSTS
// anything, and the decision rule is written against this arm by name --
// "specialized paths within 5% of a hand-written binary-only implementation".
//
// WHAT "NO GENERICITY" MEANS HERE, CONCRETELY. This file includes NO binCV
// header. It has:
//
//   * no QuantMat, SignedQuantMat, BinMat or TernaryMat -- the buffers are
//     `uint32_t*` plus a stride, and the two-plane destination layout is written
//     out as pointer arithmetic rather than named by a container;
//   * no BinMatView / BinMatConstView -- nothing is bundled into a struct and
//     unpacked again;
//   * no template over N. `N` does not appear. There is no plane array, no plane
//     loop, no `mag[N]` / `a[N]` / `b[N]` register file indexed by plane;
//   * no template over the word type. `uint32_t` and 32 are written literally, as
//     a person who has picked a word width writes them;
//   * no route selector, no `if constexpr`, no ForceGeneric -- there is exactly
//     one spelling of the arithmetic and it is the ternary one, because binary is
//     the only case this file knows about;
//   * no argument contract -- no BINCV_ASSERT, no aliasing predicate, no stride
//     validation, no BorderType parameter. BORDER_REFLECT_101 is compiled in,
//     because that is the border the VIO frontend needs (ops/derivative.hpp's
//     header says why) and a hand-written kernel supports the one it needs;
//   * no impl:: helpers -- bitMask, rowTailMask, minRowWords, borderIndex,
//     regionFromExtent and visitRowWords are all re-derived inline as the two or
//     three lines of arithmetic they are.
//
// The word-level ARITHMETIC is necessarily the same in all three arms, and it
// would be dishonest to pretend otherwise: `mag = a ^ b`, `sign = b & ~a` is the
// ternary difference, and there is no second way to compute it. That is exactly
// why this arm is a fair control -- with the arithmetic held equal, a difference
// in the numbers is the machinery AROUND the arithmetic, which is what E-4 asks
// about. Note that even the spelling here is arrived at independently: the
// library writes `pos = a & ~b`, `neg = b & ~a`, `mag = pos | neg`, which is the
// same function of the same inputs in four operations where this file uses three.
//
// ONE DELIBERATE EXCEPTION, AND IT IS CONFOUND CONTROL RATHER THAN CONVENIENCE.
// The reductions below call `__builtin_popcountll` on a widened word, exactly as
// impl::popcountWord does, and NOT the narrower `__builtin_popcount` that a
// uint32_t-only author might reach for. X-7 / X-7b establish that binCV builds
// with no -march, so the popcount LOWERING (a PLT call on x86_64; fmov/cnt/
// uaddlv/fmov on aarch64) dominates a bulk count -- and X-21's pre-registration
// names that lowering as a confound to be HELD FIXED, not varied. Two arms that
// differ in the builtin would be measuring the builtin. The narrow spelling is
// not measured here and no claim is made about it.

#include <cstddef>
#include <cstdint>

#include "genericn_arms.hpp"

namespace {

/// @brief Population count of one 32-bit word, lowered identically to the
///        library's. See the confound note in the file header.
inline size_t popcount32(uint32_t w) {
    return static_cast<size_t>(__builtin_popcountll(static_cast<unsigned long long>(w)));
}

/// @brief Mask of the bits of a word that lie at column offset >= k.
inline uint32_t headMaskFrom(unsigned k) {
    return static_cast<uint32_t>(~((UINT32_C(1) << k) - UINT32_C(1)));
}

/// @brief Mask of the low `k` bits, k in [1, 32].
inline uint32_t lowMask(unsigned k) {
    return k >= 32u ? UINT32_C(0xFFFFFFFF)
                    : static_cast<uint32_t>((UINT32_C(1) << k) - UINT32_C(1));
}

/// @brief d/dx and d/dy of a binary frame, reflect-101 border, ternary output.
///
/// Correlation, not convolution: the +1 tap is the RIGHT neighbour and the row
/// BELOW, matching cv::filter2D with [-1, 0, 1]. Under reflect-101 the neighbour
/// of column -1 is column 1 and the neighbour of column `width` is column
/// width-2, which is why the first and last column of dx come out exactly zero.
///
/// The destination is one buffer per axis holding two planes back to back:
/// magnitude at offset 0, sign at offset height*stride. Written out here rather
/// than named by a container.
void derivativeHandWritten(const t39::Word* src, size_t strideWords, int width, int height,
                           t39::Word* dstX, t39::Word* dstY) {
    if (width <= 0 || height <= 0) return;

    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    const size_t words = (w + 31u) / 32u;
    const uint32_t tailMask = lowMask(static_cast<unsigned>(w % 32u == 0u ? 32u : w % 32u));
    const uint32_t lastLiveBit = static_cast<uint32_t>(UINT32_C(1) << ((w - 1u) % 32u));

    uint32_t* magX = dstX;
    uint32_t* sgnX = dstX + h * strideWords;
    uint32_t* magY = dstY;
    uint32_t* sgnY = dstY + h * strideWords;

    // ---- d/dx -------------------------------------------------------------
    // Reflect-101 columns, resolved once for the frame: they depend on the width,
    // not on the row.
    const size_t leftCol = (w > 1u) ? size_t{1} : size_t{0};
    const size_t rightCol = (w > 1u) ? w - 2u : size_t{0};

    for (size_t y = 0; y < h; ++y) {
        const uint32_t* s = src + y * strideWords;
        uint32_t* m = magX + y * strideWords;
        uint32_t* g = sgnX + y * strideWords;

        // The left border carried as a synthetic "word before word 0", so the
        // shift recurrence below needs no test for i == 0.
        const bool leftBit = ((s[leftCol / 32u] >> (leftCol % 32u)) & UINT32_C(1)) != 0;
        uint32_t prev = leftBit ? UINT32_C(0x80000000) : UINT32_C(0);

        for (size_t i = 0; i < words; ++i) {
            const bool last = (i + 1u == words);
            const uint32_t cur = s[i];
            const uint32_t nxt = last ? UINT32_C(0) : s[i + 1u];

            uint32_t a = static_cast<uint32_t>((cur >> 1) | (nxt << 31));
            const uint32_t b = static_cast<uint32_t>((cur << 1) | (prev >> 31));
            prev = cur;

            if (last) {
                const bool bit = ((s[rightCol / 32u] >> (rightCol % 32u)) & UINT32_C(1)) != 0;
                a = static_cast<uint32_t>((a & ~lastLiveBit) |
                                          (bit ? lastLiveBit : UINT32_C(0)));
            }

            const uint32_t mm = static_cast<uint32_t>(a ^ b);
            const uint32_t ss = static_cast<uint32_t>(b & ~a);
            m[i] = last ? static_cast<uint32_t>(mm & tailMask) : mm;
            g[i] = last ? static_cast<uint32_t>(ss & tailMask) : ss;
        }
    }

    // ---- d/dy -------------------------------------------------------------
    // A vertical tap is a row index and moves no bits, so there is no shifting
    // and no per-word border work at all.
    for (size_t y = 0; y < h; ++y) {
        size_t ya = y + 1u;
        if (ya >= h) ya = (h > 1u) ? h - 2u : size_t{0};
        size_t yb = (y == 0u) ? ((h > 1u) ? size_t{1} : size_t{0}) : y - 1u;

        const uint32_t* ra = src + ya * strideWords;
        const uint32_t* rb = src + yb * strideWords;
        uint32_t* m = magY + y * strideWords;
        uint32_t* g = sgnY + y * strideWords;

        for (size_t i = 0; i < words; ++i) {
            const uint32_t a = ra[i];
            const uint32_t b = rb[i];
            const uint32_t mm = static_cast<uint32_t>(a ^ b);
            const uint32_t ss = static_cast<uint32_t>(b & ~a);
            const bool last = (i + 1u == words);
            m[i] = last ? static_cast<uint32_t>(mm & tailMask) : mm;
            g[i] = last ? static_cast<uint32_t>(ss & tailMask) : ss;
        }
    }
}

/// @brief Set pixels in the whole frame. Padding past `width` is masked off, as
///        it must be for any word-wise count (D-13).
size_t countWholeHandWritten(const t39::Word* src, size_t strideWords, int width, int height) {
    if (width <= 0 || height <= 0) return 0;

    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    const size_t words = (w + 31u) / 32u;
    const uint32_t tailMask = lowMask(static_cast<unsigned>(w % 32u == 0u ? 32u : w % 32u));

    size_t total = 0;
    for (size_t y = 0; y < h; ++y) {
        const uint32_t* r = src + y * strideWords;
        for (size_t i = 0; i + 1u < words; ++i) total += popcount32(r[i]);
        total += popcount32(static_cast<uint32_t>(r[words - 1u] & tailMask));
    }
    return total;
}

/// @brief One word's contribution to the four covariance numbers.
/// @note Split out so the row loop below can run the interior words with a
///       compile-time all-ones mask instead of testing for the head and the tail
///       word on every iteration -- the head/tail special case belongs outside
///       the loop, which is the same shape a careful hand-written kernel takes.
inline void covarianceWord(const uint32_t* mx, const uint32_t* my, const uint32_t* sx,
                           const uint32_t* sy, size_t i, uint32_t mask, size_t& xx, size_t& yy,
                           size_t& both, size_t& whenSet) {
    const uint32_t a = static_cast<uint32_t>(mx[i] & mask);
    const uint32_t b = static_cast<uint32_t>(my[i] & mask);
    const uint32_t ab = static_cast<uint32_t>(a & b);
    const uint32_t sel = static_cast<uint32_t>(sx[i] ^ sy[i]);
    xx += popcount32(a);
    yy += popcount32(b);
    both += popcount32(ab);
    whenSet += popcount32(static_cast<uint32_t>(ab & sel));
}

/// @brief The 2x2 gradient covariance over one window, from the two ternary
///        derivatives. One pass; the four numbers come out of the same loads.
///
/// `whenClear` is popcount(both) - popcount(both & selector) rather than
/// popcount(both & ~selector): the complement of a masked word has its padding
/// bits set, and a second AND against the region mask to undo that is work the
/// subtraction does not need.
t39::Cov covarianceWindowHandWritten(const t39::Word* dx, const t39::Word* dy, size_t strideWords,
                                     int width, int height, int wx, int wy, int wsize) {
    t39::Cov out;
    if (width <= 0 || height <= 0 || wsize <= 0) return out;

    // Clip the window to the frame, in long long so a negative origin or a large
    // extent cannot wrap on the way to an index.
    const long long lx0 = (wx > 0) ? wx : 0;
    const long long lx1 = (static_cast<long long>(wx) + wsize < width)
                              ? static_cast<long long>(wx) + wsize
                              : width;
    const long long ly0 = (wy > 0) ? wy : 0;
    const long long ly1 = (static_cast<long long>(wy) + wsize < height)
                              ? static_cast<long long>(wy) + wsize
                              : height;
    if (lx0 >= lx1 || ly0 >= ly1) return out;

    const size_t x0 = static_cast<size_t>(lx0);
    const size_t x1 = static_cast<size_t>(lx1);
    const size_t y0 = static_cast<size_t>(ly0);
    const size_t y1 = static_cast<size_t>(ly1);

    const size_t firstWord = x0 / 32u;
    const size_t lastWord = (x1 - 1u) / 32u;
    const uint32_t headMask = headMaskFrom(static_cast<unsigned>(x0 % 32u));
    const uint32_t tailMask = lowMask(static_cast<unsigned>(((x1 - 1u) % 32u) + 1u));

    const size_t planeWords = static_cast<size_t>(height) * strideWords;
    const uint32_t* magX = dx;
    const uint32_t* sgnX = dx + planeWords;
    const uint32_t* magY = dy;
    const uint32_t* sgnY = dy + planeWords;

    size_t xx = 0, yy = 0, both = 0, whenSet = 0;

    for (size_t y = y0; y < y1; ++y) {
        const uint32_t* mx = magX + y * strideWords;
        const uint32_t* my = magY + y * strideWords;
        const uint32_t* sx = sgnX + y * strideWords;
        const uint32_t* sy = sgnY + y * strideWords;

        if (firstWord == lastWord) {
            covarianceWord(mx, my, sx, sy, firstWord,
                           static_cast<uint32_t>(headMask & tailMask), xx, yy, both, whenSet);
            continue;
        }
        covarianceWord(mx, my, sx, sy, firstWord, headMask, xx, yy, both, whenSet);
        for (size_t i = firstWord + 1u; i < lastWord; ++i) {
            covarianceWord(mx, my, sx, sy, i, UINT32_C(0xFFFFFFFF), xx, yy, both, whenSet);
        }
        covarianceWord(mx, my, sx, sy, lastWord, tailMask, xx, yy, both, whenSet);
    }

    out.xx = xx;
    out.yy = yy;
    out.whenSet = whenSet;
    out.whenClear = both - whenSet;
    return out;
}

const t39::Arm kArm{"hand-written", &derivativeHandWritten, &countWholeHandWritten,
                    &covarianceWindowHandWritten};

}  // namespace

namespace t39 {
const Arm& handWrittenArm() { return kArm; }
}  // namespace t39
