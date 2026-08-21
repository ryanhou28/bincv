// X-31 arms B1 and B2 share this body VERBATIM by inclusion, differing only in the
// `Skip` template argument, so the sparsity skip is the ONLY difference between
// them and their separation measures exactly that.
//
// ===========================================================================
// THE REFORMULATION
//
// For blockSize 3 every quantity the response needs is a 3x3 BOX SUM OF A
// BIT-PLANE over the window `Rect(x-1, y-1, 3, 3)` clipped to the frame:
//
//     xx        = sum over the window of magX
//     yy        = sum over the window of magY
//     crossTerm = sum of (magX & magY & ~sel) - sum of (magX & magY & sel)
//                 with sel = signX ^ signY
//
// A box sum of bits is word-parallel. Horizontally, `h = a(x-1) + a(x) + a(x+1)`
// is ONE FULL ADDER into two bit-planes (0..3). Vertically, three of those sum
// into four bit-planes (0..9, and 3+3+3 = 9 fits). So 32 pixels are summed per
// word, against the shipped form's ~12 popcounts PER PIXEL over a window with
// three valid bits in thirty-two.
//
// EXACTNESS AT THE BORDER, WHICH IS WHERE THIS WOULD FAIL IF IT FAILED.
// A clipped window counts only pixels inside the frame. Here:
//   * a row above row 0 or below the last row is a ZERO plane, so it contributes
//     nothing -- identical to clipping it away;
//   * the bit shifted in from a word that does not exist is 0, so at x = 0 the
//     x-1 term vanishes and at the right edge the x+1 term does;
//   * bits past `width` in the trailing word are zero by D-13, so they cannot
//     contribute either.
// The three counts are therefore the same EXACT INTEGERS the per-pixel form
// produces, and `minEigenValue` turns the same integers into the same float.
// ===========================================================================

#include "cornerresp_arms.hpp"

namespace {

using W = uint32_t;
constexpr size_t kBits = 32;

/// One row's four source planes at word `w`, with the cross-word neighbours.
struct Planes {
    W a = 0;   ///< magX
    W b = 0;   ///< magY
    W p = 0;   ///< magX & magY & ~sel  -- the positive cross-term contribution
    W n = 0;   ///< magX & magY &  sel  -- the negative one
};

inline W wordAt(const W* row, size_t words, long long w) {
    if (row == nullptr || w < 0 || static_cast<size_t>(w) >= words) return 0;
    return row[static_cast<size_t>(w)];
}

/// Bit at column x-1 brought to position x: shift LEFT, pulling in the previous
/// word's bit 31. (Pixel x lives at bit x % 32, so x-1 is one bit lower.)
inline W shiftedLeft(W cur, W prev) { return static_cast<W>((cur << 1) | (prev >> (kBits - 1))); }
inline W shiftedRight(W cur, W next) { return static_cast<W>((cur >> 1) | (next << (kBits - 1))); }

/// h = L + C + R for one bit-plane: one full adder, two output planes (0..3).
inline void horizontal3(W l, W c, W r, W& h0, W& h1) {
    h0 = static_cast<W>(l ^ c ^ r);
    h1 = static_cast<W>((l & c) | (c & r) | (l & r));
}

/// Sum three 2-bit numbers into four planes (0..9).
inline void vertical3(const W (&a)[2], const W (&b)[2], const W (&c)[2], W (&out)[4]) {
    // a + b -> three planes
    const W s0 = static_cast<W>(a[0] ^ b[0]);
    const W k0 = static_cast<W>(a[0] & b[0]);
    const W s1 = static_cast<W>(a[1] ^ b[1] ^ k0);
    const W k1 = static_cast<W>((a[1] & b[1]) | (b[1] & k0) | (a[1] & k0));
    // (a+b) + c
    out[0] = static_cast<W>(s0 ^ c[0]);
    const W m0 = static_cast<W>(s0 & c[0]);
    out[1] = static_cast<W>(s1 ^ c[1] ^ m0);
    const W m1 = static_cast<W>((s1 & c[1]) | (c[1] & m0) | (s1 & m0));
    out[2] = static_cast<W>(k1 ^ m1);
    out[3] = static_cast<W>(k1 & m1);
}

/// The value of plane-set `p` at bit `bit`, as an integer 0..9.
inline long long valueAt(const W (&p)[4], size_t bit) {
    const W m = static_cast<W>(1u) << bit;
    return static_cast<long long>(((p[0] & m) != 0 ? 1 : 0) + ((p[1] & m) != 0 ? 2 : 0) +
                                  ((p[2] & m) != 0 ? 4 : 0) + ((p[3] & m) != 0 ? 8 : 0));
}

template <bool Skip>
void slicedImpl(const bincv::TernaryMat<W>& dx, const bincv::TernaryMat<W>& dy, int blockSize,
                bincv::ResponseMap out, size_t* skippedWords) {
    const auto magX = dx.constMagnitude(0);
    const auto magY = dy.constMagnitude(0);
    const auto sgnX = dx.constSign();
    const auto sgnY = dy.constSign();
    const size_t width = magX.width, height = magX.height;
    if (width == 0 || height == 0) return;
    (void)blockSize;   // this arm implements blockSize 3, which is the shipped default

    const size_t words = bincv::impl::minRowWords<W>(width);
    size_t skipped = 0;

    for (size_t y = 0; y < height; ++y) {
        float* dst = out.data + y * out.stride;
        // The three source rows; a row outside the frame is a null pointer and
        // therefore a zero plane, which is exactly what clipping it away does.
        const W* mxr[3]; const W* myr[3]; const W* sxr[3]; const W* syr[3];
        for (int k = -1; k <= 1; ++k) {
            const long long r = static_cast<long long>(y) + k;
            const bool ok = r >= 0 && r < static_cast<long long>(height);
            const size_t i = static_cast<size_t>(k + 1);
            mxr[i] = ok ? magX.row(static_cast<size_t>(r)) : nullptr;
            myr[i] = ok ? magY.row(static_cast<size_t>(r)) : nullptr;
            sxr[i] = ok ? sgnX.row(static_cast<size_t>(r)) : nullptr;
            syr[i] = ok ? sgnY.row(static_cast<size_t>(r)) : nullptr;
        }

        for (size_t w = 0; w < words; ++w) {
            if constexpr (Skip) {
                // A word's 32 outputs depend on source columns [32w-1, 32w+32],
                // which live in words w-1, w and w+1. If magX | magY is zero across
                // all three in all three rows, every box sum is zero, so minEig is
                // exactly 0 for all 32 pixels and no sqrt is taken.
                W any = 0;
                for (size_t i = 0; i < 3; ++i) {
                    for (long long d = -1; d <= 1; ++d) {
                        const long long ww = static_cast<long long>(w) + d;
                        any = static_cast<W>(any | wordAt(mxr[i], words, ww) |
                                             wordAt(myr[i], words, ww));
                    }
                }
                if (any == 0) {
                    const size_t lo = w * kBits;
                    const size_t hi = lo + kBits < width ? lo + kBits : width;
                    for (size_t x = lo; x < hi; ++x) dst[x] = 0.0f;
                    ++skipped;
                    continue;
                }
            }

            // Horizontal pass: four planes x three rows -> two bit-planes each.
            W hA[3][2], hB[3][2], hP[3][2], hN[3][2];
            for (size_t i = 0; i < 3; ++i) {
                Planes cur, prv, nxt;
                auto load = [&](long long ww, Planes& into) {
                    const W a = wordAt(mxr[i], words, ww);
                    const W b = wordAt(myr[i], words, ww);
                    const W sel = static_cast<W>(wordAt(sxr[i], words, ww) ^
                                                 wordAt(syr[i], words, ww));
                    const W both = static_cast<W>(a & b);
                    into.a = a;
                    into.b = b;
                    into.n = static_cast<W>(both & sel);
                    into.p = static_cast<W>(both ^ into.n);   // both & ~sel
                };
                load(static_cast<long long>(w), cur);
                load(static_cast<long long>(w) - 1, prv);
                load(static_cast<long long>(w) + 1, nxt);
                horizontal3(shiftedLeft(cur.a, prv.a), cur.a, shiftedRight(cur.a, nxt.a),
                            hA[i][0], hA[i][1]);
                horizontal3(shiftedLeft(cur.b, prv.b), cur.b, shiftedRight(cur.b, nxt.b),
                            hB[i][0], hB[i][1]);
                horizontal3(shiftedLeft(cur.p, prv.p), cur.p, shiftedRight(cur.p, nxt.p),
                            hP[i][0], hP[i][1]);
                horizontal3(shiftedLeft(cur.n, prv.n), cur.n, shiftedRight(cur.n, nxt.n),
                            hN[i][0], hN[i][1]);
            }

            W vA[4], vB[4], vP[4], vN[4];
            vertical3(hA[0], hA[1], hA[2], vA);
            vertical3(hB[0], hB[1], hB[2], vB);
            vertical3(hP[0], hP[1], hP[2], vP);
            vertical3(hN[0], hN[1], hN[2], vN);

            const size_t lo = w * kBits;
            const size_t hi = lo + kBits < width ? lo + kBits : width;
            for (size_t x = lo; x < hi; ++x) {
                const size_t bit = x - lo;
                const long long xx = valueAt(vA, bit);
                const long long yy = valueAt(vB, bit);
                const long long xy = valueAt(vP, bit) - valueAt(vN, bit);
                dst[x] = bincv::impl::minEigenValue(xx, yy, xy);
            }
        }
    }
    if (skippedWords != nullptr) *skippedWords = skipped;
}

} // namespace
