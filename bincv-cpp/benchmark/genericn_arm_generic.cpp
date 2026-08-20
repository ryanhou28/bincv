// T3.9 / E-4 (X-21) -- ARM 1: the GENERIC-N route, instantiated at N = 1.
//
// Everything here is written as if N were unknown: the derivative takes the
// ripple-borrow subtract that serves any N (impl::signedDifferenceRipple, reached
// by ForceGeneric = true), and both reductions are wrapped in a compile-time loop
// over the N planes with the 2^p place weights an N-bit reading needs.
//
// N IS THEN FIXED TO 1 BY INSTANTIATION, not by a different spelling. That is the
// whole point of the arm: if the plane loop and the ripple collapse at N = 1, the
// numbers here match arm 2's, and "arbitrary N is free" is a measurement rather
// than an argument about what the compiler probably does.
//
// A NOTE ABOUT THE TWO REDUCTIONS, SO THE RESULT IS NOT OVER-READ. ops/reduce.hpp
// has no N-generic route to force -- reductions are per-PLANE by construction
// (D-5, D-6), and there is no `ForceGeneric` to flip. So the generic arm's
// reductions are the plane loop a generic-N CALLER writes, which at N = 1 runs
// one iteration with weight 1 and calls exactly what arm 2 calls. Those two rows
// are therefore expected to be identical, and the reduction comparison that
// carries information is arm 2 against arm 3. Stated here rather than discovered
// in the numbers.

#include <cstddef>

#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/reduce.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "genericn_arms.hpp"

namespace {

using t39::Word;

/// @brief The number of bit-planes this arm is instantiated at.
/// @note A named constant so that the generic code below reads as generic. Change
///       it and everything here still compiles; that is the property being tested.
constexpr size_t kN = 1;

/// @brief dx and dy through the FORCED-GENERIC route.
/// @note impl::derivativeContainer with ForceGeneric = true is the same kernel the
///       shipped path uses with it false -- one template argument apart -- so this
///       arm differs from arm 2 in exactly the decision under test and in nothing
///       else.
/// @note The const_cast is the benchmark's, not the library's: QuantMat's wrapping
///       constructor takes a mutable pointer because a wrapped matrix is normally
///       written through, and the driver owns this buffer as non-const storage. No
///       kernel below writes through `src`.
void derivativeGeneric(const Word* src, size_t strideWords, int width, int height, Word* dstX,
                       Word* dstY) {
    bincv::QuantMat<kN, Word> in(const_cast<Word*>(src), width, height, strideWords);
    bincv::SignedQuantMat<kN, Word> outX(dstX, width, height, strideWords);
    bincv::SignedQuantMat<kN, Word> outY(dstY, width, height, strideWords);
    bincv::impl::derivativeContainer<kN, Word, /*ForceGeneric=*/true, /*Horizontal=*/true>(
        in, outX, bincv::BORDER_REFLECT_101, false);
    bincv::impl::derivativeContainer<kN, Word, /*ForceGeneric=*/true, /*Horizontal=*/false>(
        in, outY, bincv::BORDER_REFLECT_101, false);
}

/// @brief Sum of pixel VALUES over the whole frame, for an N-bit image.
/// @note The generic-N reading of "count": plane p carries place value 2^p, so the
///       sum is a weighted sum of plane popcounts. At N = 1 the weight is 1 and
///       the sum is the pixel count, which is what the other two arms return.
size_t countWholeGeneric(const Word* src, size_t strideWords, int width, int height) {
    const bincv::QuantMat<kN, Word> in(const_cast<Word*>(src), width, height, strideWords);
    size_t total = 0;
    for (size_t p = 0; p < kN; ++p) {
        total += (size_t{1} << p) * bincv::countNonZero<Word>(in.constPlane(p));
    }
    return total;
}

/// @brief The LK covariance over one window, summed over the magnitude planes.
/// @note The place weight on a product of plane p with plane p is 2^(2p), which is
///       1 at p = 0. Only the diagonal terms are formed: the cross-plane products
///       an N-bit covariance also needs do not exist in binCV today (X-20 records
///       that), and inventing them here would measure a kernel that does not ship.
///       What this arm tests is the PLANE LOOP around the shipping reduction.
t39::Cov covarianceWindowGeneric(const Word* dx, const Word* dy, size_t strideWords, int width,
                                 int height, int wx, int wy, int wsize) {
    const bincv::SignedQuantMat<kN, Word> gx(const_cast<Word*>(dx), width, height, strideWords);
    const bincv::SignedQuantMat<kN, Word> gy(const_cast<Word*>(dy), width, height, strideWords);
    const bincv::Rect window(wx, wy, wsize, wsize);

    t39::Cov out;
    for (size_t p = 0; p < kN; ++p) {
        const size_t weight = size_t{1} << (2 * p);
        const bincv::CovarianceCount c = bincv::countCovariance<Word>(
            gx.constMagnitude(p), gy.constMagnitude(p), gx.constSign(), gy.constSign(), window);
        out.xx += weight * c.xx;
        out.yy += weight * c.yy;
        out.whenClear += weight * c.xy.whenClear;
        out.whenSet += weight * c.xy.whenSet;
    }
    return out;
}

const t39::Arm kArm{"generic-N", &derivativeGeneric, &countWholeGeneric,
                    &covarianceWindowGeneric};

}  // namespace

namespace t39 {
const Arm& genericArm() { return kArm; }
}  // namespace t39
