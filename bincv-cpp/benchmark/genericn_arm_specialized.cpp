// T3.9 / E-4 (X-21) -- ARM 2: WHAT SHIPS.
//
// The public entry points, called the way a caller calls them: bincv::derivativeX
// / derivativeY on a QuantMat<1> into a SignedQuantMat<1> -- which is TernaryMat,
// the same type spelled for the reader -- and the single-plane reductions on the
// planes those containers name.
//
// Against arm 1 this differs in ONE template argument (ForceGeneric false rather
// than true, which selects impl::ternaryDifference's three word operations over
// impl::signedDifferenceRipple's borrow chain) and in the absence of a plane loop
// around the reductions. Nothing else about the two files differs, deliberately:
// a difference in the numbers is then attributable to the route and not to how
// the call site was written.

#include <cstddef>

#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/reduce.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "genericn_arms.hpp"

namespace {

using t39::Word;

/// @brief dx and dy through the SHIPPED route: BinMat in, TernaryMat out.
/// @note bincv::BinMat<Word> IS bincv::QuantMat<1, Word> and bincv::TernaryMat<Word>
///       IS bincv::SignedQuantMat<1, Word> (core/types.hpp, quantMat.hpp). The
///       aliases are used here because they are what a caller writes.
/// @note The const_cast is the benchmark's; see arm 1's note. Nothing writes `src`.
void derivativeSpecialized(const Word* src, size_t strideWords, int width, int height, Word* dstX,
                           Word* dstY) {
    bincv::BinMat<Word> in(const_cast<Word*>(src), width, height, strideWords);
    bincv::TernaryMat<Word> outX(dstX, width, height, strideWords);
    bincv::TernaryMat<Word> outY(dstY, width, height, strideWords);
    bincv::derivativeX<1, Word>(in, outX, bincv::BORDER_REFLECT_101, false);
    bincv::derivativeY<1, Word>(in, outY, bincv::BORDER_REFLECT_101, false);
}

/// @brief Set pixels in the whole frame, the single-plane way.
size_t countWholeSpecialized(const Word* src, size_t strideWords, int width, int height) {
    const bincv::BinMat<Word> in(const_cast<Word*>(src), width, height, strideWords);
    return bincv::countNonZero<Word>(in.constPlane(0));
}

/// @brief The LK covariance over one window: ARCHITECTURE 7.5's call, verbatim.
/// @note This is the five-argument fused form T3.6 calls -- magnitude x,
///       magnitude y, the two sign planes, the window -- and it is the shape
///       X-11 / X-11b settled.
t39::Cov covarianceWindowSpecialized(const Word* dx, const Word* dy, size_t strideWords, int width,
                                     int height, int wx, int wy, int wsize) {
    const bincv::TernaryMat<Word> gx(const_cast<Word*>(dx), width, height, strideWords);
    const bincv::TernaryMat<Word> gy(const_cast<Word*>(dy), width, height, strideWords);
    const bincv::CovarianceCount c =
        bincv::countCovariance<Word>(gx.constMagnitude(0), gy.constMagnitude(0), gx.constSign(),
                                     gy.constSign(), bincv::Rect(wx, wy, wsize, wsize));
    t39::Cov out;
    out.xx = c.xx;
    out.yy = c.yy;
    out.whenClear = c.xy.whenClear;
    out.whenSet = c.xy.whenSet;
    return out;
}

const t39::Arm kArm{"specialized", &derivativeSpecialized, &countWholeSpecialized,
                    &covarianceWindowSpecialized};

}  // namespace

namespace t39 {
const Arm& specializedArm() { return kArm; }
}  // namespace t39
