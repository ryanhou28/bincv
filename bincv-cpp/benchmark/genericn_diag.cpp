// T3.9 / E-4 (X-21) -- THE DECOMPOSITION ARM.
//
// ADDED AFTER THE HEADLINE THREE-ARM RUN, AND THAT IS RECORDED RATHER THAN
// smoothed over. The three arms of genericn_arms.hpp were committed at 066a339
// and run on the reference device before this file existed; nothing here changes
// the comparison the decision rule is written against, and the rule itself is
// untouched. This exists because the rule's second band says a regression is to
// be reported "before acting", with the remedy depending on WHERE THE COST COMES
// FROM -- and the headline run localizes nothing on its own.
//
// The headline run found the generic-N route and the specialization
// indistinguishable (within 0.1-0.9% at every size and workload) while BOTH sat
// 8-43% above the hand-written control. So the cost is not genericity in N. This
// file splits what is left into the two layers it could be:
//
//   hand-written  ->  views only  ->  specialized
//                  |              |
//                  |              +-- CONTAINER: constructing a QuantMat<1> /
//                  |                  SignedQuantMat<1> around the caller's
//                  |                  buffer and calling plane() / magnitude() /
//                  |                  sign(), once per frame.
//                  +-- KERNEL SHAPE: the ops/ kernel's own generic form -- the
//                      per-plane arrays a[N] / b[N] / m[N] / srcRow[N] passed by
//                      reference, the `for p < N` loops around every word, the
//                      argument-contract call, the impl:: helpers -- all of it
//                      compiled at N = 1.
//
// "Views only" calls exactly the kernel the specialized arm calls, through the
// public view entry points (D-5: kernels take views, never containers), with the
// views built by hand from the raw pointers. So
//
//     views_only - hand_written  =  kernel shape
//     specialized - views_only   =  container
//
// and the two add up to the gap the rule fired on.

#include <cstddef>

#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/core/view.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/reduce.hpp"
#include "genericn_arms.hpp"

namespace {

using t39::Word;

}  // namespace

namespace t39 {

/// @brief dx and dy through the SHIPPED kernel, reached as views rather than
///        through a container.
/// @note bincv::derivativeX / derivativeY, the public view entry points -- the
///       same instantiation the specialized arm reaches through
///       impl::derivativeContainer, with ForceGeneric false. The only thing
///       removed is the container.
void derivativeViewsOnly(const Word* src, size_t strideWords, int width, int height, Word* dstX,
                         Word* dstY) {
    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);

    const bincv::BinMatConstView<Word> srcPlanes[1] = {
        bincv::BinMatConstView<Word>{src, w, h, strideWords}};

    bincv::BinMatView<Word> magX[1] = {bincv::BinMatView<Word>{dstX, w, h, strideWords}};
    bincv::BinMatView<Word> signX{dstX + h * strideWords, w, h, strideWords};
    bincv::BinMatView<Word> magY[1] = {bincv::BinMatView<Word>{dstY, w, h, strideWords}};
    bincv::BinMatView<Word> signY{dstY + h * strideWords, w, h, strideWords};

    bincv::derivativeX<1, Word>(srcPlanes, magX, signX, bincv::BORDER_REFLECT_101, false);
    bincv::derivativeY<1, Word>(srcPlanes, magY, signY, bincv::BORDER_REFLECT_101, false);
}

/// @brief The covariance through the shipped reduction, reached as views.
/// @note The reductions were never reached through a container in the shipped
///       code either -- ops/reduce.hpp takes views only (D-5) -- so this row and
///       the specialized arm's differ ONLY in the TernaryMat construction and the
///       magnitude() / sign() calls the specialized arm makes to name the planes.
Cov covarianceWindowViewsOnly(const Word* dx, const Word* dy, size_t strideWords, int width,
                              int height, int wx, int wy, int wsize) {
    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    const size_t planeWords = h * strideWords;

    const bincv::BinMatConstView<Word> magX{dx, w, h, strideWords};
    const bincv::BinMatConstView<Word> sgnX{dx + planeWords, w, h, strideWords};
    const bincv::BinMatConstView<Word> magY{dy, w, h, strideWords};
    const bincv::BinMatConstView<Word> sgnY{dy + planeWords, w, h, strideWords};

    const bincv::CovarianceCount c = bincv::countCovariance<Word>(
        magX, magY, sgnX, sgnY, bincv::Rect(wx, wy, wsize, wsize));
    Cov out;
    out.xx = c.xx;
    out.yy = c.yy;
    out.whenClear = c.xy.whenClear;
    out.whenSet = c.xy.whenSet;
    return out;
}

}  // namespace t39
