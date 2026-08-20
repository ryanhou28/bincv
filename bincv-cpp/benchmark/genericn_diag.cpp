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
//
// ---------------------------------------------------------------------------
// A SECOND DECOMPOSITION POINT, ADDED AT TRIAGE: `scalarized`.
//
// X-21 as first written attributed the whole `views_only - hand_written` gap to
// the per-plane arrays a[N] / b[N] / m[N] / srcRow[N] and the `for p < N` loops.
// That attribution was not measured. The hand-written arm drops THREE kinds of
// genericity at once -- genericity in N, genericity in the BorderType (which is
// a RUNTIME parameter in ops/, and pays for it inside the word loop: the d/dy
// kernel's `a = haveA ? rowA[p][i] : fill` select exists only so BORDER_CONSTANT
// can be requested at run time) and genericity in the word TYPE (`B - 1` from
// bitsPerWord<WordType>() where the hand-written arm writes the literal 31) --
// and no comparison in the original run separates them.
//
// `scalarized` removes EXACTLY ONE of the three: N. It is the shipped kernel with
// the plane arrays replaced by scalars and the plane loops deleted, and NOTHING
// else changed -- same views, same impl:: helpers, same BINCV_ASSERT contract,
// same RUNTIME BorderType, same template over WordType, same ternaryDifference
// spelling. So
//
//     views_only  - scalarized    =  the N-plane array plumbing, alone
//     scalarized  - hand_written  =  everything else the library kernel carries
//
// and the two add up to what X-21 called "the kernel's generic SHAPE at N = 1".
// This is also the candidate remedy X-21's Decision Q3 named, so measuring it
// costs one decomposition point and answers whether that remedy is worth an
// experiment at all.

#include <cstddef>

#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/core/view.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/shift.hpp"
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

namespace {

/// @brief d/dx through the SHIPPED kernel's code, with the N-plane arrays
///        replaced by scalars and nothing else changed.
/// @note A line-for-line copy of impl::derivativeXRoute at N = 1: same views,
///       same impl:: helpers, same argument contract, same RUNTIME BorderType,
///       same template over WordType, and the same impl::ternaryDifference the
///       shipped route reaches through signedDifference<1, WordType, false>. The
///       ONLY difference is that `srcRow`, `magRow`, `prev`, `a`, `b` and `m` are
///       scalars rather than arrays of one, so the `for p < N` loops disappear.
///       Deliberately NOT a library change -- it lives here so the question can
///       be measured before anything is done about it.
template <typename WordType>
void derivativeXScalarized(const bincv::BinMatConstView<WordType>& src,
                           bincv::BinMatView<WordType> mag, bincv::BinMatView<WordType> sign,
                           bincv::BorderType borderType, bool borderValue) {
    const bincv::BinMatConstView<WordType> srcArr[1] = {src};
    bincv::BinMatView<WordType> magArr[1] = {mag};
    bincv::impl::checkDerivativeArgs<1, WordType>(srcArr, magArr, sign, borderType);

    const size_t width = src.width;
    const size_t height = src.height;
    if (width == 0 || height == 0) return;

    constexpr size_t B = bincv::impl::bitsPerWord<WordType>();
    const size_t words = bincv::impl::minRowWords<WordType>(width);
    const WordType tailMask = bincv::impl::rowTailMask<WordType>(width);
    const WordType lastLiveBit = bincv::impl::bitMask<WordType>(width - 1);
    const WordType topBit = static_cast<WordType>(static_cast<WordType>(1) << (B - 1));

    const ptrdiff_t leftSrc =
        bincv::impl::borderIndex(static_cast<ptrdiff_t>(-1), width, borderType);
    const ptrdiff_t rightSrc =
        bincv::impl::borderIndex(static_cast<ptrdiff_t>(width), width, borderType);

    for (size_t y = 0; y < height; ++y) {
        const WordType* srcRow = src.row(y);
        WordType* magRow = mag.row(y);
        const bool leftBit =
            (leftSrc < 0) ? borderValue
                          : bincv::impl::rowBit<WordType>(srcRow, static_cast<size_t>(leftSrc));
        WordType prev = leftBit ? topBit : static_cast<WordType>(0);
        WordType* signRow = sign.row(y);

        for (size_t i = 0; i < words; ++i) {
            const bool last = (i + 1 == words);
            const WordType cur = srcRow[i];
            const WordType nxt = last ? static_cast<WordType>(0) : srcRow[i + 1];

            WordType a = static_cast<WordType>(static_cast<WordType>(cur >> 1) |
                                               static_cast<WordType>(nxt << (B - 1)));
            const WordType b = static_cast<WordType>(static_cast<WordType>(cur << 1) |
                                                     static_cast<WordType>(prev >> (B - 1)));
            prev = cur;

            if (last) {
                const bool bit =
                    (rightSrc < 0)
                        ? borderValue
                        : bincv::impl::rowBit<WordType>(srcRow, static_cast<size_t>(rightSrc));
                a = static_cast<WordType>(
                    static_cast<WordType>(a & static_cast<WordType>(~lastLiveBit)) |
                    (bit ? lastLiveBit : static_cast<WordType>(0)));
            }

            WordType m = static_cast<WordType>(0);
            WordType s = static_cast<WordType>(0);
            bincv::impl::ternaryDifference<WordType>(a, b, m, s);

            magRow[i] = last ? static_cast<WordType>(m & tailMask) : m;
            signRow[i] = last ? static_cast<WordType>(s & tailMask) : s;
        }
    }
}

/// @brief d/dy, same treatment: impl::derivativeYRoute at N = 1 with the plane
///        arrays scalarized and nothing else touched.
/// @note The `haveA ? rowA[p][i] : fill` per-word select stays, because it is
///       BorderType genericity rather than N genericity and this point removes
///       only N.
template <typename WordType>
void derivativeYScalarized(const bincv::BinMatConstView<WordType>& src,
                           bincv::BinMatView<WordType> mag, bincv::BinMatView<WordType> sign,
                           bincv::BorderType borderType, bool borderValue) {
    const bincv::BinMatConstView<WordType> srcArr[1] = {src};
    bincv::BinMatView<WordType> magArr[1] = {mag};
    bincv::impl::checkDerivativeArgs<1, WordType>(srcArr, magArr, sign, borderType);

    const size_t width = src.width;
    const size_t height = src.height;
    if (width == 0 || height == 0) return;

    const size_t words = bincv::impl::minRowWords<WordType>(width);
    const WordType tailMask = bincv::impl::rowTailMask<WordType>(width);
    const WordType fill =
        borderValue ? static_cast<WordType>(~static_cast<WordType>(0)) : static_cast<WordType>(0);

    for (size_t y = 0; y < height; ++y) {
        const ptrdiff_t ya =
            bincv::impl::borderIndex(static_cast<ptrdiff_t>(y) + 1, height, borderType);
        const ptrdiff_t yb =
            bincv::impl::borderIndex(static_cast<ptrdiff_t>(y) - 1, height, borderType);
        const bool haveA = ya >= 0;
        const bool haveB = yb >= 0;

        const WordType* rowA = haveA ? src.row(static_cast<size_t>(ya)) : nullptr;
        const WordType* rowB = haveB ? src.row(static_cast<size_t>(yb)) : nullptr;
        WordType* magRow = mag.row(y);
        WordType* signRow = sign.row(y);

        for (size_t i = 0; i < words; ++i) {
            const bool last = (i + 1 == words);
            const WordType a = haveA ? rowA[i] : fill;
            const WordType b = haveB ? rowB[i] : fill;

            WordType m = static_cast<WordType>(0);
            WordType s = static_cast<WordType>(0);
            bincv::impl::ternaryDifference<WordType>(a, b, m, s);

            magRow[i] = last ? static_cast<WordType>(m & tailMask) : m;
            signRow[i] = last ? static_cast<WordType>(s & tailMask) : s;
        }
    }
}

}  // namespace

namespace t39 {

/// @brief The `scalarized` decomposition point: the shipped kernel with N's
///        array plumbing removed and every other kind of genericity kept.
void derivativeScalarized(const Word* src, size_t strideWords, int width, int height, Word* dstX,
                          Word* dstY) {
    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);

    const bincv::BinMatConstView<Word> srcView{src, w, h, strideWords};
    bincv::BinMatView<Word> magX{dstX, w, h, strideWords};
    bincv::BinMatView<Word> signX{dstX + h * strideWords, w, h, strideWords};
    bincv::BinMatView<Word> magY{dstY, w, h, strideWords};
    bincv::BinMatView<Word> signY{dstY + h * strideWords, w, h, strideWords};

    derivativeXScalarized<Word>(srcView, magX, signX, bincv::BORDER_REFLECT_101, false);
    derivativeYScalarized<Word>(srcView, magY, signY, bincv::BORDER_REFLECT_101, false);
}

}  // namespace t39
