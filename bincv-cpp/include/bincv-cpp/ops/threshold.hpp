#pragma once

/// @file threshold.hpp
/// @brief Producing the 1-bit frame from a higher-precision source (T3.2).
///
/// ARCHITECTURE 7.3: in a deployed system the binarization may happen in-sensor;
/// binCV provides it for pipelines that binarize on the host. Two sources, two
/// tiers, and the tier difference is the whole reason they have different names.
///
///   threshold(const cv::Mat&, dst, thresh)   CV_8U in, 1 bit out.  **TIER 1.**
///   binarize(planes, dst, thresh)            N-bit in, 1 bit out.  **TIER 3.**
///
/// ---------------------------------------------------------------------------
/// THE COMPARISON IS STRICTLY GREATER THAN, AND IT IS NOT A DETAIL
///
/// `cv::threshold` with `THRESH_BINARY` is documented and implemented as
///
///     dst(x, y) = (src(x, y) >  thresh) ? maxval : 0
///
/// **`>`, not `>=`.** Getting that backwards moves exactly one value class --
/// the pixels equal to `thresh` -- from 0 to 1. On a natural image at a
/// mid-range threshold that is a fraction of a percent of the pixels, which a
/// sampled test can miss entirely and a visual check certainly will; at
/// `thresh == 0` it is the difference between "every non-black pixel" and "every
/// pixel". Both entry points here use `>`, and tests/test_threshold.cpp pins it
/// at the boundary values (thresh - 1, thresh, thresh + 1 present in the same
/// image) rather than only in the middle of the range, at thresh = 0, 254 and 255
/// where an off-by-one is a whole-image difference.
///
/// The consequence at the ends, stated so a caller does not have to derive it:
///
///   thresh = 0    every NON-ZERO pixel is set. This is `src != 0`, the usual
///                 "make a mask" call, and it is the one an implementation using
///                 `>=` would turn into "all ones".
///   thresh = 255  NOTHING is set for a CV_8U source: no uint8 exceeds 255. Not
///                 an error, and not clamped to something more useful -- it is
///                 what cv::threshold returns, and Tier 1 means matching it.
///
/// ---------------------------------------------------------------------------
/// TIER 1 FOR CV_8U, TIER 3 FOR QuantMat<N>
///
/// The CV_8U entry point IS `cv::threshold(src, dst, thresh, 255, THRESH_BINARY)`
/// on the same content, so it takes OpenCV's name (ARCHITECTURE 5.1) and
/// tests/test_threshold.cpp proves bit-exactness through T2.1's harness across
/// its full size and fill matrix. What differs is the OUTPUT CONTAINER, not the
/// answer: a bit-packed BinMat rather than a CV_8U matrix, which is the entire
/// point -- 640x480 in 38400 bytes rather than 307200 (ARCHITECTURE 4.6). The
/// harness compares by unpacking to CV_8U {0, 255}, so "bit-exact" is asserted
/// against OpenCV's actual bytes and not against a normalisation of them.
///
/// `maxval` is therefore NOT a parameter. In a one-bit destination the set value
/// is 1 by construction; a `maxval` argument could only be ignored or asserted
/// on, and an ignored argument that looks like OpenCV's is worse than an absent
/// one. Nor is there a `type` parameter: THRESH_BINARY is the operation
/// ARCHITECTURE 7.3 names, THRESH_BINARY_INV is `bitwiseNot` of it (ops/logic.hpp),
/// and the four truncating types cannot be expressed in a 1-bit destination at
/// all. A ThresholdType enum whose every other value asserted would be a promise
/// this file cannot keep.
///
/// The QuantMat<N> entry point has no OpenCV counterpart -- OpenCV has no N-bit
/// image type -- so it is Tier 3 and must NOT borrow the name (ARCHITECTURE 5.1).
/// It is `binarize`, and it is checked against a per-pixel reference:
/// `src.at(y, x) > thresh`, the same comparison as above, so the two entry points
/// cannot drift into disagreeing about their boundary.
///
/// ---------------------------------------------------------------------------
/// THE DOMAIN OF THE TIER 1 PROMISE: `|thresh| < 2^31`
///
/// `threshold` is bit-exact against `cv::threshold` for **every `thresh` whose
/// floor is representable as an `int`**, which is every threshold a caller of a
/// CV_8U threshold can mean. It is NOT bit-exact outside that range, and the
/// reason is that `cv::threshold` is not defined there either: for a CV_8U source
/// it reduces the double with `cvFloor`, whose `(int)value` conversion is
/// undefined once the value leaves `int`'s range. Measured on OpenCV 4.5.4, x86-64:
///
///   thresh      cv::threshold      binCV        who is right
///   ---------------------------------------------------------------------
///   +1e300      every pixel set    none set     binCV (nothing exceeds 1e300)
///   -1e300      none set           every pixel  binCV (everything exceeds it)
///   +/-inf      same inversion     as above     binCV
///   +2^31       every pixel set    none set     binCV
///
/// OpenCV's answers there are the *opposite* of the arithmetic, and they flip
/// direction with the compiler's conversion behaviour rather than with the
/// threshold. binCV does NOT chase that: the reduction below is exact over the
/// whole real line, so a huge positive threshold selects nothing and a huge
/// negative one selects everything. `NaN` -- which is not a threshold at all --
/// selects nothing, because `p > NaN` is false for every `p`.
///
/// So the tier claim, stated at full strength: **bit-exact against cv::threshold
/// wherever cv::threshold is defined**, and arithmetically correct beyond it.
/// tests/test_threshold.cpp pins both halves -- the in-domain half against
/// cv::threshold itself over the enumerated (value, threshold) space, and the
/// out-of-domain half against the arithmetic, so binCV's choice cannot drift
/// silently.
///
/// ---------------------------------------------------------------------------
/// WHAT THE KERNELS PROMISE (the ops/ contract)
///
///  1. **Views, never containers** (D-5). `binarize` takes the N plane views, and
///     the QuantMat overload is a thin wrapper that names them -- not a second
///     implementation. The CV_8U overload's source is a `const cv::Mat&` because
///     that IS the foreign format being read; it is used strictly as
///     {ptr, rows, cols, step} and is never resized, reallocated or written.
///  2. **No heap allocation.** `binarize` gathers N plane words into a
///     stack array whose size is the template parameter N, so there is nothing to
///     allocate and nothing for a caller to provide.
///  3. **PADDING BITS STAY ZERO.** Both kernels write WHOLE WORDS -- a threshold
///     produces a bit per pixel and stores 8..64 of them at a time -- so the
///     trailing partial word of every destination row is stored masked. It is
///     load-bearing in `binarize`: `thresholdGE` returns a full word with no
///     notion of `width` and answers every lane, including lanes past the last
///     pixel, and at `threshold == 0` it returns all ones whatever the planes
///     hold (ops/bitslice.hpp says so in as many words). Storing that unmasked
///     would leave phantom set bits past `width` and every later word-wise
///     reduction would over-count them (D-13).
///  4. **Never throws** (ARCHITECTURE 5.3). Dimension mismatches, a short stride,
///     a source of the wrong cv::Mat type and a null destination are programming
///     errors, reported by BINCV_ASSERT in debug builds and undefined in release.
///
/// ---------------------------------------------------------------------------
/// PRECONDITION ON `dst`, identical to ops/logic.hpp's: it must span its image's
/// full width, or end on a word boundary. The trailing partial word is stored
/// masked, so in a sub-width window onto a WIDER image the bits between `width`
/// and the end of that word are that image's next pixels and are destroyed.
///
/// ALIASING does not arise for the CV_8U overload -- a cv::Mat and a packed
/// destination are different objects by construction. For `binarize` the source
/// planes and the destination MUST NOT OVERLAP, and unlike ops/logic.hpp the
/// exact-alias case is not legal either: plane p is read at word i AFTER the
/// destination's word i has been written for p' < p, so writing into a plane of
/// the source would corrupt the values still to be read. The assert is
/// impl::viewsShareNoWord, the same predicate ops/shift.hpp uses.
///
/// Empty views (width or height 0) are a no-op, not an error.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/view.hpp"
// impl::rowTailMask, impl::strideCoversARow, impl::viewsShareNoWord.
#include "../impl/kernel_util.hpp"
// thresholdGE -- T2.7's bit-sliced comparison, and the entire arithmetic of
// binarize(). Its whole (value, threshold) input space is enumerated by
// tests/test_bitslice.cpp, so this file supplies the row geometry and nothing
// else.
#include "bitslice.hpp"
// impl::minRowWords, impl::bitMask, and QuantMat<N> for the container wrapper.
#include "../quantMat.hpp"

#ifdef BINCV_WITH_OPENCV
// <cmath> only where it is used: the CV_8U entry point's threshold reduction.
// The core-only, no-exceptions and Debug configurations compile neither.
#include <cmath>

#include <opencv2/core.hpp>
#endif

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

// ---------------------------------------------------------------------------
// N-bit -> 1-bit. **API TIER 3** (no OpenCV equivalent; see the tier note above)
// ---------------------------------------------------------------------------

/// @brief dst = (src > thresh), pixel for pixel, over an N-plane bit-sliced
///        source. **API TIER 3.**
/// @tparam N Number of source planes, deduced from the array argument. Plane 0 is
///         the LEAST significant bit, matching QuantMat (ARCHITECTURE 4.1).
/// @param planes The N source plane views, all of the same dimensions as `dst`.
/// @param dst Destination view; must have the planes' dimensions and share no
///        word with any of them.
/// @param thresh The constant each pixel is compared against, STRICTLY GREATER
///        THAN -- see the comparison section at the top of this file. Values at
///        or above the largest an N-bit pixel can hold select nothing, which is
///        reached by arithmetic (a sweep from 0 to 2^N) rather than by choice and
///        is therefore defined rather than asserted against.
/// @note One pass, word by word, no scratch and no allocation: each word of the
///        destination is one `thresholdGE` over N gathered plane words, so 8..64
///        pixels are compared per call and the comparison itself is branch-free.
/// @note `thresh + 1` is what reaches thresholdGE, because it answers `>=` and
///        this operation is `>`. The `thresh >= MaxValue` shortcut above it is
///        not an optimisation -- it is what stops that increment from wrapping
///        when a caller passes the largest `unsigned`.
/// @note The destination's padding bits are zero on return. That is not
///        automatic here: thresholdGE answers every lane in the word, including
///        the ones past `width`, so the trailing word is masked before it is
///        stored. See property 3 at the top of this file.
/// @note Source padding bits are never read as pixels -- they only ever land in
///        destination bits past `width`, which the tail mask clears.
/// @note Never throws and never allocates.
template <size_t N, typename WordType>
inline void binarize(const BinMatConstView<WordType> (&planes)[N], BinMatView<WordType> dst,
                     unsigned thresh) {
    static_assert(N >= 1, "binarize: a source needs at least one plane");
    // `thresh` is an `unsigned`, and the comparison this kernel performs is
    // `value > thresh`, i.e. `value >= thresh + 1`. With more than 32 planes the
    // cutoff a caller may need (2^32) is not representable in that parameter, and
    // the answer would be silently wrong rather than merely unavailable: at
    // thresh == UINT_MAX the shortcut below reports "nothing passes" for pixels
    // that DO exceed UINT_MAX, so the result stops being monotone in `thresh`
    // (measured: 33 planes, one pixel holding 2^32, thresh UINT_MAX - 1 selects it
    // and thresh UINT_MAX does not). Rejected at compile time rather than
    // documented, because there is no correct answer to give. N == 32 is the
    // largest width `unsigned` can express and is supported; QuantMat itself caps
    // N at 8, so this bound only ever binds on the plane-view entry point.
    static_assert(N <= sizeof(unsigned) * 8,
                  "binarize: more than 32 planes cannot be thresholded through an "
                  "`unsigned` cutoff");

    for (size_t p = 0; p < N; ++p) {
        BINCV_ASSERT(planes[p].width == dst.width && planes[p].height == dst.height,
                     "binarize: every source plane must have dst's dimensions");
        BINCV_ASSERT(impl::strideCoversARow<WordType>(planes[p].width, planes[p].height,
                                                      planes[p].stride),
                     "binarize: every view's stride must cover a whole row");
        BINCV_ASSERT(impl::viewsShareNoWord(planes[p], dst),
                     "binarize: dst must share no word with any source plane");
    }
    BINCV_ASSERT(impl::strideCoversARow<WordType>(dst.width, dst.height, dst.stride),
                 "binarize: every view's stride must cover a whole row");

    if (dst.width == 0 || dst.height == 0) return;

    BINCV_ASSERT(dst.ptr != nullptr, "binarize: a non-empty view needs a non-null pointer");

    const size_t words = impl::minRowWords<WordType>(dst.width);
    const WordType tailMask = impl::rowTailMask<WordType>(dst.width);

    // A threshold no N-bit pixel can exceed selects nothing. Handled before the
    // loop so that `thresh + 1` below cannot wrap, and so the common all-zero
    // answer costs one store per word rather than N loads.
    constexpr unsigned maxValue = (N >= sizeof(unsigned) * 8)
                                      ? ~0u
                                      : static_cast<unsigned>((1ull << N) - 1ull);
    if (thresh >= maxValue) {
        for (size_t y = 0; y < dst.height; ++y) {
            WordType* dstRow = dst.row(y);
            for (size_t i = 0; i < words; ++i) dstRow[i] = static_cast<WordType>(0);
        }
        return;
    }

    // `>` against `thresh` is `>=` against `thresh + 1`; thresholdGE answers the
    // latter. See the comparison section at the top of this file.
    const unsigned geThreshold = thresh + 1u;

    for (size_t y = 0; y < dst.height; ++y) {
        WordType* dstRow = dst.row(y);
        // N is a template parameter, so this is an array on the stack and not an
        // allocation -- "no heap allocation inside kernels" with nothing for the
        // caller to provide.
        const WordType* planeRows[N];
        for (size_t p = 0; p < N; ++p) planeRows[p] = planes[p].row(y);

        WordType gathered[N];
        for (size_t i = 0; i + 1 < words; ++i) {
            for (size_t p = 0; p < N; ++p) gathered[p] = planeRows[p][i];
            dstRow[i] = thresholdGE<WordType>(gathered, N, geThreshold);
        }
        for (size_t p = 0; p < N; ++p) gathered[p] = planeRows[p][words - 1];
        dstRow[words - 1] =
            static_cast<WordType>(thresholdGE<WordType>(gathered, N, geThreshold) & tailMask);
    }
}

/// @brief dst = (src > thresh) over an N-bit binCV image. **API TIER 3.**
/// @note A thin wrapper: it names `src`'s planes and calls the view kernel above,
///        which is where the work and the contract live (D-5). A caller holding
///        views never reaches this overload.
/// @note N == 1 comes here too -- BinMat IS QuantMat<1> (core/types.hpp) -- where
///        the only meaningful threshold is 0 and the result is a copy of the
///        source. Defined rather than rejected, because a caller sweeping N does
///        not want a special case.
template <size_t N, typename WordType>
inline void binarize(const QuantMat<N, WordType>& src, BinMatView<WordType> dst, unsigned thresh) {
    BinMatConstView<WordType> planes[N];
    for (size_t p = 0; p < N; ++p) planes[p] = src.constPlane(p);
    binarize<N, WordType>(planes, dst, thresh);
}

#ifdef BINCV_WITH_OPENCV

// ---------------------------------------------------------------------------
// CV_8U -> 1-bit. **API TIER 1** -- bit-exact against cv::threshold
// ---------------------------------------------------------------------------

/// @brief dst = (src > thresh), packing a CV_8U image into one bit per pixel.
///        **API TIER 1** -- bit-exact against
///        `cv::threshold(src, tmp, thresh, 255, cv::THRESH_BINARY)` on the same
///        content, pixel for pixel, for every `thresh` with `|thresh| < 2^31`.
///        Beyond that range cv::threshold is itself undefined and binCV answers
///        the arithmetic instead; see "THE DOMAIN OF THE TIER 1 PROMISE" at the
///        top of this file.
/// @param src The source image. Must be CV_8UC1 and must have `dst`'s dimensions.
///        Read as a view -- rows, cols and step -- never written or resized.
/// @param dst Destination view; must have src's dimensions.
/// @param thresh The constant each pixel is compared against. **STRICTLY GREATER
///        THAN**: a pixel EQUAL to `thresh` is 0. See the comparison section at
///        the top of this file, including what happens at 0 and at 255.
/// @note `thresh` is `double` for one reason: it is cv::threshold's parameter
///        type, and a caller porting a call passes the same expression. It is
///        compared against the integer pixel directly, so the comparison is
///        exact for every value a CV_8U pixel can take -- a `double` holds every
///        uint8 without rounding, and a fractional threshold such as 127.5 does
///        what a reader expects rather than being truncated. This is the ONE
///        place binCV takes a floating-point argument, and it takes it to be
///        drop-in.
/// @note EVERY `double` IS DEFINED HERE, including the ones no caller means. A
///        threshold at or above 255 -- however large, up to +infinity -- selects
///        nothing; one below 0 -- however small, down to -infinity -- selects
///        everything; `NaN` selects nothing, because `p > NaN` is false for every
///        `p`. That is the arithmetic, not cv::threshold's answer, for the values
///        where cv::threshold has no defined answer to match.
/// @note THIS IS THE ONLY ENTRY POINT IN ops/ THAT NAMES A cv:: TYPE, and it is
///        behind BINCV_WITH_OPENCV: the core-only, no-exceptions and Debug
///        configurations never see it. Nothing the embedded claim rests on
///        depends on OpenCV (ARCHITECTURE 2).
/// @note The destination's padding bits are zero on return: the row's trailing
///        partial word is assembled from live pixels only, and the bits past
///        `width` are never set.
/// @note PRECONDITION ON `dst`: it must span its image's full width, or end on a
///        word boundary -- the trailing word is STORED, not merged, so bits past
///        `width` in a sub-width window onto a wider image are zeroed.
/// @note Never throws and never allocates. A src of the wrong type, mismatched
///        dimensions and a short stride are programming errors: BINCV_ASSERT
///        reports them in debug builds and they are undefined in release.
template <typename WordType>
inline void threshold(const cv::Mat& src, BinMatView<WordType> dst, double thresh) {
    BINCV_ASSERT(src.empty() || src.type() == CV_8UC1,
                 "threshold: src must be CV_8UC1");
    BINCV_ASSERT(static_cast<size_t>(src.cols) == dst.width &&
                     static_cast<size_t>(src.rows) == dst.height,
                 "threshold: src and dst must have the same dimensions");
    BINCV_ASSERT(impl::strideCoversARow<WordType>(dst.width, dst.height, dst.stride),
                 "threshold: dst's stride must cover a whole row");

    if (dst.width == 0 || dst.height == 0) return;

    BINCV_ASSERT(dst.ptr != nullptr, "threshold: a non-empty view needs a non-null pointer");

    constexpr size_t wordBits = sizeof(WordType) * 8;
    const size_t words = impl::minRowWords<WordType>(dst.width);

    // The comparison, reduced to ONE INTEGER CUTOFF before the loops. For an
    // integer pixel `p > thresh` is `p >= floor(thresh) + 1`, and the two ends
    // are exact rather than approximated: no CV_8U pixel exceeds 255, and every
    // one exceeds a negative threshold.
    //
    // Inside `|thresh| < 2^31` this is also the reduction cv::threshold performs
    // for CV_8U -- it floors `thresh` to an int and handles `< 0` and `>= 255` as
    // whole-image answers before dispatching -- which is why a fractional
    // threshold such as 127.5 produces the same image on both sides, and why the
    // tier claim survives a `double` parameter. OUTSIDE that range cv::threshold
    // is undefined (its cvFloor converts an out-of-range double to int) and binCV
    // deliberately does not reproduce its answer; see "THE DOMAIN OF THE TIER 1
    // PROMISE" at the top of this file.
    //
    // THE THREE BRANCHES ARE ORDERED SO THAT THE CAST NEVER SEES A VALUE IT
    // CANNOT REPRESENT. `std::floor` is reached only for thresh in [0, 255), so
    // the `int` conversion is exact and this kernel has no undefined behaviour of
    // its own for ANY double -- including the infinities and NaN, which reach it
    // through the second branch. The second test is written `!(thresh < 255.0)`
    // rather than `thresh >= 255.0` for exactly one input: NaN, which is less
    // than nothing and not greater than anything, and must land on "nothing
    // passes" (`p > NaN` is false for every p) rather than falling through to a
    // cast of NaN.
    //
    // 256 is "nothing passes": no uint8 reaches it. 0 is "everything passes".
    int cutoff;
    if (thresh < 0.0) {
        cutoff = 0;
    } else if (!(thresh < 255.0)) {
        cutoff = 256;
    } else {
        cutoff = static_cast<int>(std::floor(thresh)) + 1;
    }

    for (size_t y = 0; y < dst.height; ++y) {
        const uint8_t* srcRow = src.ptr<uint8_t>(static_cast<int>(y));
        WordType* dstRow = dst.row(y);

        // Whole words are ASSEMBLED and stored, never read-modify-written: the
        // destination's previous contents are irrelevant, and the bits past
        // `width` in the trailing word are simply never set -- which is the
        // padding invariant with no mask needed.
        for (size_t i = 0; i < words; ++i) {
            const size_t base = i * wordBits;
            const size_t bits = (dst.width - base < wordBits) ? (dst.width - base) : wordBits;
            WordType word = 0;
            for (size_t b = 0; b < bits; ++b) {
                // `>= cutoff` IS `> thresh`; see the derivation above. Getting
                // this boundary wrong by one moves the pixels equal to `thresh`
                // from 0 to 1 -- the failure the top of this file is about.
                if (static_cast<int>(srcRow[base + b]) >= cutoff) {
                    word = static_cast<WordType>(word | impl::bitMask<WordType>(b));
                }
            }
            dstRow[i] = word;
        }
    }
}

#endif // BINCV_WITH_OPENCV

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
