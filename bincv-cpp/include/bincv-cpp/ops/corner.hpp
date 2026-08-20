#pragma once

/// @file corner.hpp
/// @brief The minimum-eigenvalue corner response and the good-features selection
///        it feeds (T3.7). **API TIER 2** -- see the tier note below.
///
/// This is the operation the whole covariance machinery was built for. T3.6 gives
/// the 2x2 Lucas-Kanade matrix `[sumXX, sumXY; sumXY, sumYY]` over ONE window;
/// this file evaluates its smaller eigenvalue at EVERY pixel of a frame and then
/// performs the selection `cv::goodFeaturesToTrack` performs -- a quality
/// threshold, a 3x3 non-maximum suppression, and a minimum-distance spacing
/// filter.
///
/// ---------------------------------------------------------------------------
/// THE TIER, STATED FIRST BECAUSE IT IS THE THING A CALLER MUST NOT GET WRONG
///
/// **API TIER 2** (ARCHITECTURE 5.1): the same call shape and the same ROLE as
/// `cv::goodFeaturesToTrack` / `cv::cornerMinEigenVal`, with deliberately
/// different numerics. It is **NOT** bit-exact against OpenCV and no test here
/// claims it is. Two independent reasons, and either one alone would be enough:
///
///   * **The derivatives are binarized, not Sobel.** binCV's `Ix` and `Iy` are
///     `[-1, 0, 1]` taps over a ONE-BIT image (T3.5, D-3), so every gradient is in
///     {-1, 0, +1}. `cv::cornerMinEigenVal` runs a 3x3 Sobel over a byte image.
///     Those are different numbers before any window is summed. That is the
///     reference pipeline's own choice -- `gftt_corner_derivative_type: BINARIZED`
///     in SEAL/seal_params.yaml -- and this file reproduces THAT, not OpenCV's
///     default path.
///   * **`cv::cornerMinEigenVal` works in float and cannot be compared exactly.**
///     Its `eig` map is `CV_32F` produced by a float box filter over float
///     products of float Sobel outputs. binCV's three sums are EXACT INTEGERS from
///     population counts, so the only rounding on this side is the square root.
///     An exact integer comparison against `cv::cornerMinEigenVal` is therefore
///     not available even in principle, and the validation here is against a
///     per-pixel reference of this file's own (tests/test_corner.cpp) plus the
///     reference pipeline's documented behaviour.
///
/// Nothing in this file is Tier 1 and nothing in it promises OpenCV bit-exactness.
///
/// ---------------------------------------------------------------------------
/// THE OPERATION, READ OUT OF THE REFERENCE RATHER THAN INFERRED
///
/// SEAL/src/keypoint_detection/gftt.cpp and SEAL/src/keypoint_detection/corner.cpp,
/// with `SEAL/seal_params.yaml`'s values (`gftt_max_corners: 200`,
/// `gftt_quality_level: 0.01`, `gftt_min_distance: 33.33333333333`,
/// `gftt_block_size: 3`, `gftt_use_harris_detector: 0` -- so MINIMUM EIGENVALUE,
/// not Harris; GoodFeaturesParams' defaults are those four values and say so).
///
/// `cornerEigenValsVecs()` forms, per pixel, `dx*dx`, `dx*dy`, `dy*dy`; box-filters
/// the three planes with `normalize = false` (so a SUM over the block, not a mean);
/// and calls `calcMinEigenVal`:
///
///     float a = cov[j*3]*0.5f;      // sumXX / 2
///     float b = cov[j*3+1];         // sumXY
///     float c = cov[j*3+2]*0.5f;    // sumYY / 2
///     dst[j] = (a + c) - std::sqrt((a - c)*(a - c) + b*b);
///
/// which is `(sumXX + sumYY)/2 - sqrt(((sumXX - sumYY)/2)^2 + sumXY^2)`, the
/// smaller eigenvalue of `[[xx, xy], [xy, yy]]`. Those three sums are exactly
/// T3.6's `GradientCovariance` over a `blockSize x blockSize` window, so this file
/// computes no products and no box filter: it asks ops/covariance.hpp's popcounts
/// for the same three integers.
///
/// **The reference's scale factor is dropped, deliberately and without effect.**
/// `cornerEigenValsVecs` scales both derivatives by `1/(block_size * 255)` for a
/// `CV_8U` source, so its `eig` is this file's response times a single positive
/// constant `s^2`. The min-eigenvalue is homogeneous of degree one in the matrix,
/// so every response scales by the same `s^2` -- which changes no ordering, no
/// 3x3 maximum, and no `qualityLevel * maxVal` decision, because that threshold is
/// RELATIVE to the map's own maximum. The scale exists in the reference to keep a
/// byte image's products inside float range; binCV's pixels are already {0, 1}.
/// What a caller must not do is compare a response from this file against an
/// absolute threshold taken from the reference.
///
/// ---------------------------------------------------------------------------
/// THE SLIDING FORM IS WHY X-11 AND T2.11 EXIST, AND IT IS USED HERE
///
/// A response map sweeps a window over EVERY pixel. That is X-11's "DENSE" access
/// pattern verbatim -- the one ops/reduce.hpp's table and ops/covariance.hpp's
/// docstring both point at this operation for -- so `impl::cornerResponseColumn`
/// sweeps a COLUMN at a time with two `SlidingWindowCount`s alive:
///
///     SlidingWindowCount<WordType> xx(magX, firstWindow);   // ΣIx² , slid
///     SlidingWindowCount<WordType> yy(magY, firstWindow);   // ΣIy² , slid
///     for (y ...) {
///         const SplitCount xy = countAndSplit(magX, magY, signX, signY, window);
///         response = minEigenValue(xx.count(), yy.count(), xy.crossTerm());
///         xx.slideDown(); yy.slideDown();
///     }
///
/// **TWO OF THE THREE NUMBERS SLIDE AND THE THIRD DOES NOT, and that is not an
/// oversight here -- it is the property T3.6's docstring says T3.7 has to know.**
/// `SlidingWindowCount` slides ONE plane's popcount, so `sumXX` and `sumYY` each
/// get one accumulator and cost two row counts per position whatever `blockSize`
/// is. `sumXY` needs `magX & magY` split by `signX ^ signY`, nothing in
/// ops/reduce.hpp slides a split, and making it slide would cost TWO frame-sized
/// planes per pyramid level -- more than the one plane D-15 axis 3 already
/// declined on memory grounds. So the cross term is recomputed per position
/// through the four-argument `countAndSplit`, which is the widest form that needs
/// no scratch.
///
/// **The sweep is column-major because the accumulator only slides DOWNWARD.**
/// A row-major sweep would need one accumulator per column -- a `width`-long
/// scratch array the caller would have to own, which is a second shape and a
/// second decision (X-11 declined exactly that for the box accumulator). Two live
/// accumulators and zero scratch is what this costs instead. That traversal order
/// is not free, and the measurement below prices it.
///
/// ---------------------------------------------------------------------------
/// AND IT IS NOT A WIN AT THE REFERENCE PIPELINE'S OWN BLOCK SIZE. READ THIS
/// BEFORE QUOTING 15.9x AT THIS OPERATION.
///
/// X-11b's 15.9x is a single-plane `countNonZero` dense sweep. It applies to the
/// two numbers that slide and to nothing else, and when the shape is embedded in
/// THIS caller the advantage does not merely shrink -- below `blockSize` 15 it
/// reverses. Measured on the reference device at 640x480,
/// `benchmark/corner_benchmark.cpp`, spreads 0.04-3.4%
/// ([X-18](../../../EXPERIMENTS.md), `results/corner_benchmark_pi4.log`):
///
///     blockSize   sliding      recompute    net       incremental   traversal
///                 ns/pixel     ns/pixel     ratio     alone         alone
///        3         101.25        84.83      0.84x      0.94x         1.12x
///        7         142.84       138.55      0.97x      1.04x         1.07x
///       15         252.89       278.78      1.10x      1.14x         1.03x
///       31         581.44       704.39      1.21x      1.22x         1.01x
///
/// The last two columns separate the two things this shape changes at once, using
/// a third variant that recomputes COLUMN-MAJOR: the incremental state alone is
/// worth 0.93x at `blockSize` 3 (a LOSS), 1.04x at 7 and 1.24x at 31, while the
/// column-major traversal the accumulator forces costs 12% at 3, 7% at 7 and 1% at
/// 31. **The two effects cross over at DIFFERENT sizes** -- the incremental one
/// between 3 and 7, the net between 7 and 15 -- because the traversal penalty is
/// still 7% where the incremental win is only 4%. Their sum is **20% SLOWER than
/// the obvious row-major recomputation at `blockSize = 3`, which is exactly what
/// SEAL/seal_params.yaml runs.**
///
/// The spreads above are WITHIN-run. RUN-TO-RUN scatter was measured separately
/// over four device runs of the same binary (`results/corner_benchmark_pi4_scatter.log`):
/// 0.18-0.34% on the net ratio at `blockSize` 3, 7 and 15, and 3.3% at 31. **The
/// ranking holds in every run at every block size** -- the net ratio never reaches
/// 1.00 at 3 or 7 and never falls to 1.00 at 15 or 31 -- so the crossover is a
/// measured boundary and not an artefact of one run. The `blockSize` 7 row, the
/// smallest gap in the table, has a net gap about seven times the run-to-run range
/// of the two rows it is taken from.
///
/// **That contradicts documented guidance** -- ops/reduce.hpp's "WHICH SHAPE TO
/// REACH FOR" table, ops/covariance.hpp's docstring and D-15 all send a dense
/// sweep to the incremental form without a window-size qualification, and T3.7's
/// spec did too. CLAUDE.md's rule for a measurement that contradicts a documented
/// claim is to report it rather than adjust the code to fit the doc, so: the
/// sliding form is what ships, the number above is what it costs, and **whether
/// this operation should select on `blockSize` is an OPEN DECISION** that X-18
/// registers and does not take. One device and one frame size is not enough to
/// hard-code a threshold -- the x86 run has the opposite sign at `blockSize` 3
/// (1.19x in the sliding form's favour) and spreads past 50%, which is why it is
/// filed as indicative only.
///
/// ---------------------------------------------------------------------------
/// AND AGAINST OPENCV, WHICH IS THE TIER 2 DENOMINATOR AND IS NOT THE TABLE ABOVE
///
/// The table above is binCV against binCV. CLAUDE.md's denominator rule asks for
/// something else -- OpenCV doing the SAME semantic operation on the SAME binary
/// content stored as `CV_8U`, with PEAK WORKING SET beside speed -- and
/// [X-19](../../../EXPERIMENTS.md) measures it
/// (`benchmark/corner_opencv_benchmark.cpp`,
/// `results/corner_opencv_benchmark_pi4.log`). The denominator is the reference
/// pipeline written out in stock OpenCV: two `filter2D` taps, three product planes,
/// a `boxFilter` SUM, the min eigenvalue, then gftt.cpp's selection. Reference
/// device, 640x480:
///
///     variant                        ns/pixel   vs denom   B/pixel
///     binCV                            138.11      0.55x     16.54  (5.14 sized)
///     OpenCV binarized (denominator)    76.06      1.00x     36.94  (29.35 sized)
///     OpenCV Sobel (stock, different numerics)  59.50  1.28x  29.00
///
/// **binCV is 2.23x smaller (5.71x once both candidate buffers are sized to the
/// measured survivor count) and 1.82x slower.** Roughly a third of that 1.82x is
/// the sliding form's own 1.20x loss at `blockSize` 3, measured above. The rest is
/// that the OpenCV side spends SEVEN frame-sized `float` planes where this file
/// spends ONE -- which is the trade, stated with both numbers because neither
/// settles it alone.
///
/// **And the two agree exactly on which corners those are.** Over four synthetic
/// frames, 723 corners of 723 at identical positions, worst displacement 0.00 px;
/// on the repository's real frame the response map is BIT-IDENTICAL over 360 960
/// pixels and all 163 corners match
/// (`tests/test_opencv_interop.cpp`, `OpenCVInterop.RealFrameCorners`). That is
/// stronger than tier 2 owes and it is measured, not promised: the tier 2 caveat
/// is about `cv::cornerMinEigenVal`'s SOBEL path, which computes different numbers.
///
/// ---------------------------------------------------------------------------
/// PRECISION: EXACT INTEGERS IN, ONE SQUARE ROOT, `float` OUT
///
/// `sumXX`, `sumYY` and `sumXY` are exact integers bounded by `blockSize^2`. The
/// response is evaluated as
///
///     (S - sqrt(D)) / 2     with  S = xx + yy,  D = (xx - yy)^2 + 4*xy^2
///
/// rather than as the halved form above, so that **both operands of the square
/// root are integers and no rounding happens before it**. `S` and `D` are exact in
/// `double` for any window this library can express (`D <= 5*blockSize^4`, and
/// `double` is exact to 2^53). `std::sqrt` is required by IEEE-754 to be CORRECTLY
/// ROUNDED, so the `double` result carries one rounding from the root and at most
/// one from the subtraction -- and it is bit-identical on every IEEE-754 platform
/// and for every `WordType`, which tests/test_corner.cpp checks rather than
/// assumes. `double` and not `float` for the arithmetic: `float` would round `D`
/// itself once `blockSize` passes 16, which is a rounding BEFORE the root and
/// therefore not bounded by it.
///
/// **Two consequences worth stating, because the selection below compares
/// responses with `==`:**
///
///   * **The response is exactly 0.0 if and only if the window's matrix is
///     singular** (`det = xx*yy - xy^2 == 0`, which includes every empty and every
///     rank-one window -- a straight edge, or no gradient at all). `S^2 - D =
///     4*det`, so `det == 0` makes `D` the perfect square `S^2`, the root exact,
///     and the difference exactly zero. No clamping, no epsilon.
///   * **Otherwise the response is at least `1/(2*blockSize^2)`**: with `det >= 1`,
///     `S - sqrt(D) = 4*det/(S + sqrt(D)) >= 2/S >= 1/blockSize^2`. So a positive
///     response can never round to zero, in `double` or in `float`, and the
///     `val > threshold` test needs no tolerance.
///
/// **The map is stored as `float`, which is the reference's own `eig` type and
/// half the footprint of `double` (2 B/pixel saved -- 614 kB at 640x480, which is
/// FOUR TIMES the whole four-plane derivative working set the operation reads).
/// Its margin is wide at the reference's `blockSize = 3` and it NARROWS AS
/// `blockSize` GROWS, which is measured, not waved at.** Two distinct responses
/// differ by at least about `1/(2*sqrt(5)*blockSize^2)` while a `float` ulp at the
/// top of the range is about `2*blockSize^2 * 2^-23`; the headroom is ~1.3e4 at
/// `blockSize = 3` and falls to ~2 by `blockSize = 31`, so at large block sizes
/// `float` storage CAN merge two responses that differ in `double` and manufacture
/// a 3x3 tie. tests/test_corner.cpp counts exactly that -- positions where the
/// `float` map merges neighbours the `double` oracle separates, and NMS survivors
/// that differ between the two maps -- at `blockSize` 3, 7, 15 and 31, and prints
/// the counts. A caller running a large block and needing `double`'s separation
/// should take the response map from `cornerMinEigenVal` and select on its own.
///
/// ---------------------------------------------------------------------------
/// THE SELECTION, AND WHY ITS ORDER IS THE WHOLE SPECIFICATION
///
/// `goodFeaturesToTrack` performs FOUR steps and their order decides which corners
/// survive. The reference's order, read out of gftt.cpp:
///
///   1. `maxVal = minMaxLoc(eig)` over the WHOLE map, border included.
///   2. `threshold(eig, eig, maxVal*qualityLevel, 0, THRESH_TOZERO)` -- strictly
///      greater than the threshold survives.
///   3. `dilate(eig, tmp, Mat())` and keep `x` where `val != 0 && val == tmp[x]`,
///      scanning `y` in `[1, height-1)` and `x` in `[1, width-1)`. That is a 3x3
///      NON-MAXIMUM SUPPRESSION, and it EXCLUDES the outermost row and column from
///      ever being a corner.
///   4. `std::sort` descending by response, then the greedy MINIMUM-DISTANCE
///      spacing filter, accepting until `maxCorners`.
///
/// **NMS comes BEFORE the sort and the spacing filter, and swapping those two
/// stages changes the answer.** NMS's kills are not a subset of the spacing
/// filter's: the spacing filter only removes a point that is close to an ACCEPTED
/// one, while NMS removes a point that is beside a HIGHER one whether or not that
/// higher one is ever accepted. The case that pins it, and which
/// `Corner.SelectionOrder_PinsNmsBeforeDistance` builds as a response map directly:
///
///     A = (10, 10) response 100      B = (13, 10) response 99
///     C = (14, 10) response 98       minDistance = 3.5
///
///   * **NMS first (this file, and the reference).** C is beside B, which is
///     higher, so NMS deletes C. Candidates are `{A, B}`; A is accepted; B is 3
///     from A and `9 < 12.25` rejects it. Result: **{A}**.
///   * **Spacing first.** All three are ranked: A accepted; B rejected (9 < 12.25);
///     C is 4 from A and `16 >= 12.25`, so C is ACCEPTED. Result: **{A, C}** -- a
///     corner one pixel from a stronger response that was itself discarded.
///
/// This file matches the first. The test asserts `{A}` and asserts that the second
/// order would have produced `{A, C}`, so the case has teeth rather than merely
/// agreeing with whatever the code does.
///
/// **The grid is not needed for the answer, only for the reference's speed.** The
/// reference partitions the image into `cvRound(minDistance)` cells and searches
/// the 3x3 cell neighbourhood; with `cell >= minDistance - 0.5`, two points in
/// cells two apart are more than `minDistance` apart, so that search is exact and
/// an exhaustive check over the accepted points gives the IDENTICAL set. This file
/// does the exhaustive check: `maxCorners` is 200 in the reference configuration,
/// so it is at most 200*200 integer comparisons per frame, and it needs no grid --
/// which is the same as saying it needs no allocation.
///
/// **TIES ARE THE REFERENCE'S ORDER, AND THE REFERENCE DOES SPECIFY ONE.** Equal
/// responses are COMMON in this operation -- a checkerboard makes the whole
/// interior equal, and integer popcounts over a 3x3 window take few distinct
/// values -- so the tie rule decides real output. gftt.cpp sorts
/// `std::vector<const float*>`, pointers INTO the `eig` map collected in ascending
/// raster order, with
///
///     struct greaterThanPtr {                 // opencv_internal/include/gftt.hpp
///         bool operator()(const float* a, const float* b) const
///         // Ensure a fully deterministic result of the sort
///         { return (*a > *b) ? true : (*a < *b) ? false : (a > b); }
///     };
///
/// The comment is the reference's own. That third arm is a strict total order over
/// distinct addresses, and on a contiguous `CV_32F` map a larger address is a LATER
/// raster position -- so equal responses come out **later-position-first**, and the
/// reference's sort is fully determined rather than unspecified.
///
/// `impl::CornerStronger` is that rule spelled on coordinates: response
/// descending, then `y` DESCENDING, then `x` descending. It is not a refinement or
/// a deviation, it is the same order, which is what a tier 2 operation validated
/// against the reference's behaviour owes it. The direction matters: with
/// `minDistance` 3 and equal responses at `(1, 1)` and `(3, 1)` the greedy spacing
/// filter keeps whichever comes first, so an ascending tiebreak returns `(1, 1)`
/// and the reference returns `(3, 1)`. On a checkerboard, where the whole interior
/// ties, that moves every selected position.
///
/// ---------------------------------------------------------------------------
/// THE BORDER, AND THE RING OF SPURIOUS CORNERS THAT IS NOT THERE
///
/// D-19 chose BORDER_REFLECT_101 for the derivative partly BECAUSE a zero fill
/// would manufacture an edge around the whole frame for this operation to detect.
/// That reasoning is checked rather than repeated: `Corner.BorderRing_*` runs the
/// detector on a blank frame and on an all-ones frame, at every word type and
/// every block size, and requires ZERO corners -- and then runs the same frames
/// through a BORDER_CONSTANT derivative and requires the ring to APPEAR, so the
/// check can fail. With reflect-101, `src(x+1) == src(x-1)` at every edge, the
/// derivative is exactly zero there, a uniform frame has an all-zero response map,
/// `maxVal` is 0, and THRESH_TOZERO admits nothing.
///
/// **The response map's own border differs from the reference's by design.** T3.6
/// CLIPS a window at the frame edge (promise 2, D-13) where `cv::boxFilter` in the
/// reference replicates it under BORDER_REPLICATE, so this file's response is
/// smaller than the reference's in the outermost `blockSize/2` rows and columns.
/// At the reference's `blockSize = 3` that is the outermost row and column alone,
/// which step 3 above excludes from candidacy in both implementations -- so it can
/// move nothing except `maxVal`, and only downward, since a clipped sum is never
/// larger. At larger block sizes the deviation reaches inside the border and is a
/// real difference; it is recorded here rather than hidden.
///
/// ---------------------------------------------------------------------------
/// WHAT THIS OPERATION PROMISES
///
///  1. **Views, never containers, in the kernel** (D-5). The response map is a
///     caller-owned `float` buffer described by ResponseMap, in the same
///     pointer/extent/stride shape the bit views use.
///  2. **NO HEAP, ANYWHERE.** No allocation in the response map, in the NMS scan,
///     in the ranking or in the spacing filter. The candidate list IS the caller's
///     output array (see the capacity contract on selectGoodFeatures), the ranking
///     is an in-place `std::sort` and a bounded `std::push_heap`/`pop_heap`, and
///     the spacing filter compacts in place. tests/test_corner.cpp counts
///     `operator new` -- plain and C++17 over-aligned -- across every entry point
///     and requires zero.
///  3. **Never throws** (ARCHITECTURE 5.3). Mismatched dimensions, a non-positive
///     `blockSize`, a stride too short for a row and a null map are programming
///     errors reported by BINCV_ASSERT in debug builds and undefined in release.
///     There is no error return.
///  4. **Ternary planes only, i.e. pyramid level 0.** The container spelling takes
///     `TernaryMat` and rejects `SignedQuantMat<N>` for `N > 1` at compile time;
///     the view spelling cannot and does not.
///     **THIS IS NOW T3.7's OWN LIMIT AND NOT ops/covariance.hpp's.** That file's
///     promise 1 used to say the same thing and no longer does: T3.10 gave the
///     covariance a bit-sliced N-bit kernel, because X-20 found the tracker's
///     accuracy failure IS the 1-bit pyramid. The corner response has not been
///     widened to match -- it reads `countAndSplit` and `SlidingWindowCount`
///     directly rather than going through `gradientCovariance`, so widening it is
///     its own piece of work and not a re-export of T3.10's.
///  5. **Padding is never counted** (D-13), inherited from the reductions.

#include <algorithm>  // std::sort, push_heap, pop_heap -- none of which allocates
#include <cmath>      // std::sqrt, correctly rounded (see PRECISION above)
#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
#include "../quantMat.hpp"
// countAndSplit and SlidingWindowCount: the cross term that recomputes and the
// two sums that slide. Nothing here writes a word loop.
#include "reduce.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief A caller-owned, non-owning view of a `float` response map.
/// @note **API TIER 3 as a TYPE** -- OpenCV passes a `cv::Mat`, and binCV has no
///       float container. This is the same pointer/extent/stride shape
///       BinMatConstView uses (D-5), so a caller can point it at a `cv::Mat`'s
///       rows, a stack array or a pool block without a conversion.
/// @note `stride` is in ELEMENTS, not bytes, matching BinMatView's "in words".
struct ResponseMap {
    float* data = nullptr;  ///< first element of row 0
    size_t width = 0;       ///< row length in pixels
    size_t height = 0;      ///< number of rows
    size_t stride = 0;      ///< distance between rows, in floats

    bool empty() const { return data == nullptr || width == 0 || height == 0; }

    float* row(size_t y) {
        BINCV_ASSERT(stride != 0 || height <= 1, "corner: multi-row map needs a non-zero stride");
        return data + y * stride;
    }
    const float* row(size_t y) const {
        BINCV_ASSERT(stride != 0 || height <= 1, "corner: multi-row map needs a non-zero stride");
        return data + y * stride;
    }
};

/// @brief The read-only spelling of ResponseMap (D-9's two-view-types rule).
struct ConstResponseMap {
    const float* data = nullptr;
    size_t width = 0;
    size_t height = 0;
    size_t stride = 0;

    ConstResponseMap() = default;
    ConstResponseMap(const float* data_, size_t width_, size_t height_, size_t stride_)
        : data(data_), width(width_), height(height_), stride(stride_) {}
    /// @note Implicit, so `selectGoodFeatures(map, ...)` compiles with a mutable
    ///       ResponseMap. Unlike the bit views this is never a template argument,
    ///       so D-9's deduction hazard does not arise.
    ConstResponseMap(const ResponseMap& m)
        : data(m.data), width(m.width), height(m.height), stride(m.stride) {}

    bool empty() const { return data == nullptr || width == 0 || height == 0; }

    const float* row(size_t y) const {
        BINCV_ASSERT(stride != 0 || height <= 1, "corner: multi-row map needs a non-zero stride");
        return data + y * stride;
    }
};

/// @brief One detected corner: integer pixel coordinates and its response.
/// @note **Integer coordinates, matching the reference.** gftt.cpp emits
///       `cv::Point2f((float)x, (float)y)` -- a float type holding an integer
///       value, because `cv::goodFeaturesToTrack`'s signature says `Point2f`.
///       Nothing in this operation is subpixel; T3.8's optical flow is where
///       subpixel positions come from (ARCHITECTURE 7.9).
/// @note Deliberately NOT named `KeyPoint`: `cv::KeyPoint` carries size, angle,
///       octave and class fields this does not have, and Tier 2 borrows call
///       shapes, not struct layouts.
struct Corner {
    int x = 0;              ///< column
    int y = 0;              ///< row
    float response = 0.0f;  ///< minimum eigenvalue at (x, y); see PRECISION above
};

/// @brief The four parameters `goodFeaturesToTrack` takes, defaulted to the values
///        the reference pipeline actually runs.
/// @note The defaults are SEAL/seal_params.yaml verbatim: `gftt_max_corners: 200`,
///       `gftt_quality_level: 0.01`, `gftt_min_distance: 33.33333333333`,
///       `gftt_block_size: 3`. `gftt_use_harris_detector: 0` is why there is no
///       Harris option here at all -- the reference selects the minimum
///       eigenvalue, and an unused second response function is an untested one.
///       `gftt_gradient_size: 3` is the Sobel aperture, which the BINARIZED
///       derivative path ignores (it is a fixed `[-1, 0, 1]` tap), so it has no
///       field.
struct GoodFeaturesParams {
    int maxCorners = 200;                    ///< <= 0 means "no limit" (reference behaviour)
    double qualityLevel = 0.01;              ///< survivors need response > this * max response
    double minDistance = 33.33333333333;     ///< < 1 disables the spacing filter, as in gftt.cpp
    int blockSize = 3;                       ///< covariance window, square, centred as OpenCV centres it
};

/// @brief What `goodFeaturesToTrack` / `selectGoodFeatures` report back.
/// @note `candidatesTruncated` is not decoration. It is the ONLY way a caller
///       learns that the answer is a restriction of the reference's rather than
///       equal to it -- see the capacity contract on selectGoodFeatures.
struct CornerResult {
    size_t count = 0;                  ///< corners written to the caller's array
    size_t candidatesRanked = 0;       ///< NMS survivors that were ranked
    bool candidatesTruncated = false;  ///< true when `capacity` could not hold every NMS survivor
};

namespace impl {

/// @brief The smaller eigenvalue of `[[xx, xy], [xy, yy]]`, from exact integers.
/// @return `(S - sqrt(D))/2` with `S = xx + yy` and `D = (xx - yy)^2 + 4*xy^2`,
///         rounded once to `float`.
/// @note Spelled with INTEGER operands under the root on purpose: the halved form
///       `(a + c) - sqrt((a - c)^2 + b^2)` that gftt.cpp uses rounds `a`, `b` and
///       `c` before the root, which is a rounding the root's correct rounding does
///       not bound. These two agree exactly whenever the halved form is exact, and
///       this one is exact strictly more often.
/// @note Exactly 0.0f iff `xx*yy - xy*xy == 0`; otherwise at least
///       `1/(2*blockSize^2)`. See PRECISION at the top of the file -- the `!= 0`
///       test in the selection depends on it and needs no epsilon.
inline float minEigenValue(long long xx, long long yy, long long xy) {
    const double s = static_cast<double>(xx) + static_cast<double>(yy);
    const double d = static_cast<double>(xx) - static_cast<double>(yy);
    const double c = static_cast<double>(xy);
    const double disc = d * d + 4.0 * c * c;
    return static_cast<float>(0.5 * (s - std::sqrt(disc)));
}

/// @brief The window OpenCV's box filter of side `blockSize` reduces at pixel
///        `(x, y)`, anchored where `cv::Point(-1, -1)` puts it.
/// @note `cv::boxFilter`'s default anchor is `ksize/2` by integer division, so an
///       ODD `blockSize` is centred and an EVEN one leans up and left. Written out
///       rather than assumed, because "centre" is ambiguous for even sides and the
///       reference passes 3.
inline Rect blockWindow(int x, int y, int blockSize) {
    const int off = blockSize / 2;
    return Rect(x - off, y - off, blockSize, blockSize);
}

/// @brief Strict weak ordering over corners: response DESCENDING, ties broken by
///        DESCENDING raster position -- larger `y` first, then larger `x`.
/// @note **THE TIE RULE IS THE REFERENCE'S, NOT A CHOICE MADE HERE.** gftt.cpp
///       sorts `std::vector<const float*>` -- pointers INTO the `eig` map,
///       collected in ascending raster order -- with
///       `cv_internal::greaterThanPtr`, whose body is
///       `(*a > *b) ? true : (*a < *b) ? false : (a > b)` and whose comment says
///       "Ensure a fully deterministic result of the sort". On a contiguous
///       `CV_32F` map a larger address IS a later raster position, so equal
///       responses come out LATER-POSITION-FIRST there. This comparator is that
///       rule spelled on coordinates instead of addresses. See "TIES ARE THE
///       REFERENCE'S" at the top of the file.
/// @note A TOTAL order over distinct positions, as `greaterThanPtr` is over
///       distinct addresses -- so `std::sort` needs no stability from either.
struct CornerStronger {
    bool operator()(const Corner& a, const Corner& b) const {
        if (a.response != b.response) return a.response > b.response;
        if (a.y != b.y) return a.y > b.y;
        return a.x > b.x;
    }
};

} // namespace impl

// ---------------------------------------------------------------------------
// STAGE 1 -- the response map
// ---------------------------------------------------------------------------

/// @brief The minimum-eigenvalue corner response at every pixel, from binarized
///        ternary derivatives. **API TIER 2** -- `cv::cornerMinEigenVal`'s role
///        with the reference pipeline's BINARIZED numerics, NOT bit-exact against
///        it (see the tier note at the top of this file).
/// @tparam WordType The views' word type (D-1).
/// @param magX Magnitude plane of the x-derivative -- `dx.constMagnitude(0)`.
/// @param magY Magnitude plane of the y-derivative -- `dy.constMagnitude(0)`.
/// @param signX Sign plane of the x-derivative -- `dx.constSign()`; set is NEGATIVE.
/// @param signY Sign plane of the y-derivative -- `dy.constSign()`.
/// @param blockSize Side of the square covariance window, >= 1. The reference uses
///        3. Anchored as `cv::boxFilter` anchors it (impl::blockWindow).
/// @param dst Caller-owned response map with the planes' dimensions. Every pixel
///        is written; nothing is read from it.
///
/// @note **This is the sliding form.** Two `SlidingWindowCount`s carry `sumXX` and
///       `sumYY` down each column; only the cross term is recomputed per position,
///       because nothing in ops/reduce.hpp slides a split. The full argument, and
///       what it does and does not predict about the speedup, is under "THE
///       SLIDING FORM" at the top of the file.
/// @note **Windows CLIP at the frame edge** (T3.6 promise 2, D-13) where the
///       reference's box filter replicates. See "THE BORDER" above for exactly
///       which pixels that can move.
/// @note **TERNARY PLANES ONLY, and this overload cannot check it** -- a
///       `BinMatConstView` carries no plane count, so an N-bit level's planes
///       compile here and give the LSB plane's response. Prefer the container
///       spelling. ops/covariance.hpp promise 1 has the measured example.
/// @note Never throws; no allocation; no scratch beyond two accumulators.
template <typename WordType>
inline void cornerMinEigenVal(BinMatConstView<WordType> magX, BinMatConstView<WordType> magY,
                              BinMatConstView<WordType> signX, BinMatConstView<WordType> signY,
                              int blockSize, ResponseMap dst) {
    BINCV_ASSERT(magX.width == magY.width && magX.height == magY.height &&
                     magX.width == signX.width && magX.height == signX.height &&
                     magX.width == signY.width && magX.height == signY.height,
                 "corner: the four derivative planes must have the same dimensions");
    BINCV_ASSERT(dst.width == magX.width && dst.height == magX.height,
                 "corner: the response map must have the planes' dimensions");
    BINCV_ASSERT(blockSize > 0, "corner: blockSize must be positive");

    if (magX.width == 0 || magX.height == 0) return;

    BINCV_ASSERT(dst.data != nullptr, "corner: a non-empty response map needs a non-null pointer");
    BINCV_ASSERT(dst.stride >= dst.width || dst.height <= 1,
                 "corner: multi-row map needs a stride covering a whole row");

    const int width = static_cast<int>(magX.width);
    const int height = static_cast<int>(magX.height);
    const int off = blockSize / 2;

    // COLUMN-MAJOR, because SlidingWindowCount slides downward and a row-major
    // sweep would need one accumulator per column -- caller-owned scratch this
    // operation does not take. Two accumulators, rebuilt per column, is the price.
    for (int x = 0; x < width; ++x) {
        SlidingWindowCount<WordType> xxSlide(magX, Rect(x - off, -off, blockSize, blockSize));
        SlidingWindowCount<WordType> yySlide(magY, Rect(x - off, -off, blockSize, blockSize));

        for (int y = 0; y < height; ++y) {
            // The one number with no incremental form: `magX & magY` split by
            // `signX ^ signY`, recomputed per position through the four-argument
            // form -- the one that needs no selector plane (D-15 axis 3).
            const SplitCount xy = countAndSplit<WordType>(magX, magY, signX, signY,
                                                          impl::blockWindow(x, y, blockSize));
            dst.row(static_cast<size_t>(y))[static_cast<size_t>(x)] = impl::minEigenValue(
                static_cast<long long>(xxSlide.count()), static_cast<long long>(yySlide.count()),
                xy.crossTerm());
            xxSlide.slideDown();
            yySlide.slideDown();
        }
    }
}

/// @brief The container spelling of cornerMinEigenVal. **API TIER 2.**
/// @param dx Horizontal ternary derivative -- what `derivativeX` writes at level 0.
/// @param dy Vertical ternary derivative, with `dx`'s dimensions.
/// @note **Ternary only.** `TernaryMat<W>` is `SignedQuantMat<1, W>`, so an N-bit
///       level is "no matching function" here rather than a silently wrong map.
/// @note A thin naming of the view form: the container knows which plane is
///       magnitude and which is sign, and the kernel does not have to (D-5).
template <typename WordType>
inline void cornerMinEigenVal(const TernaryMat<WordType>& dx, const TernaryMat<WordType>& dy,
                              int blockSize, ResponseMap dst) {
    cornerMinEigenVal<WordType>(dx.constMagnitude(0), dy.constMagnitude(0), dx.constSign(),
                                dy.constSign(), blockSize, dst);
}

// ---------------------------------------------------------------------------
// STAGE 2 -- the selection
// ---------------------------------------------------------------------------

/// @brief The quality threshold, 3x3 non-maximum suppression and minimum-distance
///        spacing filter `cv::goodFeaturesToTrack` performs, over an existing
///        response map. **API TIER 2.**
/// @param response The map, typically from cornerMinEigenVal. Read only.
/// @param params Quality level, minimum distance and corner cap. `blockSize` is
///        unused here -- it belongs to the map, which is already computed.
/// @param corners Caller-owned output array. **It is also the candidate buffer**;
///        see the capacity contract below.
/// @param capacity Number of `Corner`s `corners` can hold.
/// @return `{count, candidatesRanked, candidatesTruncated}`. The first `count`
///         entries of `corners` are the selected corners, strongest first.
///
/// @note **THE CAPACITY CONTRACT, WHICH IS NOT `maxCorners`.** `capacity` bounds
///       the NMS survivors this call can RANK, not the corners it returns. The
///       result equals the reference's exactly when `capacity` is at least the
///       number of survivors -- whose worst case is `(width-2)*(height-2)`, since
///       every interior pixel of an all-equal map is a 3x3 maximum. When capacity
///       is smaller, the `capacity` strongest survivors are ranked, the rest are
///       dropped, and `candidatesTruncated` is set: the returned `count` is then a
///       LOWER BOUND on the reference's, because a dropped survivor might have
///       been accepted by the spacing filter after the ranked ones ran out. The
///       flag is honest at `capacity == 0` too: a zero-length buffer holds no
///       survivor, so it reports truncation whenever the map has one, and only a
///       map with no survivor at all gives `{0, 0, false}`.
///       **Sizing `capacity` to `maxCorners` is the mistake this note exists to
///       prevent** -- 200 candidates out of 5000 fill the spacing filter's first
///       few slots and then run out. The reference pays an unbounded
///       `std::vector<const float*>` for the same thing; this operation takes the
///       caller's buffer instead, because a kernel does not allocate (D and
///       CLAUDE.md's hard rules).
/// @note **The order of the four steps IS the specification.** Threshold, then
///       NMS, then rank, then spacing -- gftt.cpp's order. Swapping NMS and the
///       spacing filter changes which corners survive; the case that pins it is at
///       the top of this file and in `Corner.SelectionOrder_PinsNmsBeforeDistance`.
/// @note **The outermost row and column can never be corners.** The reference's
///       candidate scan runs `[1, height-1) x [1, width-1)` -- so does this one,
///       which is also why the map's border deviation from the reference cannot
///       reach the output.
/// @note Ties in response are broken by DESCENDING raster position, which is what
///       gftt.cpp's `greaterThanPtr` does on pointers into a contiguous map. See
///       "TIES ARE THE REFERENCE'S ORDER" at the top of this file.
/// @note No allocation: an in-place bounded heap, an in-place `std::sort`, and an
///       in-place compaction. Never throws.
inline CornerResult selectGoodFeatures(ConstResponseMap response,
                                       const GoodFeaturesParams& params, Corner* corners,
                                       size_t capacity) {
    BINCV_ASSERT(params.qualityLevel > 0.0, "corner: qualityLevel must be positive");
    BINCV_ASSERT(params.minDistance >= 0.0, "corner: minDistance must not be negative");
    BINCV_ASSERT(corners != nullptr || capacity == 0,
                 "corner: a non-zero capacity needs a non-null corner array");

    CornerResult out;
    // NOT `|| capacity == 0`. A caller passing 0 still gets an honest
    // `candidatesTruncated`, because "the buffer could not hold every NMS
    // survivor" is true of a zero-length buffer whenever a survivor exists -- and
    // that flag is the only way a caller learns the answer is a restriction of the
    // reference's. The scan below returns as soon as it finds the first survivor.
    if (response.empty()) return out;
    BINCV_ASSERT(response.stride >= response.width || response.height <= 1,
                 "corner: multi-row map needs a stride covering a whole row");

    const int width = static_cast<int>(response.width);
    const int height = static_cast<int>(response.height);

    // 1. maxVal over the WHOLE map, border included -- cv::minMaxLoc's region.
    float maxVal = response.row(0)[0];
    for (int y = 0; y < height; ++y) {
        const float* r = response.row(static_cast<size_t>(y));
        for (int x = 0; x < width; ++x) {
            if (r[static_cast<size_t>(x)] > maxVal) maxVal = r[static_cast<size_t>(x)];
        }
    }

    // 2. The THRESH_TOZERO cut, in the reference's own arithmetic: the product is
    //    formed in double (cv::threshold takes a double) and narrowed to float
    //    (cv::threshold narrows for a CV_32F source), and the comparison is
    //    STRICTLY greater. A map that is everywhere zero gives maxVal == 0, thr ==
    //    0, and no survivor -- which is the blank-frame case, and is why no
    //    special case for it exists.
    const float threshold = static_cast<float>(static_cast<double>(maxVal) * params.qualityLevel);

    // 3. NMS, fused with the threshold. `dilate` + `val == tmp[x]` on the
    //    THRESHOLDED map is exactly "val > threshold and val is the maximum of its
    //    3x3 neighbourhood in the RAW map": a neighbour at or below the threshold
    //    is zeroed and cannot exceed val, which is itself above the threshold. So
    //    the reference's second frame-sized float buffer is not needed and is not
    //    taken.
    size_t ranked = 0;
    for (int y = 1; y + 1 < height; ++y) {
        const float* prev = response.row(static_cast<size_t>(y - 1));
        const float* cur = response.row(static_cast<size_t>(y));
        const float* next = response.row(static_cast<size_t>(y + 1));
        for (int x = 1; x + 1 < width; ++x) {
            const float val = cur[static_cast<size_t>(x)];
            if (!(val > threshold)) continue;
            bool isMax = true;
            for (int dx = -1; dx <= 1 && isMax; ++dx) {
                const size_t c = static_cast<size_t>(x + dx);
                if (prev[c] > val || cur[c] > val || next[c] > val) isMax = false;
            }
            if (!isMax) continue;

            Corner candidate;
            candidate.x = x;
            candidate.y = y;
            candidate.response = val;

            if (ranked < capacity) {
                corners[ranked++] = candidate;
                std::push_heap(corners, corners + ranked, impl::CornerStronger());
            } else {
                // Full: the buffer is a max-heap under CornerStronger, and
                // CornerStronger is the SORT order -- "less" means "stronger" --
                // so the heap's maximum, `corners[0]`, is the WEAKEST retained
                // candidate. That is the one to evict, and the test below is
                // exactly "does this candidate beat the weakest we kept".
                //
                // THE COMPARATOR MUST BE THE SORT'S OWN, NOT ITS REVERSE. A heap
                // built with the reverse order puts the STRONGEST at the root, and
                // the eviction test then fires only for a new global maximum --
                // which discards the previous maximum and leaves the buffer
                // holding the first-scanned survivors instead of the strongest
                // ones. `Corner.CapacityContract` pins the prefix property that
                // catches that, not merely `corners[0]`.
                out.candidatesTruncated = true;
                if (capacity == 0) {
                    // Nothing can ever be ranked, so nothing further can change.
                    // The flag is still the honest answer: survivors exist and this
                    // call could hold none of them.
                    out.candidatesRanked = 0;
                    return out;
                }
                if (impl::CornerStronger()(candidate, corners[0])) {
                    std::pop_heap(corners, corners + ranked, impl::CornerStronger());
                    corners[ranked - 1] = candidate;
                    std::push_heap(corners, corners + ranked, impl::CornerStronger());
                }
            }
        }
    }
    out.candidatesRanked = ranked;
    if (ranked == 0) return out;

    // 4a. Rank. std::sort is in place and does not allocate; std::stable_sort
    //     would, which is why the comparator is a TOTAL order instead.
    std::sort(corners, corners + ranked, impl::CornerStronger());

    // 4b. The greedy spacing filter, compacting accepted corners to the front. The
    //     write index never passes the read index, so the compaction is safe in
    //     place. Exhaustive rather than gridded: see "The grid is not needed for
    //     the answer" at the top of the file.
    const size_t limit = (params.maxCorners > 0)
                             ? std::min(capacity, static_cast<size_t>(params.maxCorners))
                             : capacity;
    size_t kept = 0;
    if (params.minDistance >= 1.0) {
        const double minDistanceSq = params.minDistance * params.minDistance;
        for (size_t i = 0; i < ranked && kept < limit; ++i) {
            const Corner candidate = corners[i];
            bool good = true;
            for (size_t j = 0; j < kept; ++j) {
                const double dx = static_cast<double>(candidate.x) - static_cast<double>(corners[j].x);
                const double dy = static_cast<double>(candidate.y) - static_cast<double>(corners[j].y);
                if (dx * dx + dy * dy < minDistanceSq) {
                    good = false;
                    break;
                }
            }
            if (good) corners[kept++] = candidate;
        }
    } else {
        // gftt.cpp's `else` branch: no spacing at all, just the strongest
        // `maxCorners`. Already sorted and already in place.
        kept = (ranked < limit) ? ranked : limit;
    }
    out.count = kept;
    return out;
}

// ---------------------------------------------------------------------------
// STAGE 3 -- the whole operation
// ---------------------------------------------------------------------------

/// @brief `goodFeaturesToTrack` over a binarized ternary derivative pair: the
///        response map, then the selection. **API TIER 2** -- the same role and
///        call shape as `cv::goodFeaturesToTrack`, with the reference pipeline's
///        BINARIZED numerics. **Not bit-exact against OpenCV.**
/// @tparam WordType The containers' word type (D-1).
/// @param dx Horizontal ternary derivative (ops/derivative.hpp, level 0).
/// @param dy Vertical ternary derivative, with `dx`'s dimensions.
/// @param params Defaults are SEAL/seal_params.yaml's four values.
/// @param scratch Caller-owned response map with the derivatives' dimensions. It
///        is written, then read. **This is the operation's whole memory cost**:
///        4 bytes per pixel, 1 228 800 B at 640x480 -- eight times the four
///        one-bit planes it reads, and the reason it is the caller's to place,
///        reuse across frames, or point at a pool.
/// @param corners Caller-owned output array, also the candidate buffer.
/// @param capacity Entries in `corners`. **Read selectGoodFeatures' capacity
///        contract**: this is not `maxCorners`.
/// @return `{count, candidatesRanked, candidatesTruncated}`.
///
/// @note **Ternary only** -- `TernaryMat<W>` is `SignedQuantMat<1, W>`, so an
///       N-bit pyramid level does not match this overload (T3.6 promise 1).
/// @note There is deliberately no mask parameter. `cv::goodFeaturesToTrack` takes
///       one and the reference passes it through; the MVP pipeline
///       (ARCHITECTURE 7) has no caller for it, and an untested parameter is worse
///       than an absent one. A caller wanting a mask can call cornerMinEigenVal,
///       zero the masked pixels of the map, and call selectGoodFeatures -- which
///       is the same two-stage split this file already exposes.
/// @note Never throws; allocates nothing.
template <typename WordType>
inline CornerResult goodFeaturesToTrack(const TernaryMat<WordType>& dx,
                                        const TernaryMat<WordType>& dy,
                                        const GoodFeaturesParams& params, ResponseMap scratch,
                                        Corner* corners, size_t capacity) {
    cornerMinEigenVal<WordType>(dx, dy, params.blockSize, scratch);
    return selectGoodFeatures(ConstResponseMap(scratch), params, corners, capacity);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
