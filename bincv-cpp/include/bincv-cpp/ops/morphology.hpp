#pragma once

/// @file morphology.hpp
/// @brief erode / dilate / morphologyEx on bit-packed binary frames.
///
/// **API TIER 1** (the design notes): every entry point here is bit-exact
/// against `cv::erode`, `cv::dilate` and `cv::morphologyEx` on the same binary
/// content stored as `CV_8U` -- interior, edge and corner alike, for every
/// `BorderType` and every structuring element this file can express.
///
/// ---------------------------------------------------------------------------
/// THE ALGEBRA, AND WHY IT IS NOT WHAT THIS FILE RUNS
///
/// On a binary image the two primitives are shifted ORs and shifted ANDs:
///
/// dilate(src, E) = OR over (dx,dy) in E of shift(src, dx, dy, fill = 0)
/// erode (src, E) = AND over (dx,dy) in E of shift(src, dx, dy, fill = 1)
///
/// with `(dx, dy) = (ex - anchorX, ey - anchorY)` over the element's set cells --
/// the same offset sign for BOTH operations, because OpenCV does not flip the
/// kernel (its `dst(x,y) = max/min over element of src(x + x', y + y')`).
///
/// **The two fills are opposite and that is **, measured rather than
/// argued: a pixel outside the image must contribute nothing, and "nothing" is 0
/// to an OR and 1 to an AND. One fixed fill makes one of the two wrong at every
/// edge -- flipping erode's fill to `false` dropped the composition suite that
/// preceded this file from 12960/12960 to 11056/12960. OpenCV encodes the same
/// asymmetry through `morphologyDefaultBorderValue`, which resolves to the
/// depth's maximum for erosion and its minimum for dilation, and
/// `tests/test_shift.cpp` (`Shift.MorphologyFillPremise`) pins that claim
/// against the real `cv::erode` / `cv::dilate` rather than citing it.
/// Re-measured against THIS file: flipping `erode`'s default fill back to `false`
/// fails 28862 of `tests/test_morphology.cpp`'s 298541 checks.
///
/// **What this file actually runs is the FUSED form of that composition**, for
/// one reason: the composed spelling needs a frame-sized temporary between the
/// shift and the combine, and CLAUDE.md forbids a kernel allocating one. A
/// caller-provided scratch buffer for `erode` would put a frame of memory on
/// every call site in the MVP's hot path, and memory is co-equal with speed here.
/// So the accumulation happens IN THE DESTINATION ROW, word by word: for each
/// destination row the row is set to the combining operation's identity and each
/// element cell's shifted source word is folded into it. One pass, zero scratch,
/// and the source is never written.
///
/// The composition is not thereby unverified -- it is verified harder. A
/// structuring element with exactly ONE set cell reduces this kernel to a single
/// shift, so `Morphology.SingleOffsetEqualsShift_*` requires it to agree with
/// `ops/shift.hpp`'s `shift` pixel for pixel, at every offset in the element
/// and at all five `BorderType` values, in the three configurations that have no
/// OpenCV at all. The word recurrence below and the one in `ops/shift.hpp` are
/// separately written; that test is what says they mean the same thing.
///
/// ---------------------------------------------------------------------------
/// THE BORDER IS THE RISK, AND IT IS HANDLED IN TWO PIECES
///
/// * VERTICALLY, always exactly. A destination row's source row is
/// `impl::borderIndex(y + dy, height, type)` -- the same closed form
/// `ops/shift.hpp` is Tier 1 against `cv::borderInterpolate` for. Under
/// `BORDER_CONSTANT` an out-of-image row folds the constant in, and when the
/// constant IS the identity (the morphological default) it is skipped.
///
/// * HORIZONTALLY, in the word path for `BORDER_CONSTANT` and in a per-pixel
/// fixup for the other four. `impl::extendedRowWord` reads the constant for
/// everything past the row's pixels, which is exact; the four non-constant
/// types map each out-of-range column to a DIFFERENT source column, so there
/// is no word-wide answer. Those columns are recomputed one pixel at a time
/// after the row is accumulated -- at most `maxOffsetX` columns at each edge,
/// where `maxOffsetX` is the element's horizontal reach (1 for every 3x3).
/// The interior stays word-parallel, which is the whole point of the packed
/// representation.
///
/// A column strictly inside `[maxOffsetX, width - maxOffsetX)` cannot reach the
/// border under ANY cell of the element, which is what makes the split safe: the
/// fill the word path used there is never read.
///
/// ---------------------------------------------------------------------------
/// PRECONDITION ON `dst`, IDENTICAL TO ops/logic.hpp's AND ops/shift.hpp's: it
/// must span its image's full width, or end on a word boundary
/// (`width % WordBits == 0`). Every destination row's trailing partial word is
/// stored masked -- that is CLAUDE.md's padding-bits-stay-zero rule, and it is
/// what `morphApply` does after each row -- which writes zeros into bits
/// [width, rowWords * WordBits). Those are padding in the usual case, and a WIDER
/// image's next 1..WordBits-1 live pixels when `dst` is a sub-width window onto
/// one. Measured: a 70-pixel-wide destination windowed onto a 640-wide `uint32_t`
/// image cleared all 26 live pixels in columns 70..95 of each row. NOTHING
/// DIAGNOSES IT -- every address written is inside the parent image, so it is
/// neither an out-of-bounds write nor an aliasing violation, and a pixel
/// comparison against the window itself sees the right answer.
///
/// The same holds for `scratch`, which the compound ops write with exactly these
/// kernels. Sources are unaffected: nothing past `width` is ever read as a pixel.
///
/// ---------------------------------------------------------------------------
/// SCRATCH: WHO NEEDS IT, AND EXACTLY ONE FRAME
///
/// `erode` and `dilate` need none. The five compound operations each need one
/// intermediate image, and it is the CALLER'S -- `morphologyEx` takes it as a
/// view like every other argument. One frame is enough for all five, which
/// is not obvious for TOPHAT and BLACKHAT and is worth writing down:
///
/// OPEN erode(src -> scratch); dilate(scratch -> dst)
/// CLOSE dilate(src -> scratch); erode(scratch -> dst)
/// GRADIENT dilate(src -> dst); erode(src -> scratch); dst &= ~scratch
/// TOPHAT open(src -> dst) using scratch; dst = src & ~dst
/// BLACKHAT close(src -> dst) using scratch; scratch = ~src; dst &= scratch
///
/// TOPHAT and BLACKHAT would each need a second frame if the subtraction were
/// written the obvious way; they do not, because `ops/logic.hpp` supports the
/// exact-alias in-place case and the scratch is dead by the time the
/// subtraction runs.
///
/// **The subtractions are bitwise and that is exact, not an approximation.**
/// `cv::morphologyEx` computes GRADIENT, TOPHAT and BLACKHAT with `cv::subtract`
/// on `CV_8U`, which SATURATES. On content in {0, 255}, `a - b` saturating is 255
/// exactly when `a == 255 && b == 0`, i.e. `a & ~b`. That equivalence holds
/// whatever the element is -- it does not rest on opening being anti-extensive,
/// which is false for an element whose anchor cell is not set.
///
/// ---------------------------------------------------------------------------
/// WHAT IS NOT HERE
///
/// `iterations`. `cv::erode(..., iterations = n)` is `n` sequential erosions, and
/// n > 1 needs a second buffer to ping-pong through. A caller that wants it can
/// write the loop with the scratch it already owns, and the MVP's pipeline does
/// not (SEAL uses single-pass 3x3 morphology). Adding a parameter that silently
/// requires more memory than the signature shows would be the wrong default for
/// this project.
///
/// A custom `borderValue` on `morphologyEx`. Each step uses the morphological
/// default for ITS operation, which is what `cv::morphologyEx` does when it is
/// passed `morphologyDefaultBorderValue` -- its own default. A caller who wants
/// a literal constant on both steps composes `erode`/`dilate` directly, where the
/// parameter is exposed.

#include <cmath>
#include <cstddef>
#include <cstdint>

// impl::rowTailMask / strideCoversARow / viewsShareNoWord, and through it
// binMat.hpp's word helpers and the two view types.
#include "../impl/kernel_util.hpp"

// impl::borderIndex, impl::extendedRowWord, impl::isKnownBorderType -- the Tier 1
// border mapping this file inherits rather than re-derives -- and shift itself,
// which the single-cell equivalence test compares against.
#include "shift.hpp"

// bitwiseAnd / bitwiseNot for the three compound operations' subtraction.
#include "logic.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

// ---------------------------------------------------------------------------
// The structuring element
// ---------------------------------------------------------------------------

/// @brief A morphological structuring element: a shape, an extent and an anchor.
///
/// @note **Mirrors `cv::getStructuringElement` exactly**, including the parts of
/// it that are surprising, because Tier 1 means the two must agree cell for
/// cell (`Morphology.ElementMatchesOpenCv_*` compares every cell of every
/// shape at every size in the sweep against the real function):
///
/// - `MORPH_ELLIPSE` at 3x3 is a PLUS, not a filled square. OpenCV's
/// half-axis is `rows/2` and `cols/2`, so at 3x3 the row offsets +/-1
/// admit only the center column.
/// - `MORPH_CROSS` is centerd on the ANCHOR, not on the element. That is
/// `getStructuringElement`'s behavior and it is why the anchor is a
/// field here rather than a separate argument.
/// - A 1x1 element is a filled 1x1 whatever the shape says.
///
/// @note **Nothing is stored per cell and nothing is allocated.** The parametric
/// shapes are evaluated by `activeAt` on demand -- a few integer
/// comparisons, or one `sqrt` per element ROW for the ellipse, hoisted by
/// the kernels into a span so the inner loop never sees it. That keeps this
/// a 32-byte value type that can be a constant, a member, or a temporary in
/// a call expression, at any odd size, with no capacity limit and no heap.
///
/// @note **`mask` is the escape hatch, and it is a view, not a container**.
/// When non-null it names a caller-owned, row-major, `cols * rows` array of
/// bytes -- non-zero means set -- and `shape` is ignored. The memory must
/// outlive the element. This exists because the three parametric shapes are
/// all symmetric about their center, and a suite built only from them
/// cannot catch an inverted offset sign: negating a symmetric offset set
/// gives the same set back. See `Morphology.Asymmetric_*`.
///
/// @note An off-center ANCHOR breaks that symmetry too, and is the cheaper way to
/// get an asymmetric case: the offsets are `cell - anchor`, so a 3x3 rect
/// anchored at (0,0) reaches {0,1,2} in each axis and its negation reaches
/// {0,-1,-2}. Both routes are swept.
///
/// @note Aggregate-initializable, but prefer the named factories -- they document
/// which of five ints is which, and `-1` for an anchor means "center",
/// spelled `cv::Point(-1, -1)` in OpenCV.
struct StructuringElement {
    MorphShape shape = MORPH_RECT;  ///< Ignored when `mask` is non-null.
    int cols = 3;                   ///< Width in cells; >= 1.
    int rows = 3;                   ///< Height in cells; >= 1.
    int anchorX = -1;               ///< Column of the anchor; -1 means cols / 2.
    int anchorY = -1;               ///< Row of the anchor; -1 means rows / 2.

    /// Optional caller-owned mask, row-major, `cols * rows` bytes. Non-owning.
    const uint8_t* mask = nullptr;

    /// @brief `cv::getStructuringElement(MORPH_RECT, {c, r}, anchor)`.
    static StructuringElement rect(int c, int r, int ax = -1, int ay = -1) {
        return StructuringElement{MORPH_RECT, c, r, ax, ay, nullptr};
    }
    /// @brief `cv::getStructuringElement(MORPH_CROSS, {c, r}, anchor)`.
    /// @note The arms pass through the ANCHOR, not the center. OpenCV's choice.
    static StructuringElement cross(int c, int r, int ax = -1, int ay = -1) {
        return StructuringElement{MORPH_CROSS, c, r, ax, ay, nullptr};
    }
    /// @brief `cv::getStructuringElement(MORPH_ELLIPSE, {c, r}, anchor)`.
    static StructuringElement ellipse(int c, int r, int ax = -1, int ay = -1) {
        return StructuringElement{MORPH_ELLIPSE, c, r, ax, ay, nullptr};
    }
    /// @brief An arbitrary caller-owned mask; `m` must outlive the element.
    static StructuringElement custom(const uint8_t* m, int c, int r, int ax = -1, int ay = -1) {
        return StructuringElement{MORPH_RECT, c, r, ax, ay, m};
    }

    /// @brief The anchor column with OpenCV's `-1 == center` resolved.
    int anchorCol() const { return anchorX < 0 ? cols / 2 : anchorX; }
    /// @brief The anchor row with OpenCV's `-1 == center` resolved.
    int anchorRow() const { return anchorY < 0 ? rows / 2 : anchorY; }

    /// @brief True when cell (col, row) is part of the element.
    /// @param col Column in [0, cols). @param row Row in [0, rows).
    /// @note Argument order is (x, y) -- OpenCV's `Point` order, not `Mat::at`'s.
    /// @note Transcribed from `cv::getStructuringElement`'s row loop, branch for
    /// branch, so the two agree by construction and not by coincidence. The
    /// ellipse's `saturate_cast<int>` is `cvRound`, i.e. nearest with ties
    /// to even, which is what `std::lrint` is under the default rounding
    /// mode -- `std::round` would differ on a tie.
    bool activeAt(int col, int row) const {
        if (mask != nullptr) {
            const size_t index =
                static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col);
            return mask[index] != 0;
        }
        int first = 0;
        int last = 0;
        spanOfRow(row, first, last);
        return col >= first && col < last;
    }

    /// @brief The half-open column range `[first, last)` of row `row` that MAY be
    /// set: exact for the parametric shapes, `[0, cols)` for a mask.
    /// @note Every parametric shape's row is one CONTIGUOUS run -- rect is the
    /// whole row, ellipse is a centerd run, and cross is either the whole
    /// row (the anchor row) or the single anchor column. That is what lets
    /// the kernels hoist the ellipse's `sqrt` out of the pixel loop and
    /// skip the empty part of a cross's rows entirely, without storing a
    /// cell table anywhere.
    void spanOfRow(int row, int& first, int& last) const {
        if (mask != nullptr) {
            first = 0;
            last = cols;
            return;
        }

        // OpenCV: a 1x1 element is a filled rect whatever the shape argument was.
        MorphShape effective = shape;
        if (cols == 1 && rows == 1) effective = MORPH_RECT;

        if (effective == MORPH_RECT || (effective == MORPH_CROSS && row == anchorRow())) {
            first = 0;
            last = cols;
            return;
        }
        if (effective == MORPH_CROSS) {
            first = anchorCol();
            last = first + 1;
            return;
        }

        // MORPH_ELLIPSE, in OpenCV's own arithmetic and its own order.
        const int r = rows / 2;
        const int c = cols / 2;
        const double invR2 = (r != 0) ? 1.0 / (static_cast<double>(r) * static_cast<double>(r)) : 0.0;
        const int dy = row - r;
        if (dy < -r || dy > r) {
            first = 0;
            last = 0;
            return;
        }
        const double radicand = static_cast<double>(r * r - dy * dy) * invR2;
        const int dx = static_cast<int>(std::lrint(static_cast<double>(c) * std::sqrt(radicand)));
        first = (c - dx > 0) ? (c - dx) : 0;
        last = (c + dx + 1 < cols) ? (c + dx + 1) : cols;
        if (last < first) last = first;
    }

    /// @brief True when every cell inside `spanOfRow` is set, so a kernel that
    /// iterates the span needs no per-cell test at all.
    /// @note That is the case for ALL THREE parametric shapes -- each one's row is
    /// a solid run -- and only a custom mask can have holes. It matters
    /// because the per-cell test is `activeAt`, and for MORPH_ELLIPSE that
    /// is a `sqrt`: measured, calling it once per (word, cell) rather than
    /// once per element row made a 5x5 ellipse erosion of a 640x480 frame
    /// 4.23 ns/pixel, i.e. 17x SLOWER than cv::erode. The shape query is not
    /// the operation and must not be inside the word loop.
    /// @note Morphology.ElementStructure asserts both halves of this -- every set
    /// cell inside the span, and every cell inside the span set -- because
    /// the kernels now depend on the second.
    bool spanIsDense() const { return mask == nullptr; }

    /// @brief Extents positive, anchor inside the element, at least one set cell.
    /// @note Called only from BINCV_ASSERT, so the cell scan is a debug-build cost.
    bool valid() const {
        if (cols <= 0 || rows <= 0) return false;
        if (anchorX < -1 || anchorX >= cols) return false;
        if (anchorY < -1 || anchorY >= rows) return false;
        for (int row = 0; row < rows; ++row) {
            int first = 0;
            int last = 0;
            spanOfRow(row, first, last);
            for (int col = first; col < last; ++col) {
                if (activeAt(col, row)) return true;
            }
        }
        return false;
    }
};

/// @brief The 3x3 rectangle -- `cv::Mat` passed to `cv::erode`, i.e. its default.
inline StructuringElement rect3x3() { return StructuringElement::rect(3, 3); }
/// @brief The 3x3 plus -- what BOTH `MORPH_CROSS` and `MORPH_ELLIPSE` give at 3x3.
inline StructuringElement cross3x3() { return StructuringElement::cross(3, 3); }

namespace impl {

/// @brief Which implementation `morphApply` may take. Internal; tests and the
/// benchmark only.
/// @note `Generic` exists so `Morphology.FastPathEqualsGeneric_*` can require the
/// 3x3 special case to agree with the code it replaced on every swept
/// image, and so benchmark/morphology_benchmark.cpp can price what the
/// special case is worth. A special case nobody compares against the
/// general one is a second implementation with no test of its own.
/// @note IT IS A TEMPLATE PARAMETER OF `morphApply`, NOT AN ARGUMENT, and that is
/// a measurement result rather than a style preference. As a runtime
/// argument it was constant-folded only while every call site in a
/// translation unit passed the same value; adding the benchmark's one
/// `Generic` call site made `use3x3` a live branch in the SHIPPED path and
/// measured erode 3x3 at 640x480 13% slower (0.143-0.159 against
/// 0.126-0.129 ns/pixel, x86, same header). A benchmark that changes the
/// code it measures is not measuring it. Two instantiations cannot do that
/// to each other.
enum class MorphPath { Auto, Generic };

/// @brief The combining operation and its identity, as a compile-time choice.
/// @note `IsErode` rather than a functor so that both the fold and the identity
/// come from one place: an identity that does not match the fold silently
/// produces an image that is right in the interior and wrong at the edge,
/// which is the failure this file exists to avoid.
template <bool IsErode, typename WordType>
struct MorphFold {
    /// @note The outer static_cast is not decoration: at uint8_t and uint16_t both
    /// arms of the conditional are promoted to int, and returning that int
    /// is exactly the narrowing -Wconversion is on to catch ( compiles
    /// every kernel at all four widths).
    static WordType identity() {
        return static_cast<WordType>(IsErode ? static_cast<WordType>(~static_cast<WordType>(0))
                                             : static_cast<WordType>(0));
    }
    static WordType apply(WordType a, WordType b) {
        return static_cast<WordType>(IsErode ? static_cast<WordType>(a & b)
                                             : static_cast<WordType>(a | b));
    }
};

/// @brief Word `i` of `srcRow` shifted so that destination column `c` reads
/// source column `c + dx`, with everything outside the row reading `fill`.
/// @note The recurrence of `impl::shiftRowHorizontal`, evaluated for ONE word
/// instead of a whole row, because this kernel interleaves many different
/// `dx` values over the same destination word and cannot make a pass per
/// offset without a temporary. The two are required to agree by
/// `Morphology.SingleOffsetEqualsShift_*`.
/// @note `bitShift == 0` is a separate branch because `x << WordBits` is
/// undefined behavior, not merely wrong -- see ops/shift.hpp.
template <typename WordType>
inline WordType morphShiftedWord(const WordType* srcRow, size_t i, size_t rowWords, size_t width,
                                 WordType tailMask, ptrdiff_t dx, WordType fill) {
    constexpr size_t wordBits = bitsPerWord<WordType>();

    const size_t k = (dx < 0) ? (static_cast<size_t>(0) - static_cast<size_t>(dx))
                              : static_cast<size_t>(dx);
    if (k >= width) return fill;

    const size_t wordShift = k / wordBits;
    const size_t bitShift = k % wordBits;

    if (dx >= 0) {
        const WordType lo = extendedRowWord(srcRow, static_cast<ptrdiff_t>(i + wordShift),
                                            rowWords, tailMask, fill);
        if (bitShift == 0) return lo;
        const WordType hi = extendedRowWord(srcRow, static_cast<ptrdiff_t>(i + wordShift + 1),
                                            rowWords, tailMask, fill);
        return static_cast<WordType>(static_cast<WordType>(lo >> bitShift) |
                                     static_cast<WordType>(hi << (wordBits - bitShift)));
    }

    const ptrdiff_t base = static_cast<ptrdiff_t>(i) - static_cast<ptrdiff_t>(wordShift);
    const WordType hi = extendedRowWord(srcRow, base, rowWords, tailMask, fill);
    if (bitShift == 0) return hi;
    const WordType lo = extendedRowWord(srcRow, base - 1, rowWords, tailMask, fill);
    return static_cast<WordType>(static_cast<WordType>(hi << bitShift) |
                                 static_cast<WordType>(lo >> (wordBits - bitShift)));
}

/// @brief The element's horizontal reach: max |cell - anchor| over set cells.
/// @note This is the width of the band at each edge that the per-pixel border
/// fixup has to rewrite, and it is computed ONCE per call rather than per
/// row. For every 3x3 element it is 1.
inline size_t morphMaxOffsetX(const StructuringElement& se) {
    const int ax = se.anchorCol();
    int reach = 0;
    for (int row = 0; row < se.rows; ++row) {
        int first = 0;
        int last = 0;
        se.spanOfRow(row, first, last);
        for (int col = first; col < last; ++col) {
            if (!se.activeAt(col, row)) continue;
            const int d = (col >= ax) ? (col - ax) : (ax - col);
            if (d > reach) reach = d;
        }
    }
    return static_cast<size_t>(reach);
}

/// @brief One destination pixel, recomputed from the whole element with every
/// source coordinate mapped through `borderIndex`.
/// @note Recomputes rather than repairs the word path's answer, because more than
/// one element cell can reach past the same edge and their contributions
/// cannot be unpicked.
/// @note Writes only column `c < width`, so it cannot dirty a padding bit.
template <bool IsErode, typename WordType>
inline void morphFixupPixel(BinMatConstView<WordType> src, WordType* dstRow, size_t y, size_t c,
                            const StructuringElement& se, BorderType borderType, int ax, int ay,
                            bool dense) {
    constexpr size_t wordBits = bitsPerWord<WordType>();
    using Fold = MorphFold<IsErode, WordType>;

    WordType acc = Fold::identity();
    for (int ey = 0; ey < se.rows; ++ey) {
        int first = 0;
        int last = 0;
        se.spanOfRow(ey, first, last);
        if (first >= last) continue;

        const ptrdiff_t dy = static_cast<ptrdiff_t>(ey) - static_cast<ptrdiff_t>(ay);
        const ptrdiff_t sy = borderIndex(static_cast<ptrdiff_t>(y) + dy, src.height, borderType);
        // Unreachable for the four non-constant types, which is all that gets
        // here; a constant border never reaches this function.
        if (sy < 0) continue;
        const WordType* srcRow = src.row(static_cast<size_t>(sy));

        for (int ex = first; ex < last; ++ex) {
            if (!dense && !se.activeAt(ex, ey)) continue;
            const ptrdiff_t dx = static_cast<ptrdiff_t>(ex) - static_cast<ptrdiff_t>(ax);
            const ptrdiff_t sx =
                borderIndex(static_cast<ptrdiff_t>(c) + dx, src.width, borderType);
            if (sx < 0) continue;
            const size_t sxu = static_cast<size_t>(sx);
            const bool bit = (srcRow[sxu / wordBits] & bitMask<WordType>(sxu)) != 0;
            const WordType contribution =
                static_cast<WordType>(bit ? static_cast<WordType>(~static_cast<WordType>(0))
                                          : static_cast<WordType>(0));
            acc = Fold::apply(acc, contribution);
        }
    }

    const size_t wordAt = c / wordBits;
    const WordType bit = bitMask<WordType>(c);
    dstRow[wordAt] = (acc != 0)
                         ? static_cast<WordType>(dstRow[wordAt] | bit)
                         : static_cast<WordType>(dstRow[wordAt] & static_cast<WordType>(~bit));
}

/// @brief Rewrites the destination columns whose source column can leave the row,
/// one pixel at a time, for the four NON-CONSTANT border types.
///
/// @note TWO EXPLICIT BAND LOOPS, NOT ONE LOOP OVER THE ROW WITH THE INTERIOR
/// SKIPPED. This function rewrites `2 * reach` pixels of a `width`-pixel row
/// -- 2 of 640 for a 3x3 element -- and an `if (interior) continue;` inside
/// a `for (c = 0; c < width; ++c)` still pays `width` iterations to do it.
/// Measured on x86 at 640x480, `uint64_t`, `rect3x3`, best of 5 x 200
/// calls (indicative only -- see EXPERIMENTS.md on measurement platforms):
/// the skipping form ran 19.5 us under BORDER_CONSTANT, which never calls
/// this, against 241-260 us under the other four. The fixup cost 12x the
/// entire word path to rewrite 960 of 307200 pixels, and made binCV 6-10x
/// SLOWER than `cv::erode` (24-25 us) instead of faster. Banded, the same
/// four types cost 40-45 us against the same 20-23 us baseline. The border
/// is a boundary, and its cost must scale with the boundary, not the frame.
///
/// @param bandLeft Columns `[0, bandLeft)` can reach past the left edge.
/// @param bandRightStart Columns `[bandRightStart, width)` can reach past the right.
/// @note The two ranges OVERLAP when `2 * reach >= width` (and `bandRightStart`
/// is 0 with `bandLeft == width` when the element out-reaches the frame
/// entirely), so the right band starts at `max(bandLeft, bandRightStart)`
/// and every column is visited exactly once. The wide-element cases in
/// `tests/test_morphology.cpp` -- a 129-wide element over a 3-wide frame --
/// are that path.
template <bool IsErode, typename WordType>
inline void morphFixupRowBorder(BinMatConstView<WordType> src, WordType* dstRow, size_t y,
                                const StructuringElement& se, BorderType borderType,
                                size_t bandLeft, size_t bandRightStart) {
    const int ax = se.anchorCol();
    const int ay = se.anchorRow();
    const bool dense = se.spanIsDense();

    const size_t leftEnd = (bandLeft < src.width) ? bandLeft : src.width;
    for (size_t c = 0; c < leftEnd; ++c) {
        morphFixupPixel<IsErode>(src, dstRow, y, c, se, borderType, ax, ay, dense);
    }

    const size_t rightStart = (bandRightStart > leftEnd) ? bandRightStart : leftEnd;
    for (size_t c = rightStart; c < src.width; ++c) {
        morphFixupPixel<IsErode>(src, dstRow, y, c, se, borderType, ax, ay, dense);
    }
}

/// @brief One destination row, general element. Accumulates in `dstRow`.
template <bool IsErode, typename WordType>
inline void morphRowGeneric(BinMatConstView<WordType> src, WordType* dstRow, size_t y,
                            const StructuringElement& se, BorderType borderType,
                            WordType constantFill, WordType horizontalFill) {
    constexpr size_t wordBits = bitsPerWord<WordType>();
    using Fold = MorphFold<IsErode, WordType>;

    const size_t rowWords = minRowWords<WordType>(src.width);
    const WordType tailMask = rowTailMask<WordType>(src.width);
    const int ax = se.anchorCol();
    const int ay = se.anchorRow();

    for (size_t i = 0; i < rowWords; ++i) dstRow[i] = Fold::identity();

    for (int ey = 0; ey < se.rows; ++ey) {
        int first = 0;
        int last = 0;
        se.spanOfRow(ey, first, last);
        if (first >= last) continue;

        const ptrdiff_t dy = static_cast<ptrdiff_t>(ey) - static_cast<ptrdiff_t>(ay);
        const ptrdiff_t sy = borderIndex(static_cast<ptrdiff_t>(y) + dy, src.height, borderType);

        if (sy < 0) {
            // Wholly outside the image under BORDER_CONSTANT: every cell of this
            // element row contributes the constant. When the constant is the
            // fold's identity -- the morphological default -- that is nothing.
            if (constantFill == Fold::identity()) continue;
            bool anySet = se.spanIsDense();
            for (int ex = first; ex < last && !anySet; ++ex) anySet = se.activeAt(ex, ey);
            if (!anySet) continue;
            for (size_t i = 0; i < rowWords; ++i) {
                dstRow[i] = Fold::apply(dstRow[i], constantFill);
            }
            continue;
        }

        const WordType* srcRow = src.row(static_cast<size_t>(sy));
        // The span is solid for every parametric shape, so the only element that
        // needs a per-cell test is a custom mask -- and that test is a byte load,
        // not a shape query. See StructuringElement::spanIsDense.
        const bool dense = se.spanIsDense();

        // THE WINDOW. Every cell of this element row reads the same source row at
        // a DIFFERENT horizontal offset, and when the row's widest offset is
        // smaller than a word every one of those offsets is a shift of the same
        // three source words. Fetching them once per destination word and shifting
        // per cell replaces two extendedRowWord calls per CELL with one per
        // WORD: for a 5x5 ellipse that is 34 masked, bounds-checked loads per word
        // reduced to 1. The general recurrence below stays for the case the window
        // cannot express -- an element reaching a whole word or more sideways --
        // where it is also no longer the inner loop of anything.
        const int reachLeft = ax - first;
        const int reachRight = (last - 1) - ax;
        const size_t rowReach = static_cast<size_t>((reachLeft > reachRight ? reachLeft
                                                                            : reachRight) > 0
                                                        ? (reachLeft > reachRight ? reachLeft
                                                                                  : reachRight)
                                                        : 0);

        if (rowReach < wordBits) {
            WordType prev = horizontalFill;
            WordType cur = extendedRowWord(srcRow, static_cast<ptrdiff_t>(0), rowWords, tailMask,
                                           horizontalFill);
            for (size_t i = 0; i < rowWords; ++i) {
                const WordType next = extendedRowWord(srcRow, static_cast<ptrdiff_t>(i + 1),
                                                      rowWords, tailMask, horizontalFill);
                WordType acc = dstRow[i];
                for (int ex = first; ex < last; ++ex) {
                    if (!dense && !se.activeAt(ex, ey)) continue;
                    const int d = ex - ax;
                    WordType w;
                    if (d == 0) {
                        w = cur;
                    } else if (d > 0) {
                        const size_t b = static_cast<size_t>(d);
                        w = static_cast<WordType>(static_cast<WordType>(cur >> b) |
                                                  static_cast<WordType>(next << (wordBits - b)));
                    } else {
                        const size_t b = static_cast<size_t>(-d);
                        w = static_cast<WordType>(static_cast<WordType>(cur << b) |
                                                  static_cast<WordType>(prev >> (wordBits - b)));
                    }
                    acc = Fold::apply(acc, w);
                }
                dstRow[i] = acc;
                prev = cur;
                cur = next;
            }
            continue;
        }

        for (size_t i = 0; i < rowWords; ++i) {
            WordType acc = dstRow[i];
            for (int ex = first; ex < last; ++ex) {
                if (!dense && !se.activeAt(ex, ey)) continue;
                const ptrdiff_t dx = static_cast<ptrdiff_t>(ex) - static_cast<ptrdiff_t>(ax);
                acc = Fold::apply(acc, morphShiftedWord(srcRow, i, rowWords, src.width, tailMask,
                                                        dx, horizontalFill));
            }
            dstRow[i] = acc;
        }
    }
}

/// @brief One destination row, 3x3 element anchored at its center.
///
/// @note THE SPECIAL CASE ASKS FOR, and it is a special case of the loop
/// above rather than a different algorithm: the offsets are known to be
/// {-1, 0, +1} in both axes, so the horizontal recurrence collapses to two
/// one-bit shifts with a carry from the neighbouring word, and the three
/// source words a destination word needs slide along the row instead of
/// being fetched per offset.
/// @note WHAT IT ACTUALLY REMOVES, since the reason first written here was wrong.
/// That reason was "one `extendedRowWord` per word per element row where the
/// general path pays two per SET CELL"; morphRowGeneric's window branch
/// hoists `extendedRowWord` per WORD too, for any element whose row reaches
/// less than a word sideways -- which every 3x3 element does -- so the load
/// counts are the same and the claim described a path neither one takes.
/// What this kernel removes is the inner loop over element cells, the
/// data-dependent shift count each iteration computes, and the per-row span
/// queries: three cells become three predictable branches on values hoisted
/// to the whole call, and `<< 1` / `>> 1` become constants.
/// @note AND IT IS MEASURED, on the reference device, by
/// benchmark/morphology_path_benchmark.cpp -- which runs this kernel against
/// `MorphPath::Generic` on the same images and requires them to agree first.
/// At 640x480 the general path costs 2.12x (rect3x3 erode, `uint32_t`),
/// 3.17x (rect3x3 dilate), 2.47x / 3.69x at `uint64_t`, and 2.78x-3.67x for
/// cross3x3; across the whole pyramid ladder the range is 2.1x-3.7x, at
/// batch spreads under 4%. That is the number a Phase 5 reader deciding
/// whether to vectorize one path or both should start from, and it is why
/// the duplicated code stays.
/// @note Driven by the element's own cells, so it serves rect, cross, ellipse and
/// any 3x3 mask alike; a cleared cell simply contributes nothing.
/// @param cells The element's nine cells, row-major, computed ONCE per call by
/// morphApply -- not per row. A 3x3 MORPH_ELLIPSE queried per row would
/// pay nine `sqrt` per image row for an answer that cannot change.
template <bool IsErode, typename WordType>
inline void morphRow3x3(BinMatConstView<WordType> src, WordType* dstRow, size_t y,
                        const bool* cells, BorderType borderType, WordType constantFill,
                        WordType horizontalFill) {
    constexpr size_t wordBits = bitsPerWord<WordType>();
    using Fold = MorphFold<IsErode, WordType>;

    const size_t rowWords = minRowWords<WordType>(src.width);
    const WordType tailMask = rowTailMask<WordType>(src.width);

    for (size_t i = 0; i < rowWords; ++i) dstRow[i] = Fold::identity();

    for (int ey = 0; ey < 3; ++ey) {
        const bool* cell0 = cells + ey * 3;
        const bool cellLeft = cell0[0];
        const bool cellCenter = cell0[1];
        const bool cellRight = cell0[2];
        if (!cellLeft && !cellCenter && !cellRight) continue;

        const ptrdiff_t sy =
            borderIndex(static_cast<ptrdiff_t>(y) + (ey - 1), src.height, borderType);
        if (sy < 0) {
            if (constantFill == Fold::identity()) continue;
            for (size_t i = 0; i < rowWords; ++i) {
                dstRow[i] = Fold::apply(dstRow[i], constantFill);
            }
            continue;
        }

        const WordType* srcRow = src.row(static_cast<size_t>(sy));

        // The sliding window: word i needs words i-1, i and i+1 of the source, and
        // each is the previous iteration's `cur` / `next`. Outside the row both
        // read the fill, which is exactly what extendedRowWord returns for an
        // out-of-range index.
        WordType prev = horizontalFill;
        WordType cur = extendedRowWord(srcRow, static_cast<ptrdiff_t>(0), rowWords, tailMask,
                                       horizontalFill);
        for (size_t i = 0; i < rowWords; ++i) {
            const WordType next = extendedRowWord(srcRow, static_cast<ptrdiff_t>(i + 1), rowWords,
                                                  tailMask, horizontalFill);
            WordType acc = dstRow[i];
            // dx = -1: destination column c reads source column c - 1.
            if (cellLeft) {
                acc = Fold::apply(acc, static_cast<WordType>(
                                           static_cast<WordType>(cur << 1) |
                                           static_cast<WordType>(prev >> (wordBits - 1))));
            }
            if (cellCenter) acc = Fold::apply(acc, cur);
            // dx = +1: destination column c reads source column c + 1.
            if (cellRight) {
                acc = Fold::apply(acc, static_cast<WordType>(
                                           static_cast<WordType>(cur >> 1) |
                                           static_cast<WordType>(next << (wordBits - 1))));
            }
            dstRow[i] = acc;
            prev = cur;
            cur = next;
        }
    }
}

/// @brief erode (IsErode) or dilate, whole image. The one kernel both call.
/// @note Internal. `MorphPath::Generic` forbids the 3x3 special case, for the test
/// that compares the two and the benchmark row that prices it. It is a
/// TEMPLATE parameter -- see MorphPath's note on why.
template <bool IsErode, MorphPath Path, typename WordType>
inline void morphApply(BinMatConstView<WordType> src, BinMatView<WordType> dst,
                       const StructuringElement& se, BorderType borderType, bool borderValue) {
    using Fold = MorphFold<IsErode, WordType>;

    if (dst.width == 0 || dst.height == 0) return;

    const size_t rowWords = minRowWords<WordType>(dst.width);
    const WordType tailMask = rowTailMask<WordType>(dst.width);
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));
    const WordType constantFill = borderValue ? allOnes : static_cast<WordType>(0);

    // Under a non-constant BorderType the word path's horizontal fill never
    // survives: the columns it reaches are exactly the ones the fixup rewrites.
    // The fold's identity is used there so a missing fixup shows up as a plainly
    // wrong edge rather than a plausible one.
    const WordType horizontalFill =
        (borderType == BORDER_CONSTANT) ? constantFill : Fold::identity();

    const size_t reach = morphMaxOffsetX(se);
    const size_t bandLeft = (reach < dst.width) ? reach : dst.width;
    const size_t bandRightStart = (dst.width > reach) ? (dst.width - reach) : 0;

    const bool use3x3 = (Path == MorphPath::Auto) && se.cols == 3 && se.rows == 3 &&
                        se.anchorCol() == 1 && se.anchorRow() == 1;

    // Once per call, never per row: for MORPH_ELLIPSE activeAt evaluates a
    // square root.
    bool cells[9] = {false, false, false, false, false, false, false, false, false};
    if (use3x3) {
        for (int ey = 0; ey < 3; ++ey) {
            for (int ex = 0; ex < 3; ++ex) cells[ey * 3 + ex] = se.activeAt(ex, ey);
        }
    }

    for (size_t y = 0; y < dst.height; ++y) {
        WordType* dstRow = dst.row(y);
        if (use3x3) {
            morphRow3x3<IsErode>(src, dstRow, y, cells, borderType, constantFill, horizontalFill);
        } else {
            morphRowGeneric<IsErode>(src, dstRow, y, se, borderType, constantFill, horizontalFill);
        }

        // CLAUDE.md's hard rule: whatever the fold's identity was, the bits past
        // `width` are zero on return.
        dstRow[rowWords - 1] = static_cast<WordType>(dstRow[rowWords - 1] & tailMask);

        if (borderType != BORDER_CONSTANT) {
            morphFixupRowBorder<IsErode>(src, dstRow, y, se, borderType, bandLeft, bandRightStart);
        }
    }
}

/// @brief The preconditions `erode` and `dilate` share, in one place.
template <typename WordType>
inline bool morphArgumentsAreSane(BinMatConstView<WordType> src, BinMatView<WordType> dst) {
    return src.width == dst.width && src.height == dst.height &&
           strideCoversARow<WordType>(src.width, src.height, src.stride) &&
           strideCoversARow<WordType>(dst.width, dst.height, dst.stride);
}

}  // namespace impl

// ---------------------------------------------------------------------------
// The kernels ( views, never containers)
// ---------------------------------------------------------------------------

/// @brief Morphological erosion: `dst(x,y) = AND over the element of src(x+dx, y+dy)`.
/// **API TIER 1** -- bit-exact against `cv::erode` on the same binary
/// content as `CV_8U`, borders and corners included.
///
/// @param src Source view.
/// @param dst Destination view; must have src's dimensions and must share no word
/// with src. In place is NOT supported (a destination word is built from
/// several source words).
/// @param element The structuring element. `cv::erode(src, dst, cv::Mat)` is
/// `rect3x3`.
/// @param borderType How coordinates outside the image extrapolate; OpenCV's five.
/// @param borderValue The pixel value outside the image under `BORDER_CONSTANT`.
/// **Defaults to `true`, which is `morphologyDefaultBorderValue`** -- the
/// maximum, so the frame's edge is not eaten away. Passing `false`
/// is legal and matches `cv::erode(..., BORDER_CONSTANT, cv::Scalar(0))`.
///
/// @note One pass over the destination, NO SCRATCH BUFFER and no allocation. The
/// accumulation happens in the destination row (see the fused-form note at
/// the top of this file).
/// @note The destination's padding bits are zero on return, and the source's
/// padding bits are never read as pixels even when the source wraps a buffer
/// whose padding is dirty.
/// @note **`dst` must span its image's full width or end on a word boundary**, as
/// in ops/logic.hpp and ops/shift.hpp. The bits of its trailing partial word
/// past `width` are CLEARED: padding in the usual case, and a wider parent's
/// next 1..WordBits-1 live pixels when `dst` is a sub-width window onto one.
/// Nothing diagnoses that -- every address written is inside the parent.
/// @note Never throws and never allocates (the design notes). Mismatched
/// dimensions, a stride shorter than a row, an unknown `BorderType`, an
/// invalid element, and any overlap between src and dst are programming
/// errors: `BINCV_ASSERT` reports them in debug builds and they are
/// undefined in release.
template <typename WordType>
inline void erode(BinMatConstView<WordType> src, BinMatView<WordType> dst,
                  const StructuringElement& element, BorderType borderType = BORDER_CONSTANT,
                  bool borderValue = true) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "erode: src and dst must have the same dimensions");
    BINCV_ASSERT(impl::morphArgumentsAreSane(src, dst),
                 "erode: every view's stride must cover a whole row");
    BINCV_ASSERT(impl::viewsShareNoWord(src, dst),
                 "erode: dst must share no word with src (in place is not supported)");
    BINCV_ASSERT(impl::isKnownBorderType(borderType), "erode: unknown BorderType");
    BINCV_ASSERT(element.valid(),
                 "erode: the structuring element needs positive extents, an in-range anchor "
                 "and at least one set cell");
    if (dst.width == 0 || dst.height == 0) return;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "erode: a non-empty view needs a non-null pointer");

    impl::morphApply<true, impl::MorphPath::Auto>(src, dst, element, borderType, borderValue);
}

/// @brief Morphological dilation: `dst(x,y) = OR over the element of src(x+dx, y+dy)`.
/// **API TIER 1** -- bit-exact against `cv::dilate`.
/// @param borderValue Defaults to `false`, which is `morphologyDefaultBorderValue`
/// for a dilation -- the minimum, so the frame does not grow a border
///. Everything else: see erode, INCLUDING the precondition that
/// `dst` span its image's full width or end on a word boundary.
template <typename WordType>
inline void dilate(BinMatConstView<WordType> src, BinMatView<WordType> dst,
                   const StructuringElement& element, BorderType borderType = BORDER_CONSTANT,
                   bool borderValue = false) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "dilate: src and dst must have the same dimensions");
    BINCV_ASSERT(impl::morphArgumentsAreSane(src, dst),
                 "dilate: every view's stride must cover a whole row");
    BINCV_ASSERT(impl::viewsShareNoWord(src, dst),
                 "dilate: dst must share no word with src (in place is not supported)");
    BINCV_ASSERT(impl::isKnownBorderType(borderType), "dilate: unknown BorderType");
    BINCV_ASSERT(element.valid(),
                 "dilate: the structuring element needs positive extents, an in-range anchor "
                 "and at least one set cell");
    if (dst.width == 0 || dst.height == 0) return;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "dilate: a non-empty view needs a non-null pointer");

    impl::morphApply<false, impl::MorphPath::Auto>(src, dst, element, borderType, borderValue);
}

/// @brief True when `morphologyEx(op,...)` reads and writes its scratch view.
/// @note `MORPH_ERODE` and `MORPH_DILATE` are single kernels and need none; the
/// other five need exactly one frame. Callers that size a scratch buffer
/// from an `op` chosen at runtime ask this rather than hard-coding the list.
inline bool morphologyExNeedsScratch(MorphOp op) {
    return !(op == MORPH_ERODE || op == MORPH_DILATE);
}

/// @brief The seven `MorphOp` compositions. **API TIER 1** -- bit-exact against
/// `cv::morphologyEx` for every op and every element.
///
/// @param src Source view.
/// @param dst Destination view; src's dimensions, sharing no word with src.
/// @param op Any `MorphOp`: ERODE, DILATE, OPEN, CLOSE, GRADIENT, TOPHAT, BLACKHAT.
/// @param element The structuring element.
/// @param scratch **CALLER-PROVIDED intermediate**, src's dimensions, sharing no
/// word with src or dst. Its contents on entry are irrelevant and on
/// return are unspecified. Required for every op except `MORPH_ERODE` and
/// `MORPH_DILATE`, which ignore it and accept an empty view --
/// `morphologyExNeedsScratch` is that predicate.
/// @param borderType OpenCV's five; passed to every step, as `cv::morphologyEx` does.
///
/// @note **No allocation, and exactly ONE frame of scratch for all five compound
/// ops** -- see the scratch section at the top of this file for why TOPHAT
/// and BLACKHAT do not need a second.
/// @note Each step uses the morphological default fill for ITS OWN operation
/// (ones outside for an erosion, zeros for a dilation), which is what
/// `cv::morphologyEx` does with its default `borderValue`. A literal
/// constant on both steps is not offered here; compose erode/dilate.
/// @note **`dst` AND `scratch` must each span their image's full width or end on a
/// word boundary**, for the reason erode states: both are written by these
/// kernels, so both have their trailing partial word's bits past `width`
/// cleared -- a wider parent's live pixels when either is a sub-width window
/// onto one, and undiagnosable.
/// @note Never throws and never allocates. Every precondition above is a
/// `BINCV_ASSERT` (the design notes).
template <typename WordType>
inline void morphologyEx(BinMatConstView<WordType> src, BinMatView<WordType> dst, MorphOp op,
                         const StructuringElement& element, BinMatView<WordType> scratch,
                         BorderType borderType = BORDER_CONSTANT) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "morphologyEx: src and dst must have the same dimensions");
    BINCV_ASSERT(impl::viewsShareNoWord(src, dst),
                 "morphologyEx: dst must share no word with src");
    BINCV_ASSERT(op == MORPH_ERODE || op == MORPH_DILATE || op == MORPH_OPEN ||
                     op == MORPH_CLOSE || op == MORPH_GRADIENT || op == MORPH_TOPHAT ||
                     op == MORPH_BLACKHAT,
                 "morphologyEx: unknown MorphOp");
    BINCV_ASSERT(!morphologyExNeedsScratch(op) ||
                     (scratch.width == src.width && scratch.height == src.height),
                 "morphologyEx: this op needs a caller-provided scratch view of src's size");
    BINCV_ASSERT(!morphologyExNeedsScratch(op) ||
                     (impl::viewsShareNoWord(src, scratch) &&
                      impl::viewsShareNoWord(BinMatConstView<WordType>(dst), scratch)),
                 "morphologyEx: scratch must share no word with src or dst");

    switch (op) {
        case MORPH_ERODE:
            erode(src, dst, element, borderType);
            return;

        case MORPH_DILATE:
            dilate(src, dst, element, borderType);
            return;

        case MORPH_OPEN:
            erode(src, scratch, element, borderType);
            dilate(BinMatConstView<WordType>(scratch), dst, element, borderType);
            return;

        case MORPH_CLOSE:
            dilate(src, scratch, element, borderType);
            erode(BinMatConstView<WordType>(scratch), dst, element, borderType);
            return;

        case MORPH_GRADIENT:
            // dilate - erode, saturating on CV_8U == dilate AND NOT erode.
            dilate(src, dst, element, borderType);
            erode(src, scratch, element, borderType);
            bitwiseNot(BinMatConstView<WordType>(scratch), scratch);
            bitwiseAnd(BinMatConstView<WordType>(dst), BinMatConstView<WordType>(scratch), dst);
            return;

        case MORPH_TOPHAT:
            // src - open(src). The opening lands in dst, then dst = src AND NOT dst.
            erode(src, scratch, element, borderType);
            dilate(BinMatConstView<WordType>(scratch), dst, element, borderType);
            bitwiseNot(BinMatConstView<WordType>(dst), dst);
            bitwiseAnd(src, BinMatConstView<WordType>(dst), dst);
            return;

        case MORPH_BLACKHAT:
        default:
            // close(src) - src. The closing lands in dst; scratch is dead by then
            // and holds ~src for the subtraction.
            dilate(src, scratch, element, borderType);
            erode(BinMatConstView<WordType>(scratch), dst, element, borderType);
            bitwiseNot(src, scratch);
            bitwiseAnd(BinMatConstView<WordType>(dst), BinMatConstView<WordType>(scratch), dst);
            return;
    }
}

}  // inline namespace BINCV_ABI_NAMESPACE
}  // namespace bincv
