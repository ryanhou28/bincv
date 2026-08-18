#pragma once

/// @file reduce.hpp
/// @brief Bulk population counts over regions and masks (T2.5, T2.6).
///        **API TIER 1** for countNonZero, **TIER 3** for the masked forms.
///
/// Four entry points, and the shape of all four is the point of the file:
///
///   countNonZero(src)                 popcount over a whole image      Tier 1
///   countNonZero(src, region)         popcount over a rectangle        Tier 1
///   countAnd(a, b, region)            popcount(a & b)                  Tier 3
///   countAndSplit(a, b, c, region)    popcount(a & b & ~c) and
///                                     popcount(a & b &  c), one pass   Tier 3
///
/// The last one is the Lucas-Kanade covariance cross-term written out
/// (ARCHITECTURE 7.5). It is here in the MVP, next to the plain count, because
/// designing it later would mean designing a *second* reduction interface after
/// callers had been written against the first.
///
/// ---------------------------------------------------------------------------
/// D-6: THERE IS NO PER-WORD POPCOUNT IN THIS FILE'S PUBLIC SURFACE, AND THAT IS
/// THE WHOLE INTERFACE DECISION
///
/// A reasonable-looking library would export `popcount(WordType)` and let callers
/// write their own loops. binCV must not, and the reason is measured aarch64
/// codegen rather than taste (ARCHITECTURE 6.2, D-6). aarch64 -- the primary
/// target -- has no scalar population-count instruction, so every popcount has to
/// travel to the NEON register file and its result has to come back, and the cost
/// is dominated by those register-domain crossings rather than by `cnt` itself.
///
/// **The sequences below are what THIS FILE's kernels emit**, read out of
/// `g++ -O2 -DNDEBUG -S` on the reference device (aarch64, g++ 14.2), not the
/// ISA-level illustration in EXPERIMENTS.md X-3 -- which measured a minimal
/// standalone `__builtin_popcountll` under clang and is right about the ISA. The
/// count that matters is CROSSINGS PER WORD, and it is not the same for all four
/// entry points:
///
///     countNonZero   1 crossing/word    ldr  d31, [x2], 8      ; loaded straight
///                                       cnt  v31.8b, v31.8b    ;   into NEON --
///                                       addv b31, v31.8b       ;   no inbound fmov
///                                       fmov x1, d31           ; NEON -> GPR
///                                       add  x0, x0, x1        ; accumulate in a GPR
///
///     countAnd       2 crossings/word   the AND is done in GPRs, so the word has
///                                       to be moved in (`fmov d31, x1`) as well as
///                                       out (`fmov x1, d31`)
///
///     countAndSplit  4 crossings/word   two values in (`fmov d30, x2`,
///                                       `fmov d31, x0`) and two out (`fmov x2, d30`,
///                                       `fmov x0, d31`) -- and this is the LK
///                                       covariance path (ARCHITECTURE 7.5)
///
/// Note what the compiler did NOT do: it kept the accumulators in general-purpose
/// registers and crossed back on every word. That is the caller's data flow made
/// out of the loop body, and it is exactly the cost a per-word popcount in the
/// PUBLIC API would hand to every caller, forever, with no way to amortize it.
///
/// A bulk entry point is what makes the fix possible at all: the words can stay in
/// vector registers, `cnt` can accumulate into a vector total with `uadalp`, and
/// one crossing per row -- or per region -- pays for the whole row. The same
/// interface lowers to `popcntq` in a loop on x86-64 (when the build enables it;
/// see impl::popcountWord) and to the SWAR sequence on Cortex-M, so one API shape
/// stays right on all three.
///
/// **Today's implementation is the scalar one that T2.5 asks for** -- one
/// `__builtin_popcountll` per word, in impl::popcountWord -- so the loops below do
/// NOT yet keep anything in vector registers, and on the reference device they run
/// at the speed of the per-word loop D-6 forbids exposing. That is measured, and it
/// is recorded rather than papered over: EXPERIMENTS.md X-7b, the aarch64 half of
/// X-7, measured 0.99x against that loop with 1.8x available from a vector
/// accumulator. The interface decision and the implementation status are two
/// different things, and only the first is settled here.
///
/// The point of the interface is that it never made a caller write that loop, so
/// replacing the loop body with NEON in Phase 5 touches this file and nothing else.
/// impl::popcountWord is internal (`impl::`, no stability promise) and is
/// deliberately NOT in impl/kernel_util.hpp, which is the vocabulary every kernel
/// header reaches for: a per-word popcount that is one include away from every
/// kernel is a per-word popcount that will be called from one.
///
/// ---------------------------------------------------------------------------
/// WHAT A REDUCTION IN THIS FILE PROMISES
///
///  1. **Views, never containers** (D-5). Strides are read per row and may differ
///     between the arguments.
///
///     **Spell the arguments `constMagnitude()` / `constSign()` / `constView()`,
///     not `magnitude()` / `sign()` / `view()`.** Every entry point below takes
///     BinMatConstView<W>, and template argument deduction does not consider
///     user-defined conversions (D-9, and the note on
///     BinMatView::operator BinMatConstView), so handing one a BinMatView from a
///     NON-const container is a deduction failure, not an implicit conversion:
///
///         countNonZero(dx.magnitude(0), w)       // does NOT compile
///         countNonZero(dx.constMagnitude(0), w)  // this is the spelling
///
///     The diagnostic says "BinMatView<W> is not derived from BinMatConstView<W>"
///     and does not mention the conversion, which is why it is written out here
///     rather than left to be met at a call site. SignedQuantMat::constMagnitude,
///     constSign and BinMat::constView exist for exactly this, and every example
///     in this file, in ARCHITECTURE 7.5 and in tests/test_reduce.cpp uses them.
///     (Reductions take no destination, so unlike ops/logic.hpp there is no
///     container overload here that would hide the distinction.)
///
///  2. **No allocation and no throw** (ARCHITECTURE 5.3). Mismatched dimensions
///     and a stride too short to hold a row are programming errors, reported by
///     BINCV_ASSERT in debug builds and undefined in release. There is no error
///     return: a count is a `size_t`, and a sentinel would have to be checked by
///     every caller of an operation that cannot fail on valid views.
///
///  3. **Read-only, so ALIASING IS UNRESTRICTED.** This is the one family of
///     kernels D-11 does not bind: nothing is written, so `a`, `b` and `c` may be
///     the same view, may overlap partially, or may be unrelated.
///     `countAndSplit(m, m, m, r)` is well defined (it reports every set pixel as
///     `whenSet`). Callers of the covariance do exactly this kind of thing --
///     `countNonZero(mag_x, w)` and `countAnd(mag_x, mag_y, w)` over overlapping
///     windows of one frame -- and an alias assertion here would reject it.
///
///  4. **Counts are exact and unsaturated.** `size_t` holds the population of any
///     image that fits in memory (one bit per pixel means the count cannot exceed
///     the bit count of the buffer).
///
/// ---------------------------------------------------------------------------
/// THE REGION CONTRACT: HALF-OPEN, AND CLIPPED RATHER THAN REJECTED
///
/// `Rect{x, y, width, height}` covers columns [x, x+width) and rows [y, y+height),
/// exactly as cv::Rect does, and is **intersected with the image**. A region may
/// start at a negative coordinate, may extend past the right or bottom edge, and
/// may lie wholly outside; the pixels that exist are counted and the rest
/// contribute nothing. A region with `width <= 0` or `height <= 0` counts zero.
///
/// This is not leniency, it is the calling convention T3.6 needs. The LK window is
/// 31x31 centred on a keypoint, so every keypoint within 15 pixels of an edge
/// produces an out-of-range rectangle. The alternative -- assert and make the
/// caller clamp -- puts the same four `min`/`max` expressions in every call site,
/// and a wrong one there is a silently wrong covariance rather than a diagnostic.
/// Clipping once, here, is also what makes "correct for windows clipped at image
/// edges" (T2.6) a property of the library rather than of its callers.
///
/// Word alignment is irrelevant to a caller: a region may begin and end at any
/// column. The first and last words of each row are masked, and only the words
/// strictly between them are accumulated whole -- which is the loop Phase 5
/// vectorizes, and the reason the masks are hoisted out of it rather than applied
/// per word.
///
/// ---------------------------------------------------------------------------
/// THE PADDING-BIT CONTRACT, WHICH IS A DECISION AND NOT AN INHERITANCE
///
/// **A reduction here counts only bits inside the requested region intersected
/// with the image. A bit at or past `width` is never counted, whatever it holds.**
///
/// T2.5 specifies "whole-word accumulation; correctness depends on padding bits
/// being zero". Taken literally that means accumulating a row's trailing partial
/// word unmasked and relying on the invariant. This file **does not** do that, and
/// the deviation is deliberate:
///
///   - A source with dirty padding is a SUPPORTED construction, not a bug. BinMat's
///     wrap constructor says in as many words that a wrapped buffer's padding bits
///     are the caller's (sensor DMA, a sub-region of a larger frame), and
///     tests/test_logic.cpp already sweeps sources built that way. A reduction that
///     over-counts on such a view is a wrong answer produced from a legal input.
///   - ops/shift.hpp reached the same conclusion for the same reason and pays a
///     good deal more for it: every source word is substituted through
///     impl::extendedRowWord. Two kernel families disagreeing about whether a
///     source's padding is trustworthy would be worse than either rule.
///   - The cost is one AND per ROW, not per word. The trailing word is masked
///     outside the interior loop exactly as in ops/logic.hpp, so the loop that
///     matters carries no mask and Phase 5's vectorization is unaffected.
///
/// The invariant is still load-bearing for everything else: a region's INTERIOR
/// words are accumulated unmasked, which is correct only because every bit of a
/// word strictly inside a row is a pixel. And a destination binCV wrote is still
/// clean, because ops/logic.hpp and ops/shift.hpp mask their trailing words.
///
/// Restated as the property a test can pin: for any view `v`,
/// `countNonZero(v)` equals `cv::countNonZero` of the same content as CV_8U --
/// including when `v` wraps a buffer whose padding bits are all ones.
///
/// ---------------------------------------------------------------------------
/// countAndSplit IS SINGLE PASS, AND WHY THAT IS SPELLED OUT
///
/// T2.6 requires each word to be read once. The row body reads `a[i]`, `b[i]` and
/// `c[i]` once each per word index and produces both halves of the answer from
/// them:
///
///     both  = a[i] & b[i] & mask
///     total = popcount(both)              // pixels where both magnitudes are set
///     set   = popcount(both & c[i])       // ... and the signs disagree
///     whenSet   += set
///     whenClear += total - set
///
/// Two popcounts per word, not three: `whenClear` is `total - set` rather than
/// `popcount(both & ~c[i])`. That is not only cheaper on the target (the popcount
/// is the expensive operation, see D-6), it removes a padding hazard -- `~c[i]`
/// SETS every padding bit of the trailing word, so the three-popcount spelling
/// would count phantom pixels unless the mask were applied a second time.
///
/// The skeleton that walks a row -- head word, unmasked interior, tail word -- is
/// impl::visitRowWords, shared by all four entry points, so "each word index is
/// visited exactly once, in ascending order, under the right mask" is one function
/// that tests/test_reduce.cpp drives directly with a recording visitor rather than
/// a property asserted about four separate loops.
///
/// **Single pass is a property of each CALL, not of the covariance built out of
/// them, and the difference is measured.** ARCHITECTURE 7.5 needs four numbers over
/// one window; through these primitives that is three calls -- countNonZero(mag_x),
/// countNonZero(mag_y), countAndSplit(...) -- and therefore three traversals of the
/// same window, issuing the same popcounts a fused traversal would issue once. On
/// the reference device that composition costs 1.30x a fused pass, reproducibly,
/// with the popcount count identical on both sides (EXPERIMENTS.md X-8). It is
/// recorded and NOT fixed here: a covariance-shaped entry point is an interface
/// change, 1.30x is past T2.10's own 15% threshold, and T2.10 is the experiment
/// whose stated gate is this interface. T3.6 depends on it settling first.
///
/// **The MVP RECOMPUTES per window. There is no incremental or sliding state
/// here, on purpose.** LK windows are large (31x31 in practice) and overlap
/// heavily -- consecutive keypoints, and consecutive pyramid iterations on one
/// keypoint, re-read almost the same words -- which is exactly the regime where a
/// sliding accumulator could win. Whether it does is E-3 (T2.10), and it is
/// unmeasured. Building the accumulator now would fix this interface around an
/// answer nobody has, which is the thing TASKS.md T2.6 asks not to do; if E-3 says
/// incremental wins, the state is added here without any recompute caller
/// changing shape.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
// impl::bitsPerWord / lowBitsMask / minRowWords, and impl::strideCoversARow --
// the row-geometry vocabulary every kernel under ops/ is written in. Note that
// the D-11 alias predicates it also carries are NOT used here: see promise 3.
#include "../impl/kernel_util.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

namespace impl {

// ---------------------------------------------------------------------------
// INTERNAL. Nothing in this namespace is public API or carries any stability
// promise, and popcountWord in particular exists only so that the bulk entry
// points below have something to call -- see the D-6 section at the top of this
// file before exporting, re-exporting, or "just using" it from another header.
// ---------------------------------------------------------------------------

/// @brief Population count of one word without any compiler builtin.
///        **INTERNAL (D-6).**
/// @note Not decoration, and not dead source. binCV's claim is that it needs a
///       C++17 compiler and nothing else; MSVC has no __builtin_popcountll, so
///       without this the header would silently be GCC/clang-only. It is also the
///       sequence a Cortex-M build compiles to, where no popcount instruction
///       exists at any width (ARCHITECTURE 6.3).
/// @note **Compiled unconditionally, on every toolchain, and tested against the
///       builtin** by Reduce.PortablePopcount_*. Guarding the definition itself
///       behind the #if would make it source that no configuration this project
///       verifies ever compiles -- the shape of dead gate CLAUDE.md and T1.4 are
///       both written about. It costs one unreferenced inline template.
template <typename WordType>
inline size_t popcountWordPortable(WordType w) {
    uint64_t v = static_cast<uint64_t>(w);
    v = v - ((v >> 1) & UINT64_C(0x5555555555555555));
    v = (v & UINT64_C(0x3333333333333333)) + ((v >> 2) & UINT64_C(0x3333333333333333));
    v = (v + (v >> 4)) & UINT64_C(0x0F0F0F0F0F0F0F0F);
    return static_cast<size_t>((v * UINT64_C(0x0101010101010101)) >> 56);
}

/// @brief Population count of one word. **INTERNAL (D-6).**
/// @note The scalar form T2.5 specifies. On aarch64 this is two register-domain
///       crossings around a `cnt`, which is precisely why no caller outside this
///       file may reach it: the crossings are amortized by the bulk loops here
///       and cannot be amortized by anyone calling this per word.
/// @note And on the x86-64 baseline binCV actually ships -- no `-march`, since
///       the build detects AVX2/AVX-512 without enabling them until runtime
///       dispatch lands -- `__builtin_popcountll` is not `popcntq` at all. GCC
///       emits a CALL to libgcc's `__popcountdi2`, once per word, and clang
///       inlines the SWAR sequence above. Measured cost: 4.2x slower than
///       cv::countNonZero, against 2.0x faster once the instruction is available
///       (EXPERIMENTS.md X-7). A third target tier whose per-word cost dwarfs the
///       count itself, which is D-6's argument rather than an exception to it.
/// @note Phase 5 replaces the LOOPS below, not this function. A vectorized
///       reduction keeps whole vectors in NEON registers and never forms a
///       per-word count at all.
template <typename WordType>
inline size_t popcountWord(WordType w) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<size_t>(__builtin_popcountll(static_cast<unsigned long long>(w)));
#else
    return popcountWordPortable<WordType>(w);
#endif
}

/// @brief A region clipped to a view, expressed in the words a row loop walks.
/// @note Rows [y0, y1) and, within each row, word indices [firstWord, lastWord]
///       INCLUSIVE -- inclusive because the head and tail words are the two that
///       carry masks, and a half-open end would make "the tail word" an index that
///       is not in the range.
/// @note `headMask` and `tailMask` are both applied when firstWord == lastWord,
///       i.e. when the whole region-row lives in a single word. That case is not
///       rare -- a 1-pixel-wide region is the degenerate LK window and every
///       region narrower than a word hits it.
template <typename WordType>
struct RegionWords {
    size_t y0 = 0;         ///< first row, inclusive
    size_t y1 = 0;         ///< one past the last row
    size_t firstWord = 0;  ///< first word index of each row, inclusive
    size_t lastWord = 0;   ///< last word index of each row, INCLUSIVE
    WordType headMask = 0; ///< bits of firstWord that are inside the region
    WordType tailMask = 0; ///< bits of lastWord that are inside the region
    bool isEmpty = true;   ///< true when the clipped region covers no pixel
};

/// @brief Region geometry from an already-clipped, non-empty pixel extent.
/// @param x0, x1 Column range, half-open, with 0 <= x0 < x1 <= width.
/// @param y0, y1 Row range, half-open, with 0 <= y0 < y1 <= height.
/// @note The masks are the only arithmetic here that can be subtly wrong, so both
///       are written as low-bit masks of the SAME helper rather than as shifts:
///       `headMask` keeps bits [x0 % WordBits, WordBits), `tailMask` keeps bits
///       [0, (x1-1) % WordBits + 1). The `x1 - 1` is what makes the tail
///       half-open-safe -- x1 % WordBits == 0 would otherwise produce an empty
///       mask for a region ending exactly on a word boundary.
/// @note The non-empty precondition is ASSERTED, not merely documented, and the
///       reason is that `x1 - 1` underflows: regionFromExtent(0, 0, 0, 1) would
///       otherwise return lastWord = SIZE_MAX / WordBits with isEmpty == false,
///       and visitRowWords would walk ~5.8e17 word indices off the end of the
///       buffer. Both callers in this file guard it already (wholeViewWords
///       returns early on width == 0, clipRegion on x0 >= x1), so this costs
///       nothing in release and catches the next caller instead of the next user.
template <typename WordType>
inline RegionWords<WordType> regionFromExtent(size_t x0, size_t x1, size_t y0, size_t y1) {
    BINCV_ASSERT(x0 < x1 && y0 < y1, "reduce: regionFromExtent needs a non-empty extent");
    constexpr size_t wordBits = bitsPerWord<WordType>();
    RegionWords<WordType> out;
    out.y0 = y0;
    out.y1 = y1;
    out.firstWord = x0 / wordBits;
    out.lastWord = (x1 - 1) / wordBits;
    out.headMask = static_cast<WordType>(~lowBitsMask<WordType>(x0 % wordBits));
    out.tailMask = lowBitsMask<WordType>(((x1 - 1) % wordBits) + 1);
    out.isEmpty = false;
    return out;
}

/// @brief The whole of a view: every pixel, no region.
/// @note Goes through the same geometry as a region so that the trailing partial
///       word is masked here too -- that is the padding-bit contract at the top of
///       this file, and routing the no-region overload through a different path
///       would be the obvious way for the two to drift apart.
template <typename WordType>
inline RegionWords<WordType> wholeViewWords(size_t width, size_t height) {
    if (width == 0 || height == 0) return RegionWords<WordType>();
    return regionFromExtent<WordType>(0, width, 0, height);
}

/// @brief Intersects a Rect with a view's extent, in words.
/// @note All the arithmetic is in `long long` before anything becomes a size_t.
///       `x + width` overflows `int` for a large enough rectangle, and a negative
///       origin converted to size_t is not "clipped to zero", it is 2^64 - k --
///       both would turn a clipping contract into an out-of-bounds read.
/// @note Returns isEmpty for every rectangle that survives clipping as nothing:
///       non-positive extents, wholly left/above the image, wholly right/below it.
template <typename WordType>
inline RegionWords<WordType> clipRegion(size_t width, size_t height, const Rect& region) {
    // Rect::empty() rather than a re-spelling of `width <= 0 || height <= 0`.
    // One predicate, one definition: a second copy here is a second thing to keep
    // in agreement with core/types.hpp, and the copy is the one nothing tests.
    if (width == 0 || height == 0 || region.empty()) {
        return RegionWords<WordType>();
    }

    const long long rx0 = static_cast<long long>(region.x);
    const long long ry0 = static_cast<long long>(region.y);
    const long long rx1 = rx0 + static_cast<long long>(region.width);
    const long long ry1 = ry0 + static_cast<long long>(region.height);
    if (rx1 <= 0 || ry1 <= 0) return RegionWords<WordType>();

    const size_t x0 = (rx0 > 0) ? static_cast<size_t>(rx0) : size_t{0};
    const size_t y0 = (ry0 > 0) ? static_cast<size_t>(ry0) : size_t{0};

    // rx1 and ry1 are positive here, so the cast is defined; the min() is what
    // clips a region running past the right or bottom edge.
    const size_t rx1u = static_cast<size_t>(rx1);
    const size_t ry1u = static_cast<size_t>(ry1);
    const size_t x1 = (rx1u < width) ? rx1u : width;
    const size_t y1 = (ry1u < height) ? ry1u : height;

    if (x0 >= x1 || y0 >= y1) return RegionWords<WordType>();
    return regionFromExtent<WordType>(x0, x1, y0, y1);
}

/// @brief Visits every word index of one region-row, ascending, exactly once.
/// @param r A non-empty clipped region.
/// @param visit Called as `visit(wordIndex, mask)`; `mask` selects the bits of
///        that word which lie inside the region.
///
/// @note THE SINGLE-PASS SKELETON. All four entry points share it, so T2.6's
///       "reads each word once" is a property of one function rather than of four
///       loops that happen to agree today, and tests/test_reduce.cpp drives it
///       with a recording visitor to check exactly that.
/// @note The interior mask is the compile-time all-ones constant, so the interior
///       loop -- the one that runs for all but two words of a row -- folds to an
///       unmasked accumulation. Applying the head/tail masks per word instead
///       would be simpler to write and would put a loop-carried select in the only
///       loop Phase 5 cares about.
template <typename WordType, typename Visit>
inline void visitRowWords(const RegionWords<WordType>& r, Visit visit) {
    if (r.firstWord == r.lastWord) {
        visit(r.firstWord, static_cast<WordType>(r.headMask & r.tailMask));
        return;
    }
    visit(r.firstWord, r.headMask);
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));
    for (size_t i = r.firstWord + 1; i < r.lastWord; ++i) {
        visit(i, allOnes);
    }
    visit(r.lastWord, r.tailMask);
}

/// @brief popcount of one view over an already-clipped region.
/// @note Shared by both countNonZero overloads, which differ only in how the
///       region was built. The row pointer is hoisted per row rather than
///       recomputed per word: `row(y)` is loop-invariant and a release build would
///       hoist it anyway, but writing it out is what makes "each word is read
///       once" visible in three lines instead of inferable from an inlining
///       argument.
template <typename WordType>
inline size_t countViewRegion(const BinMatConstView<WordType>& src,
                              const RegionWords<WordType>& r) {
    size_t total = 0;
    for (size_t y = r.y0; y < r.y1; ++y) {
        const WordType* rs = src.row(y);
        visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
            total += popcountWord<WordType>(static_cast<WordType>(rs[i] & mask));
        });
    }
    return total;
}

} // namespace impl

/// @brief The two halves of a split count: pixels where the selector `c` was
///        clear, and pixels where it was set.
/// @note Named for what they select rather than for what a caller does with them.
///       In the LK covariance (ARCHITECTURE 7.5) `c` is `sign_x ^ sign_y`, so
///       `whenClear` counts agreeing signs (products of +1) and `whenSet` counts
///       opposing ones (products of -1), and the cross term is
///       `whenClear - whenSet`.
/// @note **That subtraction is SIGNED, and the two fields are not**, which is why
///       crossTerm() exists and why callers should use it. `whenClear - whenSet`
///       written directly on the fields is `size_t` arithmetic: a negatively
///       correlated window (ΣIxIy < 0, half of them in practice) wraps to ~1.8e19,
///       and the project's warning set -- `-Wconversion -Wsign-conversion -Werror`
///       included -- does not diagnose it, because unsigned wraparound is defined
///       behaviour rather than a conversion. Nothing in a type named for counts
///       announces that its difference is not a count. So the short spelling is
///       made the correct one instead of being warned about in prose.
struct SplitCount {
    size_t whenClear = 0;  ///< pixels in the region where a & b is set and c is clear
    size_t whenSet = 0;    ///< pixels in the region where a & b is set and c is set

    /// @brief The LK cross term: `whenClear - whenSet`, signed (ARCHITECTURE 7.5).
    /// @return ΣIxIy over the region when this came from countAndSplit(mag_x,
    ///         mag_y, sign_x ^ sign_y, window). Negative is ordinary, not an error.
    /// @note `long long` because the difference is signed and because it is exact:
    ///       each half is at most the region's pixel count, and an image with more
    ///       than LLONG_MAX pixels does not fit in an addressable buffer at one bit
    ///       per pixel.
    /// @note Not an operator- on the struct: a reader who sees `crossTerm()` looks
    ///       it up, whereas `a - b` on two size_t fields reads as obviously fine.
    long long crossTerm() const {
        return static_cast<long long>(whenClear) - static_cast<long long>(whenSet);
    }
};

// ---------------------------------------------------------------------------
// The reductions (D-5: views, never containers; D-6: bulk, never per word)
// ---------------------------------------------------------------------------

/// @brief Number of set pixels in `src`. **API TIER 1** -- equal to
///        cv::countNonZero on the same binary content stored as CV_8U.
/// @param src Source view.
/// @return The population count, exactly; never saturated, never an error code.
///
/// @note Whole-word accumulation: one popcount per WordBits pixels, reading the
///       view's own stride per row (D-5). The words strictly inside a row are
///       accumulated unmasked; the trailing partial word is masked to `width`.
/// @note **Bits at or past `width` are never counted**, so a view wrapping a
///       buffer whose padding bits are dirty -- a supported construction -- gives
///       the same answer as a clean one. See the padding-bit section at the top of
///       this file for why this file spends one AND per row on that.
/// @note Reads only, so there is no aliasing precondition of any kind (D-11 binds
///       kernels that write).
/// @note Never throws and never allocates (ARCHITECTURE 5.3). A stride shorter
///       than one row is a programming error: BINCV_ASSERT reports it in debug
///       builds, and it is undefined in release, exactly as an out-of-range at()
///       is.
/// @note An empty view counts 0. That is a value, not an error -- a 0-column frame
///       has no set pixels.
template <typename WordType>
inline size_t countNonZero(BinMatConstView<WordType> src) {
    BINCV_ASSERT(impl::strideCoversARow<WordType>(src.width, src.height, src.stride),
                 "reduce: every view's stride must cover a whole row");

    const impl::RegionWords<WordType> r = impl::wholeViewWords<WordType>(src.width, src.height);
    if (r.isEmpty) return 0;

    BINCV_ASSERT(src.ptr != nullptr, "reduce: a non-empty view needs a non-null pointer");
    return impl::countViewRegion<WordType>(src, r);
}

/// @brief Number of set pixels of `src` inside `region`. **API TIER 1** -- equal to
///        cv::countNonZero(src(region)) on the same content as CV_8U, **for any
///        region contained in the image**. A region that is not contained is
///        clipped here rather than rejected, where cv::Mat::operator()(Rect)
///        throws ("0 <= roi.x && ... roi.x + roi.width <= m.cols"); see THE REGION
///        CONTRACT at the top of this file. The Tier 1 equality is stated over the
///        rectangles OpenCV can express, because outside them OpenCV has no answer
///        to be equal to.
/// @param src Source view.
/// @param region Rectangle in pixels, half-open, **intersected with the image**.
/// @return The population count over the intersection; 0 when it is empty.
///
/// @note The region need not be word-aligned in either coordinate, and usually is
///       not: the first and last words of each region-row are masked and only the
///       words between them are accumulated whole.
/// @note A region may start at a negative coordinate, extend past the right or
///       bottom edge, or lie wholly outside -- see the region contract at the top
///       of this file. This is the calling convention the 31x31 LK window needs at
///       image edges, and clipping here is what stops each caller from writing its
///       own.
/// @note Same padding, aliasing and error contract as the whole-image overload.
template <typename WordType>
inline size_t countNonZero(BinMatConstView<WordType> src, Rect region) {
    BINCV_ASSERT(impl::strideCoversARow<WordType>(src.width, src.height, src.stride),
                 "reduce: every view's stride must cover a whole row");

    const impl::RegionWords<WordType> r = impl::clipRegion<WordType>(src.width, src.height, region);
    if (r.isEmpty) return 0;

    BINCV_ASSERT(src.ptr != nullptr, "reduce: a non-empty view needs a non-null pointer");
    return impl::countViewRegion<WordType>(src, r);
}

/// @brief Number of pixels set in BOTH `a` and `b` inside `region`.
///        **API TIER 3** -- a masked reduction, which OpenCV has no equivalent of
///        (ARCHITECTURE 5.1); `cv::countNonZero(a & b)` needs an intermediate
///        image and this needs none.
/// @param a First source view.
/// @param b Second source view; must have a's dimensions.
/// @param region Rectangle in pixels, half-open, intersected with the image.
/// @return popcount(a & b) over the intersection; 0 when it is empty.
///
/// @note **No temporary.** The AND happens a word at a time inside the reduction,
///       which is the entire reason this exists rather than being spelled
///       `bitwiseAnd` into a scratch image followed by countNonZero: the scratch
///       would be a heap allocation in a kernel (forbidden) or a caller-provided
///       buffer whose only purpose is to be counted and discarded.
/// @note This is ΣIx² / ΣIy² territory for the LK covariance when a == b, and the
///       `mag_x & mag_y` factor of the cross term otherwise (ARCHITECTURE 7.5).
/// @note MVP RECOMPUTES PER WINDOW. LK windows are 31x31 and overlap heavily, so
///       consecutive calls re-read almost the same words; whether a sliding
///       accumulator beats that is E-3 (T2.10) and is deliberately unmeasured
///       here. This signature is chosen so that answering E-3 either way leaves it
///       unchanged.
/// @note Reads only: `a` and `b` may be the same view or overlap arbitrarily.
/// @note Never throws, never allocates. Mismatched dimensions and a stride shorter
///       than a row are BINCV_ASSERT programming errors.
template <typename WordType>
inline size_t countAnd(BinMatConstView<WordType> a, BinMatConstView<WordType> b, Rect region) {
    BINCV_ASSERT(a.width == b.width && a.height == b.height,
                 "reduce: masked reductions need views of the same size");
    BINCV_ASSERT(impl::strideCoversARow<WordType>(a.width, a.height, a.stride) &&
                     impl::strideCoversARow<WordType>(b.width, b.height, b.stride),
                 "reduce: every view's stride must cover a whole row");

    const impl::RegionWords<WordType> r = impl::clipRegion<WordType>(a.width, a.height, region);
    if (r.isEmpty) return 0;

    BINCV_ASSERT(a.ptr != nullptr && b.ptr != nullptr,
                 "reduce: a non-empty view needs a non-null pointer");

    size_t total = 0;
    for (size_t y = r.y0; y < r.y1; ++y) {
        const WordType* ra = a.row(y);
        const WordType* rb = b.row(y);
        impl::visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
            const WordType both = static_cast<WordType>(static_cast<WordType>(ra[i] & rb[i]) & mask);
            total += impl::popcountWord<WordType>(both);
        });
    }
    return total;
}

/// @brief popcount(a & b & ~c) and popcount(a & b & c) over `region`, in ONE pass.
///        **API TIER 3** -- ARCHITECTURE 5.1 names "masked window reductions" as
///        the Tier 3 example, and this is that operation.
/// @param a First source view.
/// @param b Second source view; must have a's dimensions.
/// @param c Selector view; must have a's dimensions.
/// @param region Rectangle in pixels, half-open, intersected with the image.
/// @return `{whenClear, whenSet}` -- the pixels of `a & b` split by `c`.
///
/// @note **THIS IS THE LK COVARIANCE CROSS-TERM** (ARCHITECTURE 7.5), and the
///       reason T2.6 is in the MVP rather than added when T3.6 needs it. With
///       `a = mag_x`, `b = mag_y` and `c = sign_x ^ sign_y`:
///
///           ΣIxIy = whenClear - whenSet
///
///       because a ternary product is +1 exactly where both magnitudes are set and
///       the signs agree (c clear), and -1 where both are set and they disagree
///       (c set). The subtraction is the caller's, in a signed type.
/// @note **Single pass, two popcounts per word.** `a[i]`, `b[i]` and `c[i]` are
///       read once each and both halves come out of them: `whenClear` is
///       `popcount(a&b&mask) - popcount(a&b&mask&c)`, never `popcount(a&b&~c)`.
///       Besides being one popcount cheaper on a target where the popcount is the
///       expensive part (D-6), it never forms `~c` -- which would set every
///       padding bit of a trailing word and count phantom pixels.
/// @note MVP RECOMPUTES PER WINDOW; see countAnd() and E-3 (T2.10).
/// @note **`c` IS A FRAME-SIZED PLANE, NOT A WINDOW-SIZED ONE**, and T3.6 should
///       plan for that. All three views are indexed by the SAME `region`, in the
///       image's coordinate frame, so `c` must carry `a`'s dimensions -- asserted.
///       A 31x31 scratch buffer holding just this window's `sign_x ^ sign_y` is
///       therefore not accepted; the selector is formed once per pyramid level
///       with bitwiseXor (T2.2) into one caller-owned plane and every window reads
///       a rectangle of it:
///
///           BinMat<W> signXor(width, height);                      // once
///           bitwiseXor(dx.constSign(), dy.constSign(), signXor.view());
///           for (kp : keypoints)                                   // per window
///               countAndSplit(dx.constMagnitude(0), dy.constMagnitude(0),
///                             signXor.constView(), window(kp));
///
///       That plane costs one bit per pixel (38400 B at 640x480) on top of the
///       derivative planes, and it is a CALLER allocation -- no kernel here
///       allocates. A four-argument form taking `c0` and `c1` and XOR-ing them in
///       the word loop would need no plane at all; it is deliberately NOT added
///       yet, because choosing between "one plane, three arguments" and "no plane,
///       four arguments" is a memory-versus-speed call, this project settles those
///       with a measurement and a decision rule written first (CLAUDE.md), and the
///       experiment whose stated gate is "T2.6's interface" is E-3 (T2.10), still
///       TODO. Adding it now would fix the interface ahead of the measurement --
///       the same mistake T2.6 forbids for incremental state.
/// @note Reads only: `a`, `b` and `c` may be the same view or overlap arbitrarily.
/// @note Never throws, never allocates. Mismatched dimensions and a stride shorter
///       than a row are BINCV_ASSERT programming errors.
template <typename WordType>
inline SplitCount countAndSplit(BinMatConstView<WordType> a, BinMatConstView<WordType> b,
                                BinMatConstView<WordType> c, Rect region) {
    BINCV_ASSERT(a.width == b.width && a.height == b.height && a.width == c.width &&
                     a.height == c.height,
                 "reduce: masked reductions need views of the same size");
    BINCV_ASSERT(impl::strideCoversARow<WordType>(a.width, a.height, a.stride) &&
                     impl::strideCoversARow<WordType>(b.width, b.height, b.stride) &&
                     impl::strideCoversARow<WordType>(c.width, c.height, c.stride),
                 "reduce: every view's stride must cover a whole row");

    SplitCount out;
    const impl::RegionWords<WordType> r = impl::clipRegion<WordType>(a.width, a.height, region);
    if (r.isEmpty) return out;

    BINCV_ASSERT(a.ptr != nullptr && b.ptr != nullptr && c.ptr != nullptr,
                 "reduce: a non-empty view needs a non-null pointer");

    for (size_t y = r.y0; y < r.y1; ++y) {
        const WordType* ra = a.row(y);
        const WordType* rb = b.row(y);
        const WordType* rc = c.row(y);
        impl::visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
            const WordType both = static_cast<WordType>(static_cast<WordType>(ra[i] & rb[i]) & mask);
            const size_t total = impl::popcountWord<WordType>(both);
            const size_t set = impl::popcountWord<WordType>(static_cast<WordType>(both & rc[i]));
            out.whenSet += set;
            out.whenClear += total - set;
        });
    }
    return out;
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
