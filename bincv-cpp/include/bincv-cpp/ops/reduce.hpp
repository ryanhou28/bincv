#pragma once

/// @file reduce.hpp
/// @brief Bulk population counts over regions and masks.
/// **API TIER 1** for countNonZero, **TIER 3** for the masked forms.
///
/// Seven entry points and one accumulator, and the shape of all of them is the
/// point of the file:
///
/// countNonZero(src) popcount over a whole image Tier 1
/// countNonZero(src, region) popcount over a rectangle Tier 1
/// countAnd(a, b, region) popcount(a & b) Tier 3
/// countAndSplit(a, b, c, region) popcount(a & b & ~c) and
/// popcount(a & b & c), one pass Tier 3
/// countAndSplit(a, b, c0, c1, r) the same split, selector XOR-ed
/// in the word loop, no plane Tier 3
/// countCovariance(a, b, c, region) xx, yy and that split, ONE pass Tier 3
/// countCovariance(a, b, c0, c1, r) the same, and no plane either Tier 3
/// SlidingWindowCount one window sum slid DOWNWARD --
/// one row out, one row in Tier 3
///
/// The split forms are the Lucas-Kanade covariance cross-term written out, and
/// countCovariance is the whole 2x2 matrix (the design notes). They are here in
/// the MVP, next to the plain count, because designing them later would mean
/// designing a *second* reduction interface after callers had been written
/// against the first.
///
/// ---------------------------------------------------------------------------
/// WHICH SHAPE TO REACH FOR, AND WHY -- THE ACCESS PATTERN DECIDES
///
/// Several of those entry points can produce numbers another one also produces,
/// so a caller choosing between them needs the argument and not just the
/// signatures. Every clause below is measured, on the reference device, three
/// runs, against a rule written before the measurement ( /
///, and; ):
///
/// ONE window, or windows that do not overlap
/// countNonZero / countAnd / countAndSplit, once per window. There is
/// nothing to slide across: at 200 isolated keypoints the incremental form
/// issues exactly the same popcounts over exactly the same words and wins
/// nothing (0.98x at W=7, ~1.0x at 31). **One window is a countNonZero.**
///
/// A COLUMN of window positions -- a search sweep, a dense scan, the corner
/// response of the design notes -- AT A LARGE ENOUGH WINDOW
/// SlidingWindowCount. Consecutive windows differ by two rows out of W, and
/// recomputing all W of them re-reads what the previous position already
/// counted. Measured 7.3x on an 8x8 search sweep and 20x on a dense scan at
/// 31x31 (the axis 1) -- but that was against the PRE-accumulator-split
/// baseline. Against the countNonZero this file actually ships,
/// re-measured **5.96x** (search) and **15.9x** at 31x31, which is where
/// predicted the split would put them and still far past the 15% line
/// that selected this branch.
///
/// **THOSE ARE ONE-PLANE, ONE-NUMBER SWEEPS, AND THE QUALIFICATION "AT A
/// LARGE ENOUGH WINDOW" IS MEASURED RATHER THAN CAUTIOUS.** this is the
/// first real caller of this shape, and it sweeps a 2x2 covariance of which
/// only two numbers slide., on the reference device at 640x480: the
/// incremental sweep is 1.22x FASTER at a 31x31 window and 1.20x SLOWER at
/// 3x3 -- the size the reference pipeline runs -- over FOUR runs whose
/// ranking never changes (run-to-run scatter 0.18-0.34% on the ratio below
/// blockSize 31, 3.3% at 31).
/// The incremental state alone is 0.93x at 3, 1.04x at 7 and 1.24x at 31 --
/// so IT crosses over between 3 and 7 while the NET crosses between 7 and 15
/// -- and this class
/// slides only DOWNWARD, so a caller that wants it must sweep COLUMN-MAJOR,
/// which costs a further 12% at 3, 7% at 7 and 1% at 31. Read 15.9x as what it is: a
/// measurement of `countNonZero` on one plane, not a promise to a caller
/// with a different mix of work per position.
///
/// ALL FOUR covariance numbers over one window
/// countCovariance, not countNonZero + countNonZero + countAndSplit. The
/// three-call composition issues exactly the same popcounts but makes
/// THREE traversals of the region and 5 word loads per word index against
/// the fused pass's 3 -- countNonZero(a), countNonZero(b), then a, b and c
/// again in the split. That redundancy measured 1.27x (`uint32_t`) and
/// 1.29x (`uint64_t`) at 31x31 pre-split (the axis 2, reproducing that measurement’s
/// 1.30x); against the shipped code measures **1.20x** and **1.27x**
/// there, and 1.20x-1.65x across the three window sizes.
///
/// THE SELECTOR: `c`, or `c0` and `c1`
/// The four-argument forms XOR the two sign planes inside the word loop and
/// need no selector plane at all; the three-argument forms read a plane the
/// caller already holds. The plane is FASTER per frame even after paying to
/// form it -- 16-18% at earlier work, **11-14% against the shipped code** (
/// axis 3) -- and costs a fifth frame-sized plane at
/// every pyramid level, +25% of the derivative working set, ~51 kB over four
/// levels. CLAUDE.md's tiebreak takes the memory when the two goals
/// disagree, so **the four-argument form is the default and calls it**;
/// the three-argument one stays for a caller that formed a plane for other
/// reasons and should not have to unform it.
///
/// The one thing none of these choices changes is the answer. Every form counts
/// the same pixels under the same region contract, and tests/test_reduce.cpp
/// checks the incremental and fused paths against the recompute and composed ones
/// window for window, edge-clipped positions included.
///
/// ---------------------------------------------------------------------------
/// THERE IS NO PER-WORD POPCOUNT IN THIS FILE'S PUBLIC SURFACE, AND THAT IS
/// THE WHOLE INTERFACE DECISION
///
/// A reasonable-looking library would export `popcount(WordType)` and let callers
/// write their own loops. binCV must not, and the reason is measured aarch64
/// codegen rather than taste (the design notes). aarch64 -- the primary
/// target -- has no scalar population-count instruction, so every popcount has to
/// travel to the NEON register file and its result has to come back, and the cost
/// is dominated by those register-domain crossings rather than by `cnt` itself.
///
/// **The sequences below are what THIS FILE's kernels emit**, read out of
/// `g++ -O2 -DNDEBUG -S` on the reference device (aarch64, g++ 14.2), not the
/// ISA-level illustration in -- which measured a minimal
/// standalone `__builtin_popcountll` under clang and is right about the ISA. The
/// count that matters is CROSSINGS PER WORD, and it is not the same for every
/// entry point -- it rises with the number of source streams a kernel ANDs
/// together, which is why countCovariance and the four-argument splits sit at the
/// expensive end and why nothing here hands a caller the per-word primitive:
///
/// countNonZero 1 crossing/word ldr d31, [x2], 8; loaded straight
/// cnt v31.8b, v31.8b; into NEON --
/// addv b31, v31.8b; no inbound fmov
/// fmov x1, d31; NEON -> GPR
/// add x0, x0, x1; accumulate in a GPR
///
/// countAnd 2 crossings/word the AND is done in GPRs, so the word has
/// to be moved in (`fmov d31, x1`) as well as
/// out (`fmov x1, d31`)
///
/// countAndSplit 4 crossings/word two values in (`fmov d30, x2`,
/// `fmov d31, x0`) and two out (`fmov x2, d30`,
/// `fmov x0, d31`) -- and this is the LK
/// covariance path (the design notes)
///
/// countCovariance four popcounts per word rather than two,
/// over the same three loads. It costs MORE
/// per word than countAndSplit and is still
/// 1.20-1.27x faster than the three calls it
/// replaces (the axis 2), because those
/// three traverse the region three times and
/// make 5 loads per word index to this one's
/// 3.
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
/// **Today's implementation is the scalar one that asks for** -- one
/// `__builtin_popcountll` per word, in impl::popcountWord -- so the loops below do
/// NOT yet keep anything in vector registers, and on the reference device they run
/// at the speed of the per-word loop the design rule forbids exposing. That is measured, and it
/// is recorded rather than papered over:, the aarch64 half of
///, measured 0.99x against that loop with 1.8x available from a vector
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
/// 1. **Views, never containers**. Strides are read per row and may differ
/// between the arguments.
///
/// **Spell the arguments `constMagnitude` / `constSign` / `constView`,
/// not `magnitude` / `sign` / `view`.** Every entry point below takes
/// BinMatConstView<W>, and template argument deduction does not consider
/// user-defined conversions (, and the note on
/// BinMatView::operator BinMatConstView), so handing one a BinMatView from a
/// NON-const container is a deduction failure, not an implicit conversion:
///
/// countNonZero(dx.magnitude(0), w) // does NOT compile
/// countNonZero(dx.constMagnitude(0), w) // this is the spelling
///
/// The diagnostic says "BinMatView<W> is not derived from BinMatConstView<W>"
/// and does not mention the conversion, which is why it is written out here
/// rather than left to be met at a call site. SignedQuantMat::constMagnitude,
/// constSign and BinMat::constView exist for exactly this, and every example
/// in this file, in the design notes and in tests/test_reduce.cpp uses them.
/// (Reductions take no destination, so unlike ops/logic.hpp there is no
/// container overload here that would hide the distinction.)
///
/// 2. **No allocation and no throw** (the design notes). Mismatched dimensions
/// and a stride too short to hold a row are programming errors, reported by
/// BINCV_ASSERT in debug builds and undefined in release. There is no error
/// return: a count is a `size_t`, and a sentinel would have to be checked by
/// every caller of an operation that cannot fail on valid views.
///
/// 3. **Read-only, so ALIASING IS UNRESTRICTED.** This is the one family of
/// kernels this does not bind: nothing is written, so `a`, `b` and `c` may be
/// the same view, may overlap partially, or may be unrelated.
/// `countAndSplit(m, m, m, r)` is well defined (it reports every set pixel as
/// `whenSet`). Callers of the covariance do exactly this kind of thing --
/// `countNonZero(mag_x, w)` and `countAnd(mag_x, mag_y, w)` over overlapping
/// windows of one frame -- and an alias assertion here would reject it.
///
/// 4. **Counts are exact and unsaturated.** `size_t` holds the population of any
/// image that fits in memory (one bit per pixel means the count cannot exceed
/// the bit count of the buffer).
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
/// This is not leniency, it is the calling convention needs. The LK window is
/// 31x31 centerd on a keypoint, so every keypoint within 15 pixels of an edge
/// produces an out-of-range rectangle. The alternative -- assert and make the
/// caller clamp -- puts the same four `min`/`max` expressions in every call site,
/// and a wrong one there is a silently wrong covariance rather than a diagnostic.
/// Clipping once, here, is also what makes "correct for windows clipped at image
/// edges" a property of the library rather than of its callers.
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
/// specifies "whole-word accumulation; correctness depends on padding bits
/// being zero". Taken literally that means accumulating a row's trailing partial
/// word unmasked and relying on the invariant. This file **does not** do that, and
/// the deviation is deliberate:
///
/// - A source with dirty padding is a SUPPORTED construction, not a bug. BinMat's
/// wrap constructor says in as many words that a wrapped buffer's padding bits
/// are the caller's (sensor DMA, a sub-region of a larger frame), and
/// tests/test_logic.cpp already sweeps sources built that way. A reduction that
/// over-counts on such a view is a wrong answer produced from a legal input.
/// - ops/shift.hpp reached the same conclusion for the same reason and pays a
/// good deal more for it: every source word is substituted through
/// impl::extendedRowWord. Two kernel families disagreeing about whether a
/// source's padding is trustworthy would be worse than either rule.
/// - The cost is one AND per ROW, not per word. The trailing word is masked
/// outside the interior loop exactly as in ops/logic.hpp, so the loop that
/// matters carries no mask and Phase 5's vectorization is unaffected.
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
/// requires each word to be read once. The row body reads `a[i]`, `b[i]` and
/// `c[i]` once each per word index and produces both halves of the answer from
/// them:
///
/// both = a[i] & b[i] & mask
/// total = popcount(both) // pixels where both magnitudes are set
/// set = popcount(both & c[i]) //... and the signs disagree
/// whenSet += set
/// whenClear += total - set
///
/// Two popcounts per word, not three: `whenClear` is `total - set` rather than
/// `popcount(both & ~c[i])`. That is not only cheaper on the target (the popcount
/// is the expensive operation, see), it removes a padding hazard -- `~c[i]`
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
/// them, and the difference is measured.** the design notes needs four numbers over
/// one window; through these primitives that is three calls -- countNonZero(mag_x),
/// countNonZero(mag_y), countAndSplit(...) -- and therefore three traversals of the
/// same window, issuing the same popcounts a fused traversal would issue once, and
/// making 5 word loads per word index where the fused pass makes 3. On the
/// reference device that composition costs 1.30x a fused pass, reproducibly, with
/// the popcount count identical on both sides (, reproduced at
/// 1.27-1.29x across two word widths and three window sizes by earlier work, and at
/// 1.20-1.65x against the SHIPPED entry points by earlier work).
///
/// ** IS SETTLED, AND IT WENT AGAINST THE SHAPE THIS FILE ORIGINALLY SHIPPED**
/// ( /, ). All three of its axes
/// moved off the simpler form, and landed all three plus a fourth finding:
///
/// 1. a vertically-sliding accumulator beats recompute by 7.3x on a search sweep
/// and 20x on a dense scan at 31x31 -> SlidingWindowCount;
/// 2. the fused covariance entry point wins 1.27-1.29x -> countCovariance;
/// 3. a four-argument countAndSplit costs 16-18% of one operation's time and
/// saves a frame-sized plane at every pyramid level, and the project's
/// tiebreak takes the memory -> the (a, b, c0, c1, region) overloads;
/// 4. and the accumulator that carried one sum across a whole region was a
/// single dependency chain through the popcount latency -> the row bodies in
/// impl:: each return their own partial sum.
///
/// The interface was deliberately NOT changed in the same commit as the
/// measurement that gated it; the measurement is and the interface is.
/// Item 4 landed FIRST of the four, so items 1-3's re-measured ratios report the
/// gain the shipped code actually has rather than a gain against a baseline that
/// no longer exists.
///
/// **THOSE FOUR NUMBERS ARE that measurement’s, AND RE-MEASURED THEM AGAINST THE CODE
/// THAT SHIPPED.** Every ratio quoted below carries both where they differ, because
/// a reader picking an entry point wants the shipped one and a reader auditing the
/// decision wants the one the rule was applied to. Item 1 becomes 5.96x / 15.9x;
/// item 2 becomes 1.20x / 1.27x at 31x31 (1.20-1.65x across the three window
/// sizes); item 3's gap narrows to 11-14%. Item 4's own figure moved furthest and
/// is the one to be careful with: recorded 1.15-1.32x, but never measured it
/// directly -- it inferred it from axis 1's isolated-keypoint column, which differs
/// from a per-window countNonZero in TWO ways rather than one. Timed directly and
/// interleaved, puts the split at **1.03-1.09x** at the LK window sizes and a
/// 5-6% LOSS at W=7 on the overlapping patterns. No decision moves -- the split
/// costs no memory and no interface and still wins at W=15 and W=31 -- but
/// 1.03-1.09x is the number to quote (, amended).

#include <climits>  // INT_MIN / INT_MAX -- SlidingWindowCount::window asserts its range
#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
// impl::bitsPerWord / lowBitsMask / minRowWords, and impl::strideCoversARow --
// the row-geometry vocabulary every kernel under ops/ is written in. Note that
// the alias predicates it also carries are NOT used here: see promise 3.
#include "../impl/kernel_util.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief The two halves of a split count: pixels where the selector `c` was
/// clear, and pixels where it was set.
/// @note Named for what they select rather than for what a caller does with them.
/// In the LK covariance (the design notes) `c` is `sign_x ^ sign_y`, so
/// `whenClear` counts agreeing signs (products of +1) and `whenSet` counts
/// opposing ones (products of -1), and the cross term is
/// `whenClear - whenSet`.
/// @note **That subtraction is SIGNED, and the two fields are not**, which is why
/// crossTerm exists and why callers should use it. `whenClear - whenSet`
/// written directly on the fields is `size_t` arithmetic: a negatively
/// correlated window (ΣIxIy < 0, half of them in practice) wraps to ~1.8e19,
/// and the project's warning set -- `-Wconversion -Wsign-conversion -Werror`
/// included -- does not diagnose it, because unsigned wraparound is defined
/// behavior rather than a conversion. Nothing in a type named for counts
/// announces that its difference is not a count. So the short spelling is
/// made the correct one instead of being warned about in prose.
struct SplitCount {
    size_t whenClear = 0;  ///< pixels in the region where a & b is set and c is clear
    size_t whenSet = 0;    ///< pixels in the region where a & b is set and c is set

    /// @brief The LK cross term: `whenClear - whenSet`, signed (the design notes).
    /// @return ΣIxIy over the region when this came from countAndSplit(mag_x,
    /// mag_y, sign_x ^ sign_y, window). Negative is ordinary, not an error.
    /// @note `long long` because the difference is signed and because it is exact:
    /// each half is at most the region's pixel count, and an image with more
    /// than LLONG_MAX pixels does not fit in an addressable buffer at one bit
    /// per pixel.
    /// @note Not an operator- on the struct: a reader who sees `crossTerm` looks
    /// it up, whereas `a - b` on two size_t fields reads as obviously fine.
    long long crossTerm() const {
        return static_cast<long long>(whenClear) - static_cast<long long>(whenSet);
    }
};

/// @brief The four numbers of a 2x2 gradient covariance over one region:
/// popcount(a), popcount(b), and the split of `a & b` by the selector.
/// @note **API TIER 3.** Returned by countCovariance, which produces all four
/// from ONE traversal ( item 2). The three-call composition --
/// countNonZero(a) + countNonZero(b) + countAndSplit(a, b, c) -- produces
/// the same four numbers with the same popcounts, three traversals and 5
/// word loads per word index to the fused pass's 3, and measured 1.27x
/// (`uint32_t`) / 1.29x (`uint64_t`) slower at 31x31 pre-split
/// ( axis 2, reproducing that measurement’s 1.30x) and **1.20x /
/// 1.27x** against the shipped code.
/// @note With `a = mag_x`, `b = mag_y` and the selector `sign_x ^ sign_y`
/// (the design notes): `xx` is the Sigma-Ix-squared term, `yy` the
/// Sigma-Iy-squared term, and `crossTerm` is Sigma-IxIy.
/// @note `xy` is a whole SplitCount rather than a pre-subtracted difference so
/// that the two halves stay available -- and so that the SIGNED
/// subtraction has exactly one implementation, in SplitCount::crossTerm.
struct CovarianceCount {
    size_t xx = 0;      ///< pixels of `a` set inside the region
    size_t yy = 0;      ///< pixels of `b` set inside the region
    SplitCount xy;      ///< pixels of `a & b`, split by the selector

    /// @brief The LK cross term, signed: `xy.whenClear - xy.whenSet`.
    /// @note Forwards to SplitCount::crossTerm rather than re-deriving it. A
    /// second copy of that subtraction is a second place for it to be
    /// written unsigned, which is the whole hazard crossTerm exists for.
    long long crossTerm() const { return xy.crossTerm(); }
};


namespace impl {

// ---------------------------------------------------------------------------
// INTERNAL. Nothing in this namespace is public API or carries any stability
// promise, and popcountWord in particular exists only so that the bulk entry
// points below have something to call -- see the section at the top of this
// file before exporting, re-exporting, or "just using" it from another header.
// ---------------------------------------------------------------------------

/// @brief Population count of one word without any compiler builtin.
/// **INTERNAL.**
/// @note Not decoration, and not dead source. binCV's claim is that it needs a
/// C++17 compiler and nothing else; MSVC has no __builtin_popcountll, so
/// without this the header would silently be GCC/clang-only. It is also the
/// sequence a Cortex-M build compiles to, where no popcount instruction
/// exists at any width (the design notes).
/// @note **Compiled unconditionally, on every toolchain, and tested against the
/// builtin** by Reduce.PortablePopcount_*. Guarding the definition itself
/// behind the #if would make it source that no configuration this project
/// verifies ever compiles -- the shape of dead gate CLAUDE.md and are
/// both written about. It costs one unreferenced inline template.
template <typename WordType>
inline size_t popcountWordPortable(WordType w) {
    uint64_t v = static_cast<uint64_t>(w);
    v = v - ((v >> 1) & UINT64_C(0x5555555555555555));
    v = (v & UINT64_C(0x3333333333333333)) + ((v >> 2) & UINT64_C(0x3333333333333333));
    v = (v + (v >> 4)) & UINT64_C(0x0F0F0F0F0F0F0F0F);
    return static_cast<size_t>((v * UINT64_C(0x0101010101010101)) >> 56);
}

/// @brief Population count of one word. **INTERNAL.**
/// @note The scalar form specifies. On aarch64 this is two register-domain
/// crossings around a `cnt`, which is precisely why no caller outside this
/// file may reach it: the crossings are amortized by the bulk loops here
/// and cannot be amortized by anyone calling this per word.
/// @note And on the x86-64 baseline binCV actually ships -- no `-march`, since
/// the build detects AVX2/AVX-512 without enabling them until runtime
/// dispatch lands -- `__builtin_popcountll` is not `popcntq` at all. GCC
/// emits a CALL to libgcc's `__popcountdi2`, once per word, and clang
/// inlines the SWAR sequence above. Measured cost: 4.2x slower than
/// cv::countNonZero, against 2.0x faster once the instruction is available
///. A third target tier whose per-word cost dwarfs the
/// count itself, which is the design rule’s argument rather than an exception to it.
/// @note Phase 5 replaces the LOOPS below, not this function. A vectorized
/// reduction keeps whole vectors in NEON registers and never forms a
/// per-word count at all.
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
/// INCLUSIVE -- inclusive because the head and tail words are the two that
/// carry masks, and a half-open end would make "the tail word" an index that
/// is not in the range.
/// @note `headMask` and `tailMask` are both applied when firstWord == lastWord,
/// i.e. when the whole region-row lives in a single word. That case is not
/// rare -- a 1-pixel-wide region is the degenerate LK window and every
/// region narrower than a word hits it.
template <typename WordType>
struct RegionWords {
    size_t y0 = 0;         ///< first row, inclusive
    size_t y1 = 0;         ///< one past the last row
    size_t firstWord = 0;  ///< first word index of each row, inclusive
    size_t lastWord = 0;   ///< last word index of each row, INCLUSIVE
    WordType headMask = 0; ///< bits of firstWord that are inside the region
    WordType tailMask = 0; ///< bits of lastWord that are inside the region
    bool isEmpty = true;   ///< true when the clipped region covers no pixel

    /// The CLIPPED column range, half-open. Carried because a kernel that extracts
    /// the whole region into ONE aligned word needs the pixel bounds, and
    /// `regionFromExtent` is handed them anyway -- recovering them from the masks
    /// afterwards would cost a count-trailing-zeros to rediscover what was already
    /// known.
    size_t x0 = 0;
    size_t x1 = 0;
};

/// @brief Region geometry from an already-clipped, non-empty pixel extent.
/// @param x0, x1 Column range, half-open, with 0 <= x0 < x1 <= width.
/// @param y0, y1 Row range, half-open, with 0 <= y0 < y1 <= height.
/// @note The masks are the only arithmetic here that can be subtly wrong, so both
/// are written as low-bit masks of the SAME helper rather than as shifts:
/// `headMask` keeps bits [x0 % WordBits, WordBits), `tailMask` keeps bits
/// [0, (x1-1) % WordBits + 1). The `x1 - 1` is what makes the tail
/// half-open-safe -- x1 % WordBits == 0 would otherwise produce an empty
/// mask for a region ending exactly on a word boundary.
/// @note The non-empty precondition is ASSERTED, not merely documented, and the
/// reason is that `x1 - 1` underflows: regionFromExtent(0, 0, 0, 1) would
/// otherwise return lastWord = SIZE_MAX / WordBits with isEmpty == false,
/// and visitRowWords would walk ~5.8e17 word indices off the end of the
/// buffer. Both callers in this file guard it already (wholeViewWords
/// returns early on width == 0, clipRegion on x0 >= x1), so this costs
/// nothing in release and catches the next caller instead of the next user.
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
    out.x0 = x0;
    out.x1 = x1;
    out.isEmpty = false;
    return out;
}

/// @brief The whole of a view: every pixel, no region.
/// @note Goes through the same geometry as a region so that the trailing partial
/// word is masked here too -- that is the padding-bit contract at the top of
/// this file, and routing the no-region overload through a different path
/// would be the obvious way for the two to drift apart.
template <typename WordType>
inline RegionWords<WordType> wholeViewWords(size_t width, size_t height) {
    if (width == 0 || height == 0) return RegionWords<WordType>();
    return regionFromExtent<WordType>(0, width, 0, height);
}

/// @brief The COLUMN half of a clip: a band of columns over EVERY row of a view.
/// @param x, bandWidth The column extent in pixels, half-open, unclipped.
/// @return Word range and masks for columns [x, x + bandWidth) intersected with
/// the image, over rows [0, height).
///
/// @note Split out of clipRegion because SlidingWindowCount needs exactly this
/// and nothing else: every window in a vertical column of positions shares
/// one x extent, so the masks are built ONCE and the sliding is pure row
/// arithmetic. Sharing the function rather than copying its four lines is
/// the point -- the head/tail masks are the arithmetic in this file most
/// able to be subtly wrong, and a second copy is the one nothing tests.
/// @note All the arithmetic is in `long long` before anything becomes a size_t.
/// `x + bandWidth` overflows `int` for a large enough rectangle, and a
/// negative origin converted to size_t is not "clipped to zero", it is
/// 2^64 - k -- both would turn a clipping contract into an out-of-bounds
/// read.
template <typename WordType>
inline RegionWords<WordType> clipColumns(size_t width, size_t height, int x, int bandWidth) {
    if (width == 0 || height == 0 || bandWidth <= 0) return RegionWords<WordType>();

    const long long rx0 = static_cast<long long>(x);
    const long long rx1 = rx0 + static_cast<long long>(bandWidth);
    if (rx1 <= 0) return RegionWords<WordType>();

    const size_t x0 = (rx0 > 0) ? static_cast<size_t>(rx0) : size_t{0};
    // rx1 is positive here, so the cast is defined; the min is what clips a
    // band running past the right edge.
    const size_t rx1u = static_cast<size_t>(rx1);
    const size_t x1 = (rx1u < width) ? rx1u : width;

    if (x0 >= x1) return RegionWords<WordType>();
    return regionFromExtent<WordType>(x0, x1, 0, height);
}

/// @brief Intersects a Rect with a view's extent, in words.
/// @note The column half is clipColumns; only the row range is computed here,
/// so the two callers cannot drift apart on the masks.
/// @note Returns isEmpty for every rectangle that survives clipping as nothing:
/// non-positive extents, wholly left/above the image, wholly right/below it.
template <typename WordType>
inline RegionWords<WordType> clipRegion(size_t width, size_t height, const Rect& region) {
    // Rect::empty rather than a re-spelling of `width <= 0 || height <= 0`.
    // One predicate, one definition: a second copy here is a second thing to keep
    // in agreement with core/types.hpp, and the copy is the one nothing tests.
    if (region.empty()) return RegionWords<WordType>();

    RegionWords<WordType> out = clipColumns<WordType>(width, height, region.x, region.width);
    if (out.isEmpty) return out;

    const long long ry0 = static_cast<long long>(region.y);
    const long long ry1 = ry0 + static_cast<long long>(region.height);
    if (ry1 <= 0) return RegionWords<WordType>();

    const size_t y0 = (ry0 > 0) ? static_cast<size_t>(ry0) : size_t{0};
    const size_t ry1u = static_cast<size_t>(ry1);
    const size_t y1 = (ry1u < height) ? ry1u : height;

    if (y0 >= y1) return RegionWords<WordType>();
    out.y0 = y0;
    out.y1 = y1;
    return out;
}

/// @brief Visits every word index of one region-row, ascending, exactly once.
/// @param r A non-empty clipped region.
/// @param visit Called as `visit(wordIndex, mask)`; `mask` selects the bits of
/// that word which lie inside the region.
///
/// @note THE SINGLE-PASS SKELETON. All four entry points share it, so that work’s
/// "reads each word once" is a property of one function rather than of four
/// loops that happen to agree today, and tests/test_reduce.cpp drives it
/// with a recording visitor to check exactly that.
/// @note The interior mask is the compile-time all-ones constant, so the interior
/// loop -- the one that runs for all but two words of a row -- folds to an
/// unmasked accumulation. Applying the head/tail masks per word instead
/// would be simpler to write and would put a loop-carried select in the only
/// loop Phase 5 cares about.
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

// ---------------------------------------------------------------------------
// THE ROW BODIES, AND WHY EACH RETURNS ITS OWN PARTIAL SUM
//
// Every reduction in this file is a loop over rows whose body is one of the four
// functions below. They return a row's count rather than adding into a
// caller-held accumulator, and that is a MEASURED choice, not a stylistic one
// ( item 4;, ).
//
// The previous shape carried ONE size_t across every row and every word of a
// region. On the reference device that makes a window traversal a single
// serialized dependency chain through the popcount latency -- and the popcount is
// the expensive operation on aarch64, so the chain, not the work, sets the
// pace. Per-row partial sums give the core one independent chain per row and cost
// nothing: addition of counts is associative and none of these sums can overflow,
// since each is bounded by the region's pixel count.
//
// HOW MUCH IT IS WORTH, AND THE FIGURE THAT WAS WITHDRAWN. recorded
// 1.15-1.32x at LK window sizes, read off its isolated-keypoint column on the
// argument that the sliding path never executes there, so the accumulator was the
// only thing left to explain the gap. That argument has a hole: the sliding form
// differs from a per-window countNonZero in TWO ways, not one -- where the sum
// lands, AND that it clips its column band once at construction instead of running
// the full region clip per position. a measurement timed the split directly and
// interleaved, identical popcounts over identical words with only the accumulator
// changed, and measured **1.03-1.09x at the LK window sizes (1.08x at W=31) and a
// 5-6% LOSS at W=7 on the overlapping patterns**. That is the number to quote;
// this is amended to it.
//
// The decision is unchanged -- the split costs no memory, no interface and no
// popcount, and it is a gain at both window sizes LK uses. What changed is only
// the magnitude, and the dependency-chain reasoning is what survives: the gain
// growing with the row count is what that explanation predicts, and it does.
//
// a measurement measured it on countViewRegion. It is applied to all four row bodies
// because the argument is about where a sum lands and not about which kernel is
// summing, and because leaving countAnd/countAndSplit unsplit would have made the
// re-measured fused-versus-composed ratio a mix of two effects instead of the
// traversal count it is meant to isolate.
// ---------------------------------------------------------------------------

/// @brief popcount of ONE row of an already-clipped region. **INTERNAL.**
/// @param rs First word of the row -- the caller hoists `src.row(y)`, which is
/// loop-invariant, so that "each word is read once" is visible rather than
/// inferable from an inlining argument.
template <typename WordType>
inline size_t countRowRegion(const WordType* rs, const RegionWords<WordType>& r) {
    size_t rowTotal = 0;
    visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
        rowTotal += popcountWord<WordType>(static_cast<WordType>(rs[i] & mask));
    });
    return rowTotal;
}

/// @brief popcount(a & b) over ONE row of an already-clipped region. **INTERNAL.**
template <typename WordType>
inline size_t countAndRowRegion(const WordType* ra, const WordType* rb,
                                const RegionWords<WordType>& r) {
    size_t rowTotal = 0;
    visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
        rowTotal += popcountWord<WordType>(
            static_cast<WordType>(static_cast<WordType>(ra[i] & rb[i]) & mask));
    });
    return rowTotal;
}

/// @brief The split of `a & b` by a selector, over ONE row. **INTERNAL.**
/// @param selector Called as `selector(wordIndex)` and returns that word of the
/// selector: either `c[i]` (the precomputed-plane form) or `c0[i] ^ c1[i]`
/// (the four-argument form). One loop body, two selectors -- so "single
/// pass, two popcounts per word, whenClear is total - set" is ONE
/// definition rather than a property four loops happen to share.
/// @note `whenClear` is `total - set`, never `popcount(both & ~sel)`. Besides
/// being one popcount cheaper on a target where the popcount is the
/// expensive part, it never forms `~sel` -- which would set every
/// padding bit of a trailing word and count phantom pixels.
template <typename WordType, typename Selector>
inline SplitCount splitRowRegion(const WordType* ra, const WordType* rb,
                                              Selector selector,
                                              const RegionWords<WordType>& r) {
    SplitCount row;
    visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
        const WordType both = static_cast<WordType>(static_cast<WordType>(ra[i] & rb[i]) & mask);
        const size_t total = popcountWord<WordType>(both);
        const size_t set = popcountWord<WordType>(static_cast<WordType>(both & selector(i)));
        row.whenSet += set;
        row.whenClear += total - set;
    });
    return row;
}

/// @brief All four covariance numbers over ONE row, from one visit per word.
/// **INTERNAL.**
/// @note This is the fused body the axis 2 selected: `a[i]`, `b[i]` and the
/// selector are read once each and produce xx, yy, whenClear and whenSet.
/// The composition out of countNonZero x2 + countAndSplit issues exactly
/// these popcounts but traverses and loads the region three times.
template <typename WordType, typename Selector>
inline CovarianceCount covarianceRowRegion(const WordType* ra, const WordType* rb,
                                                        Selector selector,
                                                        const RegionWords<WordType>& r) {
    CovarianceCount row;
    visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
        const WordType wa = static_cast<WordType>(ra[i] & mask);
        const WordType wb = static_cast<WordType>(rb[i] & mask);
        const WordType both = static_cast<WordType>(wa & wb);
        const size_t total = popcountWord<WordType>(both);
        const size_t set = popcountWord<WordType>(static_cast<WordType>(both & selector(i)));
        row.xx += popcountWord<WordType>(wa);
        row.yy += popcountWord<WordType>(wb);
        row.xy.whenSet += set;
        row.xy.whenClear += total - set;
    });
    return row;
}

/// @brief popcount of one view over an already-clipped region.
/// @note Shared by both countNonZero overloads, which differ only in how the
/// region was built.
template <typename WordType>
inline size_t countViewRegion(const BinMatConstView<WordType>& src,
                              const RegionWords<WordType>& r) {
    size_t total = 0;
    for (size_t y = r.y0; y < r.y1; ++y) {
        total += countRowRegion<WordType>(src.row(y), r);
    }
    return total;
}

} // namespace impl

// ---------------------------------------------------------------------------
// The reductions ( views, never containers; bulk, never per word)
// ---------------------------------------------------------------------------

/// @brief Number of set pixels in `src`. **API TIER 1** -- equal to
/// cv::countNonZero on the same binary content stored as CV_8U.
/// @param src Source view.
/// @return The population count, exactly; never saturated, never an error code.
///
/// @note Whole-word accumulation: one popcount per WordBits pixels, reading the
/// view's own stride per row. The words strictly inside a row are
/// accumulated unmasked; the trailing partial word is masked to `width`.
/// @note **Bits at or past `width` are never counted**, so a view wrapping a
/// buffer whose padding bits are dirty -- a supported construction -- gives
/// the same answer as a clean one. See the padding-bit section at the top of
/// this file for why this file spends one AND per row on that.
/// @note Reads only, so there is no aliasing precondition of any kind ( binds
/// kernels that write).
/// @note Never throws and never allocates (the design notes). A stride shorter
/// than one row is a programming error: BINCV_ASSERT reports it in debug
/// builds, and it is undefined in release, exactly as an out-of-range at
/// is.
/// @note An empty view counts 0. That is a value, not an error -- a 0-column frame
/// has no set pixels.
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
/// cv::countNonZero(src(region)) on the same content as CV_8U, **for any
/// region contained in the image**. A region that is not contained is
/// clipped here rather than rejected, where cv::Mat::operator(Rect)
/// throws ("0 <= roi.x &&... roi.x + roi.width <= m.cols"); see THE REGION
/// CONTRACT at the top of this file. The Tier 1 equality is stated over the
/// rectangles OpenCV can express, because outside them OpenCV has no answer
/// to be equal to.
/// @param src Source view.
/// @param region Rectangle in pixels, half-open, **intersected with the image**.
/// @return The population count over the intersection; 0 when it is empty.
///
/// @note The region need not be word-aligned in either coordinate, and usually is
/// not: the first and last words of each region-row are masked and only the
/// words between them are accumulated whole.
/// @note A region may start at a negative coordinate, extend past the right or
/// bottom edge, or lie wholly outside -- see the region contract at the top
/// of this file. This is the calling convention the 31x31 LK window needs at
/// image edges, and clipping here is what stops each caller from writing its
/// own.
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
/// **API TIER 3** -- a masked reduction, which OpenCV has no equivalent of
/// (the design notes); `cv::countNonZero(a & b)` needs an intermediate
/// image and this needs none.
/// @param a First source view.
/// @param b Second source view; must have a's dimensions.
/// @param region Rectangle in pixels, half-open, intersected with the image.
/// @return popcount(a & b) over the intersection; 0 when it is empty.
///
/// @note **No temporary.** The AND happens a word at a time inside the reduction,
/// which is the entire reason this exists rather than being spelled
/// `bitwiseAnd` into a scratch image followed by countNonZero: the scratch
/// would be a heap allocation in a kernel (forbidden) or a caller-provided
/// buffer whose only purpose is to be counted and discarded.
/// @note This is ΣIx² / ΣIy² territory for the LK covariance when a == b, and the
/// `mag_x & mag_y` factor of the cross term otherwise (the design notes).
/// @note RECOMPUTES PER WINDOW, and that is the right thing for ONE window. LK
/// windows are 31x31 and overlap heavily, so consecutive calls over a
/// COLUMN of positions re-read almost the same words -- SlidingWindowCount
/// is the entry point for that pattern (7.3x on a search sweep, 20x on a
/// dense scan at 31x31; the axis 1). This signature is unchanged by it: a
/// caller with one window still wants exactly this.
/// @note Reads only: `a` and `b` may be the same view or overlap arbitrarily.
/// @note Never throws, never allocates. Mismatched dimensions and a stride shorter
/// than a row are BINCV_ASSERT programming errors.
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
        total += impl::countAndRowRegion<WordType>(a.row(y), b.row(y), r);
    }
    return total;
}

/// @brief popcount(a & b & ~c) and popcount(a & b & c) over `region`, in ONE pass.
/// **API TIER 3** -- the design notes names "masked window reductions" as
/// the Tier 3 example, and this is that operation.
/// @param a First source view.
/// @param b Second source view; must have a's dimensions.
/// @param c Selector view; must have a's dimensions.
/// @param region Rectangle in pixels, half-open, intersected with the image.
/// @return `{whenClear, whenSet}` -- the pixels of `a & b` split by `c`.
///
/// @note **THIS IS THE LK COVARIANCE CROSS-TERM** (the design notes), and the
/// reason this is in the MVP rather than added when needs it. With
/// `a = mag_x`, `b = mag_y` and `c = sign_x ^ sign_y`:
///
/// ΣIxIy = whenClear - whenSet
///
/// because a ternary product is +1 exactly where both magnitudes are set and
/// the signs agree (c clear), and -1 where both are set and they disagree
/// (c set). The subtraction is the caller's, in a signed type.
/// @note **Single pass, two popcounts per word.** `a[i]`, `b[i]` and `c[i]` are
/// read once each and both halves come out of them: `whenClear` is
/// `popcount(a&b&mask) - popcount(a&b&mask&c)`, never `popcount(a&b&~c)`.
/// Besides being one popcount cheaper on a target where the popcount is the
/// expensive part, it never forms `~c` -- which would set every
/// padding bit of a trailing word and count phantom pixels.
/// @note RECOMPUTES PER WINDOW; see countAnd. The sliding form
/// (SlidingWindowCount) stands alongside this one, not instead of it.
/// @note **Wanting xx and yy as well? Call countCovariance instead.** Composing
/// the 2x2 covariance out of countNonZero x2 plus this measured 1.27-1.29x
/// slower at 31x31 for redundant traversal alone (the axis 2), and
/// 1.20-1.27x slower there against the shipped code.
/// @note **`c` IS A FRAME-SIZED PLANE, NOT A WINDOW-SIZED ONE.** All three views
/// are indexed by the SAME `region`, in the image's coordinate frame, so `c`
/// must carry `a`'s dimensions -- asserted. A 31x31 scratch buffer holding
/// just this window's `sign_x ^ sign_y` is therefore not accepted; the
/// selector is formed once per pyramid level with bitwiseXor into one
/// caller-owned plane and every window reads a rectangle of it:
///
/// BinMat<W> signXor(width, height); // once
/// bitwiseXor(dx.constSign, dy.constSign, signXor.view);
/// for (kp : keypoints) // per window
/// countAndSplit(dx.constMagnitude(0), dy.constMagnitude(0),
/// signXor.constView, window(kp));
///
/// That plane costs one bit per pixel (38400 B at 640x480) on top of the
/// derivative planes, and it is a CALLER allocation -- no kernel here
/// allocates.
/// @note **NOTHING OBLIGES A CALLER TO FORM THAT PLANE ANY MORE.** The
/// four-argument overload below takes `c0` and `c1` and XORs them in the
/// word loop, needing no plane at any pyramid level, and it is the form
/// calls: axis 3 measured the plane faster per frame INCLUDING its
/// formation cost -- 16-18% at earlier work, 11-14% against the shipped code
/// -- against a fifth frame-sized plane at every level
/// (+25% of the derivative working set), and CLAUDE.md's tiebreak takes the
/// memory. This three-argument form stays for a caller that already has a
/// plane for other reasons and should not be made to unform it -- and when
/// it does have one, it is the faster call. See.
/// @note Reads only: `a`, `b` and `c` may be the same view or overlap arbitrarily.
/// @note Never throws, never allocates. Mismatched dimensions and a stride shorter
/// than a row are BINCV_ASSERT programming errors.
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
        const WordType* rc = c.row(y);
        const SplitCount row = impl::splitRowRegion<WordType>(
            a.row(y), b.row(y), [rc](size_t i) { return rc[i]; }, r);
        out.whenSet += row.whenSet;
        out.whenClear += row.whenClear;
    }
    return out;
}


/// @brief popcount(a & b & ~(c0 ^ c1)) and popcount(a & b & (c0 ^ c1)) over
/// `region`, in ONE pass and with NO selector plane. **API TIER 3.**
/// @param a First source view.
/// @param b Second source view; must have a's dimensions.
/// @param c0, c1 The two selector planes; the selector is their XOR, formed a
/// word at a time inside the loop. Both must have a's dimensions.
/// @param region Rectangle in pixels, half-open, intersected with the image.
/// @return `{whenClear, whenSet}` -- the pixels of `a & b` split by `c0 ^ c1`.
///
/// @note **THIS IS THE FORM THE COVARIANCE CALLS** (the design notes),
/// with `c0 = sign_x` and `c1 = sign_y`. It is identical in value to
/// `countAndSplit(a, b, xorPlane, region)` where `xorPlane` was filled by
/// bitwiseXor(c0, c1,...) -- and needs no such plane to exist.
/// @note **IT IS ALSO THE SLOWER OF THE TWO, AND THAT IS THE DECISION.** Axis 3
/// measured the precomputed plane faster per frame at 200 keypoints
/// INCLUDING the cost of forming it -- 16-18% at earlier work, and 11-14% against
/// the code that shipped (, 125.7 us against 139.1 us per frame at
/// W=31 with formation included). The four-argument form loads
/// a fourth stream and XORs it in the word loop, and that costs more than
/// one streaming pass over the frame. What it buys is the plane itself --
/// one bit per pixel of every pyramid level, 38400 B at 640x480 and ~51 kB
/// over a four-level pyramid, a FIFTH plane on top of the four the
/// covariance already reads (+25% of the derivative working set), held for
/// the frame's lifetime. Speed and footprint disagree here, and CLAUDE.md's
/// tiebreak is explicit: memory wins. So this is the default; the
/// three-argument overload stays for a caller that has a plane already.
/// @note Same single-pass, padding, aliasing and error contract as the
/// three-argument overload. `~(c0 ^ c1)` is never formed, for the same
/// reason `~c` is not.
template <typename WordType>
inline SplitCount countAndSplit(BinMatConstView<WordType> a, BinMatConstView<WordType> b,
                                BinMatConstView<WordType> c0, BinMatConstView<WordType> c1,
                                Rect region) {
    BINCV_ASSERT(a.width == b.width && a.height == b.height && a.width == c0.width &&
                     a.height == c0.height && a.width == c1.width && a.height == c1.height,
                 "reduce: masked reductions need views of the same size");
    BINCV_ASSERT(impl::strideCoversARow<WordType>(a.width, a.height, a.stride) &&
                     impl::strideCoversARow<WordType>(b.width, b.height, b.stride) &&
                     impl::strideCoversARow<WordType>(c0.width, c0.height, c0.stride) &&
                     impl::strideCoversARow<WordType>(c1.width, c1.height, c1.stride),
                 "reduce: every view's stride must cover a whole row");

    SplitCount out;
    const impl::RegionWords<WordType> r = impl::clipRegion<WordType>(a.width, a.height, region);
    if (r.isEmpty) return out;

    BINCV_ASSERT(a.ptr != nullptr && b.ptr != nullptr && c0.ptr != nullptr && c1.ptr != nullptr,
                 "reduce: a non-empty view needs a non-null pointer");

    for (size_t y = r.y0; y < r.y1; ++y) {
        const WordType* r0 = c0.row(y);
        const WordType* r1 = c1.row(y);
        const SplitCount row = impl::splitRowRegion<WordType>(
            a.row(y), b.row(y),
            [r0, r1](size_t i) { return static_cast<WordType>(r0[i] ^ r1[i]); }, r);
        out.whenSet += row.whenSet;
        out.whenClear += row.whenClear;
    }
    return out;
}

/// @brief All four numbers of the 2x2 gradient covariance over `region`, from ONE
/// traversal. **API TIER 3.**
/// @param a First source view -- `mag_x` for the LK covariance.
/// @param b Second source view -- `mag_y`; must have a's dimensions.
/// @param c Selector plane -- `sign_x ^ sign_y`; must have a's dimensions.
/// @param region Rectangle in pixels, half-open, intersected with the image.
/// @return `{xx, yy, xy}`: popcount(a), popcount(b), and the split of `a & b`.
///
/// @note **REACH FOR THIS RATHER THAN COMPOSING** when a caller wants more than
/// one of the four numbers over the same window, which the covariance
/// always does. `countNonZero(a, w) + countNonZero(b, w) +
/// countAndSplit(a, b, c, w)` returns the same four numbers and issues
/// exactly the same popcounts, but makes THREE traversals of the region and
/// 5 word loads per word index against this one's 3 (`a` and `b` are each
/// read twice, `c` once). That measured 1.27x (`uint32_t`) and 1.29x
/// (`uint64_t`) slower at 31x31 on the reference device pre-split
/// ( axis 2, reproducing that measurement’s 1.30x; ARCHITECTURE.md
///), and **1.20x / 1.27x against the code that shipped**, holding from
/// 1.20x to 1.65x across the three window sizes and two word widths
/// (the axis 2). The delta is redundant traversal and nothing else, which
/// is why it is an entry point rather than an optimization hint.
/// @note **Single pass, four popcounts per word**: `popcount(a & mask)`,
/// `popcount(b & mask)`, `popcount(a & b & mask)` and
/// `popcount(a & b & mask & c)`; `xy.whenClear` is the difference of the
/// last two, never `popcount(... & ~c)`.
/// @note Extra memory: none. The four counters are returned by value.
/// @note Same region, padding, aliasing and error contract as every other
/// reduction here. `a`, `b` and `c` may alias arbitrarily.
template <typename WordType>
inline CovarianceCount countCovariance(BinMatConstView<WordType> a, BinMatConstView<WordType> b,
                                       BinMatConstView<WordType> c, Rect region) {
    BINCV_ASSERT(a.width == b.width && a.height == b.height && a.width == c.width &&
                     a.height == c.height,
                 "reduce: masked reductions need views of the same size");
    BINCV_ASSERT(impl::strideCoversARow<WordType>(a.width, a.height, a.stride) &&
                     impl::strideCoversARow<WordType>(b.width, b.height, b.stride) &&
                     impl::strideCoversARow<WordType>(c.width, c.height, c.stride),
                 "reduce: every view's stride must cover a whole row");

    CovarianceCount out;
    const impl::RegionWords<WordType> r = impl::clipRegion<WordType>(a.width, a.height, region);
    if (r.isEmpty) return out;

    BINCV_ASSERT(a.ptr != nullptr && b.ptr != nullptr && c.ptr != nullptr,
                 "reduce: a non-empty view needs a non-null pointer");

    for (size_t y = r.y0; y < r.y1; ++y) {
        const WordType* rc = c.row(y);
        const CovarianceCount row = impl::covarianceRowRegion<WordType>(
            a.row(y), b.row(y), [rc](size_t i) { return rc[i]; }, r);
        out.xx += row.xx;
        out.yy += row.yy;
        out.xy.whenSet += row.xy.whenSet;
        out.xy.whenClear += row.xy.whenClear;
    }
    return out;
}

/// @brief All four covariance numbers over `region`, from ONE traversal and with
/// NO selector plane. **API TIER 3.**
/// @param a First source view -- `mag_x`.
/// @param b Second source view -- `mag_y`; must have a's dimensions.
/// @param c0, c1 The two sign planes; the selector is `c0 ^ c1`, formed a word at
/// a time inside the loop. Both must have a's dimensions.
/// @param region Rectangle in pixels, half-open, intersected with the image.
///
/// @note **THIS IS THE SIGNATURE CALLS.** It is the conjunction of the two
/// decisions recorded: fused rather than composed (axis 2, 1.27-1.29x
/// pre-split, 1.20-1.27x at 31x31 against the shipped code by design) and
/// four-argument rather than plane (axis 3, memory wins the tiebreak).
/// Composing those two decisions leaves exactly one entry point that is both
/// single-pass and scratch-free, and this is it -- the LK covariance over a
/// window with **0 B** of caller memory beyond the derivative planes it must
/// read anyway.
/// @note It is the SLOWER selector form, deliberately; see the four-argument
/// countAndSplit above for the 11-14% (16-18% at earlier work) and the +25% it is
/// traded against.
/// A caller that already holds a `sign_x ^ sign_y` plane should call the
/// four-argument overload of this function instead and keep the speed.
template <typename WordType>
inline CovarianceCount countCovariance(BinMatConstView<WordType> a, BinMatConstView<WordType> b,
                                       BinMatConstView<WordType> c0, BinMatConstView<WordType> c1,
                                       Rect region) {
    BINCV_ASSERT(a.width == b.width && a.height == b.height && a.width == c0.width &&
                     a.height == c0.height && a.width == c1.width && a.height == c1.height,
                 "reduce: masked reductions need views of the same size");
    BINCV_ASSERT(impl::strideCoversARow<WordType>(a.width, a.height, a.stride) &&
                     impl::strideCoversARow<WordType>(b.width, b.height, b.stride) &&
                     impl::strideCoversARow<WordType>(c0.width, c0.height, c0.stride) &&
                     impl::strideCoversARow<WordType>(c1.width, c1.height, c1.stride),
                 "reduce: every view's stride must cover a whole row");

    CovarianceCount out;
    const impl::RegionWords<WordType> r = impl::clipRegion<WordType>(a.width, a.height, region);
    if (r.isEmpty) return out;

    BINCV_ASSERT(a.ptr != nullptr && b.ptr != nullptr && c0.ptr != nullptr && c1.ptr != nullptr,
                 "reduce: a non-empty view needs a non-null pointer");

    for (size_t y = r.y0; y < r.y1; ++y) {
        const WordType* r0 = c0.row(y);
        const WordType* r1 = c1.row(y);
        const CovarianceCount row = impl::covarianceRowRegion<WordType>(
            a.row(y), b.row(y),
            [r0, r1](size_t i) { return static_cast<WordType>(r0[i] ^ r1[i]); }, r);
        out.xx += row.xx;
        out.yy += row.yy;
        out.xy.whenSet += row.xy.whenSet;
        out.xy.whenClear += row.xy.whenClear;
    }
    return out;
}

// ---------------------------------------------------------------------------
// INCREMENTAL WINDOW STATE -- THE INC-ROW FORM ( item 1, earlier work)
// ---------------------------------------------------------------------------

/// @brief A window count slid DOWNWARD one pixel row at a time: the sum gains the
/// incoming row's windowed popcount and loses the outgoing row's.
/// **API TIER 3** -- OpenCV has no equivalent, so this borrows no OpenCV
/// name.
/// @tparam WordType The view's word type.
///
/// @note **WHEN TO REACH FOR THIS, AND WHEN NOT TO.** It is worth exactly what
/// the caller's access pattern makes it worth, and all three patterns the
/// MVP contains were measured at 640x480, `uint32_t`, on the reference
/// device. The column is against the PRE-accumulator-split recompute
/// baseline; **the column is against the countNonZero this file
/// actually ships**, and is the one a caller should plan with:
///
///
/// DENSE every window position in a frame 20x 15.9x
/// (the corner response of the design notes)
/// SEARCH 200 keypoints x an 8x8 sweep 7.3x 5.96x
/// SPARSE 200 isolated keypoints 1.32x 1.10x
/// (the LK covariance of 7.5)
///
/// DENSE and SEARCH are far past the 15% line that selected this branch of
///, at either baseline. They fell because item 4 -- the per-row
/// accumulator split in impl's row bodies -- made the DENOMINATOR faster,
/// which is exactly what predicted when it required item 4 to land
/// first ("roughly 15x and 5.6x"); the measurement met the prediction.
///
/// SPARSE is the row to read carefully. **Treat it as nothing.** With one
/// window per keypoint the sliding path never executes, so this issues
/// exactly the popcounts countNonZero would, over exactly the same words --
/// and the 1.10x that remains is the call structure (this clips its column
/// band once at construction; countNonZero runs the full region clip per
/// call), not incremental state. **One window is a countNonZero, not a
/// SlidingWindowCount.**
/// @note **It keeps ONE scalar of state and takes NO caller scratch**, and that
/// is why it is the incremental form binCV exposes. The per-column box
/// accumulator is faster still on a dense sweep (36x, since it issues no
/// popcount at all) but is 12x SLOWER on isolated keypoints and needs a
/// `sweepWidth + W - 1` counter array the caller would have to own -- a
/// second shape, and a second decision explicitly declines to take.
/// @note **The column masks are built once, in the constructor.** Every window in
/// a vertical column shares one x extent, so sliding is pure row
/// arithmetic: one impl::countRowRegion out, one in.
/// @note **Clipping is exact at every edge** and is what makes this agree with
/// countNonZero window for window rather than only in the interior. The
/// columns are clipped once; the rows are clipped per position, because the
/// row that leaves or enters simply does not exist when it falls outside
/// [0, height). A window wholly above, below, left or right of the image
/// counts 0, exactly as `countNonZero(src, window)` does.
/// @note Reads only; no allocation; never throws. It holds a view, so the pixels
/// must outlive it, and it must not be slid while the underlying
/// words are being written.
template <typename WordType>
class SlidingWindowCount {
public:
    /// @brief Binds to `src` and counts `firstWindow`.
    /// @param src Source view.
    /// @param firstWindow The topmost window of the column, in pixels, half-open,
    /// intersected with the image exactly as every region here is. Its `x`
    /// and `width` are fixed for the whole column; `y` is what slideDown
    /// advances.
    SlidingWindowCount(BinMatConstView<WordType> src, Rect firstWindow)
        : src_(src),
          cols_(impl::clipColumns<WordType>(src.width, src.height, firstWindow.x,
                                            firstWindow.width)),
          top_(static_cast<long long>(firstWindow.y)),
          rows_(firstWindow.height),
          x_(firstWindow.x),
          width_(firstWindow.width),
          total_(0) {
        BINCV_ASSERT(impl::strideCoversARow<WordType>(src.width, src.height, src.stride),
                     "reduce: every view's stride must cover a whole row");
        if (!alive()) return;
        BINCV_ASSERT(src.ptr != nullptr, "reduce: a non-empty view needs a non-null pointer");

        const long long imageRows = static_cast<long long>(src_.height);
        long long first = (top_ > 0) ? top_ : 0;
        long long last = top_ + static_cast<long long>(rows_);
        if (last > imageRows) last = imageRows;
        for (long long y = first; y < last; ++y) {
            total_ += impl::countRowRegion<WordType>(src_.row(static_cast<size_t>(y)), cols_);
        }
    }

    /// @brief Set pixels inside the current window, intersected with the image.
    /// @return Exactly `countNonZero(src, window)`, for every position of the
    /// column including the clipped ones. That equality is the operation's
    /// contract and tests/test_reduce.cpp sweeps whole frames checking it
    /// position by position -- an accumulator that drifts after many
    /// slides, or only at a border, is the failure this shape can have.
    size_t count() const { return total_; }

    /// @brief Advances the window one pixel row down.
    /// @note Two row counts, whatever the window height: the row at the old top
    /// leaves and the row one past the old bottom enters. Either is skipped
    /// when it falls outside the image, which is precisely how the clipped
    /// positions stay exact.
    void slideDown() {
        if (alive()) {
            const long long imageRows = static_cast<long long>(src_.height);
            const long long leaving = top_;
            const long long entering = top_ + static_cast<long long>(rows_);
            if (leaving >= 0 && leaving < imageRows) {
                total_ -=
                    impl::countRowRegion<WordType>(src_.row(static_cast<size_t>(leaving)), cols_);
            }
            if (entering >= 0 && entering < imageRows) {
                total_ +=
                    impl::countRowRegion<WordType>(src_.row(static_cast<size_t>(entering)), cols_);
            }
        }
        ++top_;
    }

    /// @brief The window this accumulator is currently reporting, unclipped.
    /// @note Handed back so a caller can compare against countNonZero(src, rect)
    /// without re-deriving the rectangle, which is what the window-for-window
    /// test does.
    Rect window() const {
        BINCV_ASSERT(top_ >= static_cast<long long>(INT_MIN) &&
                         top_ <= static_cast<long long>(INT_MAX),
                     "reduce: the sliding window's top row has left the range of int");
        return Rect(x_, static_cast<int>(top_), width_, rows_);
    }

private:
    /// @brief False when no position of this column can ever count anything: an
    /// empty column band, or a non-positive window height.
    bool alive() const { return !cols_.isEmpty && rows_ > 0; }

    BinMatConstView<WordType> src_;
    impl::RegionWords<WordType> cols_;  ///< column masks; built once, never re-clipped
    long long top_;                     ///< current window's top row, unclipped
    int rows_;                          ///< window height in pixels
    int x_;                             ///< window left edge, unclipped
    int width_;                         ///< window width in pixels
    size_t total_;                      ///< THE ONE SCALAR OF STATE
};

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
