// Deliberately violates one debug-checked precondition, and does not survive it.
//
// THIS FILE FORCES THE CHECKED CONFIGURATION. The #undef below runs before any
// binCV header is seen, so BINCV_DEBUG_CHECKS is 1 here no matter which build
// type compiled it. Without that, this suite would be dead source in every
// configuration the project actually verifies: all three V-ALL builds are
// Release, so NDEBUG is set in all of them, and the debug half of the error
// policy -- the live BINCV_ASSERT, and the bounds checks that at() and set() are
// specified to have -- would never be compiled or run.
//
// WHAT IT REPLACES: before T1.4, out-of-range at()/set() threw std::out_of_range
// and tests/test_binMat.cpp asserted that with three BINCV_CHECK_THROWS lines
// (plus one in tests/test_opencv_interop.cpp). D-7 removed the throw, and those
// four assertions were deleted with it and not replaced -- deleting both bounds
// checks from binMat_impl.hpp left Release, Debug and -fno-exceptions all
// reporting 100% passed. These cases are that coverage, restored at the checked
// configuration's terms: the failure is now an abort, so it is observed from
// outside the process by tests/expect_fatal.cmake.
//
// Each case is registered in tests/CMakeLists.txt with the diagnostic it must
// print, so the message is covered too -- an assert that fires with the wrong
// message points at the wrong mistake.

#undef NDEBUG

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/view.hpp"
#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/bitslice.hpp"
#include "bincv-cpp/ops/logic.hpp"
#include "bincv-cpp/ops/reduce.hpp"
#include "bincv-cpp/ops/shift.hpp"
#include "bincv-cpp/quantMat.hpp"

#if !BINCV_DEBUG_CHECKS
#error "test_assert_abort must compile with BINCV_DEBUG_CHECKS == 1; the #undef NDEBUG above is what guarantees it"
#endif

namespace {

using Mat = bincv::BinMat<uint32_t>;

// 20 pixels in a 32-bit word, so the out-of-range columns below are still inside
// the row's own allocation: the case proves the CHECK fires, and does not depend
// on undefined behaviour to do it.
Mat makeMat() { return Mat(20, 4); }

uint32_t g_buffer[8] = {0};

int caseAtRow() {
    const Mat m = makeMat();
    return m.at(999, 0) ? 1 : 0;
}

int caseAtCol() {
    const Mat m = makeMat();
    return m.at(0, 999) ? 1 : 0;
}

int caseAtNegative() {
    const Mat m = makeMat();
    return m.at(-1, 0) ? 1 : 0;
}

int caseSetRow() {
    Mat m = makeMat();
    m.set(-1, 5, true);
    return m.countNonZero();
}

int caseSetCol() {
    Mat m = makeMat();
    // Column 25 is past `width` but inside the first word. In release this
    // silently sets a padding bit that countNonZero() cannot see, which is the
    // invariant break the check exists to catch (CLAUDE.md: padding bits stay
    // zero).
    m.set(0, 25, true);
    return m.countNonZero();
}

// The views are the type kernels bind to, so their preconditions are the other
// half of the hot-path policy. A missing stride is the one field whose omission
// is not self-announcing: every row silently aliases row 0.
int caseViewStride() {
    bincv::BinMatView<uint32_t> v{g_buffer, 64, 4, 0};
    return static_cast<int>(*v.row(2));
}

int caseConstViewStride() {
    bincv::BinMatConstView<uint32_t> v{g_buffer, 64, 4, 0};
    return static_cast<int>(*v.row(2));
}

// The N-plane container's own per-pixel preconditions (T1.5/T1.6). Same policy as
// at() and set() above: debug-checked, gone in release, so the only place they can
// be observed is a process that does not survive them.
//
// NOT here, deliberately: the plane-index checks. plane(), constPlane() and
// magnitude() are view factories rather than per-pixel access, and are validated
// through BINCV_THROW in every build -- so their cases live in
// tests/test_error_abort.cpp with the other validation sites.
using Quant = bincv::QuantMat<3, uint32_t>;
using Ternary = bincv::SignedQuantMat<2, uint32_t>;

Quant makeQuant() { return Quant(20, 4); }

int caseQuantAtRow() {
    const Quant m = makeQuant();
    return static_cast<int>(m.at(999, 0));
}

int caseQuantAtCol() {
    const Quant m = makeQuant();
    return static_cast<int>(m.at(0, 999));
}

int caseQuantSetRow() {
    Quant m = makeQuant();
    m.set(-1, 0, 1u);
    return static_cast<int>(m.at(0, 0));
}

// A value with bits above N would be written to planes this container does not
// have; the low N bits alone would land, silently storing something else.
int caseQuantSetValue() {
    Quant m = makeQuant();
    m.set(0, 0, 8u);                            // 3 planes hold 0..7
    return static_cast<int>(m.at(0, 0));
}

// The same value-range precondition at N == 1, which is where it used to go
// missing: BinMat::set takes bool, so an out-of-range value from code written
// generically over QuantMat<N> narrowed to 1 instead of being reported. The
// integral overload added for that is what fires here.
int caseBinMatSetValue() {
    Mat m = makeMat();
    m.set(0, 0, 3u);                            // one plane holds 0..1
    return m.countNonZero();
}

int caseSignedSetValue() {
    Ternary m(20, 4);
    m.set(0, 0, -4);                            // 2 magnitude planes hold -3..3
    return m.at(0, 0);
}

// The extreme of the same range check. Worth its own case because the value that
// gets past it is the one that used to be undefined behaviour rather than merely
// wrong: the magnitude was computed as `-value`, and negating INT_MIN is signed
// overflow. impl::signedMagnitude now does that in unsigned; this case is the
// other half, proving the guard that keeps INT_MIN out is still here.
int caseSignedSetIntMin() {
    Ternary m(20, 4);
    volatile int mostNegative = INT_MIN;        // volatile: not folded at compile time
    m.set(0, 0, mostNegative);
    return m.at(0, 0);
}

// The T2.2 kernel preconditions. Kernels never throw (ARCHITECTURE 5.3), so
// mismatched dimensions and an unsafe destination alias are BINCV_ASSERT sites --
// which means they are observable only from outside the process, exactly like the
// per-pixel checks above.
//
// The alias case is the one worth having. A destination that overlaps a source at
// a DIFFERENT offset reads words the row loop has already overwritten: every
// address involved is valid memory, so no sanitizer sees anything, and the result
// is simply wrong for part of the image. Exact aliasing (dst IS a, same stride) is
// supported and is covered in-process by tests/test_logic.cpp.
uint32_t g_logicA[8] = {0};
uint32_t g_logicB[8] = {0};
uint32_t g_logicDst[8] = {0};

int caseLogicDims() {
    const bincv::BinMatConstView<uint32_t> a{g_logicA, 64, 3, 2};
    const bincv::BinMatConstView<uint32_t> b{g_logicB, 32, 3, 1};    // narrower
    const bincv::BinMatView<uint32_t> dst{g_logicDst, 64, 3, 2};
    bincv::bitwiseAnd(a, b, dst);
    return static_cast<int>(g_logicDst[0]);
}

int caseLogicDimsNot() {
    const bincv::BinMatConstView<uint32_t> src{g_logicA, 64, 3, 2};
    const bincv::BinMatView<uint32_t> dst{g_logicDst, 32, 3, 1};     // narrower
    bincv::bitwiseNot(src, dst);
    return static_cast<int>(g_logicDst[0]);
}

int caseLogicAlias() {
    // Same shape, same buffer, one word apart: word i of dst is word i+1 of src.
    //
    // Half a row apart, deliberately. The predicate accepts two views over one
    // buffer whose ROWS never meet -- interleaved row bands and column tiles are
    // legal (D-5) and are covered in-process by Logic.AliasAccepts_* -- so a case
    // that merely shared a buffer would no longer prove anything. This one shares
    // words: dst row 0 is src words 1..2, and word 1 is read after it was written.
    const bincv::BinMatConstView<uint32_t> src{g_logicA, 64, 3, 2};
    const bincv::BinMatView<uint32_t> dst{g_logicA + 1, 64, 3, 2};
    bincv::bitwiseNot(src, dst);
    return static_cast<int>(g_logicA[0]);
}

// A stride shorter than the row it has to hold: consecutive rows overlap, so the
// kernel reads each row through the previous one's tail. BinMat's wrap constructor
// rejects the same numbers by name; a hand-built view is what a kernel takes
// (D-5), so the kernel carries the check. Measured before it existed: this call
// produced three rows of ffff0000 from a source of 0000ffff, in every build, with
// nothing reported.
int caseLogicShortStride() {
    const bincv::BinMatConstView<uint32_t> src{g_logicA, 64, 3, 1};   // needs 2 words
    const bincv::BinMatView<uint32_t> dst{g_logicDst, 64, 3, 1};
    bincv::bitwiseNot(src, dst);
    return static_cast<int>(g_logicDst[0]);
}

// The T2.3/T2.4 shift preconditions. Two of them say something the logic cases
// above do not.
//
// `shift-in-place` is the important one. ops/logic.hpp ACCEPTS a destination that
// is exactly a source -- D-11's in-place idiom -- because those kernels are
// pointwise in the word index. A shift is not: word i of the destination is built
// from words i +/- wordShift of the source, so the same call that is legal for
// bitwiseAnd is undefined here. That difference is invisible in a release build
// (the answer is merely wrong for part of the image, with every address valid and
// no sanitizer able to see it), which is exactly what makes it worth a case.
//
// `shift-border-type` covers the argument no other kernel has: BorderType is an
// enum, so an int can arrive as one, and an unchecked unknown value would silently
// behave as BORDER_CONSTANT -- a plausible answer to a question nobody asked.
uint32_t g_shiftSrc[8] = {0};
uint32_t g_shiftDst[8] = {0};

int caseShiftDims() {
    const bincv::BinMatConstView<uint32_t> src{g_shiftSrc, 64, 3, 2};
    const bincv::BinMatView<uint32_t> dst{g_shiftDst, 32, 3, 1};      // narrower
    bincv::shiftLeft(src, dst, 1);
    return static_cast<int>(g_shiftDst[0]);
}

int caseShiftInPlace() {
    const bincv::BinMatConstView<uint32_t> src{g_shiftSrc, 64, 3, 2};
    const bincv::BinMatView<uint32_t> dst{g_shiftSrc, 64, 3, 2};      // exactly src
    bincv::shiftLeft(src, dst, 1);
    return static_cast<int>(g_shiftSrc[0]);
}

int caseShiftOverlap() {
    // Same shape, same buffer, one word apart: word i of dst is word i+1 of src.
    // Half a row apart, deliberately -- the predicate accepts two views over one
    // buffer whose ROWS never meet (Shift.DisjointViews_* covers those), so a case
    // that merely shared a buffer would no longer prove anything.
    const bincv::BinMatConstView<uint32_t> src{g_shiftSrc, 64, 3, 2};
    const bincv::BinMatView<uint32_t> dst{g_shiftSrc + 1, 64, 3, 2};
    bincv::shiftUp(src, dst, 1);
    return static_cast<int>(g_shiftSrc[0]);
}

int caseShiftShortStride() {
    const bincv::BinMatConstView<uint32_t> src{g_shiftSrc, 64, 3, 1};   // needs 2 words
    const bincv::BinMatView<uint32_t> dst{g_shiftDst, 64, 3, 1};
    bincv::shiftRight(src, dst, 1);
    return static_cast<int>(g_shiftDst[0]);
}

int caseShiftBorderType() {
    const bincv::BinMatConstView<uint32_t> src{g_shiftSrc, 64, 3, 2};
    const bincv::BinMatView<uint32_t> dst{g_shiftDst, 64, 3, 2};
    // 7, not 99. BorderType's enumerators span 0..4, so the smallest bit-field
    // that holds them all is 3 bits and the type's value range is 0..7: casting
    // 99 into it is UNSPECIFIED, and -Wconversion says so ("the result of the
    // conversion is unspecified because '99' is outside the range of type"). 7 is
    // a representable value that is not one of the five, which is exactly the
    // input isKnownBorderType() exists to reject.
    bincv::shift(src, dst, 1, 1, static_cast<bincv::BorderType>(7), false);
    return static_cast<int>(g_shiftDst[0]);
}

// The T2.5/T2.6 reduction preconditions. A reduction READS ONLY, so it has no
// aliasing case at all -- D-11 binds kernels that write, and countAndSplit(m, m,
// m, r) is a supported call that an alias predicate copied in from ops/logic.hpp
// would abort on. What is left is the two checks a reduction can still fail:
// dimensions that disagree between the masked arguments, and a stride too short
// to hold a row.
//
// The dimension check gets a case per entry point because each carries its own:
// countAnd compares two views and countAndSplit three, and a third view that
// nobody compared would be read at the first two's geometry -- inside the buffer,
// wrong answer, nothing to see.
uint32_t g_reduceA[8] = {0};
uint32_t g_reduceB[8] = {0};

int caseReduceDims() {
    const bincv::BinMatConstView<uint32_t> a{g_reduceA, 64, 3, 2};
    const bincv::BinMatConstView<uint32_t> b{g_reduceB, 32, 3, 1};  // narrower
    return static_cast<int>(bincv::countAnd(a, b, bincv::Rect(0, 0, 64, 3)));
}

int caseReduceSplitDims() {
    const bincv::BinMatConstView<uint32_t> a{g_reduceA, 64, 3, 2};
    const bincv::BinMatConstView<uint32_t> c{g_reduceB, 64, 2, 2};  // one row short
    return static_cast<int>(bincv::countAndSplit(a, a, c, bincv::Rect(0, 0, 64, 3)).whenSet);
}

// T2.11 added a FOURTH view argument and two more entry points, each with its own
// dimension assert, and each therefore its own way to be deleted unnoticed. That
// is not hypothetical: deleting `&& a.width == c1.width && a.height == c1.height`
// from both four-argument overloads and the stride assert from
// SlidingWindowCount's constructor left all four gate configurations green with
// byte-identical CHECKS counts (Debug test_reduce: 594184/594184 either way). An
// assert has no in-process test, so these three cases are the only place those
// checks can be observed firing.
int caseReduceXorSplitDims() {
    const bincv::BinMatConstView<uint32_t> a{g_reduceA, 64, 3, 2};
    const bincv::BinMatConstView<uint32_t> c1{g_reduceB, 64, 2, 2};  // one row short
    // c1 is the argument nobody compared before this case existed: it would be
    // read at a's geometry -- row 2 of a two-row view -- which is off the end of
    // its buffer, in Release, returning a plausible-looking count.
    return static_cast<int>(
        bincv::countAndSplit(a, a, a, c1, bincv::Rect(0, 0, 64, 3)).whenSet);
}

int caseReduceCovarianceDims() {
    const bincv::BinMatConstView<uint32_t> a{g_reduceA, 64, 3, 2};
    const bincv::BinMatConstView<uint32_t> c{g_reduceB, 64, 2, 2};  // one row short
    return static_cast<int>(bincv::countCovariance(a, a, c, bincv::Rect(0, 0, 64, 3)).xx);
}

// The four-argument countCovariance carries its OWN copy of the same clause --
// four entry points, four copies, four separate deletions. Covered separately for
// exactly the reason the three-argument cases are.
int caseReduceCovarianceXorDims() {
    const bincv::BinMatConstView<uint32_t> a{g_reduceA, 64, 3, 2};
    const bincv::BinMatConstView<uint32_t> c1{g_reduceB, 64, 2, 2};  // one row short
    return static_cast<int>(bincv::countCovariance(a, a, a, c1, bincv::Rect(0, 0, 64, 3)).xx);
}

// SlidingWindowCount is the one entry point that is a CLASS, so its precondition
// is checked in a constructor rather than at a call. Same short stride as
// reduce-short-stride, different site.
int caseReduceSlidingShortStride() {
    const bincv::BinMatConstView<uint32_t> a{g_reduceA, 64, 3, 1};  // needs 2 words
    bincv::SlidingWindowCount<uint32_t> w(a, bincv::Rect(0, 0, 31, 3));
    return static_cast<int>(w.count());
}

int caseReduceShortStride() {
    const bincv::BinMatConstView<uint32_t> a{g_reduceA, 64, 3, 1};  // needs 2 words
    return static_cast<int>(bincv::countNonZero(a));
}

// impl::regionFromExtent's non-empty precondition, called DIRECTLY rather than
// through an entry point -- because no entry point can reach it. Both callers in
// ops/reduce.hpp guard the empty extent themselves (wholeViewWords returns early
// on width == 0, clipRegion on x0 >= x1), so this is a trap laid for the next
// caller, not a live bug: x1 == 0 underflows `x1 - 1` and yields lastWord =
// SIZE_MAX / WordBits with isEmpty == false, which visitRowWords would walk
// straight off the end of the buffer. The assertion is what stops that being
// discovered by a segfault in Phase 3; this case is what stops the assertion from
// being deleted as unreachable.
int caseReduceEmptyExtent() {
    const bincv::impl::RegionWords<uint32_t> r = bincv::impl::regionFromExtent<uint32_t>(0, 0, 0, 1);
    return static_cast<int>(r.lastWord);
}

struct Case {
    const char* name;
    int (*run)();
};

// The T2.7 bit-sliced preconditions. Two of them say something no case above does.
//
// `bitslice-dims` is the three-source dimension check. ops/logic.hpp's kernels
// compare two views, majority3 compares three, and a check that forgot the third
// would read `c` at the other two's geometry -- inside its own buffer, with only
// the answer wrong. Measured by deleting `c.width == dst.width && c.height ==
// dst.height` from the assert: every one of tests/test_bitslice.cpp's 348061
// checks still passed, in all four configurations. That is the whole reason this
// case exists rather than being assumed covered by logic-dims.
//
// `bitslice-sum-overlap` is a precondition on RAW POINTERS rather than on views,
// which nothing else in this file has. bitSlicedSum accumulates in its output
// planes, so an outPlanes array overlapping inputs rewrites inputs the ripple has
// not read yet -- again valid memory, again only the answer wrong.
uint32_t g_majA[8] = {0};
uint32_t g_majB[8] = {0};
uint32_t g_majC[8] = {0};
uint32_t g_majDst[8] = {0};

int caseBitSliceDims() {
    const bincv::BinMatConstView<uint32_t> a{g_majA, 64, 3, 2};
    const bincv::BinMatConstView<uint32_t> b{g_majB, 64, 3, 2};
    const bincv::BinMatConstView<uint32_t> c{g_majC, 32, 3, 1};       // narrower
    const bincv::BinMatView<uint32_t> dst{g_majDst, 64, 3, 2};
    bincv::majority3(a, b, c, dst);
    return static_cast<int>(g_majDst[0]);
}

int caseBitSliceAlias() {
    // Same shape, same buffer, one word apart -- half a row, so the two views
    // really do share words rather than merely sharing a buffer (the disjoint
    // cases are legal under D-5 and are covered in-process).
    const bincv::BinMatConstView<uint32_t> a{g_majA, 64, 3, 2};
    const bincv::BinMatConstView<uint32_t> b{g_majB, 64, 3, 2};
    const bincv::BinMatConstView<uint32_t> c{g_majC, 64, 3, 2};
    const bincv::BinMatView<uint32_t> dst{g_majA + 1, 64, 3, 2};
    bincv::majority3(a, b, c, dst);
    return static_cast<int>(g_majA[0]);
}

int caseBitSliceShortStride() {
    const bincv::BinMatConstView<uint32_t> a{g_majA, 64, 3, 1};       // needs 2 words
    const bincv::BinMatConstView<uint32_t> b{g_majB, 64, 3, 1};
    const bincv::BinMatConstView<uint32_t> c{g_majC, 64, 3, 1};
    const bincv::BinMatView<uint32_t> dst{g_majDst, 64, 3, 1};
    bincv::majority3(a, b, c, dst);
    return static_cast<int>(g_majDst[0]);
}

uint32_t g_sumBuffer[8] = {0};

int caseBitSliceSumOverlap() {
    // Four inputs at [0, 4), three output planes at [2, 5) -- the accumulator
    // would overwrite inputs 2 and 3 before it read them.
    bincv::bitSlicedSum<uint32_t>(g_sumBuffer, 4, g_sumBuffer + 2);
    return static_cast<int>(g_sumBuffer[0]);
}

const Case kCases[] = {
    {"at-row", caseAtRow},
    {"at-col", caseAtCol},
    {"at-negative", caseAtNegative},
    {"set-row", caseSetRow},
    {"set-col", caseSetCol},
    {"quant-at-row", caseQuantAtRow},
    {"quant-at-col", caseQuantAtCol},
    {"quant-set-row", caseQuantSetRow},
    {"quant-set-value", caseQuantSetValue},
    {"binmat-set-value", caseBinMatSetValue},
    {"signed-set-value", caseSignedSetValue},
    {"signed-set-int-min", caseSignedSetIntMin},
    {"view-stride", caseViewStride},
    {"const-view-stride", caseConstViewStride},
    {"logic-dims", caseLogicDims},
    {"logic-dims-not", caseLogicDimsNot},
    {"logic-alias", caseLogicAlias},
    {"logic-short-stride", caseLogicShortStride},
    {"shift-dims", caseShiftDims},
    {"shift-in-place", caseShiftInPlace},
    {"shift-overlap", caseShiftOverlap},
    {"shift-short-stride", caseShiftShortStride},
    {"shift-border-type", caseShiftBorderType},
    {"reduce-dims", caseReduceDims},
    {"reduce-split-dims", caseReduceSplitDims},
    {"reduce-xor-split-dims", caseReduceXorSplitDims},
    {"reduce-covariance-dims", caseReduceCovarianceDims},
    {"reduce-covariance-xor-dims", caseReduceCovarianceXorDims},
    {"reduce-sliding-short-stride", caseReduceSlidingShortStride},
    {"reduce-short-stride", caseReduceShortStride},
    {"reduce-empty-extent", caseReduceEmptyExtent},
    {"bitslice-dims", caseBitSliceDims},
    {"bitslice-alias", caseBitSliceAlias},
    {"bitslice-short-stride", caseBitSliceShortStride},
    {"bitslice-sum-overlap", caseBitSliceSumOverlap},
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <case>\ncases:\n", argv[0]);
        for (const Case& c : kCases) std::fprintf(stderr, "  %s\n", c.name);
        return 2;
    }

    for (const Case& c : kCases) {
        if (std::strcmp(argv[1], c.name) == 0) {
            const int evidence = c.run();
            std::fprintf(stderr, "REACHED: the check did not fire for case '%s' (evidence %d)\n",
                         c.name, evidence);
            std::fflush(stderr);
            return 0;
        }
    }

    std::fprintf(stderr, "unknown case '%s'\n", argv[1]);
    return 2;
}
