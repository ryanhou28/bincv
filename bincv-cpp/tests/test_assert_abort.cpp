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
#include "bincv-cpp/ops/logic.hpp"
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

struct Case {
    const char* name;
    int (*run)();
};

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
