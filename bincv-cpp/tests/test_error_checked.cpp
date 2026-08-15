// The debug half of the error policy (T1.4, ARCHITECTURE 5.3), compiled in every
// configuration.
//
// THIS FILE FORCES THE CHECKED CONFIGURATION, exactly as tests/test_assert_abort.cpp
// does and for the same reason: all three V-ALL builds are Release, so without
// the #undef below BINCV_DEBUG_CHECKS is 0 everywhere and every claim about the
// live BINCV_ASSERT is dead source. tests/test_error.cpp covers the other half --
// what the macro does when it is compiled away -- and can only ever cover one of
// the two, because a translation unit is in one configuration or the other.
//
// This is the half that does NOT die: the checks that hold while assertions are
// live. The ones that fire are separate processes (test_assert_abort), because a
// firing assert aborts.

#undef NDEBUG

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/error.hpp"
#include "bincv-cpp/core/view.hpp"
#include "test_util.hpp"

#if !BINCV_DEBUG_CHECKS
#error "test_error_checked must compile with BINCV_DEBUG_CHECKS == 1; the #undef NDEBUG above is what guarantees it"
#endif

namespace {

int g_conditionEvaluations = 0;

bool countedCondition() {
    ++g_conditionEvaluations;
    return true;
}

// 1. A live BINCV_ASSERT evaluates its condition, exactly once.
//
// The counterpart of tests/test_error.cpp's release-build claim that the
// condition is not evaluated at all. Together they pin both ends: the macro is
// not a no-op in debug, and not a branch in release.
void testAssertIsLiveInDebug() {
    std::cout << "\n--- BINCV_ASSERT is live when checks are on ---\n";

    BINCV_CHECK_EQ(BINCV_DEBUG_CHECKS, 1);

    g_conditionEvaluations = 0;
    BINCV_ASSERT(countedCondition(), "holds, and was actually checked");
    BINCV_CHECK_EQ(g_conditionEvaluations, 1);

    // Twice, to show the count tracks the number of asserts rather than some
    // one-time initialization.
    BINCV_ASSERT(countedCondition(), "checked again");
    BINCV_CHECK_EQ(g_conditionEvaluations, 2);

    // A passing assert is transparent: it yields void and changes nothing.
    const int value = 7;
    BINCV_ASSERT(value == 7, "sanity");
    BINCV_CHECK_EQ(value, 7);

    // The documented workaround for a condition containing a top-level comma
    // (error.hpp). Without the extra parentheses this line does not preprocess,
    // so it is a compile-time regression test for the note as much as a runtime
    // one -- and kernel preconditions over template parameters are the shape
    // that needs it.
    BINCV_ASSERT((std::is_same<uint32_t, uint32_t>::value), "matching word types");
    BINCV_CHECK(true);
}

// 2. In-range access still works with the checks live.
//
// The bounds check must reject what is outside [0, height) x [0, width) and pass
// everything inside it, including the last valid index in each direction -- an
// off-by-one in the check itself would be caught here rather than by a death
// test that only proves something aborted.
void testInRangeAccessIsUnaffected() {
    std::cout << "\n--- checked at()/set() accept every in-range index ---\n";

    // 20 pixels in a 32-bit word: one word per row, 12 padding bits.
    bincv::BinMat<uint32_t> m(20, 4);
    BINCV_CHECK_EQ(m.getAlignedWidth(), size_t(1));

    m.set(0, 0, true);              // first pixel
    m.set(3, 19, true);             // last pixel: last row, last column
    m.set(2, 10, true);
    BINCV_CHECK(m.at(0, 0));
    BINCV_CHECK(m.at(3, 19));
    BINCV_CHECK(m.at(2, 10));
    BINCV_CHECK(!m.at(1, 0));
    BINCV_CHECK_EQ(m.countNonZero(), 3);

    // Clearing goes through the same check.
    m.set(2, 10, false);
    BINCV_CHECK(!m.at(2, 10));
    BINCV_CHECK_EQ(m.countNonZero(), 2);

    // Nothing leaked past `width`: the padding bits of row 3 are still zero, so
    // a word-wise reduction would agree with countNonZero().
    BINCV_CHECK(m.data()[3] == (uint32_t(1) << 19));
}

// 3. The view precondition passes on the shapes that are legal.
//
// BinMatView::row asserts that a multi-row view carries a stride. Both legal
// shapes must survive it: a real multi-row view, and the degenerate single-row
// view where a zero stride is meaningless rather than wrong.
void testViewRowPreconditionAcceptsLegalShapes() {
    std::cout << "\n--- view row() precondition accepts legal views ---\n";

    uint32_t buffer[8] = {0};
    buffer[0] = 0x1;
    buffer[2] = 0x2;

    bincv::BinMatView<uint32_t> multi{buffer, 64, 4, 2};
    BINCV_CHECK(multi.row(0) == buffer);
    BINCV_CHECK(multi.row(1) == buffer + 2);
    BINCV_CHECK_EQ(static_cast<int>(*multi.row(1)), 2);

    // height <= 1: a zero stride is not a mistake, there is no second row.
    bincv::BinMatView<uint32_t> single{buffer, 64, 1, 0};
    BINCV_CHECK(single.row(0) == buffer);

    bincv::BinMatConstView<uint32_t> constMulti = multi;
    BINCV_CHECK(constMulti.row(1) == buffer + 2);

    // A view onto a real matrix carries that matrix's stride, so the container's
    // own views are always legal ones.
    bincv::BinMat<uint32_t> m(20, 4);
    m.set(2, 3, true);
    bincv::BinMatView<uint32_t> owned = m.view();
    BINCV_CHECK_EQ(owned.stride, m.getAlignedWidth());
    BINCV_CHECK_EQ(static_cast<int>(*owned.row(2)), 8);
}

} // namespace

int main() {
    std::cout << "=== Error policy tests, checked configuration (T1.4) ===\n";
    std::cout << "  BINCV_DEBUG_CHECKS       = " << BINCV_DEBUG_CHECKS << "\n";
    std::cout << "  BINCV_EXCEPTIONS_ENABLED = " << BINCV_EXCEPTIONS_ENABLED << "\n";

    testAssertIsLiveInDebug();
    testInRangeAccessIsUnaffected();
    testViewRowPreconditionAcceptsLegalShapes();

    return bincv::test::summarize("Error policy tests (checked)");
}
