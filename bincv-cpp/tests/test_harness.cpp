// Regression-check of the test harness itself (T1.7 "Done when": a deliberately
// failing assertion still produces a non-zero exit code).
//
// The hazard this closes is not hypothetical. The harness this file checks
// replaced hand-rolled tests that printed failures to stderr and returned 0
// anyway -- so every regression they found was invisible to the build. A test
// framework that cannot fail is worth less than no framework, because it also
// consumes the attention that would otherwise notice.
//
// Built TWICE from this one source:
//   test_harness           -- ordinary suite, must pass
//   test_harness_failing   -- same file with BINCV_HARNESS_EXPECT_FAILURE, so one
//                             case fails on purpose; ctest runs it with
//                             WILL_FAIL TRUE, which passes only if it exits
//                             non-zero.
// Both are built in EVERY configuration, so the claim is checked against both
// backends -- Google Test where it is used, the built-in harness in the
// dependency-free build.

#include <cstdint>

#include "bincv-cpp/binMat.hpp"
#include "test_util.hpp"

namespace {

int sideEffects = 0;

bool countingTrue() {
    ++sideEffects;
    return true;
}

} // namespace

// A check that holds must pass, and must be counted.
BINCV_TEST(Harness, PassingChecksAreCounted) {
    const int before = ::bincv::test::checkCount();
    BINCV_CHECK(true);
    BINCV_CHECK_EQ(1 + 1, 2);
    const int after = ::bincv::test::checkCount();
    // Two checks recorded above; this one is the third and is not in the window.
    BINCV_CHECK_EQ(after - before, 2);
}

// The check macros evaluate their argument exactly once. A macro that evaluated
// it twice would make every check with a side effect quietly wrong.
BINCV_TEST(Harness, ChecksEvaluateArgumentOnce) {
    sideEffects = 0;
    BINCV_CHECK(countingTrue());
    BINCV_CHECK_EQ(sideEffects, 1);
}

// The failure count starts at zero and stays there while everything passes --
// which is what makes the non-zero exit code in the failing twin meaningful.
BINCV_TEST(Harness, NoFailuresWhileEverythingPasses) {
    BINCV_CHECK_EQ(::bincv::test::failureCount(), 0);
}

// The harness reaches the library it is meant to be testing.
BINCV_TEST(Harness, ExercisesTheLibrary) {
    bincv::BinMat<uint32_t> m(8, 2);
    m.set(1, 7, true);
    BINCV_CHECK(m.at(1, 7));
    BINCV_CHECK_EQ(m.countNonZero(), 1);
}

#ifdef BINCV_HARNESS_EXPECT_FAILURE
// The whole point of the second binary. Every check macro is exercised failing,
// so a backend that dropped any one of them on the floor is caught rather than
// only the first.
BINCV_TEST(Harness, DeliberateFailure) {
    BINCV_CHECK(1 == 2);
    BINCV_CHECK_EQ(6 * 7, 43);
    BINCV_CHECK(::bincv::test::failureCount() > 0);   // the failures were recorded
}
#endif

BINCV_TEST_MAIN("Harness regression tests")
