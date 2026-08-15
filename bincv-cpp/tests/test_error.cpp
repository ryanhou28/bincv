// Error-policy tests (T1.4, ARCHITECTURE 5.3). Core-only on purpose: the whole
// point of the policy is the configuration that has no OpenCV and no exceptions,
// so this suite has to build and run there.
//
// Covers the four claims the policy makes:
//   1. BINCV_THROW throws the named type, with the message, where exceptions exist
//   2. BINCV_ASSERT compiles away ENTIRELY under NDEBUG
//   3. Validation still rejects bad arguments
//   4. at() / set() are unchecked in a release build
//
// This suite covers the configuration it is COMPILED in, which is never the
// checked one in practice: all three verified builds are Release. Two companion
// suites cover what this one structurally cannot --
//   tests/test_error_checked.cpp  the live BINCV_ASSERT and the checked
//                                 accessors, forced on with #undef NDEBUG so
//                                 they run in every configuration
//   tests/test_error_abort.cpp    every BINCV_THROW site, and
//   tests/test_assert_abort.cpp   every BINCV_ASSERT site, as death tests: a
//                                 failed check terminates the process, so it
//                                 cannot be observed from inside one.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/error.hpp"
#include "test_util.hpp"

#if BINCV_EXCEPTIONS_ENABLED
#include <stdexcept>
#include <string>
#endif

namespace {

// ---------------------------------------------------------------------------
// 1. BINCV_THROW
// ---------------------------------------------------------------------------

#if BINCV_EXCEPTIONS_ENABLED

// Not routed through BINCV_CHECK_THROWS: that macro only reports *whether*
// something of the right type was thrown, and the message is half of what
// BINCV_THROW is responsible for carrying into the no-exceptions build.
void testThrowCarriesTypeAndMessage() {
    std::cout << "\n--- BINCV_THROW throws, with its message ---\n";

    bool caught = false;
    std::string what;
    try {
        BINCV_THROW(std::invalid_argument, "a deliberate message");
    } catch (const std::invalid_argument& e) {
        caught = true;
        what = e.what();
    } catch (...) {
    }
    BINCV_CHECK(caught);
    BINCV_CHECK(what == "a deliberate message");

    // The exception type is a macro argument, not a fixed choice: a caller can
    // report through whichever type fits, and the library uses two.
    bool caughtRuntime = false;
    try {
        BINCV_THROW(std::runtime_error, "a different type");
    } catch (const std::runtime_error& e) {
        caughtRuntime = (std::string(e.what()) == "a different type");
    } catch (...) {
    }
    BINCV_CHECK(caughtRuntime);
}

#endif // BINCV_EXCEPTIONS_ENABLED

// ---------------------------------------------------------------------------
// 2. BINCV_ASSERT compiles away under NDEBUG
// ---------------------------------------------------------------------------

int g_conditionEvaluations = 0;

// Deliberately has a side effect, so that "was the condition evaluated?" is a
// question the running test can answer rather than one only a disassembler can.
bool countedCondition() {
    ++g_conditionEvaluations;
    return true;
}

#if !BINCV_DEBUG_CHECKS
// Compile-time half of the same claim, and the stronger half: this type has no
// conversion to bool, so `if (!(cond))` would not compile. The release build
// compiles only because the NDEBUG expansion mentions neither argument -- the
// condition is not evaluated, not branched on, and not even parsed. A "compiles
// away" macro that still parsed its argument would fail here.
struct NotConvertibleToBool {};

void assertDiscardsItsConditionEntirely() {
    BINCV_ASSERT(NotConvertibleToBool{}, "never evaluated, never parsed");
}
#endif

void testAssertCompilesAwayInRelease() {
    std::cout << "\n--- BINCV_ASSERT is debug-only ---\n";

    g_conditionEvaluations = 0;
    BINCV_ASSERT(countedCondition(), "holds either way");

#if BINCV_DEBUG_CHECKS
    // Debug build: the check is live, so the condition ran exactly once. This
    // branch WAS unverified when it was written -- all three configurations were
    // Release, so nothing compiled it. T1.8 added the Debug configuration to
    // scripts/verify.sh, and this is now one of the lines that configuration
    // exists to run. The same claim is also made unconditionally in
    // tests/test_error_checked.cpp, which forces the checks on regardless.
    BINCV_CHECK_EQ(g_conditionEvaluations, 1);
#else
    // Release build: nothing ran. This is the claim that lets at() reduce to a
    // shift and a mask.
    BINCV_CHECK_EQ(g_conditionEvaluations, 0);
    assertDiscardsItsConditionEntirely();

    // Called once here, after the check above, to show the condition really is a
    // live function with an observable side effect -- so the zero above means the
    // assert dropped it, not that the counter never moves. It also keeps
    // -Wunused-function quiet in the build where nothing else calls it.
    BINCV_CHECK(countedCondition());
    BINCV_CHECK_EQ(g_conditionEvaluations, 1);
#endif

    // A passing assert is transparent in both configurations.
    const int value = 7;
    BINCV_ASSERT(value == 7, "sanity");
    BINCV_CHECK_EQ(value, 7);
}

// ---------------------------------------------------------------------------
// 3. Validation still rejects bad arguments
// ---------------------------------------------------------------------------
//
// Routing the throws through a macro is only worth anything if the checks
// themselves survived it. Guarded as a whole rather than per line: without
// exceptions BINCV_CHECK_THROWS does not evaluate its expression, so the locals
// below would be unused and every one of these calls would abort the suite.
//
// This is therefore NOT the coverage that makes the no-exceptions build able to
// fail -- it is compiled out there. tests/test_error_abort.cpp drives the same
// sites as death tests, one ctest test each, in every configuration.

#if BINCV_EXCEPTIONS_ENABLED
void testValidationStillRejects() {
    std::cout << "\n--- Validation rejects bad arguments (BINCV_THROW path) ---\n";

    using Mat = bincv::BinMat<uint32_t>;

    // Constructors
    BINCV_CHECK_THROWS(Mat(-1, 4), std::invalid_argument);
    BINCV_CHECK_THROWS(Mat(4, -1), std::invalid_argument);
    BINCV_CHECK_THROWS(Mat(4, 4, 0), std::invalid_argument);   // alignment not a power of two
    BINCV_CHECK_THROWS(Mat(4, 4, 6), std::invalid_argument);   // ditto
    BINCV_CHECK_THROWS(Mat(4, 4, 2), std::invalid_argument);   // not a multiple of the word size

    // Wrapping constructor
    uint32_t buffer[8] = {0};
    BINCV_CHECK_THROWS(Mat(buffer, 64, 2, 1), std::invalid_argument);   // stride too small
    BINCV_CHECK_THROWS(Mat(buffer, -1, 2, 2), std::invalid_argument);
    BINCV_CHECK_THROWS(Mat(nullptr, 8, 2, 1), std::invalid_argument);

    // Reallocating operations
    Mat m(8, 4);
    BINCV_CHECK_THROWS(m.resize(-1, 4), std::invalid_argument);
    BINCV_CHECK_THROWS(m.pad(0, 0, -1, 0), std::invalid_argument);

    // Operations with no answer for an empty matrix
    Mat empty;
    BINCV_CHECK_THROWS(empty.sparsity(), std::runtime_error);
    BINCV_CHECK_THROWS(empty.forEachNonZero([](int, int) {}), std::runtime_error);

    // ...and the valid forms of the same calls still work, so the checks are
    // rejecting the argument rather than the call.
    m.resize(16, 2);
    BINCV_CHECK_EQ(m.getWidth(), size_t(16));
    m.pad(1, 1, 1, 1);
    BINCV_CHECK_EQ(m.getWidth(), size_t(18));
}
#endif // BINCV_EXCEPTIONS_ENABLED

// ---------------------------------------------------------------------------
// 4. at() and set() are unchecked in release
// ---------------------------------------------------------------------------

#if !BINCV_DEBUG_CHECKS
// The behaviour change D-7 sanctions, pinned down so it cannot drift back.
//
// Every index used here is outside [0, width) but inside the row's own word, so
// the reads and writes stay within the allocation and the test is well-defined.
// That is the only kind of out-of-range index a portable test can use: a truly
// out-of-bounds one is undefined behaviour now, which is the point.
void testAtIsUncheckedInRelease() {
    std::cout << "\n--- at() / set() are unchecked in release ---\n";

    // 20 pixels in a 32-bit word: columns 20..31 are padding the row physically
    // has and the container does not admit to.
    bincv::BinMat<uint32_t> m(20, 2);
    BINCV_CHECK_EQ(m.getAlignedWidth(), size_t(1));

    // Reading past `width` returns a padding bit instead of reporting the error.
    // In a debug build this line aborts; that is the trade.
    BINCV_CHECK(!m.at(0, 25));
    BINCV_CHECK(!m.at(1, 31));

    // Writing past `width` is not stopped either, and it breaks the padding-bit
    // invariant: the pixel is readable through at(), yet countNonZero() -- which
    // only walks columns [0, width) -- never sees it.
    m.set(0, 25, true);
    BINCV_CHECK(m.at(0, 25));
    BINCV_CHECK_EQ(m.countNonZero(), 0);
    BINCV_CHECK(m.data()[0] == (uint32_t(1) << 25));

    // In-range access is unaffected by any of this.
    m.set(1, 19, true);
    BINCV_CHECK(m.at(1, 19));
    BINCV_CHECK_EQ(m.countNonZero(), 1);
}
#endif // !BINCV_DEBUG_CHECKS

// The configuration under test, printed so a passing log says which paths ran.
void reportConfiguration() {
    std::cout << "\n--- Error policy configuration ---\n";
    std::cout << "  BINCV_EXCEPTIONS_ENABLED = " << BINCV_EXCEPTIONS_ENABLED << "\n";
    std::cout << "  BINCV_DEBUG_CHECKS       = " << BINCV_DEBUG_CHECKS << "\n";

    // Both are 0/1 macros, never merely defined/undefined: `#if` on a
    // misspelled name silently yields 0, so the values are checked, not assumed.
    BINCV_CHECK(BINCV_EXCEPTIONS_ENABLED == 0 || BINCV_EXCEPTIONS_ENABLED == 1);
    BINCV_CHECK(BINCV_DEBUG_CHECKS == 0 || BINCV_DEBUG_CHECKS == 1);

#if BINCV_EXCEPTIONS_ENABLED
    // The two spellings agree: BINCV_NO_EXCEPTIONS is defined exactly when
    // exceptions are off, whether the user asked for that or the compiler was.
#  ifdef BINCV_NO_EXCEPTIONS
    BINCV_CHECK(false && "BINCV_NO_EXCEPTIONS defined while exceptions are enabled");
#  endif
#else
#  ifndef BINCV_NO_EXCEPTIONS
    BINCV_CHECK(false && "BINCV_NO_EXCEPTIONS not defined while exceptions are disabled");
#  endif
#endif
}

} // namespace

BINCV_TEST(ErrorPolicy, Configuration) { reportConfiguration(); }

#if BINCV_EXCEPTIONS_ENABLED
BINCV_TEST(ErrorPolicy, ThrowCarriesTypeAndMessage) { testThrowCarriesTypeAndMessage(); }
BINCV_TEST(ErrorPolicy, ValidationStillRejects)     { testValidationStillRejects(); }
#else
BINCV_TEST(ErrorPolicy, ThrowAbortsWithoutExceptions) {
    std::cout << "\n  exceptions disabled: BINCV_THROW's abort path is covered by"
                 " test_error_abort\n";
}
#endif

BINCV_TEST(ErrorPolicy, AssertCompilesAwayInRelease) { testAssertCompilesAwayInRelease(); }

#if !BINCV_DEBUG_CHECKS
BINCV_TEST(ErrorPolicy, AtIsUncheckedInRelease) { testAtIsUncheckedInRelease(); }
#else
BINCV_TEST(ErrorPolicy, AtIsCheckedInDebug) {
    std::cout << "\n  debug build: at()/set() are bounds-checked, so the"
                 " unchecked-access test is skipped here -- the checked"
                 " behaviour is covered by test_error_checked and"
                 " test_assert_abort\n";
}
#endif

BINCV_TEST_MAIN("Error policy tests")
