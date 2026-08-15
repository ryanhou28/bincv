#pragma once

/// @file test_util.hpp
/// @brief The suites' assertion and registration vocabulary, over two backends.
///
/// T1.7 asked for a Google Test migration. What is here is a hybrid, and the
/// reason is the shape of this project rather than a preference about frameworks:
///
///   - Google Test is the backend wherever it can be had. Suites are structured
///     as named cases, so a failure names a case, `--gtest_filter` works, and
///     ctest gets standard output from a standard runner.
///
///   - The built-in harness stays as the backend of the DEPENDENCY-FREE
///     configuration -- core-only, `-fno-exceptions`. That configuration exists
///     to demonstrate that binCV needs a C++17 compiler and nothing else
///     (ARCHITECTURE 2, Tier 2). Making its verification require a downloaded
///     desktop test framework would quietly retire that claim, and would put the
///     one gate this project most depends on behind a network fetch. See
///     tests/CMakeLists.txt.
///
/// @note The premise in TASKS.md T1.7 -- "GTest needs exceptions, so the Tier 2
///       configuration keeps a minimal assertion path" -- is not correct, and
///       this file does not rely on it. Measured: googletest v1.14.0 compiles
///       from source under -fno-exceptions (it detects the absence of __EXCEPTIONS
///       and sets GTEST_HAS_EXCEPTIONS to 0 itself), links, runs, reports a
///       deliberate failure, exits non-zero, and even EXPECT_DEATH works. So
///       `-DBINCV_USE_GTEST=ON` is a supported way to build that configuration.
///       The default keeps the built-in harness there for the dependency reason
///       above, not because GTest cannot compile.
///
/// The two backends run the SAME case bodies through the SAME check macros, and
/// both count checks and print the same summary line. That is deliberate: the
/// per-suite check totals stay directly comparable to the pre-migration numbers
/// (test_binMat 845, test_storage 544, test_quantMat 443), so "the test count did
/// not drop" is something a script can read rather than something to take on
/// trust. scripts/verify.sh reads exactly that line.

#include <iostream>
#include <string>
#include <vector>

// For BINCV_EXCEPTIONS_ENABLED: BINCV_CHECK_THROWS below cannot exist as written
// in a build without exceptions, and this suite has to compile in that build.
#include "bincv-cpp/core/error.hpp"

/// @def BINCV_TEST_WITH_GTEST
/// @brief 1 when the Google Test backend is in use. Defined by the build; the
///        fallback here keeps the header usable in a hand-run compile.
#ifndef BINCV_TEST_WITH_GTEST
#  define BINCV_TEST_WITH_GTEST 0
#endif

#if BINCV_TEST_WITH_GTEST
#  include <gtest/gtest.h>
#endif

namespace bincv {
namespace test {

inline int& failureCount() {
    static int failures = 0;
    return failures;
}

inline int& checkCount() {
    static int checks = 0;
    return checks;
}

/// @brief Checks the running configuration cannot evaluate, counted separately.
/// @note Exists so that a check which disappears in one configuration says so.
///       The no-exceptions build cannot run BINCV_CHECK_THROWS at all, and when
///       that expanded to nothing the two builds reported "801/801" and "845/845"
///       and both read as complete successes -- 44 validation checks were gone
///       with nothing in the log to say it. A skipped check is not a passing one.
inline int& skipCount() {
    static int skips = 0;
    return skips;
}

inline void reportSkipped(const char* expr, const char* file, int line) {
    ++skipCount();
    std::cout << "  [SKIP] " << file << ":" << line << "  " << expr
              << "  (not expressible without exceptions; covered as a death test"
                 " -- see tests/CMakeLists.txt)\n";
}

/// @brief Records one check and reports it if it failed.
/// @note This is the ONLY function that differs between the two backends, which
///       is why the check macros below are written once. Under Google Test the
///       failure is handed to the framework with the ORIGINAL file and line
///       through ADD_FAILURE_AT, so it is attributed to the assertion rather than
///       to this header, and the enclosing case fails. The counters are
///       maintained identically either way.
inline void reportCheck(bool ok, const char* expr, const char* file, int line, const std::string& note) {
    ++checkCount();
    if (ok) return;
    ++failureCount();
#if BINCV_TEST_WITH_GTEST
    ADD_FAILURE_AT(file, line) << expr << (note.empty() ? std::string() : "  (" + note + ")");
#else
    std::cerr << "  [FAIL] " << file << ":" << line << "  " << expr;
    if (!note.empty()) std::cerr << "  (" << note << ")";
    std::cerr << "\n";
#endif
}

/// @brief Prints the summary and returns the process exit code.
/// @note Printed by both backends, and by Google Test's main AFTER the run, so
///       the check total is emitted exactly once per process in every
///       configuration. scripts/verify.sh parses it.
inline int summarize(const char* suiteName) {
    const int failures = failureCount();
    std::cout << "\n" << suiteName << ": " << (checkCount() - failures)
              << "/" << checkCount() << " checks passed";
    if (skipCount() > 0) {
        std::cout << ", " << skipCount() << " skipped";
    }
    std::cout << "\n";
    if (failures > 0) {
        std::cerr << suiteName << ": " << failures << " CHECK(S) FAILED\n";
        return 1;
    }
    std::cout << suiteName << ": OK\n";
    return 0;
}

#if !BINCV_TEST_WITH_GTEST
// ---------------------------------------------------------------------------
// Built-in backend: the minimum that makes BINCV_TEST mean the same thing here
// as it does under Google Test -- named cases, registered at static init, run by
// a shared main, selectable by name.
// ---------------------------------------------------------------------------

struct Case {
    const char* name;
    void (*fn)();
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back(Case{name, fn}); }
};

/// @brief Runs every registered case, or those whose name contains `argv[1]`.
/// @note The substring filter is the same idea as --gtest_filter without the
///       wildcard grammar, so that a developer moving between configurations can
///       narrow a run the same way in both.
inline int runAll(const char* suiteName, int argc, char** argv) {
    const char* filter = (argc > 1) ? argv[1] : nullptr;
    std::cout << "=== " << suiteName << " ===\n";
    int ran = 0;
    for (const Case& c : registry()) {
        if (filter != nullptr && std::string(c.name).find(filter) == std::string::npos) continue;
        std::cout << "\n[ RUN      ] " << c.name << "\n";
        const int before = failureCount();
        c.fn();
        std::cout << (failureCount() == before ? "[       OK ] " : "[  FAILED  ] ") << c.name << "\n";
        ++ran;
    }
    if (ran == 0) {
        std::cerr << suiteName << ": no test case matched"
                  << (filter != nullptr ? std::string(" '") + filter + "'" : std::string())
                  << "\n";
        return 1;
    }
    return summarize(suiteName);
}
#endif // !BINCV_TEST_WITH_GTEST

} // namespace test
} // namespace bincv

// ---------------------------------------------------------------------------
// Checks. Identical text in both backends -- see reportCheck.
// ---------------------------------------------------------------------------

#define BINCV_CHECK(expr) \
    ::bincv::test::reportCheck((expr), #expr, __FILE__, __LINE__, "")

#define BINCV_CHECK_EQ(actual, expected)                                        \
    ::bincv::test::reportCheck((actual) == (expected), #actual " == " #expected, \
        __FILE__, __LINE__,                                                      \
        "got " + std::to_string(actual) + ", expected " + std::to_string(expected))

/// @brief Passes if evaluating `expr` throws an exception of type `exc`.
///
/// @note Guarded at the definition, not at each call site. Without exceptions the
///       try/catch below is not merely useless, it is ill-formed -- the catch(...)
///       handler was the single largest source of errors in the -fno-exceptions
///       build before T1.4 -- so every caller would otherwise have to repeat the
///       same #if.
/// @note In a build without exceptions this does NOT evaluate `expr`. The
///       validation it probes reports through BINCV_THROW, which aborts there:
///       evaluating the expression would take the whole test process down rather
///       than record a check. It reports a SKIP instead of expanding to nothing,
///       so the check is visibly absent rather than silently absent -- see
///       reportSkipped.
/// @note Skipping is not the same as not covering. Every validation site the
///       library has is also driven as its own process by tests/test_error_abort.cpp
///       through tests/expect_fatal.cmake, in every configuration, which is what
///       makes the no-exceptions build able to fail when a check is removed.
/// @note Deliberately not EXPECT_THROW even under Google Test: EXPECT_THROW does
///       not exist in a -fno-exceptions build, so routing through it would make
///       the two backends disagree about what this macro means in exactly the
///       configuration where the difference matters.
#if BINCV_EXCEPTIONS_ENABLED
#define BINCV_CHECK_THROWS(expr, exc)                                            \
    do {                                                                         \
        bool threw = false;                                                      \
        try { (void)(expr); } catch (const exc&) { threw = true; } catch (...) {} \
        ::bincv::test::reportCheck(threw, #expr " throws " #exc,                 \
            __FILE__, __LINE__, "no exception of expected type");                \
    } while (0)
#else
#define BINCV_CHECK_THROWS(expr, exc)                                            \
    ::bincv::test::reportSkipped(#expr " throws " #exc, __FILE__, __LINE__)
#endif

// ---------------------------------------------------------------------------
// Case registration
//
// BINCV_TEST(Suite, Case) { ... } is TEST(Suite, Case) under Google Test and a
// registered free function otherwise. Suites instantiated over several word types
// register one case per instantiation -- BinMat.Contract_uint8_t and friends --
// rather than using TYPED_TEST, so that the case NAMES are also identical across
// the two backends and a filter written for one works in the other.
// ---------------------------------------------------------------------------

#if BINCV_TEST_WITH_GTEST
#  define BINCV_TEST(suite, name) TEST(suite, name)
#else
#  define BINCV_TEST(suite, name)                                               \
      static void bincvTestBody_##suite##_##name();                             \
      static const ::bincv::test::Registrar bincvTestReg_##suite##_##name(      \
          #suite "." #name, &bincvTestBody_##suite##_##name);                   \
      static void bincvTestBody_##suite##_##name()
#endif

/// @def BINCV_TEST_MAIN(suiteLabel)
/// @brief The suite's entry point. One line per test binary, both backends.
/// @note Under Google Test the process exit code is the framework's, except that
///       a failure the framework did not see -- there is none by construction,
///       since reportCheck routes through ADD_FAILURE_AT -- would still fail the
///       run. Cheap, and it means the summary line and the exit code can never
///       disagree.
#if BINCV_TEST_WITH_GTEST
#  define BINCV_TEST_MAIN(suiteLabel)                                           \
      int main(int argc, char** argv) {                                         \
          ::testing::InitGoogleTest(&argc, argv);                               \
          const int rc = RUN_ALL_TESTS();                                       \
          const int summaryRc = ::bincv::test::summarize(suiteLabel);           \
          return (rc != 0 || summaryRc != 0) ? 1 : 0;                           \
      }
#else
#  define BINCV_TEST_MAIN(suiteLabel)                                           \
      int main(int argc, char** argv) {                                         \
          return ::bincv::test::runAll(suiteLabel, argc, argv);                 \
      }
#endif
