#pragma once

// Minimal assertion harness so tests report failures through the process exit
// code (the previous hand-rolled tests printed to stderr but always returned 0,
// so a regression could never fail a build).
//
// @todo Replace with Google Test once it is vendored in (ROADMAP 1.2).

#include <iostream>
#include <string>

// For BINCV_EXCEPTIONS_ENABLED: BINCV_CHECK_THROWS below cannot exist as written
// in a build without exceptions, and this suite has to compile in that build.
#include "bincv-cpp/core/error.hpp"

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

inline void reportCheck(bool ok, const char* expr, const char* file, int line, const std::string& note) {
    ++checkCount();
    if (ok) return;
    ++failureCount();
    std::cerr << "  [FAIL] " << file << ":" << line << "  " << expr;
    if (!note.empty()) std::cerr << "  (" << note << ")";
    std::cerr << "\n";
}

/// @brief Prints the summary and returns the process exit code.
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

} // namespace test
} // namespace bincv

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
