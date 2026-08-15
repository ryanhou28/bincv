#pragma once

// Minimal assertion harness so tests report failures through the process exit
// code (the previous hand-rolled tests printed to stderr but always returned 0,
// so a regression could never fail a build).
//
// @todo Replace with Google Test once it is vendored in (ROADMAP 1.2).

#include <iostream>
#include <string>

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
              << "/" << checkCount() << " checks passed\n";
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
#define BINCV_CHECK_THROWS(expr, exc)                                            \
    do {                                                                         \
        bool threw = false;                                                      \
        try { (void)(expr); } catch (const exc&) { threw = true; } catch (...) {} \
        ::bincv::test::reportCheck(threw, #expr " throws " #exc,                 \
            __FILE__, __LINE__, "no exception of expected type");                \
    } while (0)
